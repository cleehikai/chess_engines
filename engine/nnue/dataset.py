"""Load generated .tsv shards into a torch Dataset of packed HalfKP features.

Each sample -> (stm_feature_indices, nstm_feature_indices, target_cp).
The target blends Stockfish eval and game result via `lambda_` (the nnue-pytorch
lever): target = lambda_ * eval + (1 - lambda_) * result_scaled.

We precompute feature indices once (CPU-bound but one-time) and cache them as
Python lists; a custom collate packs them into EmbeddingBag (idx, offset) form.
"""
from __future__ import annotations

import glob
import math

import chess
import torch
from torch.utils.data import Dataset

from engine.nnue.features import both_perspectives

# A win (+1) corresponds to this many centipawns when mixing result into target.
RESULT_SCALE = 600.0


def _cp_to_wdl(cp: float) -> float:
    """Map centipawns to a win-probability-ish target in [-1, 1] via tanh."""
    return math.tanh(cp / 400.0)


class NNUEDataset(Dataset):
    def __init__(self, shard_glob: str, lambda_: float = 1.0,
                 max_samples: int | None = None, as_wdl: bool = True,
                 style_profile=None):
        """If `style_profile` (a StyleProfile) is given, each sample also carries
        a *weight* in [~0.3, ~3] reflecting how characteristic the position is of
        that style (see style_weight.py). The fine-tuner uses these weights so
        the net leans toward the style WITHOUT distorting the eval targets --
        winning stays the objective; style only reweights which positions the
        net cares most about getting right.
        """
        self.lambda_ = lambda_
        self.as_wdl = as_wdl
        self.style_profile = style_profile
        # samples: (stm_idx, nstm_idx, target, weight)
        self.samples: list[tuple[list[int], list[int], float, float]] = []
        weigh = None
        if style_profile is not None:
            from engine.nnue.style_weight import position_style_weight
            weigh = position_style_weight
        paths = sorted(glob.glob(shard_glob))
        if not paths:
            raise FileNotFoundError(f"no shards match {shard_glob!r}")
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    fen, cp_s, res_s = parts
                    cp = float(cp_s)
                    result = float(res_s)  # white-relative {1,0,-1}
                    board = chess.Board(fen)
                    stm, nstm = both_perspectives(board)
                    # Convert white-relative labels to side-to-move perspective.
                    if board.turn == chess.BLACK:
                        cp = -cp
                        result = -result
                    target = self._make_target(cp, result)
                    weight = weigh(board, style_profile) if weigh else 1.0
                    self.samples.append((stm, nstm, target, weight))
                    if max_samples and len(self.samples) >= max_samples:
                        return

    def _make_target(self, cp: float, result: float) -> float:
        if self.as_wdl:
            eval_t = _cp_to_wdl(cp)
            res_t = result  # already in [-1,1]
        else:
            eval_t = cp
            res_t = result * RESULT_SCALE
        return self.lambda_ * eval_t + (1 - self.lambda_) * res_t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    """Pack a list of (stm, nstm, target, weight) into EmbeddingBag inputs.
    Returns (stm_idx, stm_off, nstm_idx, nstm_off, targets, weights)."""
    stm_idx, stm_off = [], []
    nstm_idx, nstm_off = [], []
    targets, weights = [], []
    s_pos = n_pos = 0
    for sample in batch:
        stm, nstm, tgt = sample[0], sample[1], sample[2]
        weight = sample[3] if len(sample) > 3 else 1.0
        stm_off.append(s_pos)
        nstm_off.append(n_pos)
        stm_idx.extend(stm)
        nstm_idx.extend(nstm)
        s_pos += len(stm)
        n_pos += len(nstm)
        targets.append(tgt)
        weights.append(weight)
    return (
        torch.tensor(stm_idx, dtype=torch.long),
        torch.tensor(stm_off, dtype=torch.long),
        torch.tensor(nstm_idx, dtype=torch.long),
        torch.tensor(nstm_off, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
        torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
    )
