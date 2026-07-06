"""Fast NNUE inference for use inside alpha-beta search.

Loads a trained checkpoint and evaluates positions with plain NumPy (no torch
autograd/per-call overhead). Returns centipawns from the side-to-move
perspective, matching the convention used by engine/search.py and evaluate.py.

The model is trained in WDL space (pred = tanh(raw) in [-1,1]); we invert the
_cp_to_wdl mapping (tanh(cp/400)) to recover centipawns:  cp = 400 * atanh(pred).

This is a straightforward (non-incremental) forward pass. It's already fast
because only ~30 features are active per perspective, so the feature-transformer
step is a sum of ~30 rows. Incremental accumulator updates can come later if we
need more nodes/sec.
"""
from __future__ import annotations

import math

import chess
import numpy as np
import torch

from engine.nnue.features import both_perspectives
from engine.nnue.model import NNUE

_WDL_CLAMP = 0.999  # avoid atanh(+-1) -> inf


class NNUEEvaluator:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        model = NNUE()
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Extract weights to NumPy for a dependency-light forward pass.
        self.ft_w = model.ft.weight.detach().numpy().astype(np.float32)   # (F, HL)
        self.ft_b = model.ft_bias.detach().numpy().astype(np.float32)     # (HL,)
        self.l1_w = model.l1.weight.detach().numpy().astype(np.float32)   # (32, 2HL)
        self.l1_b = model.l1.bias.detach().numpy().astype(np.float32)
        self.l2_w = model.l2.weight.detach().numpy().astype(np.float32)
        self.l2_b = model.l2.bias.detach().numpy().astype(np.float32)
        self.out_w = model.out.weight.detach().numpy().astype(np.float32)  # (1,32)
        self.out_b = model.out.bias.detach().numpy().astype(np.float32)

    def eval_cp(self, board: chess.Board) -> int:
        """Centipawns from the side-to-move perspective (int)."""
        if board.is_checkmate():
            return -30000
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves()):
            return 0

        stm, nstm = both_perspectives(board)
        acc_stm = self.ft_w[stm].sum(axis=0) + self.ft_b if stm else self.ft_b.copy()
        acc_nstm = self.ft_w[nstm].sum(axis=0) + self.ft_b if nstm else self.ft_b.copy()

        x = np.concatenate([acc_stm, acc_nstm])
        x = np.clip(x, 0.0, 1.0)
        x = np.clip(self.l1_w @ x + self.l1_b, 0.0, 1.0)
        x = np.clip(self.l2_w @ x + self.l2_b, 0.0, 1.0)
        raw = float(self.out_w @ x + self.out_b)

        wdl = math.tanh(raw)
        wdl = max(-_WDL_CLAMP, min(_WDL_CLAMP, wdl))
        cp = 400.0 * math.atanh(wdl)
        return int(round(cp))


def make_nnue_eval_fn(checkpoint: str, device: str = "cpu"):
    """Return an eval fn (board -> cp) suitable for Searcher(eval_fn=...) and for
    wrapping with style shaping. Note: style shaping expects (score, features);
    for NNUE we provide a features-producing wrapper in board.py when enabled."""
    ev = NNUEEvaluator(checkpoint, device)
    return ev.eval_cp
