"""The NNUE network (PyTorch).

Architecture (classic small NNUE):
    feature transformer:  40960 -> HL  (shared weights, run per perspective)
    concat side-to-move + other:  2*HL
    clipped ReLU
    dense: 2*HL -> 32 -> 32 -> 1

The feature transformer is a big sparse linear layer; because only ~30 inputs
are active per perspective, we implement it as an embedding-sum for training
efficiency. Output is in centipawns (side-to-move perspective).

Device-agnostic: nothing here assumes CPU or CUDA. train.py picks the device.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from engine.nnue.features import NUM_FEATURES

HL = 256  # feature-transformer hidden size per perspective


class NNUE(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES, hl: int = HL):
        super().__init__()
        self.hl = hl
        # Feature transformer as an embedding table: summing the rows of the
        # active features == a sparse matrix-vector product. +1 bias row.
        self.ft = nn.EmbeddingBag(num_features, hl, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(hl))

        self.l1 = nn.Linear(2 * hl, 32)
        self.l2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.ft.weight, std=0.01)
        for lin in (self.l1, self.l2, self.out):
            nn.init.kaiming_uniform_(lin.weight, a=1.0)
            nn.init.zeros_(lin.bias)

    def forward(self, stm_idx: torch.Tensor, stm_off: torch.Tensor,
                nstm_idx: torch.Tensor, nstm_off: torch.Tensor) -> torch.Tensor:
        """Args are the packed sparse representation from a batch:
          *_idx: 1-D LongTensor of all active feature indices (concatenated)
          *_off: 1-D LongTensor of per-sample start offsets (EmbeddingBag format)
        Returns a (batch, 1) tensor of centipawn evals (stm perspective).
        """
        acc_stm = self.ft(stm_idx, stm_off) + self.ft_bias
        acc_nstm = self.ft(nstm_idx, nstm_off) + self.ft_bias

        x = torch.cat([acc_stm, acc_nstm], dim=1)
        x = torch.clamp(x, 0.0, 1.0)          # clipped ReLU, NNUE-style
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)
        return self.out(x)

    @torch.no_grad()
    def eval_board_indices(self, stm_feats: list[int], nstm_feats: list[int],
                           device: torch.device) -> float:
        """Convenience single-position eval (slow path, for testing/parity)."""
        stm_idx = torch.tensor(stm_feats, dtype=torch.long, device=device)
        nstm_idx = torch.tensor(nstm_feats, dtype=torch.long, device=device)
        off = torch.tensor([0], dtype=torch.long, device=device)
        out = self.forward(stm_idx, off, nstm_idx, off)
        return float(out.item())
