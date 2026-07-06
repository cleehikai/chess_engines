"""Numba-accelerated NNUE evaluator for alpha-beta search (the fast path).

Combines two wins:

* **Numba-JIT kernels** (``kernels.py``) for the feature-transformer sum, int
  quantization, and the dense head. These compile to native SIMD loops, which is
  where pure NumPy loses — especially on the small integer matmuls, for which
  NumPy has no optimized kernel. ~13x faster than the NumPy path even without
  incremental reuse.
* **Incremental accumulators.** The 256-wide per-perspective accumulator is kept
  on a stack synced to the board's move stack; between a node and its child only
  1-4 feature rows change, so we add/subtract those rows (~2us) instead of
  re-summing ~30 (a king move refreshes that side, still cheap). The board-sync
  logic here is the same design verified to exact float parity in tests.

Drop-in for ``NNUEEvaluator``: same ``eval_cp(board) -> int`` API. Terminal /
draw checks are intentionally omitted — the search generates legal moves itself
and applies its own draw guards before ever calling eval, so repeating them here
was pure overhead.
"""
from __future__ import annotations

import math

import chess
import numpy as np
import torch

from engine.nnue import kernels as K
from engine.nnue.features import (
    NUM_PIECE_KINDS,
    NUM_SQ,
    _TYPE_ORDER,
    active_features,
)
from engine.nnue.kernels import QA, QW
from engine.nnue.model import NNUE

_WDL_CLAMP = 0.999
_STRIDE = NUM_PIECE_KINDS
_KING_BLOCK = NUM_SQ * NUM_PIECE_KINDS
_MAX_DELTA = 6  # max feature rows changed by one move (castling: 2 add + rook, etc.)


class FastNNUE:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        K.warmup()  # compile kernels once (cached to disk after first run)

        model = NNUE()
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        self.ft_w = np.ascontiguousarray(
            model.ft.weight.detach().numpy().astype(np.float32))   # (F, HL)
        self.ft_b = np.ascontiguousarray(
            model.ft_bias.detach().numpy().astype(np.float32))     # (HL,)
        self.hl = self.ft_b.shape[0]

        l1_w = model.l1.weight.detach().numpy().astype(np.float32)
        l1_b = model.l1.bias.detach().numpy().astype(np.float32)
        l2_w = model.l2.weight.detach().numpy().astype(np.float32)
        l2_b = model.l2.bias.detach().numpy().astype(np.float32)
        out_w = model.out.weight.detach().numpy().astype(np.float32)
        out_b = model.out.bias.detach().numpy().astype(np.float32)

        self.l1_w = np.ascontiguousarray(np.round(l1_w * QW).astype(np.int32))
        self.l1_b = np.ascontiguousarray(np.round(l1_b * QA * QW).astype(np.int32))
        self.l2_w = np.ascontiguousarray(np.round(l2_w * QW).astype(np.int32))
        self.l2_b = np.ascontiguousarray(np.round(l2_b * QA * QW).astype(np.int32))
        self.out_w = np.ascontiguousarray(np.round(out_w * QW).astype(np.int32))
        self.out_b = np.ascontiguousarray(np.round(out_b * QA * QW).astype(np.int32))

        # Reusable scratch buffers (avoid per-eval allocation).
        self._xq = np.zeros(2 * self.hl, dtype=np.int32)
        self._added = np.zeros(_MAX_DELTA, dtype=np.int32)
        self._removed = np.zeros(_MAX_DELTA, dtype=np.int32)
        self._o_stm = np.zeros(self.hl, dtype=np.float32)
        self._o_nstm = np.zeros(self.hl, dtype=np.float32)

        # Accumulator + sync state (see _sync).
        self._acc: dict[bool, np.ndarray] = {}
        self._king: dict[bool, int] = {}
        self._occupied: int = 0
        self._stack: list[tuple] = []
        self._moves: list[chess.Move] = []

    # -- public API -----------------------------------------------------------
    def eval_cp(self, board: chess.Board) -> int:
        """Fused from-scratch eval straight from bitboards (the fast path).

        Measured faster than the incremental accumulator for this search's
        push/eval/pop access pattern: incremental only wins when the accumulator
        is carried across a deep chain without an immediate pop, whereas the
        search evaluates one position per push. So we recompute from bitboards
        each call using the fused JIT kernel — no per-eval board pop/push and no
        256-float stack copy.
        """
        K.accumulate_from_bitboards(
            self.ft_w, self.ft_b,
            np.uint64(board.pawns), np.uint64(board.knights),
            np.uint64(board.bishops), np.uint64(board.rooks),
            np.uint64(board.queens), np.uint64(board.occupied_co[chess.WHITE]),
            board.king(chess.WHITE), board.king(chess.BLACK),
            self._o_stm, self._o_nstm, board.turn == chess.WHITE)
        K.quantize_acc(self._o_stm, self._o_nstm, self._xq)
        raw = K.dense_forward(self._xq, self.l1_w, self.l1_b, self.l2_w,
                              self.l2_b, self.out_w, self.out_b)
        wdl = math.tanh(raw)
        wdl = max(-_WDL_CLAMP, min(_WDL_CLAMP, wdl))
        cp = 400.0 * math.atanh(wdl)
        return int(round(cp))

    def eval_cp_incremental(self, board: chess.Board) -> int:
        """Incremental-accumulator variant (kept for reference / deep-chain use).
        Correct but slower than eval_cp for the search's leaf-eval pattern."""
        self._sync(board)
        K.quantize_acc(self._acc[board.turn], self._acc[not board.turn], self._xq)
        raw = K.dense_forward(self._xq, self.l1_w, self.l1_b, self.l2_w,
                              self.l2_b, self.out_w, self.out_b)
        wdl = math.tanh(raw)
        wdl = max(-_WDL_CLAMP, min(_WDL_CLAMP, wdl))
        cp = 400.0 * math.atanh(wdl)
        return int(round(cp))

    # -- board sync -----------------------------------------------------------
    def _sync(self, board: chess.Board) -> None:
        if not self._acc:
            self._refresh(board)
            return

        bmoves = board.move_stack
        bn = len(bmoves)
        mn = len(self._moves)

        if bn == mn + 1 and self._prefix_matches(bmoves, mn):
            self._apply_push(board)
            return
        if bn == mn - 1 and self._stack and self._prefix_matches(bmoves, bn):
            self._apply_pop()
            self._moves.pop()
            return
        if bn == mn and self._prefix_matches(bmoves, bn) \
                and self._position_synced(board):
            return
        self._refresh(board)

    def _position_synced(self, board: chess.Board) -> bool:
        return (self._king[chess.WHITE] == board.king(chess.WHITE)
                and self._king[chess.BLACK] == board.king(chess.BLACK)
                and self._occupied == board.occupied)

    def _prefix_matches(self, bmoves, n: int) -> bool:
        for i in range(n):
            if self._moves[i] != bmoves[i]:
                return False
        return True

    def _refresh(self, board: chess.Board) -> None:
        for persp in (chess.WHITE, chess.BLACK):
            self._acc[persp] = self._compute_acc(board, persp)
            self._king[persp] = board.king(persp)
        self._occupied = board.occupied
        self._stack.clear()
        self._moves = list(board.move_stack)

    def _compute_acc(self, board: chess.Board, persp: bool) -> np.ndarray:
        feats = active_features(board, persp)
        if feats:
            return K.accumulate(self.ft_w, self.ft_b,
                                np.asarray(feats, dtype=np.int32))
        return self.ft_b.copy()

    def _apply_pop(self) -> None:
        wacc, bacc, wk, bk, occ = self._stack.pop()
        self._acc[chess.WHITE] = wacc
        self._acc[chess.BLACK] = bacc
        self._king[chess.WHITE] = wk
        self._king[chess.BLACK] = bk
        self._occupied = occ

    def _apply_push(self, board: chess.Board) -> None:
        move = board.peek()
        self._stack.append((
            self._acc[chess.WHITE].copy(),
            self._acc[chess.BLACK].copy(),
            self._king[chess.WHITE],
            self._king[chess.BLACK],
            self._occupied,
        ))
        board.pop()
        try:
            self._update_for_move(board, move)
        finally:
            board.push(move)
        self._occupied = board.occupied
        self._moves.append(move)

    # -- incremental feature delta --------------------------------------------
    def _update_for_move(self, pre: chess.Board, move: chess.Move) -> None:
        """`pre` is the position BEFORE `move`; update both accumulators."""
        mover = pre.turn
        piece = pre.piece_at(move.from_square)
        assert piece is not None, "incremental update on an illegal move"
        king_moved = piece.piece_type == chess.KING

        # (square, piece_type, color) removals/additions in board terms.
        removed: list[tuple[int, int, bool]] = [
            (move.from_square, piece.piece_type, piece.color)]
        added: list[tuple[int, int, bool]] = []

        if pre.is_en_passant(move):
            cap_sq = move.to_square + (-8 if mover == chess.WHITE else 8)
            removed.append((cap_sq, chess.PAWN, not mover))
        else:
            captured = pre.piece_at(move.to_square)
            if captured is not None:
                removed.append((move.to_square, captured.piece_type, captured.color))

        arrived_type = move.promotion if move.promotion else piece.piece_type
        added.append((move.to_square, arrived_type, piece.color))

        if pre.is_castling(move):
            rank = chess.square_rank(move.from_square)
            if chess.square_file(move.to_square) == 6:  # king-side
                rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)
            else:  # queen-side
                rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)
            removed.append((rook_from, chess.ROOK, mover))
            added.append((rook_to, chess.ROOK, mover))

        for persp in (chess.WHITE, chess.BLACK):
            if king_moved and piece.color == persp:
                # Anchor changed -> recompute this side post-move (still cheap).
                self._king[persp] = move.to_square
                pre.push(move)
                try:
                    self._acc[persp] = self._compute_acc(pre, persp)
                finally:
                    pre.pop()
                continue

            acc = self._acc[persp]
            block = _orient_king_block(self._king[persp], persp)
            n_rem = 0
            for sq, ptype, color in removed:
                idx = _feat_idx(block, sq, ptype, color, persp)
                if idx is not None:
                    self._removed[n_rem] = idx
                    n_rem += 1
            n_add = 0
            for sq, ptype, color in added:
                idx = _feat_idx(block, sq, ptype, color, persp)
                if idx is not None:
                    self._added[n_add] = idx
                    n_add += 1
            K.apply_deltas(acc, self.ft_w, self._added, n_add,
                           self._removed, n_rem)


def _orient_king_block(king_sq: int, persp: bool) -> int:
    oriented = king_sq if persp == chess.WHITE else chess.square_mirror(king_sq)
    return oriented * _KING_BLOCK


def _feat_idx(king_block_base: int, sq: int, ptype: int, color: bool,
              persp: bool) -> int | None:
    if ptype == chess.KING:
        return None
    base = _TYPE_ORDER[ptype]
    pidx = base if color == persp else base + 5
    osq = sq if persp == chess.WHITE else chess.square_mirror(sq)
    return king_block_base + osq * _STRIDE + pidx
