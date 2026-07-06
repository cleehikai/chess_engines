"""Numba-JIT numeric kernels for NNUE inference.

These compile to native SIMD loops, so the two things that are *slow* in pure
NumPy — the feature-transformer gather/sum and the small integer dense matmuls —
become fast here. This is where int quantization actually pays off: Numba emits
real integer kernels, unlike NumPy which has no optimized int matmul.

All kernels are plain functions over contiguous arrays so they JIT cleanly and
cache to disk (compiled once, reused across runs). The higher-level evaluator in
``fast_infer.py`` owns the model weights and the board-sync logic; this module is
pure arithmetic.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Fixed-point scales (see fast_infer.py). Kept here so kernels are self-contained.
QA = 255
QW = 1024


@njit(cache=True, fastmath=True)
def accumulate(ft_w, ft_b, feats):
    """Sum the active feature rows of the transformer table, add bias.

    ft_w: (F, HL) float32; ft_b: (HL,) float32; feats: (k,) int32 indices.
    Returns (HL,) float32 accumulator. This is the from-scratch path (used on a
    refresh / king move); the incremental add/sub kernels below handle deltas.
    """
    hl = ft_b.shape[0]
    out = np.empty(hl, dtype=np.float32)
    for j in range(hl):
        out[j] = ft_b[j]
    for i in range(feats.shape[0]):
        row = feats[i]
        for j in range(hl):
            out[j] += ft_w[row, j]
    return out


@njit(cache=True, fastmath=True)
def accumulate_from_bitboards(ft_w, ft_b, pawns, knights, bishops, rooks,
                              queens, white_occ, wk, bk, out_stm, out_nstm,
                              white_to_move):
    """Compute BOTH perspective accumulators directly from piece bitboards.

    Fuses HalfKP feature extraction and the transformer sum into one compiled
    pass — no Python piece_map. Piece-type bitboards are the union over both
    colors (python-chess ``board.pawns`` etc.); ``white_occ`` is
    ``board.occupied_co[WHITE]`` so we can split color by a bit test. ``wk``/
    ``bk`` are the king squares. Kings are excluded as features (they're the
    anchors). Writes into preallocated out_stm / out_nstm (HL,) float32 buffers,
    side-to-move first.

    Feature index (per perspective):
        king_sq_oriented*640 + piece_sq_oriented*10 + piece_kind
    piece_kind: own P,N,B,R,Q = 0..4 ; opp = 5..9. Black perspective mirrors
    squares vertically (sq ^ 56).
    """
    hl = ft_b.shape[0]
    # White-perspective anchor (no mirror); black-perspective anchor (mirror).
    w_base = wk * 640
    b_base = (bk ^ 56) * 640

    # Accumulate into white-persp and black-persp buffers, then assign to
    # stm/nstm at the end.
    wacc = out_stm if white_to_move else out_nstm
    bacc = out_nstm if white_to_move else out_stm
    for j in range(hl):
        wacc[j] = ft_b[j]
        bacc[j] = ft_b[j]

    # Iterate the 5 encoded piece types (pawn..queen -> base 0..4).
    for base in range(5):
        if base == 0:
            bb = pawns
        elif base == 1:
            bb = knights
        elif base == 2:
            bb = bishops
        elif base == 3:
            bb = rooks
        else:
            bb = queens
        while bb != np.uint64(0):
            sq = _trailing_zeros(bb)
            bb = bb & (bb - np.uint64(1))  # clear lowest set bit
            is_white = (white_occ >> np.uint64(sq)) & np.uint64(1)
            # White perspective: own(base) if white piece else opp(base+5).
            w_kind = base if is_white == np.uint64(1) else base + 5
            w_row = w_base + sq * 10 + w_kind
            # Black perspective: own(base) if black piece else opp(base+5),
            # square mirrored.
            b_kind = base if is_white == np.uint64(0) else base + 5
            b_row = b_base + (sq ^ 56) * 10 + b_kind
            for j in range(hl):
                wacc[j] += ft_w[w_row, j]
                bacc[j] += ft_w[b_row, j]


@njit(cache=True, fastmath=True)
def _trailing_zeros(x):
    """Index (0..63, plain int) of the lowest set bit of a uint64 (x != 0)."""
    n = 0
    one = np.uint64(1)
    while (x & one) == np.uint64(0):
        x = x >> one
        n += 1
    return n


@njit(cache=True, fastmath=True)
def add_feature(acc, ft_w, row):
    """acc += ft_w[row] (in place)."""
    for j in range(acc.shape[0]):
        acc[j] += ft_w[row, j]


@njit(cache=True, fastmath=True)
def sub_feature(acc, ft_w, row):
    """acc -= ft_w[row] (in place)."""
    for j in range(acc.shape[0]):
        acc[j] -= ft_w[row, j]


@njit(cache=True, fastmath=True)
def apply_deltas(acc, ft_w, added, n_add, removed, n_rem):
    """Apply a batch of add/remove feature-row deltas to acc in place.

    added/removed are int32 buffers; only the first n_add / n_rem entries are
    used (avoids per-call allocation from the caller).
    """
    hl = acc.shape[0]
    for i in range(n_rem):
        row = removed[i]
        for j in range(hl):
            acc[j] -= ft_w[row, j]
    for i in range(n_add):
        row = added[i]
        for j in range(hl):
            acc[j] += ft_w[row, j]


@njit(cache=True, fastmath=True)
def quantize_acc(acc_stm, acc_nstm, out):
    """Clipped-ReLU both accumulators into [0,QA] int16, concatenated into out.

    out: preallocated (2*HL,) int32 buffer.
    """
    hl = acc_stm.shape[0]
    for j in range(hl):
        v = acc_stm[j]
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[j] = np.int32(v * QA + 0.5)
    for j in range(hl):
        v = acc_nstm[j]
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        out[hl + j] = np.int32(v * QA + 0.5)


@njit(cache=True, fastmath=True)
def dense_forward(xq, l1_w, l1_b, l2_w, l2_b, out_w, out_b):
    """Quantized dense head: (2HL) -> 32 -> 32 -> 1, clipped-ReLU, fixed point.

    xq: (2HL,) int32 activations in [0,QA]. Weights are int32 (scaled by QW),
    biases int32 (scaled by QA*QW). Returns the float logit (de-quantized).
    """
    half = QW // 2

    n1 = l1_w.shape[0]
    a1 = np.empty(n1, dtype=np.int32)
    for r in range(n1):
        acc = l1_b[r]
        for c in range(xq.shape[0]):
            acc += l1_w[r, c] * xq[c]
        v = (acc + half) // QW
        if v < 0:
            v = 0
        elif v > QA:
            v = QA
        a1[r] = v

    n2 = l2_w.shape[0]
    a2 = np.empty(n2, dtype=np.int32)
    for r in range(n2):
        acc = l2_b[r]
        for c in range(n1):
            acc += l2_w[r, c] * a1[c]
        v = (acc + half) // QW
        if v < 0:
            v = 0
        elif v > QA:
            v = QA
        a2[r] = v

    acc_out = out_b[0]
    for c in range(n2):
        acc_out += out_w[0, c] * a2[c]
    return acc_out / (QA * QW)


def warmup():
    """Force compilation of all kernels (one-time JIT cost) with tiny inputs."""
    ftw = np.zeros((4, 8), np.float32)
    ftb = np.zeros(8, np.float32)
    feats = np.array([0, 1], np.int32)
    acc = accumulate(ftw, ftb, feats)
    add_feature(acc, ftw, 0)
    sub_feature(acc, ftw, 0)
    buf = np.zeros(2, np.int32)
    apply_deltas(acc, ftw, buf, 0, buf, 0)
    o1 = np.zeros(8, np.float32); o2 = np.zeros(8, np.float32)
    accumulate_from_bitboards(ftw, ftb, np.uint64(0), np.uint64(0), np.uint64(0),
                              np.uint64(0), np.uint64(0), np.uint64(0),
                              np.int64(4), np.int64(60), o1, o2, True)
    xq = np.zeros(16, np.int32)
    quantize_acc(acc, acc, xq)
    l1w = np.zeros((2, 16), np.int32); l1b = np.zeros(2, np.int32)
    l2w = np.zeros((2, 2), np.int32); l2b = np.zeros(2, np.int32)
    ow = np.zeros((1, 2), np.int32); ob = np.zeros(1, np.int32)
    dense_forward(xq, l1w, l1b, l2w, l2b, ow, ob)
