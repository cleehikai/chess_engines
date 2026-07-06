"""HalfKP feature encoding for NNUE.

HalfKP is the classic Stockfish NNUE feature set. For each side ("perspective"),
a feature is the tuple (own_king_square, piece_square, piece_type_and_color).
The board is encoded from BOTH perspectives; the network runs the same feature
transformer over each and concatenates (own-to-move first).

Index layout (per perspective):
    index = king_sq * (64 * 10) + piece_sq * 10 + piece_index
where piece_index in [0,10) encodes (piece_type, is_own_color):
    own P,N,B,R,Q  = 0..4     (own king is NOT a feature; it's the anchor)
    opp P,N,B,R,Q  = 5..9
Kings are excluded as *pieces* (there are always exactly two and one is the
anchor); the enemy king is also excluded to keep the set standard HalfKP-lite.

NUM_FEATURES = 64 (king) * 64 (piece sq) * 10 (piece kinds) = 40960 per side.

This is a pure-Python encoder used both for training-data generation and, in a
vectorized form, at inference time. It is intentionally simple and correct;
speed at inference comes from incremental updates in infer.py later.
"""
from __future__ import annotations

import chess

NUM_PIECE_KINDS = 10
NUM_SQ = 64
NUM_FEATURES = NUM_SQ * NUM_SQ * NUM_PIECE_KINDS  # 40960

# piece_type (1..5 for P..Q) -> base offset 0..4
_TYPE_ORDER = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def _piece_index(piece: chess.Piece, perspective: chess.Color) -> int | None:
    """Return the 0..9 piece-kind index for this piece from `perspective`, or
    None if the piece is a king (kings aren't encoded as features here)."""
    if piece.piece_type == chess.KING:
        return None
    base = _TYPE_ORDER[piece.piece_type]
    is_own = piece.color == perspective
    return base if is_own else base + 5


def _orient(square: int, perspective: chess.Color) -> int:
    """Mirror the square vertically for the black perspective so both sides see
    the board 'from their own side'. White perspective is unchanged."""
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def active_features(board: chess.Board, perspective: chess.Color) -> list[int]:
    """Return the list of active feature indices for one perspective."""
    king_sq = board.king(perspective)
    if king_sq is None:
        return []
    oriented_king = _orient(king_sq, perspective)
    feats: list[int] = []
    king_base = oriented_king * (NUM_SQ * NUM_PIECE_KINDS)
    for square, piece in board.piece_map().items():
        pidx = _piece_index(piece, perspective)
        if pidx is None:
            continue
        osq = _orient(square, perspective)
        feats.append(king_base + osq * NUM_PIECE_KINDS + pidx)
    return feats


def both_perspectives(board: chess.Board) -> tuple[list[int], list[int]]:
    """Return (stm_features, nstm_features) — side-to-move first.

    The network concatenates [own_accumulator, opp_accumulator] in this order so
    the evaluation is always from the side-to-move's point of view.

    Optimized single pass over the piece bitboards (avoids building the
    piece_map dict twice). Both perspectives are computed together.
    """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        stm = board.turn
        return active_features(board, stm), active_features(board, not stm)

    # King anchors (oriented) and their base offsets, per perspective.
    w_base = wk * (NUM_SQ * NUM_PIECE_KINDS)                      # white persp
    b_base = chess.square_mirror(bk) * (NUM_SQ * NUM_PIECE_KINDS)  # black persp

    white_feats: list[int] = []
    black_feats: list[int] = []
    stride = NUM_PIECE_KINDS

    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        base = _TYPE_ORDER[pt]
        # White pieces: own(0..4) for white persp, opp(5..9) for black persp.
        for sq in board.pieces(pt, chess.WHITE):
            white_feats.append(w_base + sq * stride + base)
            black_feats.append(b_base + chess.square_mirror(sq) * stride + base + 5)
        # Black pieces: opp for white persp, own for black persp.
        for sq in board.pieces(pt, chess.BLACK):
            white_feats.append(w_base + sq * stride + base + 5)
            black_feats.append(b_base + chess.square_mirror(sq) * stride + base)

    if board.turn == chess.WHITE:
        return white_feats, black_feats
    return black_feats, white_feats
