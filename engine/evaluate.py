"""Hand-crafted evaluation (Phase 1).

Returns both a scalar score AND a set of positional *features* so the style
shaping layer (style/shaping.py) can nudge move choice without re-deriving the
position. All scores are in centipawns from the side-to-move's perspective at
the call site convention documented on `evaluate`.

This is deliberately simple and fast: material + piece-square tables + a few
cheap positional terms. It will later be swapped for the NNUE eval (Phase 3),
but the *feature* dict it produces is the stable contract the style layer uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

# --- Material values (centipawns) ---------------------------------------------
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # king has no material value; safety handled separately
}

# --- Piece-square tables (from White's perspective, a1 = index 0) -------------
# Values are small nudges toward good squares. Mirror for Black.
# fmt: off
_PST_PAWN = [
     0,   0,   0,   0,   0,   0,   0,   0,
     5,  10,  10, -20, -20,  10,  10,   5,
     5,  -5, -10,   0,   0, -10,  -5,   5,
     0,   0,   0,  20,  20,   0,   0,   0,
     5,   5,  10,  25,  25,  10,   5,   5,
    10,  10,  20,  30,  30,  20,  10,  10,
    50,  50,  50,  50,  50,  50,  50,  50,
     0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_KNIGHT = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -30,   5,  10,  15,  15,  10,   5, -30,
   -30,   0,  15,  20,  20,  15,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -40, -20,   0,   0,   0,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50,
]
_PST_BISHOP = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   5,   0,   0,   0,   0,   5, -10,
   -10,  10,  10,  10,  10,  10,  10, -10,
   -10,   0,  10,  10,  10,  10,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10, -10, -10, -10, -10, -20,
]
_PST_ROOK = [
     0,   0,   0,   5,   5,   0,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     5,  10,  10,  10,  10,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
]
_PST_QUEEN = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -10,   5,   5,   5,   5,   5,   0, -10,
     0,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
   -10,   0,   5,   5,   5,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20,
]
_PST_KING_MID = [
    20,  30,  10,   0,   0,  10,  30,  20,
    20,  20,   0,   0,   0,   0,  20,  20,
   -10, -20, -20, -20, -20, -20, -20, -10,
   -20, -30, -30, -40, -40, -30, -30, -20,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
   -30, -40, -40, -50, -50, -40, -40, -30,
]
_PST_KING_END = [
   -50, -30, -30, -30, -30, -30, -30, -50,
   -30, -30,   0,   0,   0,   0, -30, -30,
   -30, -10,  20,  30,  30,  20, -10, -30,
   -30, -10,  30,  40,  40,  30, -10, -30,
   -30, -10,  30,  40,  40,  30, -10, -30,
   -30, -10,  20,  30,  30,  20, -10, -30,
   -30, -20, -10,   0,   0, -10, -20, -30,
   -50, -40, -30, -20, -20, -30, -40, -50,
]
# fmt: on

_PST = {
    chess.PAWN: _PST_PAWN,
    chess.KNIGHT: _PST_KNIGHT,
    chess.BISHOP: _PST_BISHOP,
    chess.ROOK: _PST_ROOK,
    chess.QUEEN: _PST_QUEEN,
}

# Rough total non-pawn material used to interpolate mid/endgame king tables.
_PHASE_MAX = 2 * (PIECE_VALUE[chess.KNIGHT] + PIECE_VALUE[chess.BISHOP]
                  + PIECE_VALUE[chess.ROOK]) + PIECE_VALUE[chess.QUEEN]


@dataclass
class EvalFeatures:
    """Positional signals the style layer consumes.

    All values are from White's perspective unless noted; the style layer
    flips as needed. Scores are centipawns.
    """
    material_white: int       # summed piece values, White
    material_black: int
    phase: float              # 1.0 = full material (opening), 0.0 = bare kings
    openness: float           # 0..1, higher = more open position
    king_safety_white: int    # higher = safer White king (cp-ish)
    king_safety_black: int
    mobility_white: int       # pseudo-legal move count proxy
    mobility_black: int


def _pst_value(piece_type: int, square: int, white: bool, king_end_weight: float) -> int:
    idx = square if white else chess.square_mirror(square)
    if piece_type == chess.KING:
        mid = _PST_KING_MID[idx]
        end = _PST_KING_END[idx]
        return round(mid * (1 - king_end_weight) + end * king_end_weight)
    return _PST[piece_type][idx]


def _count_open_files(board: chess.Board) -> int:
    """Number of files with no pawns of either color (a proxy for openness)."""
    pawns = board.pawns
    open_files = 0
    for f in range(8):
        file_mask = chess.BB_FILES[f]
        if not (pawns & file_mask):
            open_files += 1
    return open_files


def _king_shelter(board: chess.Board, color: bool) -> int:
    """Cheap king-safety proxy: friendly pawns near own king, minus open files
    adjacent to the king. Positive = safer."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    shelter = 0
    own_pawns = board.pieces(chess.PAWN, color)
    for df in (-1, 0, 1):
        f = kf + df
        if 0 <= f <= 7:
            file_has_pawn = bool(chess.BB_FILES[f] & int(own_pawns))
            if file_has_pawn:
                shelter += 12
            else:
                shelter -= 15  # open/half-open file next to king is dangerous
    # Reward the king being tucked toward the back for its color.
    home_rank = 0 if color == chess.WHITE else 7
    shelter -= abs(kr - home_rank) * 6
    return shelter


def evaluate(board: chess.Board) -> tuple[int, EvalFeatures]:
    """Return (score_cp, features).

    `score_cp` is from the perspective of the side to move: positive means the
    side to move is better. This matches the negamax convention in search.py.
    """
    if board.is_checkmate():
        # Side to move is checkmated -> very bad for them.
        return -30000, _empty_features()
    if board.is_stalemate() or board.is_insufficient_material() \
            or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0, _empty_features()

    # Phase for king table interpolation.
    non_pawn = 0
    mat_w = 0
    mat_b = 0
    for piece_type in PIECE_VALUE:
        if piece_type == chess.KING:
            continue
        w = len(board.pieces(piece_type, chess.WHITE))
        b = len(board.pieces(piece_type, chess.BLACK))
        mat_w += w * PIECE_VALUE[piece_type]
        mat_b += b * PIECE_VALUE[piece_type]
        if piece_type != chess.PAWN:
            non_pawn += (w + b) * PIECE_VALUE[piece_type]
    phase = min(1.0, non_pawn / _PHASE_MAX) if _PHASE_MAX else 0.0
    king_end_weight = 1.0 - phase

    score_white = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUE[piece.piece_type]
        pst = _pst_value(piece.piece_type, square, piece.color == chess.WHITE,
                         king_end_weight)
        signed = val + pst
        score_white += signed if piece.color == chess.WHITE else -signed

    # Positional/style features.
    open_files = _count_open_files(board)
    openness = open_files / 8.0
    ks_w = _king_shelter(board, chess.WHITE)
    ks_b = _king_shelter(board, chess.BLACK)
    score_white += ks_w - ks_b

    # Mobility proxy via attack spans (no move generation -> much cheaper than
    # counting legal moves for both sides). Counts the squares each side's
    # non-pawn pieces attack.
    mob_w = _attack_span(board, chess.WHITE)
    mob_b = _attack_span(board, chess.BLACK)
    score_white += (mob_w - mob_b) * 1

    feats = EvalFeatures(
        material_white=mat_w,
        material_black=mat_b,
        phase=phase,
        openness=openness,
        king_safety_white=ks_w,
        king_safety_black=ks_b,
        mobility_white=mob_w,
        mobility_black=mob_b,
    )

    score_stm = score_white if board.turn == chess.WHITE else -score_white
    return score_stm, feats


def _attack_span(board: chess.Board, color: chess.Color) -> int:
    """Total squares attacked by `color`'s knights/bishops/rooks/queens (a cheap
    mobility proxy that needs no move generation). Pawns and king excluded to
    focus on piece activity."""
    span = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            span += chess.popcount(int(board.attacks_mask(sq)))
    return span


def _empty_features() -> EvalFeatures:
    return EvalFeatures(0, 0, 0.0, 0.0, 0, 0, 0, 0)
