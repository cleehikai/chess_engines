"""Atomic-specific board helpers, and the rule facts they encode.

The rules that actually matter for engine design (Lichess ruleset, which is what
Fairy-Stockfish and the Chalmers thesis both implement):

1. A capture removes the captured piece, **the capturing piece**, and every
   **non-pawn** piece on the up-to-8 adjacent squares. Pawns are immune to
   blast damage but not to being captured directly.
2. Kings therefore cannot capture (it would be suicide), and two kings may sit
   adjacent — neither checks the other.
3. You may not make a capture that explodes your own king. You *may* ignore
   check, move a pinned piece, or move into check, provided the move explodes
   the enemy king — that ends the game immediately in your favour.
4. En passant explodes around the square the capturing pawn *lands on*.
5. Win conditions: enemy king exploded, or ordinary checkmate.

The consequences that drive everything downstream:

- **Non-local king safety.** A king can die to a capture two squares away, so
  "is my king safe" is not a function of the squares attacking it. Any eval
  that reuses standard-chess king-safety terms is measuring the wrong thing.
- **Compressed, ply-dependent material.** See `ATOMIC_PIECE_VALUES` in
  `lab.methods`: a queen is worth ~10 pawns but a knight only ~1.9, and every
  value *peaks around ply 60 and then declines* (Pav 2025, Table 10). Material
  is a weaker signal than in standard chess; position dominates.
- **Short games.** Median game length is 30 ply versus ~80 in standard chess,
  which is why self-play throughput is cheap here and why endgames are
  chronically under-represented in self-play data.
"""

from __future__ import annotations

import chess
import chess.variant


def explosion_squares(board: chess.Board, move: chess.Move) -> chess.SquareSet:
    """Squares whose occupants are destroyed if `move` is played.

    Empty for a non-capture. For a capture: the landing square (capturing piece
    and victim) plus every adjacent square holding a non-pawn. Mirrors
    `AtomicBoard._push_capture`, but as a *query* so an eval or move-orderer can
    ask "what does this blow up?" without pushing the move.
    """
    if not board.is_capture(move):
        return chess.SquareSet()
    radius = chess.BB_KING_ATTACKS[move.to_square] & ~board.pawns
    squares = chess.SquareSet(radius & board.occupied)
    # The landing square holds the capturing piece and, in an ordinary capture,
    # the victim; both die.
    squares.add(move.to_square)
    if board.is_en_passant(move):
        # En passant is the exception: the victim pawn stands beside the
        # landing square rather than on it, and it dies too.
        squares.add(move.to_square + (-8 if board.turn == chess.WHITE else 8))
    return squares


def explodes_king(board: chess.Board, move: chess.Move, color: chess.Color) -> bool:
    """Whether `move` destroys `color`'s king."""
    king = board.king(color)
    return king is not None and king in explosion_squares(board, move)


def material_after(
    board: chess.Board,
    color: chess.Color,
    values: dict[chess.PieceType, int],
) -> int:
    """Material balance from `color`'s point of view on the *current* board."""
    total = 0
    for piece_type, value in values.items():
        if not value:
            continue
        total += value * len(board.pieces(piece_type, color))
        total -= value * len(board.pieces(piece_type, not color))
    return total


def game_result(board: chess.variant.AtomicBoard) -> str | None:
    """`"1-0"`, `"0-1"`, `"1/2-1/2"`, or None if the game is still running.

    `AtomicBoard.result()` already handles king explosion, but it is worth
    routing every call through one place so the arena and any future analysis
    agree on adjudication.
    """
    if not board.is_game_over(claim_draw=True):
        return None
    return board.result(claim_draw=True)


def is_quiet(board: chess.variant.AtomicBoard) -> bool:
    """No captures available — the atomic analogue of a quiescent position.

    Stricter than the standard-chess notion on purpose: in atomic *any*
    available capture can be a multi-piece swing or an instant win, so a
    position with captures on the board is never safe to evaluate statically.
    """
    return not any(board.is_capture(m) for m in board.legal_moves)
