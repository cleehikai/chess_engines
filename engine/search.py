"""Alpha-beta search with iterative deepening, transposition table, and
quiescence (Phase 1).

The evaluation function is injected so the same search drives the hand-crafted
eval now, a style-shaped eval (Phase 2), and NNUE later (Phase 3). The style
layer works by wrapping `evaluate` with a function of the same signature.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess
import chess.polyglot  # for zobrist_hash

from engine.evaluate import PIECE_VALUE, evaluate as default_evaluate

# An eval fn takes a board and returns centipawns from side-to-move perspective.
EvalFn = Callable[[chess.Board], int]

MATE_SCORE = 30000
INF = 1 << 30

# Transposition table entry flags.
EXACT, LOWER, UPPER = 0, 1, 2


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]


@dataclass
class SearchInfo:
    """Result + telemetry from a search."""
    best_move: Optional[chess.Move]
    score: int
    depth: int
    nodes: int
    pv: list[chess.Move] = field(default_factory=list)
    elapsed: float = 0.0


def _default_eval_fn(board: chess.Board) -> int:
    score, _ = default_evaluate(board)
    return score


class Searcher:
    """Reusable searcher holding a transposition table and history heuristic.

    Pass a custom `eval_fn` (e.g. a style-shaped or NNUE eval) to change how
    positions are scored without touching the search logic.
    """

    def __init__(self, eval_fn: EvalFn | None = None, tt_size: int = 1_000_000):
        self.eval_fn: EvalFn = eval_fn or _default_eval_fn
        self._tt: dict[int, TTEntry] = {}
        self._tt_size = tt_size
        self._killers: dict[int, list[chess.Move]] = {}
        self._nodes = 0
        self._deadline = 0.0
        self._stop = False

    # -- public API -----------------------------------------------------------
    def search(self, board: chess.Board, *, depth: int = 4,
               movetime: float | None = None) -> SearchInfo:
        """Iteratively deepen up to `depth`, or until `movetime` seconds elapse
        (whichever comes first). `movetime` of None means depth-limited only."""
        start = time.perf_counter()
        self._nodes = 0
        self._stop = False
        self._deadline = start + movetime if movetime else float("inf")

        best_move: Optional[chess.Move] = None
        best_score = 0
        completed_depth = 0

        for d in range(1, depth + 1):
            try:
                score, move = self._root_search(board, d)
            except _TimeUp:
                break
            if move is not None:
                best_move, best_score, completed_depth = move, score, d
            # Stop early on forced mate.
            if abs(best_score) > MATE_SCORE - 100:
                break
            if time.perf_counter() >= self._deadline:
                break

        elapsed = time.perf_counter() - start
        pv = self._extract_pv(board, best_move)
        return SearchInfo(best_move, best_score, completed_depth,
                          self._nodes, pv, elapsed)

    # -- root -----------------------------------------------------------------
    def _root_search(self, board: chess.Board, depth: int) -> tuple[int, Optional[chess.Move]]:
        alpha, beta = -INF, INF
        best_move = None
        best_score = -INF
        for move in self._order_moves(board, depth, self._tt_move(board)):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
        self._store_tt(board, depth, best_score, EXACT, best_move)
        return best_score, best_move

    # -- negamax with alpha-beta ----------------------------------------------
    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                 ply: int) -> int:
        self._nodes += 1
        if self._nodes % 2048 == 0 and time.perf_counter() >= self._deadline:
            raise _TimeUp

        # Cheap draw checks first (repetition via the fast bitboard path, not
        # the expensive can_claim_threefold_repetition which replays the game).
        if board.is_insufficient_material() or board.is_repetition(2) \
                or board.halfmove_clock >= 100:
            return 0

        alpha_orig = alpha
        key = chess.polyglot.zobrist_hash(board)
        entry = self._tt.get(key)
        if entry is not None and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score
            if entry.flag == LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        # Generate legal moves ONCE per node. Empty => mate or stalemate.
        legal = list(board.legal_moves)
        if not legal:
            return -MATE_SCORE + ply if board.is_check() else 0

        tt_move = entry.best_move if entry else None
        best_score = -INF
        best_move = None
        for move in self._order_moves(board, depth, tt_move, legal):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self._record_killer(depth, move)
                break

        flag = EXACT
        if best_score <= alpha_orig:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        self._store_tt(board, depth, best_score, flag, best_move, key)
        return best_score

    # -- quiescence: only search captures to avoid horizon effect -------------
    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._nodes += 1
        stand_pat = self.eval_fn(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for move in self._ordered_captures(board):
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # -- move ordering --------------------------------------------------------
    def _order_moves(self, board: chess.Board, depth: int,
                     tt_move: Optional[chess.Move],
                     legal: list[chess.Move] | None = None):
        if legal is None:
            legal = list(board.legal_moves)
        killers = self._killers.get(depth, [])

        def score(move: chess.Move) -> int:
            if move == tt_move:
                return 1_000_000
            if board.is_capture(move):
                return 100_000 + self._mvv_lva(board, move)
            if move in killers:
                return 90_000
            if move.promotion:
                return 80_000 + PIECE_VALUE.get(move.promotion, 0)
            return 0

        return sorted(legal, key=score, reverse=True)

    def _ordered_captures(self, board: chess.Board):
        caps = [m for m in board.legal_moves
                if board.is_capture(m) or m.promotion]
        caps.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)
        return caps

    def _mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        """Most-valuable-victim / least-valuable-attacker capture ordering."""
        if board.is_en_passant(move):
            victim = PIECE_VALUE[chess.PAWN]
        else:
            captured = board.piece_at(move.to_square)
            victim = PIECE_VALUE[captured.piece_type] if captured else 0
        attacker_piece = board.piece_at(move.from_square)
        attacker = PIECE_VALUE[attacker_piece.piece_type] if attacker_piece else 0
        return victim * 10 - attacker

    def _record_killer(self, depth: int, move: chess.Move) -> None:
        ks = self._killers.setdefault(depth, [])
        if move not in ks:
            ks.insert(0, move)
            del ks[2:]

    # -- transposition table --------------------------------------------------
    def _tt_move(self, board: chess.Board) -> Optional[chess.Move]:
        entry = self._tt.get(chess.polyglot.zobrist_hash(board))
        return entry.best_move if entry else None

    def _store_tt(self, board: chess.Board, depth: int, score: int, flag: int,
                  best_move: Optional[chess.Move], key: int | None = None) -> None:
        if key is None:
            key = chess.polyglot.zobrist_hash(board)
        if len(self._tt) >= self._tt_size:
            # Cheap eviction: drop an arbitrary chunk when full.
            self._tt.clear()
        self._tt[key] = TTEntry(depth, score, flag, best_move)

    def _extract_pv(self, board: chess.Board, first: Optional[chess.Move],
                    max_len: int = 16) -> list[chess.Move]:
        pv: list[chess.Move] = []
        if first is None:
            return pv
        b = board.copy(stack=False)
        move = first
        for _ in range(max_len):
            if move is None or not b.is_legal(move):
                break
            pv.append(move)
            b.push(move)
            entry = self._tt.get(chess.polyglot.zobrist_hash(b))
            move = entry.best_move if entry else None
        return pv


class _TimeUp(Exception):
    """Raised internally when the movetime deadline is hit."""
