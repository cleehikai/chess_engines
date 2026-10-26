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

    def __init__(self, eval_fn: EvalFn | None = None, tt_size: int = 1_000_000,
                 *, use_nullmove: bool = False, use_lmr: bool = False,
                 use_pvs: bool = False, use_check_ext: bool = False,
                 use_aspiration: bool = True):
        self.eval_fn: EvalFn = eval_fn or _default_eval_fn
        self._tt: dict[int, TTEntry] = {}
        self._tt_size = tt_size
        self._killers: dict[int, list[chess.Move]] = {}
        self._nodes = 0
        self._deadline = 0.0
        self._stop = False
        # Search-feature toggles. These modern pruning/extension techniques are
        # tuned for engines that reach high depth; on a pure-Python engine that
        # only reaches depth ~2-4 in blitz time their per-node overhead can
        # outweigh the savings. Measured (12-game A/B vs plain baseline, hand eval
        # @0.25s), so defaults reflect what actually helps THIS engine:
        #   aspiration  6-0-6  -> clear win, ON by default
        #   check ext   3-4-5  -> slight loss, OFF
        #   null-move   6-6-0  -> neutral at this depth, OFF (kept as a toggle)
        #   LMR         6-6-0  -> neutral, OFF
        #   PVS         6-6-0  -> neutral, OFF
        # They remain switchable: once nps rises (Cython/C move-gen) and the
        # engine reaches higher depth, re-run the ablation -- they should flip
        # positive, as they do in every strong engine.
        self.use_nullmove = use_nullmove
        self.use_lmr = use_lmr
        self.use_pvs = use_pvs
        self.use_check_ext = use_check_ext
        self.use_aspiration = use_aspiration

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

        # A _TimeUp can be raised mid-node, after pushes (including a null move)
        # but before their matching pops -- leaving the caller's board mutated.
        # Record the stack depth so we can always restore it, whatever unwinds.
        base_stack = len(board.move_stack)

        for d in range(1, depth + 1):
            try:
                # Aspiration windows: after the first few plies, search a narrow
                # window around the previous score instead of (-INF, INF). Most
                # searches land inside it and prune far more; on a fail
                # high/low we widen and re-search. Cheap, ~+15-30 Elo.
                if self.use_aspiration and d >= 3 \
                        and abs(best_score) < MATE_SCORE - 100:
                    score, move = self._aspiration(board, d, best_score)
                else:
                    score, move = self._root_search(board, d, -INF, INF)
            except _TimeUp:
                # Unwind any pushes the interrupted node left on the board.
                while len(board.move_stack) > base_stack:
                    board.pop()
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
    def _aspiration(self, board: chess.Board, depth: int, prev: int
                    ) -> tuple[int, Optional[chess.Move]]:
        """Search depth `depth` with a window centered on `prev`, widening on a
        fail high/low until the true score is bracketed."""
        window = 50
        while True:
            alpha = prev - window
            beta = prev + window
            score, move = self._root_search(board, depth, alpha, beta)
            if score <= alpha:            # fail low -> widen down
                window *= 3
            elif score >= beta:           # fail high -> widen up
                window *= 3
            else:
                return score, move
            if window > 2000:             # give up narrowing -> full window
                return self._root_search(board, depth, -INF, INF)

    def _root_search(self, board: chess.Board, depth: int,
                     alpha: int, beta: int) -> tuple[int, Optional[chess.Move]]:
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
            if alpha >= beta:
                break
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

        in_check = board.is_check()

        # Check extension: if we're in check, search one ply deeper. Checks are
        # forcing and cheap to resolve; not extending them causes horizon blunders
        # (walking into a mate just past the depth limit). Applied before the
        # depth<=0 test so an in-check "leaf" is still expanded.
        if in_check and self.use_check_ext:
            depth += 1

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        # Null-move pruning: if we can "pass" and still be so good the opponent's
        # best reply can't bring us below beta, this node is a cutoff. Skipped
        # when in check (illegal to pass under check), near the frontier, in a
        # PV-ish narrow window, or when we have only pawns (zugzwang: passing may
        # genuinely be best, so the assumption breaks).
        non_pv = beta - alpha <= 1
        if (self.use_nullmove and depth >= 3 and not in_check and non_pv
                and self._has_non_pawn_material(board)
                and abs(beta) < MATE_SCORE - 100):
            R = 2 + (depth // 6)  # reduction grows slowly with depth
            board.push(chess.Move.null())
            null_score = -self._negamax(board, depth - 1 - R, -beta,
                                        -beta + 1, ply + 1)
            board.pop()
            if null_score >= beta:
                return beta

        # Generate legal moves ONCE per node. Empty => mate or stalemate.
        legal = list(board.legal_moves)
        if not legal:
            return -MATE_SCORE + ply if in_check else 0

        tt_move = entry.best_move if entry else None
        best_score = -INF
        best_move = None
        # LMR is only considered for late moves at depth>=3; gate the expensive
        # per-move predicates (is_capture, and especially gives_check, which are
        # slow in python-chess) behind those cheap conditions. Computing them for
        # every move at every node erased the pruning gain at the shallow depths
        # this engine actually reaches -- measured: it made the search *weaker*.
        lmr_allowed = self.use_lmr and depth >= 3 and not in_check

        for i, move in enumerate(self._order_moves(board, depth, tt_move, legal)):
            # Decide the reduction BEFORE pushing (gives_check/is_capture read the
            # pre-move board). Only pay for the checks on late quiet candidates.
            reduction = 0
            if lmr_allowed and i >= 3 and not move.promotion \
                    and not board.is_capture(move) and not board.gives_check(move):
                reduction = 1 + (i >= 6)  # deeper cut for very late moves

            board.push(move)

            # Principal variation search: search the first (best-ordered) move
            # with the full window; scout the rest with a null window, which
            # prunes far harder. A scout that beats alpha is re-searched at full
            # width. LMR reduction folds into the scout pass. This is also what
            # makes null-move pruning fire at interior nodes (they inherit the
            # null window), so PVS + LMR + null-move compound here.
            if i == 0 or not self.use_pvs:
                # Full-window search. (When PVS is off, LMR still applies its
                # reduction, re-searching at full depth if it beats alpha.)
                if reduction:
                    score = -self._negamax(board, depth - 1 - reduction,
                                           -beta, -alpha, ply + 1)
                    if score > alpha:
                        score = -self._negamax(board, depth - 1, -beta, -alpha,
                                               ply + 1)
                else:
                    score = -self._negamax(board, depth - 1, -beta, -alpha,
                                           ply + 1)
            else:
                score = -self._negamax(board, depth - 1 - reduction,
                                       -alpha - 1, -alpha, ply + 1)
                # If the scout (possibly reduced) beat alpha, it may be a real
                # improvement hidden by the null window or the reduction -> verify
                # with a full-depth, full-window search.
                if score > alpha and (reduction or score < beta):
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

        # Delta pruning: if even capturing the most valuable hangable piece plus
        # a safety margin can't reach alpha, this position is hopeless and we
        # stop searching captures. Disabled when in check (must resolve the
        # check, not just chase material).
        in_check = board.is_check()

        for move in self._ordered_captures(board):
            if not in_check and not move.promotion:
                victim = self._capture_victim_value(board, move)
                if stand_pat + victim + 200 < alpha:
                    continue  # this capture can't raise alpha; skip
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

    def _capture_victim_value(self, board: chess.Board, move: chess.Move) -> int:
        """Centipawn value of the piece captured by `move` (0 if none)."""
        if board.is_en_passant(move):
            return PIECE_VALUE[chess.PAWN]
        captured = board.piece_at(move.to_square)
        return PIECE_VALUE[captured.piece_type] if captured else 0

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

    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        """True if the side to move has at least one knight/bishop/rook/queen.
        Null-move pruning is unsafe in king-and-pawn positions (zugzwang)."""
        color = board.turn
        return bool(board.knights & board.occupied_co[color]
                    or board.bishops & board.occupied_co[color]
                    or board.rooks & board.occupied_co[color]
                    or board.queens & board.occupied_co[color])

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
