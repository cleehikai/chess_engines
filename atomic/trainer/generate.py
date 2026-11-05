"""Mining puzzles out of games that were already analysed.

The expensive part is already paid for. Every imported game has had a deep
Fairy-Stockfish pass over **every** ply at MultiPV 2, so the two facts a puzzle
candidate has to satisfy — "there is something big to find here" and "only one
move finds it" — are already sitting in the cache. Stage 1 is therefore a query,
not a search, and costs no engine time at all.

Stage 2 spends real compute, but only on survivors: re-confirm at a higher node
count with MultiPV 3, then walk the line forward until it stops being forced.

Two families of candidate, because atomic has two kinds of critical moment and
only recognising the first would miss half the material:

- **punish** — the opponent just blew it and there is a unique refutation. The
  classic puzzle shape.
- **save** — nobody blundered, but exactly one move holds and everything else
  loses. In a variant this forcing, "the only move that does not lose" is a
  more common skill than "the winning shot", and this account's own review data
  says as much: the refutation was a *quiet* move in 50 of 56 blunders.

The triviality filter is the part that atomic specifically needs. A large share
of big evaluation swings here are one-move king blasts that any beginner sees;
they are unique *and* obvious, so the uniqueness test does not remove them.
`M0-greedy` — the lab's one-ply explosion-aware material engine, which already
exists and is already known to be barely better than random — is used as the
"any beginner sees it" oracle. If it finds the move, the position is not a
puzzle.

Line extension is breadth-first across *all* candidates rather than
depth-first through each one: every candidate's current position is batched
into one `ensure_analysed` call, which parallelises across positions and keeps
every database write on the calling thread. Walking one line at a time would
serialise the engine behind a dependency chain that is inherently sequential.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import chess
import chess.variant

from lab.methods import ATOMIC_PIECE_VALUES
from lab.rules import material_after

from . import analysis, book, puzzles, winrate

#: Stage 2 re-checks at a higher budget than the review pass, and with a third
#: line so "unique" is tested against two alternatives rather than one.
CONFIRM_NODES = 4_000_000
CONFIRM_MULTIPV = 3

#: Win% the solver must reach for a `punish` puzzle, and — the same number, so
#: it is defined once — the line above which a confirmed puzzle is labelled a
#: win rather than a hold. See `puzzles.WIN_FLOOR`.
WIN_FLOOR = puzzles.WIN_FLOOR

#: Above this, the solver was already winning before the opponent's error, so
#: there is nothing to punish and no puzzle.
ALREADY_WON = 65.0

#: A `save` puzzle: the solver is not winning, one move holds, and the
#: alternatives lose outright.
SAVE_FLOOR = 42.0
SAVE_SECOND_CEILING = 28.0

#: Uniqueness, in win% points between the best move and the second best.
#: Lichess uses 70 points of *their* win scale; a third of that is the like-for
#: -like translation onto the fitted atomic curve, and 20 is a little stricter
#: than that. Loose at stage 1 (only two lines are known) and strict at stage 2.
SCREEN_GAP = 12.0
UNIQUE_GAP = 20.0

#: How many moves the solver may be asked to find.
MAX_SOLVER_MOVES = 5

#: The opponent's move has to have been at least this bad for a `punish`
#: candidate. `winrate.MISTAKE`, not `BLUNDER`: the size of the resulting
#: opportunity is tested directly by `WIN_FLOOR`, so requiring a big drop *and*
#: a big result would only reject positions that were already bad.
TRIGGER_LOSS = winrate.MISTAKE


# -- the triviality oracle -------------------------------------------------


def greedy_moves(board: chess.variant.AtomicBoard) -> set[str]:
    """Every move `M0-greedy` might play here, as UCI.

    `lab.methods.GreedyMaterialMethod` breaks ties randomly, which is right for
    playing games and wrong for this: a filter that rejects a puzzle only when
    the coin lands the right way is not a filter. The rule is reproduced
    exactly — win immediately if you can, otherwise maximise explosion-aware
    material one ply deep — and the whole tie set is returned.
    """
    me = board.turn
    best_score: int | None = None
    best: set[str] = set()
    wins: set[str] = set()
    for move in list(board.legal_moves):
        board.push(move)
        # Both predicates are relative to the side to move, which after the
        # push is the opponent: they have lost exactly when we have won.
        won = board.is_variant_loss() or board.is_checkmate()
        score = material_after(board, me, ATOMIC_PIECE_VALUES)
        board.pop()
        if won:
            # Greedy plays *some* immediate win, and which one depends on a
            # coin. Every one of them counts as found.
            wins.add(move.uci())
            continue
        if best_score is None or score > best_score:
            best_score, best = score, {move.uci()}
        elif score == best_score:
            best.add(move.uci())
    return wins or best


#: Greedy breaks ties by coin flip, so "greedy solves it" has to mean "greedy
#: plays it often enough to count". At two tied moves that is a 50% chance,
#: which is plenty to disqualify a puzzle; at twelve it is 8%, which is not
#: solving anything.
GREEDY_TIES = 2


def is_trivial(board: chess.variant.AtomicBoard, uci: str) -> bool:
    """Would a beginner find this without looking? Then it is not a puzzle.

    The subtlety that cost a whole corpus: in a position with no captures
    available, every legal move leaves identical material, so the greedy tie
    set is *every legal move* and a naive `uci in greedy_moves(board)` calls
    everything trivial. Endgames are mostly quiet, so this rejected 100% of
    them — 14 genuine only-move tablebase positions in the first run, every
    one of them thrown away.

    Being indifferent between twenty moves is not solving a puzzle; it is
    guessing. So the tie set has to be small enough that playing the right move
    is a real preference rather than a coincidence.
    """
    if board.legal_moves.count() <= 1:
        return True  # nothing to choose between
    picks = greedy_moves(board)
    return uci in picks and len(picks) <= GREEDY_TIES


# -- stage 1: a query over the cache ---------------------------------------


@dataclass
class Candidate:
    """A position that might be a puzzle, found without spending engine time."""

    game_id: str
    ply: int                     # ply the solver moves at
    fen: str
    solver: str
    motive: str                  # 'win' / 'save'
    setup_fen: str | None
    setup_uci: str | None
    setup_san: str | None
    win_best: float
    win_second: float | None
    mine: bool
    #: Which corpus this belongs to: `game` for your own imported games,
    #: `dump` for a stranger's rated game scraped out of the Lichess archive.
    #: Only the provenance differs — every threshold below applies to both.
    source: str = "game"

    @property
    def gap(self) -> float:
        return 999.0 if self.win_second is None else self.win_best - self.win_second


def _final_fen(moves: Sequence[str]) -> str:
    board = chess.variant.AtomicBoard()
    for san in moves:
        board.push_san(san)
    return board.fen()


def candidates_in_game(
    db: sqlite3.Connection,
    rows: Sequence[dict[str, object]],
    final_fen: str,
    *,
    model: winrate.WinModel,
    game_id: str,
    nodes: int,
    multipv: int = analysis.DEFAULT_MULTIPV,
    my_color: str | None = None,
    source: str = "game",
    saves: bool = True,
    screen_gap: float = SCREEN_GAP,
) -> list[Candidate]:
    """Every position in one judged game worth spending stage 2 on.

    `rows` is the game's plies in order, each a mapping with the keys `plies`
    carries: `ply`, `fen`, `mover`, `uci`, `san`, `loss`, `win_before`. Both
    callers — the review pass over your own games, which reads them out of the
    database, and the scraper, which has just computed them in memory and will
    never store them — hand over the same shape, so the thresholds that decide
    what a puzzle is live here once rather than twice.
    """
    found: list[Candidate] = []
    for index, row in enumerate(rows):
        # The position the *opponent* faces after this move: the next ply's own
        # `fen`, or the game's final position for the last move.
        after = str(rows[index + 1]["fen"]) if index + 1 < len(rows) else final_fen
        lines = analysis.cached_lines(db, after, nodes, multipv)
        if not lines:
            continue  # terminal, or never analysed at this budget
        win_best = lines[0].win_pct(model)
        win_second = lines[1].win_pct(model) if len(lines) > 1 else None
        gap = 999.0 if win_second is None else win_best - win_second
        if gap < screen_gap:
            continue

        solver = "black" if row["mover"] == "white" else "white"
        common = {
            "game_id": game_id,
            "ply": int(row["ply"]) + 1,
            "fen": after,
            "solver": solver,
            "setup_fen": row["fen"],
            "setup_uci": row["uci"],
            "setup_san": row["san"],
            "win_best": win_best,
            "win_second": win_second,
            "mine": my_color == solver,
            "source": source,
        }
        # `punish`: they erred, and the position is now decisive. The solver
        # must not have been winning already, or the "mistake" only widened
        # something that was over.
        solver_before = 100.0 - float(row["win_before"])
        if (
            float(row["loss"]) >= TRIGGER_LOSS
            and win_best >= WIN_FLOOR
            and solver_before <= ALREADY_WON
        ):
            found.append(Candidate(motive="win", **common))
        elif saves and SAVE_FLOOR <= win_best < WIN_FLOOR and (
            win_second is not None and win_second <= SAVE_SECOND_CEILING
        ):
            found.append(Candidate(motive="save", **common))
    return found


def candidates(
    db: sqlite3.Connection,
    *,
    model: winrate.WinModel,
    account: str | None = None,
    game_id: str | None = None,
    include_ai: bool = False,
    saves: bool = True,
    screen_gap: float = SCREEN_GAP,
) -> list[Candidate]:
    """Every position in the analysed games worth spending stage 2 on.

    Reads only the review pass's own cache entries, at the node count each game
    was analysed with, so this is free however many times it is run.
    """
    sql = "SELECT * FROM games WHERE analysed_nodes IS NOT NULL"
    params: list[object] = []
    if account:
        sql += " AND account = ?"
        params.append(account.lower())
    if game_id:
        sql += " AND id = ?"
        params.append(game_id)
    if not include_ai:
        sql += " AND (source IS NULL OR source != 'ai')"

    found: list[Candidate] = []
    for game in db.execute(sql, params):
        rows = list(
            db.execute("SELECT * FROM plies WHERE game_id = ? ORDER BY ply", (game["id"],))
        )
        if not rows:
            continue
        found.extend(
            candidates_in_game(
                db,
                [dict(row) for row in rows],
                _final_fen((game["moves"] or "").split()),
                model=model,
                game_id=game["id"],
                nodes=game["analysed_nodes"],
                my_color=game["my_color"],
                saves=saves,
                screen_gap=screen_gap,
            )
        )
    return found


# -- stage 2: confirm and extend -------------------------------------------


@dataclass
class _Walk:
    """One candidate's line, grown a ply at a time by `_extend_all`."""

    candidate: Candidate
    board: chess.variant.AtomicBoard
    line: list[str] = field(default_factory=list)
    live: bool = True
    win_before: float | None = None
    win_second: float | None = None

    @property
    def solver_moves(self) -> int:
        return (len(self.line) + 1) // 2

    @property
    def solver_to_move(self) -> bool:
        return len(self.line) % 2 == 0


def _step(
    walk: _Walk,
    db: sqlite3.Connection,
    model: winrate.WinModel,
    nodes: int,
    multipv: int,
    unique_gap: float,
    max_solver_moves: int,
) -> None:
    """Advance one line by a single ply, or mark it finished."""
    board = walk.board
    if board.is_game_over(claim_draw=True):
        walk.live = False
        return
    lines = analysis.cached_lines(db, board.fen(), nodes, multipv)
    if not lines:
        walk.live = False
        return

    if not walk.solver_to_move:
        # The defender plays their best try. There is nothing to test here:
        # whatever they do, the solver has to answer it, and their best move
        # is the answer that is hardest to meet.
        walk.board.push(chess.Move.from_uci(lines[0].uci))
        walk.line.append(lines[0].uci)
        return

    if walk.solver_moves >= max_solver_moves:
        walk.live = False
        return
    win = lines[0].win_pct(model)
    second = lines[1].win_pct(model) if len(lines) > 1 else None
    gap = 999.0 if second is None else win - second
    floor = WIN_FLOOR if walk.candidate.motive == "win" else SAVE_FLOOR
    if win < floor or gap < unique_gap:
        walk.live = False   # the line stopped being both winning and forced
        return
    if walk.win_before is None:
        walk.win_before, walk.win_second = win, second
    walk.board.push(chess.Move.from_uci(lines[0].uci))
    walk.line.append(lines[0].uci)


def _extend_all(
    db: sqlite3.Connection,
    walks: list[_Walk],
    *,
    model: winrate.WinModel,
    nodes: int,
    multipv: int,
    unique_gap: float,
    max_solver_moves: int,
    workers: int | None,
    on_round: Callable[[int, int, int], None] | None = None,
) -> None:
    """Grow every line together, one ply per round.

    Each round analyses the whole frontier in one batch. That keeps the engine
    saturated — the positions in a round are independent, while the plies of a
    single line are not — and keeps every cache write on this thread.
    """
    rounds = 0
    while True:
        live = [walk for walk in walks if walk.live]
        if not live:
            return
        rounds += 1
        analysis.ensure_analysed(
            db,
            [walk.board.fen() for walk in live if not walk.board.is_game_over(claim_draw=True)],
            nodes=nodes,
            multipv=multipv,
            workers=workers,
        )
        for walk in live:
            _step(walk, db, model, nodes, multipv, unique_gap, max_solver_moves)
        if on_round:
            on_round(rounds, len(live), sum(1 for w in walks if w.live))


def _trim(walk: _Walk) -> list[str]:
    """Cut a grown line back to something that is a fair question to ask.

    Two rules, both borrowed from `lichess-puzzler` and both still right here:
    a solution ends on the solver's move, and its last move must not be the
    only legal one — being forced to play your one legal move is not a find.
    """
    line = list(walk.line)
    if len(line) % 2 == 0:
        line = line[:-1]                     # do not end on the defender
    while line:
        board = chess.variant.AtomicBoard(walk.candidate.fen)
        for uci in line[:-1]:
            board.push(chess.Move.from_uci(uci))
        if board.legal_moves.count() > 1:
            break
        line = line[:-2]   # drop the forced move and the reply that forced it
    return line


@dataclass
class Report:
    """What a generation run did, in the terms the thresholds are set in."""

    candidates: int = 0
    duplicate: int = 0
    trivial: int = 0
    not_unique: int = 0
    too_short: int = 0
    kept: int = 0
    wins: int = 0
    saves: int = 0
    analysed: int = 0

    def line(self) -> str:
        return (
            f"{self.candidates} candidates -> {self.kept} puzzles "
            f"({self.wins} punish, {self.saves} save); rejected "
            f"{self.duplicate} duplicate positions, {self.trivial} trivial, "
            f"{self.not_unique} not unique, {self.too_short} with no line left"
        )


def generate(
    db: sqlite3.Connection,
    found: Sequence[Candidate],
    *,
    model: winrate.WinModel,
    nodes: int = CONFIRM_NODES,
    multipv: int = CONFIRM_MULTIPV,
    unique_gap: float = UNIQUE_GAP,
    max_solver_moves: int = MAX_SOLVER_MOVES,
    workers: int | None = None,
    seen: dict[str, set[str]] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[puzzles.Puzzle], Report]:
    """Confirm candidates at a higher budget and turn the survivors into puzzles.

    `seen` is `puzzles.seen_positions(db)` — every position the corpus already
    covers, with the game and ply each copy came from. Duplicates are dropped
    here, before the confirming pass, because that pass is the expensive step:
    a duplicate rejected afterwards has already cost 4M nodes. The map is
    updated as the run goes, so two candidates from different games that reach
    the same position produce one puzzle rather than two.
    """
    report = Report(candidates=len(found))
    if seen is not None:
        kept_candidates = []
        for candidate in found:
            key = puzzles.position_key(candidate.fen)
            here = puzzles.provenance(candidate.game_id, candidate.ply)
            covered = seen.get(key)
            if covered is not None and here not in covered:
                report.duplicate += 1
                continue
            seen.setdefault(key, set()).add(here)
            kept_candidates.append(candidate)
        found = kept_candidates
    if not found:
        return [], report

    # Confirm the starting positions first: everything else is wasted work on a
    # candidate whose uniqueness does not survive a deeper look.
    if on_progress:
        on_progress("confirming", 0, len(found))
    report.analysed += analysis.ensure_analysed(
        db,
        [c.fen for c in found],
        nodes=nodes,
        multipv=multipv,
        workers=workers,
        on_progress=(lambda d, t: on_progress("confirming", d, t)) if on_progress else None,
    )

    walks: list[_Walk] = []
    for candidate in found:
        lines = analysis.cached_lines(db, candidate.fen, nodes, multipv)
        if not lines:
            continue
        board = chess.variant.AtomicBoard(candidate.fen)
        win = lines[0].win_pct(model)
        second = lines[1].win_pct(model) if len(lines) > 1 else None
        gap = 999.0 if second is None else win - second
        floor = WIN_FLOOR if candidate.motive == "win" else SAVE_FLOOR
        if win < floor or gap < unique_gap:
            report.not_unique += 1
            continue
        if is_trivial(board, lines[0].uci):
            report.trivial += 1
            continue
        walks.append(_Walk(candidate=candidate, board=board))

    if on_progress:
        on_progress("extending", 0, len(walks))
    _extend_all(
        db, walks,
        model=model, nodes=nodes, multipv=multipv, unique_gap=unique_gap,
        max_solver_moves=max_solver_moves, workers=workers,
        on_round=(lambda r, n, left: on_progress("extending", r, n)) if on_progress else None,
    )

    out: list[puzzles.Puzzle] = []
    for walk in walks:
        line = _trim(walk)
        if not line:
            report.too_short += 1
            continue
        candidate = walk.candidate
        # The candidate's motive was a guess made at the review pass's budget;
        # this one is what the 4M-node confirming pass actually found. They
        # disagree often enough to matter — a position screened as "one move
        # holds" is sometimes winning outright once it is looked at properly —
        # and the solver is told the confirmed one, so it has to be stored.
        motive = puzzles.motive_of(walk.win_before, candidate.motive)
        puzzle = puzzles.Puzzle(
            fen=candidate.fen,
            solver=candidate.solver,
            solution=line,
            solution_san=puzzles.solution_sans(candidate.fen, line),
            motive=motive,
            source=candidate.source,
            game_id=candidate.game_id,
            ply=candidate.ply,
            setup_fen=candidate.setup_fen,
            setup_uci=candidate.setup_uci,
            setup_san=candidate.setup_san,
            win_before=walk.win_before,
            win_second=walk.win_second,
            unique_gap=(
                None if walk.win_second is None
                else (walk.win_before or 0.0) - walk.win_second
            ),
            nodes=nodes,
            mine=candidate.mine,
        )
        in_book = bool(book.lookup(db, book.book_fen(chess.variant.AtomicBoard(puzzle.fen))))
        out.append(puzzles.annotate(puzzle, in_book=in_book))
        report.kept += 1
        report.wins += motive == "win"
        report.saves += motive == "save"
    return out, report


def run(
    db: sqlite3.Connection,
    *,
    model: winrate.WinModel | None = None,
    account: str | None = None,
    game_id: str | None = None,
    include_ai: bool = False,
    saves: bool = True,
    nodes: int = CONFIRM_NODES,
    multipv: int = CONFIRM_MULTIPV,
    unique_gap: float = UNIQUE_GAP,
    screen_gap: float = SCREEN_GAP,
    max_solver_moves: int = MAX_SOLVER_MOVES,
    limit: int | None = None,
    workers: int | None = None,
    dedupe: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[int, Report]:
    """The whole pipeline: query the cache, confirm, extend, store."""
    model = model or winrate.load()
    found = candidates(
        db,
        model=model,
        account=account,
        game_id=game_id,
        include_ai=include_ai,
        saves=saves,
        screen_gap=screen_gap,
    )
    found.sort(key=lambda c: -c.gap)
    if limit:
        found = found[:limit]
    made, report = generate(
        db, found,
        model=model, nodes=nodes, multipv=multipv, unique_gap=unique_gap,
        max_solver_moves=max_solver_moves, workers=workers,
        seen=puzzles.seen_positions(db) if dedupe else None,
        on_progress=on_progress,
    )
    return puzzles.save(db, made), report


__all__ = [
    "Candidate",
    "Report",
    "candidates",
    "generate",
    "greedy_moves",
    "is_trivial",
    "run",
]
