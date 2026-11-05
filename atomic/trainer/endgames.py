"""Endgame puzzles with no engine in them at all.

Everything else in this trainer is downstream of Fairy-Stockfish, and therefore
downstream of its blind spots — the unresolved 1.Nh3-vs-1.Nf3 disagreement is
the standing reminder that a strong engine is not an oracle. Below six pieces,
atomic has something better: Lichess serves a **tablebase**, and a tablebase is
not an opinion. `lab/tablebase.py` already speaks the protocol and caches every
probe.

So an endgame puzzle here is exact:

> a position where **exactly one** legal move preserves the result.

No node count, no win% curve, no threshold anyone can argue with. If the
tablebase says the position is won and one move keeps it won, that move is the
answer and every other move throws the win away — as a fact, not an estimate.
The same construction gives defensive puzzles for free: in a drawn position
where one move holds the draw, "save it" is the same query with a different
category.

**Two sources, because enumeration is not on the table.** The plan's original
wording was "enumerate <=6-piece positions"; there are far too many, and the
tablebase is a free public service on a 0.25s courtesy interval, so a full
sweep would be both impossible and rude. Instead:

- *your own endgames* — every position from an imported game that is small
  enough to probe. Directly relevant, and a few hundred requests.
- *sampled material configurations* — pick a material split (KQ vs KR, KRP vs
  KB, ...), scatter the pieces at random, keep what is legal. This is sampling,
  not enumeration, and the code says so: the corpus it produces is a random
  draw from a space nobody has enumerated, which is the honest description.

Both are filtered by the same triviality oracle the game-derived generator
uses, for the same reason: in atomic a large share of only-moves are one-move
king blasts that need no thought.
"""

from __future__ import annotations

import random
import sqlite3
import time
import urllib.error
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import chess
import chess.variant

from lab import config
from lab.tablebase import Probe, Tablebase

from . import generate, puzzles

#: The tablebase covers up to six pieces, kings included.
MAX_PIECES = 6

#: How deep a tablebase line may run, in solver moves. Endgames can be forced
#: for dozens of plies; a puzzle that long is a chore rather than a lesson.
MAX_SOLVER_MOVES = 4

#: Retry policy for a throttled probe. Seconds, multiplied by the attempt.
_RETRIES = 4
_BACKOFF = 2.0

#: How far a walk follows a position before starting again somewhere else,
#: and how often it plays a deliberately random move instead of the best one.
WALK_LENGTH = 14
BLUNDER_RATE = 0.4

#: Better-is-larger ordering over tablebase verdicts, from the point of view of
#: the side to move. `cursed-win` and `blessed-loss` are the 50-move-rule
#: cases; they are grouped with the draw they actually score as.
_RANK = {
    "loss": 0,
    "blessed-loss": 1,
    "draw": 1,
    "cursed-win": 1,
    "win": 2,
}

#: Material splits worth sampling. Chosen for atomic rather than borrowed:
#: a lone queen mates on the edge, a knight is worth barely more than a pawn,
#: and pawn endings behave completely differently because pawns survive
#: explosions. Kings are implied.
CONFIGURATIONS: tuple[tuple[str, str], ...] = (
    ("Q", ""),
    ("R", ""),
    ("Q", "R"),
    ("Q", "N"),
    ("Q", "B"),
    ("R", "N"),
    ("R", "B"),
    ("R", "P"),
    ("B", "N"),
    ("N", "N"),
    ("P", ""),
    ("PP", ""),
    ("P", "P"),
    ("QP", "R"),
    ("RP", "B"),
    ("NP", "P"),
    ("BB", "R"),
)

_PIECE = {
    "P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP,
    "R": chess.ROOK, "Q": chess.QUEEN,
}


def piece_count(board: chess.Board) -> int:
    return bin(board.occupied).count("1")


# -- position sources ------------------------------------------------------


def from_games(
    db: sqlite3.Connection, *, include_ai: bool = False, max_pieces: int = MAX_PIECES
) -> Iterator[str]:
    """Every small-enough position that actually occurred in your games."""
    sql = (
        "SELECT DISTINCT p.fen FROM plies p JOIN games g ON g.id = p.game_id "
        + ("" if include_ai else "WHERE (g.source IS NULL OR g.source != 'ai') ")
    )
    seen: set[str] = set()
    for row in db.execute(sql):
        board = chess.variant.AtomicBoard(row["fen"])
        if piece_count(board) > max_pieces:
            continue
        key = board.fen()
        if key not in seen:
            seen.add(key)
            yield key


def from_cache(max_pieces: int = MAX_PIECES) -> Iterator[str]:
    """Every position the tablebase cache already holds an answer for.

    The probe cache is an asset in exactly the way the analysis cache is: a
    walk that found nothing still paid for thousands of exact verdicts, and
    re-examining them after a filter changes costs no requests at all. This is
    the source to run after touching the acceptance rules.
    """
    db = sqlite3.connect(config.TABLEBASE_CACHE)
    try:
        rows = db.execute("SELECT fen FROM probes").fetchall()
    finally:
        db.close()
    for (fen,) in rows:
        try:
            board = chess.variant.AtomicBoard(fen)
        except ValueError:
            continue
        if piece_count(board) <= max_pieces and not board.is_game_over(claim_draw=True):
            yield fen


def _place(
    rng: random.Random, white: str, black: str
) -> chess.variant.AtomicBoard | None:
    """Scatter one material configuration onto an empty board.

    Returns None whenever the draw lands somewhere the rules forbid — pawns on
    the back ranks, a side already exploded, a position that is over before it
    starts. Rejection sampling is the right shape here: the constraints are
    cheap to test and awkward to enumerate around.
    """
    board = chess.variant.AtomicBoard(None)
    squares = rng.sample(range(64), 2 + len(white) + len(black))
    board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
    index = 2
    for symbols, color in ((white, chess.WHITE), (black, chess.BLACK)):
        for symbol in symbols:
            square = squares[index]
            index += 1
            if symbol == "P" and chess.square_rank(square) in (0, 7):
                return None
            board.set_piece_at(square, chess.Piece(_PIECE[symbol], color))
    board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
    board.castling_rights = chess.BB_EMPTY
    if not board.is_valid() or board.is_game_over(claim_draw=True):
        return None
    # `is_valid()` does not catch it here: `AtomicBoard` relaxes the
    # into-check rules (moving into check is legal if it explodes the enemy
    # king), and the relaxation reaches `status()` too, so a position with the
    # side *not* to move already in check passes validation. The Lichess
    # tablebase rejects those with a 400, correctly — no such position can
    # arise in a game. Tested directly instead.
    board.turn = not board.turn
    exposed = board.is_check()
    board.turn = not board.turn
    return None if exposed else board


def sampled(
    count: int,
    *,
    seed: int | None = None,
    configurations: Iterable[tuple[str, str]] = CONFIGURATIONS,
) -> Iterator[str]:
    """Random legal positions from the listed material configurations.

    A sample, not a sweep. Yields at most `count` FENs and gives up on a
    configuration that keeps producing illegal draws rather than looping.
    """
    rng = random.Random(seed)
    configs = list(configurations)
    seen: set[str] = set()
    attempts = 0
    budget = count * 40
    while len(seen) < count and attempts < budget:
        attempts += 1
        white, black = configs[rng.randrange(len(configs))]
        # Both colours get a turn at being the stronger side.
        if rng.random() < 0.5:
            white, black = black, white
        board = _place(rng, white, black)
        if board is None:
            continue
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        yield fen


def walked(
    tb: Tablebase,
    report: Report,
    starts: Iterable[str],
    *,
    seed: int | None = None,
    walk_length: int = WALK_LENGTH,
    blunder_rate: float = BLUNDER_RATE,
) -> Iterator[str]:
    """Walk the tablebase from each start, yielding every position on the way.

    Isolated random positions do not work, and the first run said so plainly:
    3,000 of them produced **zero** puzzles, because a position scattered onto
    an empty board is almost always either hopeless or comfortably won, and a
    comfortably won position has a dozen moves that keep the win. Only-move
    positions live next door to a mistake.

    So the walk plays. It follows the tablebase's own best move most of the
    time and a random legal one otherwise, which is the point rather than a
    concession — a *blunder* is what turns a won game into a position where
    exactly one move saves it, and those are the positions worth asking about.

    The walk costs nothing extra. Every step needs a probe, every probe is a
    candidate test, and the answers are cached, so the miner's own probe of the
    same position is free.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    for start in starts:
        board = chess.variant.AtomicBoard(start)
        for _ in range(walk_length):
            if board.is_game_over(claim_draw=True) or piece_count(board) > MAX_PIECES:
                break
            fen = board.fen()
            if fen in seen:
                break        # a loop, or a line already walked
            seen.add(fen)
            yield fen

            probe = _probe(tb, board, report)
            if probe is None or not probe.known:
                break
            legal = {move.uci() for move in board.legal_moves}
            options = [
                entry["uci"] for entry in probe.moves
                if entry.get("uci") in legal
            ]
            if not options:
                break
            best = probe.best_move()
            choice = (
                rng.choice(options)
                if len(options) > 1 and rng.random() < blunder_rate
                else (best or options[0])
            )
            board.push(chess.Move.from_uci(choice))


# -- the only-move test ----------------------------------------------------


def holding_moves(board: chess.variant.AtomicBoard, probe: Probe) -> list[str]:
    """Moves that preserve the position's result, best-first.

    The tablebase reports each move's category from the *opponent's* point of
    view, so the moves that keep a win are the ones that leave the opponent
    lost. Sorted by distance-to-zeroing so the shortest conversion comes first,
    which is what the puzzle should teach when several moves tie.
    """
    want = _RANK.get(probe.category)
    if want is None or probe.category == "loss":
        # A lost position has nothing to preserve. "The only move that keeps me
        # lost" is a well-defined query and a pointless puzzle.
        return []
    legal = {move.uci() for move in board.legal_moves}
    kept = [
        entry
        for entry in probe.moves
        if entry.get("uci") in legal
        # 2 - rank flips the opponent's verdict back to ours: they are lost
        # exactly when we are winning.
        and 2 - _RANK.get(entry.get("category", ""), -5) == want
    ]
    kept.sort(key=lambda entry: abs(entry.get("dtz") or 10**6))
    return [entry["uci"] for entry in kept]


@dataclass
class Report:
    #: Tablebase lookups, cache hits included — not the number of HTTP requests.
    probed: int = 0
    out_of_range: int = 0
    unknown: int = 0
    refused: int = 0
    not_unique: int = 0
    throttled: int = 0
    trivial: int = 0
    duplicate: int = 0
    kept: int = 0
    wins: int = 0
    saves: int = 0

    def line(self) -> str:
        return (
            f"{self.probed} probed -> {self.kept} puzzles "
            f"({self.wins} win, {self.saves} save); rejected "
            f"{self.not_unique} with more than one holding move, "
            f"{self.trivial} trivial, {self.duplicate} already in the corpus, "
            f"{self.unknown} not in the tables, "
            f"{self.refused} refused as illegal"
        )


def _probe(tb: Tablebase, board: chess.variant.AtomicBoard, report: Report) -> Probe | None:
    """Probe, surviving both kinds of "no".

    Two failure modes, and conflating them cost a 3,000-position run a third of
    its budget before this existed:

    - **400** — the service considers the position unreachable. A judgement
      about one position; skip it and carry on.
    - **429** — we are asking too fast. Nothing is wrong with the position and
      dropping it wastes work, so back off and try again. This is a free
      service run by a nonprofit and a sustained 4 requests a second is the
      kind of thing it is entitled to say no to.
    """
    report.probed += 1
    for attempt in range(_RETRIES):
        try:
            return tb.probe(board)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                report.refused += 1
                return None
            report.throttled += 1
            time.sleep(_BACKOFF * (attempt + 1))
        except urllib.error.URLError:
            time.sleep(_BACKOFF)
    return None


def _extend(
    tb: Tablebase,
    board: chess.variant.AtomicBoard,
    category: str,
    max_solver_moves: int,
    report: Report,
) -> list[str]:
    """Walk the forced line: only-move by the solver, best defence in reply.

    Stops as soon as the solver has a choice. That is the honest boundary of a
    tablebase puzzle — once two moves both hold, there is no single answer to
    ask for, whatever the engine might prefer between them.
    """
    line: list[str] = []
    solver = board.turn
    work = board.copy(stack=False)
    while len(line) // 2 < max_solver_moves:
        if work.is_game_over(claim_draw=True):
            break
        probe = _probe(tb, work, report)
        if probe is None or not probe.known:
            break
        if work.turn == solver:
            if probe.category != category:
                break                    # the defence changed the result
            holding = holding_moves(work, probe)
            if len(holding) != 1:
                break                    # a choice: stop before it
            work.push(chess.Move.from_uci(holding[0]))
            line.append(holding[0])
        else:
            reply = probe.best_move()
            if reply is None:
                break
            work.push(chess.Move.from_uci(reply))
            line.append(reply)
    if len(line) % 2 == 0:
        line = line[:-1]
    return line


def mine(
    db: sqlite3.Connection,
    fens: Iterable[str],
    *,
    tb: Tablebase | None = None,
    report: Report | None = None,
    max_solver_moves: int = MAX_SOLVER_MOVES,
    limit: int | None = None,
    seen: dict[str, set[str]] | None = None,
    on_progress: Callable[[int, Report], None] | None = None,
) -> tuple[list[puzzles.Puzzle], Report]:
    """Turn candidate positions into exact endgame puzzles.

    `seen` is `puzzles.seen_positions(db)`. The walk revisits positions on
    purpose — the same endgame is reachable by many routes, which is what makes
    walking productive — so without it a second run over the probe cache
    re-proposes everything the first run already kept.
    """
    report = report if report is not None else Report()
    out: list[puzzles.Puzzle] = []
    owned = tb is None
    tb = tb or Tablebase()
    try:
        for index, fen in enumerate(fens):
            if limit and len(out) >= limit:
                break
            board = chess.variant.AtomicBoard(fen)
            if piece_count(board) > MAX_PIECES:
                report.out_of_range += 1
                continue
            if board.is_game_over(claim_draw=True):
                continue
            if seen is not None and puzzles.position_key(board.fen()) in seen:
                report.duplicate += 1
                continue
            probe = _probe(tb, board, report)
            if probe is None or not probe.known:
                report.unknown += 1
                continue
            holding = holding_moves(board, probe)
            if len(holding) != 1:
                report.not_unique += 1
                continue
            if generate.is_trivial(board, holding[0]):
                report.trivial += 1
                continue

            line = _extend(tb, board, probe.category, max_solver_moves, report)
            if not line or line[0] != holding[0]:
                report.not_unique += 1
                continue
            motive = "win" if probe.category in ("win", "cursed-win") else "save"
            puzzle = puzzles.Puzzle(
                fen=board.fen(),
                solver="white" if board.turn == chess.WHITE else "black",
                solution=line,
                solution_san=puzzles.solution_sans(board.fen(), line),
                motive=motive,
                source="tablebase",
                tb_category=probe.category,
            )
            out.append(puzzles.annotate(puzzle))
            if seen is not None:
                seen.setdefault(puzzles.position_key(board.fen()), set()).add(
                    puzzles.provenance(None, None)
                )
            report.kept += 1
            report.wins += motive == "win"
            report.saves += motive == "save"
            if on_progress:
                on_progress(index + 1, report)
    finally:
        if owned:
            tb.close()
    return out, report


def run(
    db: sqlite3.Connection,
    *,
    source: str = "walk",
    samples: int = 200,
    seed: int | None = None,
    walk_length: int = WALK_LENGTH,
    include_ai: bool = False,
    limit: int | None = None,
    max_solver_moves: int = MAX_SOLVER_MOVES,
    dedupe: bool = True,
    on_progress: Callable[[int, Report], None] | None = None,
) -> tuple[int, Report]:
    """Mine endgame puzzles and store them.

    `source` picks where positions come from: `games` (yours), `random`
    (isolated scattered positions — kept because it is the honest baseline, and
    it produced nothing), `walk` (scattered starts, then played out), or `all`.
    """
    report = Report()
    with Tablebase() as tb:
        streams: list[Iterable[str]] = []
        if source in ("cache", "all"):
            streams.append(from_cache())
        if source in ("games", "all"):
            streams.append(from_games(db, include_ai=include_ai))
        if source in ("random", "all"):
            streams.append(sampled(samples, seed=seed))
        if source in ("walk", "all"):
            streams.append(
                walked(
                    tb, report, sampled(samples, seed=seed),
                    seed=seed, walk_length=walk_length,
                )
            )

        def chained() -> Iterator[str]:
            for stream in streams:
                yield from stream

        found, report = mine(
            db, chained(), tb=tb, report=report, limit=limit,
            max_solver_moves=max_solver_moves,
            seen=puzzles.seen_positions(db) if dedupe else None,
            on_progress=on_progress,
        )
    return puzzles.save(db, found), report
