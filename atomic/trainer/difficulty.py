"""How hard a puzzle is, measured against the calibrated engine ladder.

Lichess rates a puzzle by Glicko over thousands of human solvers. There is one
solver here, so that road is closed — but this repo has something lichess does
not: `lab.cli levels` has already put a ladder of M1 settings on a measured Elo
scale (2k / 16k / 64k / 1024k nodes = 1366 / 1691 / 1704 / 1951, chained from a
single anchor). `docs/trainer-plan.md` §5 therefore defines difficulty as:

> a puzzle's difficulty is the rating of the weakest calibrated level that
> finds the whole solution line.

Five decisions carry the implementation, each with a plausible alternative that
is worse:

**The engine is held to the solver's standard, exactly.** A level "finds" the
line when it plays the stored solution move at every solver ply — the same test
`server.puzzle_move` applies to you, move for move. Accepting any move the
engine likes as long as it also wins would rate a *position*; this rates the
puzzle you are actually asked to solve.

**Only chained level ratings are used.** `PROGRESS.md` is explicit that
per-budget gauntlet ratings are not comparable across budgets — a weak setting
that survives to play more rungs can out-rate a strong one. `rating.load_levels`
is the chained ladder and is the only source read here. Without it, difficulty
would be a node count wearing an Elo costume, so an uncalibrated store raises
rather than guessing.

**Every rung is run, not just up to the first success.** The ladder is not
strictly ordered in practice: 16k and 64k nodes measured 1691 ± 131 and 1704 ±
135, which is one rung inside the other's error bar. A puzzle solved at 16k and
missed at 64k is therefore a fact about the ladder worth recording rather than
an anomaly to hide behind an early exit, and `monotone` says whether it
happened. The cost is real (the 1024k rung dominates, ~0.5 s per solver move on
this box) and is why rating runs as a job.

**A rating is a bound as often as it is a number.** Solved by the weakest rung
means *at most* 1366 — the ladder cannot see below its own floor, and the true
difficulty of a one-move blast is somewhere under it. Solved by none means above
1951. Both are reported as bounds and never silently rounded into the nearest
rung's Elo, which is the same discipline `lab.rating.Rating` already applies to
a shutout gauntlet.

**Ratings live in their own table.** `puzzles.save` is `INSERT OR REPLACE`, so a
column on `puzzles` would be erased every time the corpus was regenerated at a
higher node budget. A rating is a function of (fen, solution) — which is exactly
what the puzzle id hashes — so it stays valid across regeneration, and it
survives it in `puzzle_ratings` for the same reason `puzzle_attempts` does.

The known limitation, stated in the plan and worth restating: engine-solvable
and human-hard are different axes. An engine finds a 6-ply forced blast
instantly that no human sees, and a quiet only-move that is obvious to a person
can defeat a shallow search. This module measures the first axis honestly and
`puzzles.improvement` reports your solve rate per band beside it, so the second
is visible rather than assumed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import chess
import chess.variant

from lab import methods, rating

from . import puzzles

#: The method whose calibrated ladder defines the scale. `lab.cli levels`
#: measured this one; a different method would need its own calibration run.
LADDER_METHOD = "M1-alphabeta"

Progress = Callable[[str, int, int], None]


class NotCalibrated(RuntimeError):
    """The level ladder has not been measured, so there is no scale to use."""


@dataclass(frozen=True)
class Rung:
    """One calibrated setting of the ladder method."""

    nodes: int
    elo: float
    span: float          # the +/- half-width from the levels calibration
    anchor: bool = False

    @property
    def name(self) -> str:
        return f"{LADDER_METHOD}@{self.nodes:,}n"

    def public(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "elo": self.elo, "span": self.span,
                "anchor": self.anchor}


def ladder() -> list[Rung]:
    """The calibrated rungs, weakest first.

    Sorted by *measured Elo*, not by node count, because those two orders are
    not guaranteed to agree and the definition of difficulty above says
    "weakest level", which is a statement about Elo.
    """
    data = rating.load_levels()
    if not data or not data.get("levels"):
        raise NotCalibrated(
            f"{rating.LEVELS_FILE} is missing — run `python -m lab.cli levels` "
            "before rating puzzles; without it there is no Elo scale to rate on"
        )
    if data.get("method") != LADDER_METHOD:
        raise NotCalibrated(
            f"the level ladder was calibrated for {data.get('method')!r}, "
            f"not {LADDER_METHOD!r}"
        )
    rungs = [
        Rung(nodes=nodes, elo=float(level["elo"]), span=float(level["span"]),
             anchor=bool(level.get("anchor")))
        for nodes, level in data["levels"].items()
    ]
    return sorted(rungs, key=lambda rung: rung.elo)


def ladder_key(rungs: Sequence[Rung]) -> str:
    """A fingerprint of the ladder a verdict was made against.

    Recalibrating the levels changes what an old rating means. Storing the
    ladder with each verdict makes that visible — `stale` below is the query —
    rather than leaving two incomparable scales mixed in one column.
    """
    return json.dumps({str(rung.nodes): round(rung.elo, 1) for rung in rungs},
                      sort_keys=True)


# -- solving ---------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """What one rung did with one puzzle."""

    nodes: int
    elo: float
    solved: bool
    found: int           # solver moves played correctly before the first miss
    total: int           # solver moves the puzzle asks for
    played: str | None   # the move it went wrong with, as UCI; None if solved
    ms: int

    def public(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes, "elo": self.elo, "solved": self.solved,
            "found": self.found, "total": self.total, "played": self.played,
            "ms": self.ms,
        }


def engine(**kwargs: Any) -> methods.Method:
    """The ladder method, or a plain error if the Rust core is not built.

    `methods.build` raises `SystemExit`, which is right for a command line and
    wrong inside a job thread — `jobs.Registry` catches `Exception`, so a
    `SystemExit` here would kill the thread with the job still marked running
    and block every later job.
    """
    try:
        return methods.build(LADDER_METHOD, **kwargs)
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from None


def solve(
    method: methods.Method, puzzle: puzzles.Puzzle, rung: Rung
) -> Attempt:
    """Play one rung through a puzzle's solver moves.

    The defender's replies come from the stored line, not from the engine: the
    puzzle asserts that this is what happens, and letting the engine choose the
    defence would be testing a different claim.

    The transposition table is cleared before each search so the answer does not
    depend on which puzzle came before it. That costs the engine a little
    strength relative to how it plays in a game, and buys a rating that is
    reproducible, which matters more.
    """
    board = chess.variant.AtomicBoard(puzzle.fen)
    total = puzzle.length
    found = 0
    started = time.perf_counter()
    for index, uci in enumerate(puzzle.solution):
        if index % 2:
            board.push(chess.Move.from_uci(uci))    # the scripted defence
            continue
        method.reset()
        chosen = method.choose(board, methods.Budget(nodes=rung.nodes)).uci()
        if chosen != uci:
            return Attempt(
                nodes=rung.nodes, elo=rung.elo, solved=False, found=found,
                total=total, played=chosen,
                ms=int((time.perf_counter() - started) * 1000),
            )
        found += 1
        board.push(chess.Move.from_uci(uci))
    return Attempt(
        nodes=rung.nodes, elo=rung.elo, solved=True, found=found, total=total,
        played=None, ms=int((time.perf_counter() - started) * 1000),
    )


# -- the verdict -----------------------------------------------------------

#: What a rating is bounded by. `""` means a rung in the middle of the ladder
#: solved it and the number is a reading; the other two mean the ladder ran out.
BOUNDS = ("", "below", "above")


@dataclass
class Verdict:
    """One puzzle's difficulty, with the evidence that produced it."""

    puzzle_id: str
    elo: float
    span: float
    bound: str                       # "" / "below" / "above"
    solved_by: int | None            # node budget of the weakest rung that solved
    monotone: bool                   # did every stronger rung solve it too?
    attempts: list[Attempt] = field(default_factory=list)
    ladder: str = ""
    method: str = LADDER_METHOD
    rated_at: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        return puzzles.rating_label(self.elo, self.bound) or "?"

    def describe(self) -> str:
        if self.bound == "below":
            return (f"at or below {self.elo:.0f} Elo — the weakest calibrated "
                    f"level found it, so the ladder cannot see how much easier")
        if self.bound == "above":
            return (f"above {self.elo:.0f} Elo — no calibrated level found the "
                    f"whole line")
        return f"{self.elo:.0f} +/- {self.span:.0f} Elo"

    def public(self) -> dict[str, Any]:
        return {
            "id": self.puzzle_id,
            "elo": self.elo,
            "span": self.span,
            "bound": self.bound,
            "label": self.label,
            "describe": self.describe(),
            "solved_by": self.solved_by,
            "monotone": self.monotone,
            "method": self.method,
            "rated_at": self.rated_at,
            "levels": [attempt.public() for attempt in self.attempts],
        }


def verdict_from(puzzle_id: str, attempts: Sequence[Attempt],
                 rungs: Sequence[Rung]) -> Verdict:
    """Turn a full profile over the ladder into a rating.

    The rating is the weakest rung that solved it. When that is the bottom of
    the ladder the number is an upper bound, when nothing solved it the top rung
    is a lower bound, and `monotone` records whether the rungs above the
    successful one agreed — see the module docstring on why they need not.
    """
    first = next((i for i, a in enumerate(attempts) if a.solved), None)
    by_nodes = {rung.nodes: rung for rung in rungs}
    if first is None:
        top = rungs[-1]
        return Verdict(
            puzzle_id=puzzle_id, elo=top.elo, span=top.span, bound="above",
            solved_by=None, monotone=True, attempts=list(attempts),
            ladder=ladder_key(rungs),
        )
    rung = by_nodes[attempts[first].nodes]
    above = attempts[first + 1:]
    return Verdict(
        puzzle_id=puzzle_id,
        elo=rung.elo,
        span=rung.span,
        bound="below" if first == 0 else "",
        solved_by=rung.nodes,
        monotone=all(a.solved for a in above),
        attempts=list(attempts),
        ladder=ladder_key(rungs),
    )


def rate_puzzle(
    puzzle: puzzles.Puzzle,
    *,
    rungs: Sequence[Rung],
    method: methods.Method | None = None,
) -> Verdict:
    """Run one puzzle up the whole ladder."""
    method = method or engine()
    attempts = [solve(method, puzzle, rung) for rung in rungs]
    return verdict_from(puzzle.id, attempts, rungs)


# -- storage ---------------------------------------------------------------


def save(db: sqlite3.Connection, verdicts: Iterable[Verdict]) -> int:
    written = 0
    for verdict in verdicts:
        db.execute(
            "INSERT OR REPLACE INTO puzzle_ratings "
            "(puzzle_id, elo, span, bound, solved_by, monotone, levels, "
            " method, ladder, rated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                verdict.puzzle_id, verdict.elo, verdict.span, verdict.bound,
                verdict.solved_by, int(verdict.monotone),
                json.dumps([a.public() for a in verdict.attempts]),
                verdict.method, verdict.ladder, verdict.rated_at,
            ),
        )
        written += 1
    db.commit()
    return written


def load(db: sqlite3.Connection, puzzle_id: str) -> Verdict | None:
    row = db.execute(
        "SELECT * FROM puzzle_ratings WHERE puzzle_id = ?", (puzzle_id,)
    ).fetchone()
    return _from_row(row) if row else None


def _from_row(row: sqlite3.Row) -> Verdict:
    return Verdict(
        puzzle_id=row["puzzle_id"],
        elo=row["elo"],
        span=row["span"],
        bound=row["bound"],
        solved_by=row["solved_by"],
        monotone=bool(row["monotone"]),
        attempts=[
            Attempt(
                nodes=a["nodes"], elo=a["elo"], solved=a["solved"],
                found=a["found"], total=a["total"], played=a.get("played"),
                ms=a.get("ms", 0),
            )
            for a in json.loads(row["levels"])
        ],
        ladder=row["ladder"],
        method=row["method"],
        rated_at=row["rated_at"],
    )


def clear(db: sqlite3.Connection) -> int:
    cursor = db.execute("DELETE FROM puzzle_ratings")
    db.commit()
    return cursor.rowcount


def stale(db: sqlite3.Connection, rungs: Sequence[Rung]) -> int:
    """How many stored ratings were made against a different ladder."""
    return db.execute(
        "SELECT COUNT(*) AS n FROM puzzle_ratings WHERE ladder != ?",
        (ladder_key(rungs),),
    ).fetchone()["n"]


# -- the batch run ---------------------------------------------------------


@dataclass
class Report:
    """What a rating run did, in the terms the run is judged by."""

    rated: int = 0
    unsolved: int = 0            # no rung found the line
    floor: int = 0               # the weakest rung found it
    non_monotone: int = 0        # a stronger rung missed what a weaker one found
    seconds: float = 0.0
    bands: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        bands = ", ".join(f"{label} {n}" for label, n in sorted(self.bands.items()))
        return (
            f"rated {self.rated} puzzles in {self.seconds / 60:.1f} min — "
            f"{self.floor} at the ladder floor, {self.unsolved} above its top, "
            f"{self.non_monotone} non-monotone; {bands}"
        )


def pending(
    db: sqlite3.Connection, spec: puzzles.Filter, *, force: bool = False
) -> list[puzzles.Puzzle]:
    """The puzzles a run would rate: everything matching, minus what is done."""
    found = puzzles.search(db, spec)
    if force:
        return found
    done = {
        row["puzzle_id"]
        for row in db.execute("SELECT puzzle_id FROM puzzle_ratings")
    }
    return [puzzle for puzzle in found if puzzle.id not in done]


def run(
    db: sqlite3.Connection,
    *,
    spec: puzzles.Filter | None = None,
    force: bool = False,
    cancelled: Callable[[], bool] | None = None,
    on_progress: Progress | None = None,
) -> tuple[int, Report]:
    """Rate every unrated puzzle matching `spec` against the ladder.

    Committed in batches rather than at the end: a run over a large corpus is
    minutes to hours of Fairy-Stockfish-scale work, and losing all of it to a
    Ctrl-C would make the command something you avoid starting.
    """
    rungs = ladder()
    spec = spec or puzzles.Filter(limit=100_000, order="recent")
    todo = pending(db, spec, force=force)
    report = Report()
    if not todo:
        return 0, report

    method = engine()
    started = time.time()
    batch: list[Verdict] = []
    for index, puzzle in enumerate(todo):
        if cancelled and cancelled():
            break
        verdict = rate_puzzle(puzzle, rungs=rungs, method=method)
        batch.append(verdict)
        report.rated += 1
        report.bands[verdict.label] = report.bands.get(verdict.label, 0) + 1
        if verdict.bound == "above":
            report.unsolved += 1
        elif verdict.bound == "below":
            report.floor += 1
        if not verdict.monotone:
            report.non_monotone += 1
        if len(batch) >= 20:
            save(db, batch)
            batch = []
        if on_progress:
            on_progress("rating", index + 1, len(todo))
    save(db, batch)
    report.seconds = time.time() - started
    return report.rated, report


# -- reading the corpus back ------------------------------------------------


def distribution(db: sqlite3.Connection) -> dict[str, Any]:
    """The corpus by difficulty, plus how you do at each band.

    The second half is the plan's §5 refinement in its cheapest honest form: no
    fitted correction from engine Elo to human Elo, just the two numbers side by
    side so a disagreement between them is visible. A band where the engine
    needs 1951 and you solve 90% is the interesting cell, and it is only
    interesting if nothing has averaged it away.
    """
    rows = db.execute(
        "SELECT r.elo, r.bound, r.solved_by, r.monotone, "
        "       COUNT(*) AS n, "
        "       SUM(EXISTS (SELECT 1 FROM puzzle_attempts a "
        "                   WHERE a.puzzle_id = r.puzzle_id)) AS tried, "
        "       SUM(EXISTS (SELECT 1 FROM puzzle_attempts a "
        "                   WHERE a.puzzle_id = r.puzzle_id AND a.solved = 1)) AS solved "
        "FROM puzzle_ratings r JOIN puzzles p ON p.id = r.puzzle_id "
        "GROUP BY r.elo, r.bound ORDER BY r.elo, r.bound"
    ).fetchall()
    total = db.execute("SELECT COUNT(*) AS n FROM puzzles").fetchone()["n"]
    rated = sum(row["n"] for row in rows)
    bands = [
        {
            "elo": row["elo"],
            "bound": row["bound"],
            "label": puzzles.rating_label(row["elo"], row["bound"]),
            "nodes": row["solved_by"],
            "count": row["n"],
            "tried": row["tried"] or 0,
            "solved": row["solved"] or 0,
            "rate": (100.0 * row["solved"] / row["tried"]) if row["tried"] else None,
        }
        for row in rows
    ]
    non_monotone = db.execute(
        "SELECT COUNT(*) AS n FROM puzzle_ratings WHERE monotone = 0"
    ).fetchone()["n"]
    return {
        "rated": rated,
        "unrated": max(0, total - rated),
        "bands": bands,
        "non_monotone": non_monotone,
        "method": LADDER_METHOD,
    }


def summary(db: sqlite3.Connection) -> dict[str, Any]:
    """What the UI needs to show and filter by difficulty, ladder included."""
    try:
        rungs = ladder()
    except NotCalibrated as exc:
        return {"calibrated": False, "reason": str(exc), "rungs": [],
                **distribution(db)}
    return {
        "calibrated": True,
        "rungs": [rung.public() for rung in rungs],
        "stale": stale(db, rungs),
        **distribution(db),
    }


__all__ = [
    "LADDER_METHOD",
    "Attempt",
    "NotCalibrated",
    "Report",
    "Rung",
    "Verdict",
    "distribution",
    "ladder",
    "load",
    "rate_puzzle",
    "run",
    "save",
    "solve",
    "summary",
]
