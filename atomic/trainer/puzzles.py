"""The puzzle corpus: what one is, how it is stored, how it is asked for.

A puzzle here is a position, a forced line through it, and the claim that the
line is the *only* one that works. The generators in `generate.py` (from your
games) and `endgames.py` (from the tablebase) both produce this shape; nothing
downstream needs to know which one made it.

Three decisions worth stating, because each has an alternative that looks
reasonable and is worse:

**Ids are content hashes.** `id = sha1(fen + solution)`. Regenerating the
corpus with a bigger node budget reproduces the id of every puzzle whose
solution did not change, so the solve history stays attached to the puzzle it
belongs to. An autoincrement id would silently orphan it, and a random uuid
would produce duplicates on every run.

**The solution alternates and has odd length.** Solver, defender, solver, …,
ending on the solver. A line ending on the defender's move would ask the user
to accept a position as "solved" on someone else's move.

**The solution never leaves the server.** `Puzzle.public()` is what the page
gets, and it has no solution in it. This is the same referee rule `lab/server`
already enforces for legality, applied to the answer: a page that knows the
answer will leak it, through the DOM if not through the network tab.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import chess
import chess.variant

from . import themes as themes_mod

#: Where a puzzle came from. `game` is one of *your* imported games, `dump` is
#: a stranger's rated game scraped out of the public Lichess archive, and
#: `tablebase` is exact. The distinction is not cosmetic: only `game` puzzles
#: have a review to link back to, and only they can be "your side".
SOURCES = ("game", "dump", "tablebase")
MOTIVES = ("win", "save")
PHASES = ("opening", "middlegame", "endgame")

#: Solver win%, under best play, at or above which a puzzle is played *for the
#: win* rather than to hold. On the fitted atomic curve — 2.84x flatter than the
#: standard one — 75% is roughly +950cp, so it marks a decisive position rather
#: than a pleasant one.
#:
#: It lives here rather than in `generate`, which is where it started, because
#: three things now need the same number and they must not drift apart: the
#: generator's acceptance test, the migration that relabels an existing corpus,
#: and `Puzzle.goal`, which is what the solver is actually told to play for.
WIN_FLOOR = 75.0


def motive_of(win_before: float | None, fallback: str = "win") -> str:
    """Whether a confirmed line is a win or a hold.

    Read off the *confirmed* evaluation, never the screening one. The two
    disagree more often than they look like they should: candidates are picked
    from the review pass's budget and the line is then re-judged at 4M nodes
    with a third line, and a position that screened at 70% can confirm at 100%.
    Thirty-one puzzles in this corpus were labelled "one move holds" while the
    deeper pass said the solver was winning outright — which is exactly the
    kind of lie that makes a solver stop trusting the label.
    """
    if win_before is None:
        return fallback
    return "win" if win_before >= WIN_FLOOR else "save"


def puzzle_id(fen: str, solution: Sequence[str]) -> str:
    """A stable id for a position and its answer."""
    digest = hashlib.sha1(f"{fen}|{' '.join(solution)}".encode()).hexdigest()
    return digest[:16]


def position_key(fen: str) -> str:
    """The identity of a *position*, for deciding whether two puzzles are one.

    The first four FEN fields: placement, side to move, castling, en passant.
    The halfmove and fullmove counters are dropped for the same reason
    `book.book_fen` drops them — two games that reach the same position at
    different move numbers have reached the same position, and in atomic, where
    the opening tree is narrow and forcing, that happens constantly. Mining a
    thousand rated games without this produces the same tactic over and over
    with a different game id attached to each copy.

    En passant is kept even though it is usually irrelevant, because when it is
    relevant in atomic it is *the whole puzzle* (`ep-blast`), and merging a
    position with an available en passant into one without it would merge two
    genuinely different questions.
    """
    return " ".join(fen.split()[:4])


def rating_label(elo: float | None, bound: str = "") -> str | None:
    """How a measured difficulty is written wherever it is shown.

    Lives here rather than in `difficulty` because both the rater and every
    reader of the corpus need it, and `difficulty` imports this module. The
    inequalities are not decoration: a rating at either end of the ladder is a
    bound, and dropping the sign would turn "the weakest level I have solves
    this" into a claim that the puzzle is worth exactly that Elo.
    """
    if elo is None:
        return None
    if bound == "below":
        return f"≤{elo:.0f}"
    if bound == "above":
        return f">{elo:.0f}"
    return f"{elo:.0f}"


@dataclass
class Puzzle:
    """One puzzle, as generated and as stored."""

    fen: str
    solver: str
    solution: list[str]
    solution_san: list[str]
    themes: list[str] = field(default_factory=list)
    phase: str = "middlegame"
    motive: str = "win"
    source: str = "game"
    game_id: str | None = None
    ply: int | None = None
    setup_fen: str | None = None
    setup_uci: str | None = None
    setup_san: str | None = None
    win_before: float | None = None
    win_second: float | None = None
    unique_gap: float | None = None
    tb_category: str | None = None
    nodes: int | None = None
    mine: bool = False
    created_at: float = field(default_factory=time.time)

    #: Measured difficulty, when it has been measured. Filled in on the way out
    #: of the database from `puzzle_ratings` (see `difficulty.py`) and never
    #: written back through `save` — a rating is not part of what makes a puzzle
    #: this puzzle, and a generator that has just built one has no business
    #: claiming to know how hard it is.
    rating: float | None = None
    rating_bound: str = ""
    rating_span: float | None = None

    @property
    def rating_text(self) -> str | None:
        return rating_label(self.rating, self.rating_bound)

    @property
    def id(self) -> str:
        return puzzle_id(self.fen, self.solution)

    @property
    def pos_key(self) -> str:
        return position_key(self.fen)

    @property
    def length(self) -> int:
        """How many moves the solver has to find."""
        return (len(self.solution) + 1) // 2

    @property
    def goal(self) -> dict[str, Any]:
        """What playing this correctly *achieves*, and how well that is known.

        A solver who thinks they are hunting a win in a position that is only
        held is solving the wrong puzzle, and the corpus has both kinds in it —
        half of it is `save` material, because in a variant this forcing "the
        only move that does not lose" is the commoner skill. So the goal is
        stated rather than implied by a verdict line, and it carries its own
        evidence:

        - `exact` — the atomic tablebase said so, and there is nothing to argue
          with;
        - otherwise `win`, the engine's own win% for the solver after best
          play, so "a draw the engine is too weak to see" is at least visible
          as a number rather than hidden behind the word `win`.
        """
        if self.tb_category:
            return {
                "kind": "win" if self.tb_category in ("win", "cursed-win") else "draw",
                "exact": True,
                "win": None,
                "category": self.tb_category,
            }
        return {
            "kind": "win" if motive_of(self.win_before, self.motive) == "win" else "draw",
            "exact": False,
            "win": self.win_before,
            "category": None,
        }

    def public(self) -> dict[str, Any]:
        """Everything the page may know. Notably not the solution."""
        return {
            "id": self.id,
            "fen": self.fen,
            "solver": self.solver,
            "length": self.length,
            "themes": self.themes,
            "phase": self.phase,
            "motive": self.motive,
            "goal": self.goal,
            "source": self.source,
            "game_id": self.game_id,
            "ply": self.ply,
            "setup_fen": self.setup_fen,
            "setup_uci": self.setup_uci,
            "setup_san": self.setup_san,
            "tb_category": self.tb_category,
            "mine": self.mine,
            "rating": self.rating,
            "rating_bound": self.rating_bound,
            "rating_span": self.rating_span,
            "rating_label": self.rating_text,
        }

    def reveal(self) -> dict[str, Any]:
        """The public view plus the answer, for when it has been earned."""
        return {**self.public(), "solution": self.solution, "san": self.solution_san}


# -- storage ---------------------------------------------------------------


_COLUMNS = (
    "id", "source", "game_id", "ply", "setup_fen", "setup_uci", "setup_san",
    "fen", "solver", "solution", "solution_san", "themes", "phase", "motive",
    "win_before", "win_second", "unique_gap", "tb_category", "nodes", "mine",
    "created_at", "pos_key",
)


def _row_values(puzzle: Puzzle) -> tuple[Any, ...]:
    return (
        puzzle.id, puzzle.source, puzzle.game_id, puzzle.ply, puzzle.setup_fen,
        puzzle.setup_uci, puzzle.setup_san, puzzle.fen, puzzle.solver,
        json.dumps(puzzle.solution), json.dumps(puzzle.solution_san),
        json.dumps(puzzle.themes), puzzle.phase, puzzle.motive,
        puzzle.win_before, puzzle.win_second, puzzle.unique_gap,
        puzzle.tb_category, puzzle.nodes, int(puzzle.mine), puzzle.created_at,
        puzzle.pos_key,
    )


def _from_row(row: sqlite3.Row) -> Puzzle:
    # The rating columns are only there when the caller joined `puzzle_ratings`,
    # so read them defensively: `get` and `search` do join, and a row handed in
    # by a test or a migration need not.
    keys = set(row.keys())
    return Puzzle(
        fen=row["fen"],
        solver=row["solver"],
        solution=json.loads(row["solution"]),
        solution_san=json.loads(row["solution_san"]),
        themes=json.loads(row["themes"]),
        phase=row["phase"],
        motive=row["motive"],
        source=row["source"],
        game_id=row["game_id"],
        ply=row["ply"],
        setup_fen=row["setup_fen"],
        setup_uci=row["setup_uci"],
        setup_san=row["setup_san"],
        win_before=row["win_before"],
        win_second=row["win_second"],
        unique_gap=row["unique_gap"],
        tb_category=row["tb_category"],
        nodes=row["nodes"],
        mine=bool(row["mine"]),
        created_at=row["created_at"],
        rating=row["rating_elo"] if "rating_elo" in keys else None,
        rating_bound=(row["rating_bound"] or "") if "rating_bound" in keys else "",
        rating_span=row["rating_span"] if "rating_span" in keys else None,
    )


#: Every read of a puzzle carries its measured difficulty when there is one.
#: A LEFT JOIN rather than a second query per puzzle: the list view shows
#: ratings on 200 rows at a time.
_WITH_RATING = (
    "SELECT p.*, r.elo AS rating_elo, r.bound AS rating_bound, "
    "       r.span AS rating_span "
    "FROM puzzles p LEFT JOIN puzzle_ratings r ON r.puzzle_id = p.id"
)


def save(db: sqlite3.Connection, found: Iterable[Puzzle]) -> int:
    """Insert or refresh puzzles, keeping their theme index in step.

    `INSERT OR REPLACE` rather than `INSERT OR IGNORE`: regenerating at a
    higher node budget should update the recorded evaluations and themes of a
    puzzle whose solution is unchanged, not leave the old numbers in place.
    Attempts are untouched — `puzzle_attempts` deliberately carries no foreign
    key, so a replaced puzzle keeps the history recorded against its id.
    """
    written = 0
    for puzzle in found:
        db.execute(
            f"INSERT OR REPLACE INTO puzzles ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(_COLUMNS))})",
            _row_values(puzzle),
        )
        db.execute("DELETE FROM puzzle_themes WHERE puzzle_id = ?", (puzzle.id,))
        db.executemany(
            "INSERT INTO puzzle_themes (puzzle_id, theme) VALUES (?, ?)",
            [(puzzle.id, theme) for theme in puzzle.themes],
        )
        written += 1
    db.commit()
    return written


def provenance(game_id: str | None, ply: int | None) -> str:
    """Which moment of which game a puzzle was taken from."""
    return f"{game_id or '-'}:{ply if ply is not None else '-'}"


def seen_positions(db: sqlite3.Connection) -> dict[str, set[str]]:
    """Every position the corpus covers, and where each copy came from.

    Used by the generators to refuse a duplicate *before* spending the
    confirming pass on it, which is where the saving is: mining a thousand
    rated games at 4M nodes a survivor, duplicates are not a storage problem,
    they are hours of Fairy-Stockfish.

    The provenance is carried rather than just the key so that regenerating is
    not mistaken for duplicating. Re-running the generator over your own games
    at a higher node budget re-derives the *same* moment of the same game, and
    that has to be allowed through — `save` will refresh the row in place and
    keep its solve history. Only a position arriving from somewhere new is a
    duplicate.
    """
    out: dict[str, set[str]] = {}
    for row in db.execute(
        "SELECT pos_key, game_id, ply FROM puzzles WHERE pos_key IS NOT NULL"
    ):
        out.setdefault(row["pos_key"], set()).add(
            provenance(row["game_id"], row["ply"])
        )
    return out


def get(db: sqlite3.Connection, puzzle_id_: str) -> Puzzle | None:
    row = db.execute(
        f"{_WITH_RATING} WHERE p.id = ?", (puzzle_id_,)
    ).fetchone()
    return _from_row(row) if row else None


def clear(db: sqlite3.Connection, source: str | None = None) -> int:
    """Drop generated puzzles. Solve history survives; it is not derived."""
    cursor = (
        db.execute("DELETE FROM puzzles WHERE source = ?", (source,))
        if source
        else db.execute("DELETE FROM puzzles")
    )
    db.commit()
    return cursor.rowcount


# -- querying --------------------------------------------------------------


@dataclass
class Filter:
    """What to pick out of the corpus. Every field is optional and ANDed."""

    theme: str | None = None
    phase: str | None = None
    source: str | None = None
    motive: str | None = None
    game_id: str | None = None
    mine: bool = False
    unsolved: bool = False
    #: Never attempted at all — the queue that does not repeat itself. Stricter
    #: than `unsolved`, which keeps returning a puzzle you have failed; that is
    #: the right default for a redo queue and the wrong one for a session of
    #: new material, so the two are separate flags and the page picks.
    unseen: bool = False
    failed: bool = False
    #: Measured difficulty, in the Elo units of `difficulty.py`. `rated` is
    #: tri-state: None means "do not care", True only rated puzzles, False only
    #: the ones still waiting for the ladder — which is what the rating job
    #: asks for.
    min_rating: float | None = None
    max_rating: float | None = None
    rated: bool | None = None
    limit: int = 50
    order: str = "recent"     # recent / random / easiest / hardest / ...

    def _where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.theme:
            clauses.append(
                "p.id IN (SELECT puzzle_id FROM puzzle_themes WHERE theme = ?)"
            )
            params.append(self.theme)
        for column, value in (
            ("phase", self.phase),
            ("source", self.source),
            ("motive", self.motive),
            ("game_id", self.game_id),
        ):
            if value:
                clauses.append(f"p.{column} = ?")
                params.append(value)
        if self.mine:
            clauses.append("p.mine = 1")
        if self.unsolved:
            # Never solved cleanly. A puzzle failed three times is still
            # unsolved and should keep coming back; that is the point.
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM puzzle_attempts a "
                "WHERE a.puzzle_id = p.id AND a.solved = 1)"
            )
        if self.unseen:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM puzzle_attempts a WHERE a.puzzle_id = p.id)"
            )
        if self.failed:
            # Narrower than `unsolved`, and the one the improvement tab wants:
            # a puzzle you have *tried* and never solved. `unsolved` also
            # includes everything you have never seen, which is most of the
            # corpus and is not evidence of anything.
            clauses.append(
                "EXISTS (SELECT 1 FROM puzzle_attempts a WHERE a.puzzle_id = p.id) "
                "AND NOT EXISTS (SELECT 1 FROM puzzle_attempts a "
                "WHERE a.puzzle_id = p.id AND a.solved = 1)"
            )
        if self.rated is not None:
            clauses.append("r.elo IS NOT NULL" if self.rated else "r.elo IS NULL")
        # A rating band never matches an unrated puzzle. "I have not measured
        # this" is not evidence that it falls inside your band, and letting it
        # through would fill a "hardest puzzles" queue with the ones nothing has
        # looked at yet.
        if self.min_rating is not None:
            clauses.append("r.elo >= ?")
            params.append(self.min_rating)
        if self.max_rating is not None:
            clauses.append("r.elo <= ?")
            params.append(self.max_rating)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


_ORDERS = {
    "recent": "p.created_at DESC",
    "random": "RANDOM()",
    # Line length, which is not a difficulty and was the only ordering the
    # corpus supported before there were ratings. Kept: "give me the one-movers"
    # is a real request and `easiest` does not answer it.
    "shortest": "LENGTH(p.solution) ASC, p.created_at DESC",
    "longest": "LENGTH(p.solution) DESC, p.created_at DESC",
    # Measured difficulty. Unrated puzzles sort last in both directions rather
    # than clustering at one end, because a missing rating is not a low one.
    "easiest": "r.elo IS NULL, r.elo ASC, LENGTH(p.solution) ASC",
    "hardest": "r.elo IS NULL, r.elo DESC, LENGTH(p.solution) DESC",
}


def search(db: sqlite3.Connection, spec: Filter) -> list[Puzzle]:
    where, params = spec._where()
    order = _ORDERS.get(spec.order, _ORDERS["recent"])
    rows = db.execute(
        f"{_WITH_RATING}{where} ORDER BY {order} LIMIT ?",
        [*params, spec.limit],
    )
    return [_from_row(row) for row in rows]


def count(db: sqlite3.Connection, spec: Filter) -> int:
    where, params = spec._where()
    return db.execute(
        "SELECT COUNT(*) AS n FROM puzzles p "
        "LEFT JOIN puzzle_ratings r ON r.puzzle_id = p.id" + where,
        params,
    ).fetchone()["n"]


def theme_counts(db: sqlite3.Connection) -> dict[str, int]:
    return {
        row["theme"]: row["n"]
        for row in db.execute(
            "SELECT theme, COUNT(*) AS n FROM puzzle_themes "
            "GROUP BY theme ORDER BY n DESC"
        )
    }


def stats(db: sqlite3.Connection) -> dict[str, Any]:
    row = db.execute(
        "SELECT COUNT(*) AS n, SUM(source = 'game') AS from_games, "
        "SUM(source = 'tablebase') AS from_tb, SUM(source = 'dump') AS from_dump, "
        "SUM(mine) AS mine, SUM(motive = 'save') AS saves FROM puzzles"
    ).fetchone()
    attempts = db.execute(
        "SELECT COUNT(*) AS n, SUM(solved) AS solved, "
        "COUNT(DISTINCT puzzle_id) AS touched FROM puzzle_attempts"
    ).fetchone()
    solved = db.execute(
        "SELECT COUNT(DISTINCT puzzle_id) AS n FROM puzzle_attempts WHERE solved = 1"
    ).fetchone()["n"]
    phases = {
        row["phase"]: row["n"]
        for row in db.execute("SELECT phase, COUNT(*) AS n FROM puzzles GROUP BY phase")
    }
    rated = db.execute(
        "SELECT COUNT(*) AS n, AVG(elo) AS mean, MIN(elo) AS lo, MAX(elo) AS hi "
        "FROM puzzle_ratings r JOIN puzzles p ON p.id = r.puzzle_id"
    ).fetchone()
    return {
        "total": row["n"] or 0,
        "rated": rated["n"] or 0,
        "unrated": (row["n"] or 0) - (rated["n"] or 0),
        "rating_mean": rated["mean"],
        "rating_min": rated["lo"],
        "rating_max": rated["hi"],
        "from_games": row["from_games"] or 0,
        "from_tablebase": row["from_tb"] or 0,
        "from_dump": row["from_dump"] or 0,
        "mine": row["mine"] or 0,
        "saves": row["saves"] or 0,
        "phases": phases,
        "themes": theme_counts(db),
        "attempts": attempts["n"] or 0,
        "attempts_solved": attempts["solved"] or 0,
        "puzzles_attempted": attempts["touched"] or 0,
        "puzzles_solved": solved or 0,
    }


# -- attempts --------------------------------------------------------------


def record_attempt(
    db: sqlite3.Connection,
    puzzle_id_: str,
    *,
    solved: bool,
    hints: int = 0,
    wrong: int = 0,
    ms: int | None = None,
) -> None:
    """Log one go at a puzzle.

    `solved` means found unaided: no hint taken and no wrong move tried. A
    puzzle you got right on the third guess is not one you can solve, and
    recording it as solved would quietly empty the `unsolved` queue of exactly
    the puzzles that belong in it.
    """
    db.execute(
        "INSERT INTO puzzle_attempts (puzzle_id, at, solved, hints, wrong, ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (puzzle_id_, time.time(), int(bool(solved)), hints, wrong, ms),
    )
    db.commit()


def history(db: sqlite3.Connection, puzzle_id_: str) -> list[sqlite3.Row]:
    return list(
        db.execute(
            "SELECT * FROM puzzle_attempts WHERE puzzle_id = ? ORDER BY at",
            (puzzle_id_,),
        )
    )


# -- what you are good at --------------------------------------------------
#
# The improvement report. Two decisions carry it, and both exist to stop it
# from lying with small numbers:
#
# **The unit is the first attempt at a puzzle**, not the attempt. Puzzles come
# back — that is what the `unsolved` queue is for — and counting the fourth go
# at a position you have now memorised as evidence that you are good at
# `king-blast` would make every theme look strong the more you drilled it.
#
# **Nothing is called a strength or a weakness without an interval that says
# so.** A theme with three attempts is not a finding, and the difference
# between 60% and 70% on twenty puzzles is not either. Each rate carries a
# Wilson score interval, and a theme is only reported as strong or weak when
# its interval clears the overall rate — the same standard the Elo tables in
# `PROGRESS.md` are held to.

#: Below this many first attempts, a theme gets an interval and no verdict.
MIN_SAMPLE = 5


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% confidence interval for a proportion, as percentages.

    Wilson rather than the normal approximation because the counts here are
    small and the rates are near the edges, which is exactly where `p ± z·se`
    produces intervals that run past 0 or 100.
    """
    if n <= 0:
        return (0.0, 100.0)
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return (max(0.0, centre - spread) * 100.0, min(1.0, centre + spread) * 100.0)


@dataclass
class Skill:
    """How you do on one slice of the corpus, with an interval."""

    key: str
    seen: int            # puzzles attempted at least once
    solved: int          # of those, found unaided at the first sight
    avg_ms: float | None
    lower: float
    upper: float

    @property
    def rate(self) -> float | None:
        return 100.0 * self.solved / self.seen if self.seen else None

    def verdict(self, overall: float) -> str:
        if self.seen < MIN_SAMPLE:
            return "too few to judge"
        if self.lower > overall:
            return "strong"
        if self.upper < overall:
            return "weak"
        return "as usual"

    def public(self, overall: float) -> dict[str, Any]:
        return {
            "key": self.key,
            "seen": self.seen,
            "solved": self.solved,
            "rate": self.rate,
            "lower": self.lower,
            "upper": self.upper,
            "avg_ms": self.avg_ms,
            "verdict": self.verdict(overall),
        }


def _first_attempts(db: sqlite3.Connection) -> list[sqlite3.Row]:
    """One row per puzzle ever attempted: the first go at it, with its puzzle.

    `MIN(id)` rather than `MIN(at)`: two attempts logged in the same
    millisecond are possible and the autoincrement is not ambiguous.
    """
    return list(
        db.execute(
            "SELECT a.puzzle_id, a.solved, a.ms, a.at, a.wrong, a.hints, "
            "p.themes, p.phase, p.motive, p.source, p.mine, "
            "r.elo AS rating_elo, r.bound AS rating_bound "
            "FROM puzzle_attempts a JOIN puzzles p ON p.id = a.puzzle_id "
            "LEFT JOIN puzzle_ratings r ON r.puzzle_id = p.id "
            "WHERE a.id IN (SELECT MIN(id) FROM puzzle_attempts GROUP BY puzzle_id)"
        )
    )


def _tally(rows: Iterable[tuple[str, sqlite3.Row]]) -> list[Skill]:
    """Group (key, attempt-row) pairs into `Skill`s, commonest slice first."""
    buckets: dict[str, list[sqlite3.Row]] = {}
    for key, row in rows:
        buckets.setdefault(key, []).append(row)
    out = []
    for key, group in buckets.items():
        solved = sum(1 for row in group if row["solved"])
        times = [row["ms"] for row in group if row["ms"]]
        lower, upper = wilson(solved, len(group))
        out.append(
            Skill(
                key=key,
                seen=len(group),
                solved=solved,
                avg_ms=sum(times) / len(times) if times else None,
                lower=lower,
                upper=upper,
            )
        )
    out.sort(key=lambda skill: (-skill.seen, skill.key))
    return out


def improvement(db: sqlite3.Connection) -> dict[str, Any]:
    """The whole report: overall rate, each dimension, and what to redo."""
    rows = _first_attempts(db)
    seen = len(rows)
    solved = sum(1 for row in rows if row["solved"])
    overall = 100.0 * solved / seen if seen else 0.0
    lower, upper = wilson(solved, seen)

    dimensions = {
        "themes": _tally(
            (theme, row) for row in rows for theme in json.loads(row["themes"] or "[]")
        ),
        "phases": _tally((row["phase"], row) for row in rows),
        "motives": _tally((row["motive"], row) for row in rows),
        "sources": _tally((row["source"], row) for row in rows),
        # Your solve rate against the engine's difficulty, which is the whole
        # point of measuring difficulty at all: if the two axes agreed
        # perfectly this row would descend monotonically, and where it does not
        # is where "engine-hard" and "human-hard" come apart (plan §5).
        # Unrated puzzles are their own bucket rather than being dropped —
        # hiding them would make the sample look more complete than it is.
        "difficulty": _tally(
            (rating_label(row["rating_elo"], row["rating_bound"] or "") or "unrated",
             row)
            for row in rows
        ),
    }

    redo = search(db, Filter(failed=True, limit=40, order="recent"))
    # Every attempt, oldest first, so the page can draw the trend rather than
    # only the total. Cheap: this table has one row per go at a puzzle.
    timeline = [
        {"at": row["at"], "solved": bool(row["solved"])}
        for row in db.execute(
            "SELECT at, solved FROM puzzle_attempts ORDER BY at"
        )
    ]

    return {
        "overall": {
            "seen": seen,
            "solved": solved,
            "rate": overall,
            "lower": lower,
            "upper": upper,
            "attempts": len(timeline),
            "corpus": count(db, Filter()),
            "min_sample": MIN_SAMPLE,
        },
        **{
            name: [skill.public(overall) for skill in skills]
            for name, skills in dimensions.items()
        },
        "redo": [puzzle.public() for puzzle in redo],
        "timeline": timeline,
    }


def recent_attempts(db: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = db.execute(
        "SELECT a.*, p.fen, p.phase, p.themes, p.source FROM puzzle_attempts a "
        "LEFT JOIN puzzles p ON p.id = a.puzzle_id ORDER BY a.at DESC LIMIT ?",
        (limit,),
    )
    return [
        {
            "puzzle_id": row["puzzle_id"],
            "at": row["at"],
            "solved": bool(row["solved"]),
            "hints": row["hints"],
            "wrong": row["wrong"],
            "ms": row["ms"],
            "phase": row["phase"],
            "themes": json.loads(row["themes"]) if row["themes"] else [],
        }
        for row in rows
    ]


# -- display ---------------------------------------------------------------


def line_san(puzzle: Puzzle) -> str:
    """The solution as a readable line, numbered from the puzzle's own ply."""
    ply = puzzle.ply if puzzle.ply is not None else 0
    out: list[str] = []
    for index, san in enumerate(puzzle.solution_san):
        at = ply + index
        if at % 2 == 0:
            out.append(f"{at // 2 + 1}.{san}")
        elif not out:
            out.append(f"{at // 2 + 1}...{san}")
        else:
            out.append(san)
    return " ".join(out)


def solution_sans(fen: str, solution: Sequence[str]) -> list[str]:
    """SAN for each move of a line, which needs the board it is played on."""
    board = chess.variant.AtomicBoard(fen)
    out = []
    for uci in solution:
        move = chess.Move.from_uci(uci)
        out.append(board.san(move))
        board.push(move)
    return out


def annotate(
    puzzle: Puzzle, *, in_book: bool = False
) -> Puzzle:
    """Fill in phase and themes from the position and the line."""
    board = chess.variant.AtomicBoard(puzzle.fen)
    puzzle.phase = themes_mod.phase_of(board, puzzle.ply)
    puzzle.themes = themes_mod.detect(
        puzzle.fen,
        puzzle.solution,
        source=puzzle.source,
        in_book=in_book,
        ply=puzzle.ply,
    )
    return puzzle
