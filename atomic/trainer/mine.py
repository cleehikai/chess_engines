"""Scraping puzzles out of other people's rated atomic games.

`generate.py` mines the games you played. This mines the 28.9M rated atomic
games Lichess publishes, which is the only supply of atomic positions large
enough to build a corpus that is not a portrait of one account's habits.

**Where the games come from, and why not the API.** The Lichess API exports
games *per user*: there is no "give me rated atomic games in this rating band"
endpoint, and there was never going to be — hammering `/api/games/user` down a
crawl of usernames would be slow, rude, and against the terms. The published
monthly `.pgn.zst` dumps are the supported bulk path and are what
`database.lichess.org` exists for, so that is what this reads. `dump.py`
already streams them for the opening book, and the months it has cached are
reused rather than re-downloaded.

**The funnel, and why it is two-tier.** `docs/trainer-plan.md` §3 predicted
this: a deep 1M-node pass on every ply is affordable over your own 86 games and
is not affordable over thousands of strangers'. So the scan runs at
`SCAN_NODES` — cheap enough to sweep a game in under a second — and only its
survivors go through the same 4M-node confirming pass that `generate.py` uses.
The two-tier funnel was left in the code for exactly this moment; the
thresholds are shared with the review path (`generate.candidates_in_game`) and
are not re-tuned here, so a scraped puzzle means the same thing as a mined one.

**Trusting a cheap scan.** It is a screen, not a verdict — every position it
proposes is re-judged at `generate.CONFIRM_NODES` before becoming a puzzle, and
the calibrated ladder says 20k nodes is already a ~1900-Elo opinion in atomic.
What a cheap scan costs is *recall*: positions it misjudges never reach stage 2
and are lost silently. That is the honest trade, and it is why the scan budget
is a parameter and the node count is recorded on every row it writes.

**Deduplication.** Atomic's opening tree is narrow and forcing — 20 of one
account's 86 games open 1.Nf3 f6 — so the same critical position recurs across
unrelated games constantly. Without a position key a thousand-game scrape
produces the same tactic a dozen times with a different game id on each copy.
Puzzles are therefore deduplicated on the position itself
(`puzzles.position_key`, which ignores the move counters so a transposition
counts as the same position), against the whole existing corpus and within the
run, *before* the confirming pass is spent on them.

Scanned games are recorded in `mined_games` whether or not they produced
anything, so a second run continues rather than re-scanning. They are
deliberately **not** written to `games`: that table is your own history, and
every personal report is a query over it.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.variant

from . import analysis, dump, generate, puzzles, winrate

#: The screening budget. 20k nodes is one of the calibrated FSF rungs
#: (`PROGRESS.md`: 1892 Elo), so this is not a guess about what it can see —
#: it is a rung with a measured rating, roughly 300 Elo below the top of the
#: ladder and about 50x cheaper than the review pass.
SCAN_NODES = 20_000
SCAN_MULTIPV = 2

#: Both players at least this strong, so the losing side's mistake is a
#: mistake and not noise. The atomic pool is small; 1800 keeps roughly the top
#: slice without emptying the corpus.
MIN_ELO = 1800

#: Games shorter than this ended before anything happened; longer than this are
#: usually a bullet scramble whose "blunders" are clock artefacts. The atomic
#: median is 30 ply, so the window is generous at both ends.
MIN_PLY = 12
MAX_PLY = 120

#: Only games that ended over the board. A flag fall or an abort says nothing
#: about the position, which is the same rule the win% fit used.
TERMINATION = "Normal"


@dataclass
class Report:
    """What a scrape did, at each stage of the funnel."""

    games_read: int = 0
    games_eligible: int = 0
    games_scanned: int = 0
    positions_analysed: int = 0
    kept: int = 0
    stage2: generate.Report = field(default_factory=generate.Report)

    def line(self) -> str:
        return (
            f"read {self.games_read} games, {self.games_eligible} eligible, "
            f"{self.games_scanned} scanned at {SCAN_NODES:,} nodes "
            f"({self.positions_analysed} positions); {self.stage2.line()}"
        )


# -- stage 0: which games are worth an engine at all ------------------------


def eligible(
    game: dump.RawGame,
    *,
    min_elo: int = MIN_ELO,
    min_ply: int = MIN_PLY,
    max_ply: int = MAX_PLY,
) -> bool:
    """Is this dump game worth scanning? Decided without touching a board."""
    if game.headers.get("Termination") != TERMINATION:
        return False
    white, black = game.elo("White"), game.elo("Black")
    if white is None or black is None or min(white, black) < min_elo:
        return False
    return min_ply <= len(game.moves()) <= max_ply


def game_id(game: dump.RawGame) -> str | None:
    """The Lichess id, from the `Site` header (`https://lichess.org/<id>`)."""
    site = game.headers.get("Site", "")
    tail = site.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def months_available(directory: Path | None = None) -> list[str]:
    """Months already downloaded, newest first. No network."""
    directory = directory or dump.DUMPS
    if not directory.is_dir():
        return []
    months = [
        path.name.removeprefix("lichess_db_atomic_rated_").removesuffix(".pgn.zst")
        for path in directory.glob("lichess_db_atomic_rated_*.pgn.zst")
    ]
    return sorted(months, reverse=True)


def stream(
    months: Sequence[str],
    *,
    skip: set[str],
    min_elo: int = MIN_ELO,
    min_ply: int = MIN_PLY,
    max_ply: int = MAX_PLY,
    fetch_missing: bool = False,
    report: Report | None = None,
) -> Iterator[tuple[str, str, dump.RawGame]]:
    """Yield `(month, id, game)` for every eligible, not-yet-scanned game.

    Filtering happens on the headers, before the movetext is parsed into a
    board, which is the whole reason `dump.py` splits games rather than using
    `chess.pgn`: a run that keeps 3% of games pays board costs on 3% of them.
    """
    for month in months:
        path = dump.DUMPS / f"lichess_db_atomic_rated_{month}.pgn.zst"
        if not path.exists():
            if not fetch_missing:
                continue
            path = dump.fetch(month)
        for game in dump.read(path):
            if report is not None:
                report.games_read += 1
            if not eligible(game, min_elo=min_elo, min_ply=min_ply, max_ply=max_ply):
                continue
            identifier = game_id(game)
            if identifier is None or identifier in skip:
                continue
            if report is not None:
                report.games_eligible += 1
            yield month, identifier, game


def already_scanned(db: sqlite3.Connection) -> set[str]:
    return {row["id"] for row in db.execute("SELECT id FROM mined_games")}


def _record(
    db: sqlite3.Connection,
    identifier: str,
    month: str,
    game: dump.RawGame,
    ply: int,
    nodes: int,
    kept: int,
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO mined_games "
        "(id, month, white_elo, black_elo, ply, scanned_at, nodes, kept) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            identifier, month, game.elo("White"), game.elo("Black"), ply,
            time.time(), nodes, kept,
        ),
    )


# -- stage 1: the cheap sweep ----------------------------------------------


def scan_game(
    db: sqlite3.Connection,
    pool: analysis.EnginePool,
    executor: ThreadPoolExecutor,
    identifier: str,
    moves: Sequence[str],
    *,
    model: winrate.WinModel,
    nodes: int = SCAN_NODES,
    multipv: int = SCAN_MULTIPV,
    saves: bool = True,
    screen_gap: float = generate.SCREEN_GAP,
) -> list[generate.Candidate]:
    """Judge every ply of one dump game cheaply and return its candidates.

    `analysis.judge_game` does the judging — the same function the review pass
    over your own games uses, at a different node count. Its results are *not*
    written to `plies`: those are judgements about your play, and a stranger's
    game has no business in the table the habit clustering reads.
    """
    judged = analysis.judge_game(db, pool, moves, model, nodes, multipv, executor)
    if not judged:
        return []
    board = chess.variant.AtomicBoard()
    for san in moves:
        board.push_san(san)
    return generate.candidates_in_game(
        db,
        [
            {
                "ply": j.ply, "fen": j.fen, "mover": j.mover, "uci": j.uci,
                "san": j.san, "loss": j.loss, "win_before": j.win_before,
            }
            for j in judged
        ],
        board.fen(),
        model=model,
        game_id=identifier,
        nodes=nodes,
        multipv=multipv,
        my_color=None,          # nobody here is you
        source="dump",
        saves=saves,
        screen_gap=screen_gap,
    )


# -- the whole run ----------------------------------------------------------


def run(
    db: sqlite3.Connection,
    *,
    months: Sequence[str] | None = None,
    games: int = 200,
    model: winrate.WinModel | None = None,
    min_elo: int = MIN_ELO,
    min_ply: int = MIN_PLY,
    max_ply: int = MAX_PLY,
    scan_nodes: int = SCAN_NODES,
    confirm_nodes: int = generate.CONFIRM_NODES,
    unique_gap: float = generate.UNIQUE_GAP,
    screen_gap: float = generate.SCREEN_GAP,
    max_solver_moves: int = generate.MAX_SOLVER_MOVES,
    saves: bool = True,
    batch: int = 40,
    workers: int | None = None,
    fetch_missing: bool = False,
    cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[int, Report]:
    """Scan `games` unseen rated games and store the puzzles they yield.

    Candidates are confirmed in batches rather than all at the end, so a run
    that is interrupted — or cancelled from the UI, which is the common case —
    keeps the puzzles it has already paid for instead of throwing the lot away.
    """
    model = model or winrate.load()
    months = list(months or months_available())
    if not months:
        raise FileNotFoundError(
            "no Lichess atomic dumps in data/dumps — run `trainer book build` "
            "or pass --fetch to download a month"
        )

    report = Report()
    seen = puzzles.seen_positions(db)
    skip = already_scanned(db)
    written = 0
    pending: list[generate.Candidate] = []
    source = stream(
        months,
        skip=skip,
        min_elo=min_elo,
        min_ply=min_ply,
        max_ply=max_ply,
        fetch_missing=fetch_missing,
        report=report,
    )

    def stopped() -> bool:
        return bool(cancelled and cancelled())

    def flush() -> int:
        nonlocal pending
        if not pending:
            return 0
        made, stage2 = generate.generate(
            db, pending,
            model=model, nodes=confirm_nodes, multipv=generate.CONFIRM_MULTIPV,
            unique_gap=unique_gap, max_solver_moves=max_solver_moves,
            workers=workers, seen=seen,
        )
        for field_name in (
            "candidates", "duplicate", "trivial", "not_unique", "too_short",
            "kept", "wins", "saves", "analysed",
        ):
            setattr(
                report.stage2, field_name,
                getattr(report.stage2, field_name) + getattr(stage2, field_name),
            )
        pending = []
        stored = puzzles.save(db, made)
        report.kept += stored
        return stored

    with (
        analysis.EnginePool(scan_nodes) as pool,
        ThreadPoolExecutor(max_workers=workers or analysis.default_workers()) as executor,
    ):
        for month, identifier, game in source:
            if report.games_scanned >= games or stopped():
                break
            moves = game.moves()
            try:
                found = scan_game(
                    db, pool, executor, identifier, moves,
                    model=model, nodes=scan_nodes, saves=saves,
                    screen_gap=screen_gap,
                )
            except ValueError:
                # An unparsable game is a defect in one row of a 3M-game file,
                # not a reason to abandon the run.
                found = []
            report.games_scanned += 1
            report.positions_analysed += len(moves) + 1
            _record(db, identifier, month, game, len(moves), scan_nodes, len(found))
            pending.extend(found)
            if on_progress:
                on_progress("scanning", report.games_scanned, games)
            if len(pending) >= batch:
                if on_progress:
                    on_progress("confirming", report.games_scanned, games)
                written += flush()
            db.commit()
        if on_progress:
            on_progress("confirming", report.games_scanned, games)
        written += flush()
        db.commit()

    # Every puzzle links back to a real game, and the book knows the opening,
    # so `opening-trap` fires on scraped puzzles exactly as it does on mined
    # ones. Nothing else about a scraped puzzle differs.
    return written, report


def stats(db: sqlite3.Connection) -> dict[str, object]:
    """What has been scraped so far, and what there is left to scrape."""
    row = db.execute(
        "SELECT COUNT(*) AS n, SUM(kept) AS kept, MIN(month) AS oldest, "
        "MAX(month) AS newest, SUM(ply) AS plies FROM mined_games"
    ).fetchone()
    puzzles_made = db.execute(
        "SELECT COUNT(*) AS n FROM puzzles WHERE source = 'dump'"
    ).fetchone()["n"]
    return {
        "games": row["n"] or 0,
        "candidates": row["kept"] or 0,
        "plies": row["plies"] or 0,
        "months": [row["oldest"], row["newest"]],
        "available": months_available(),
        "puzzles": puzzles_made,
        "scan_nodes": SCAN_NODES,
        "min_elo": MIN_ELO,
    }


__all__ = ["Report", "eligible", "game_id", "months_available", "run", "scan_game", "stats"]
