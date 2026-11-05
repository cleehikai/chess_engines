"""SQLite store for the trainer.

Three kinds of data, with very different lifetimes:

- **games** — cheap to re-fetch from Lichess, so losing them costs a download.
- **plies** — the per-move judgement of a game. Derived, cheap to recompute
  *given* the analysis cache.
- **positions** — the analysis cache. This is the expensive asset: a deep pass
  over one game costs 30-60s of Fairy-Stockfish time, and every feature
  downstream (review, puzzle generation, repertoire validation) reads from it
  rather than re-analysing. It is keyed by position rather than by game, so two
  games that transpose pay once, which in atomic — where the opening tree is
  narrow and 20 of your 87 games start 1.Nf3 f6 — is a real saving.

Migrations may rebuild `plies` freely. They must never drop `positions`.

Concurrency: WAL plus a long busy timeout, because the analysis pass runs
several Fairy-Stockfish sessions at once. Writes still funnel through the
parent thread (see `analysis.py`); the timeout is belt-and-braces for two
commands running at the same time.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lab import config

DB_PATH: Path = config.DATA / "trainer.sqlite"

SCHEMA_VERSION = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per imported game.
CREATE TABLE IF NOT EXISTS games (
    id            TEXT PRIMARY KEY,   -- lichess game id
    site          TEXT NOT NULL,      -- 'lichess'
    account       TEXT NOT NULL,      -- the account this was imported for
    my_color      TEXT,               -- 'white' / 'black' / NULL if not ours
    rated         INTEGER NOT NULL,
    speed         TEXT,
    source        TEXT,               -- lobby / arena / ai / friend ...
    created_at    INTEGER,            -- ms since epoch, from Lichess
    white         TEXT,
    black         TEXT,
    white_rating  INTEGER,
    black_rating  INTEGER,
    winner        TEXT,               -- 'white' / 'black' / NULL for a draw
    status        TEXT,               -- mate / resign / variantEnd / outoftime ...
    moves         TEXT NOT NULL,      -- SAN, space separated
    ply           INTEGER NOT NULL,
    raw           TEXT NOT NULL,      -- the original NDJSON line
    analysed_at   REAL,               -- when the deep pass finished
    analysed_nodes INTEGER            -- what it was analysed at; NULL = not yet
);

CREATE INDEX IF NOT EXISTS games_account ON games(account, created_at);
CREATE INDEX IF NOT EXISTS games_pending ON games(analysed_nodes) WHERE analysed_nodes IS NULL;

-- The analysis cache. `lines` is the MultiPV result from the side to move's
-- point of view: [{"uci":..., "cp":..., "mate":..., "pv":[...]}, ...], best
-- first. Scores are POV scores — never White-relative — because every consumer
-- wants "how good is this for the player about to move".
CREATE TABLE IF NOT EXISTS positions (
    fen        TEXT NOT NULL,
    nodes      INTEGER NOT NULL,
    multipv    INTEGER NOT NULL,
    engine     TEXT NOT NULL,
    lines      TEXT NOT NULL,
    depth      INTEGER,
    created_at REAL NOT NULL,
    PRIMARY KEY (fen, nodes, multipv, engine)
);

-- One row per move played, with the judgement of it.
CREATE TABLE IF NOT EXISTS plies (
    game_id     TEXT NOT NULL,
    ply         INTEGER NOT NULL,     -- 0-based; ply 0 is White's first move
    fen         TEXT NOT NULL,        -- position BEFORE the move
    mover       TEXT NOT NULL,        -- 'white' / 'black'
    uci         TEXT NOT NULL,
    san         TEXT NOT NULL,
    win_before  REAL NOT NULL,        -- win% for the mover, before
    win_after   REAL NOT NULL,        -- win% for the mover, after
    loss        REAL NOT NULL,        -- win_before - win_after, in win% points
    judgement   TEXT NOT NULL,        -- ok / inaccuracy / mistake / blunder
    best_uci    TEXT,
    best_san    TEXT,
    win_best    REAL,                 -- win% of the best move, for the mover
    win_second  REAL,                 -- win% of the second best; NULL if none
    unique_gap  REAL,                 -- win_best - win_second; the puzzle test
    nodes       INTEGER NOT NULL,
    PRIMARY KEY (game_id, ply),
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS plies_loss ON plies(loss DESC);
CREATE INDEX IF NOT EXISTS plies_fen  ON plies(fen);

-- The opening book, counted from the Lichess rated dumps. One row per
-- (position, move). `fen` is a 4-field FEN — no halfmove or fullmove counter —
-- so a position reached at different move numbers is a single entry and
-- transposing move orders pool their statistics. Rebuildable from the dumps,
-- so it may be dropped and recounted freely.
CREATE TABLE IF NOT EXISTS book (
    fen        TEXT NOT NULL,
    san        TEXT NOT NULL,
    uci        TEXT NOT NULL,
    ply        INTEGER NOT NULL,   -- 0-based ply this move is played at
    games      INTEGER NOT NULL,
    white_wins INTEGER NOT NULL,
    draws      INTEGER NOT NULL,
    black_wins INTEGER NOT NULL,
    elo_sum    INTEGER NOT NULL,   -- game-weighted, for the average
    elo_n      INTEGER NOT NULL,
    PRIMARY KEY (fen, san)
);

CREATE INDEX IF NOT EXISTS book_games ON book(games DESC);

-- The repertoire: exactly one intended move per position you can reach, per
-- colour. That uniqueness is the whole difference between a repertoire and a
-- book — the book says what everybody plays, this says what *you* play. Keyed
-- on a 4-field FEN like `book`, so lines that transpose share one node and
-- cannot drift apart into two contradictory answers.
--
-- The `val_*` columns are the Fairy-Stockfish verdict on the intended move,
-- recorded with the node count it was reached at so a cheap verdict is never
-- mistaken for a deep one. NULL means not yet validated, which is not the same
-- as validated-and-fine.
CREATE TABLE IF NOT EXISTS repertoire (
    color     TEXT NOT NULL,      -- whose repertoire: 'white' / 'black'
    fen       TEXT NOT NULL,      -- 4-field FEN; `color` is to move
    san       TEXT NOT NULL,      -- the move you intend to play
    uci       TEXT NOT NULL,
    ply       INTEGER NOT NULL,
    source    TEXT NOT NULL,      -- derived / authored / study
    note      TEXT,
    games     INTEGER,            -- book games behind this choice
    score     REAL,               -- book score of the intended move, mover POV
    win_move  REAL,               -- engine win% after the intended move
    win_best  REAL,               -- engine win% of the engine's own choice
    best_san  TEXT,
    val_loss  REAL,               -- win_best - win_move; the flag
    val_nodes INTEGER,
    val_at    REAL,
    PRIMARY KEY (color, fen)
);

CREATE INDEX IF NOT EXISTS repertoire_ply ON repertoire(color, ply);

-- Generated puzzles. `id` is a content hash of the position and the solution,
-- so regenerating the corpus reproduces the same ids and a puzzle keeps the
-- attempts already recorded against it. Derived: droppable and rebuildable.
--
-- `fen` is the position the *solver* moves in. A puzzle mined from a game also
-- carries the position before the opponent's error (`setup_fen`) and the move
-- that made it (`setup_uci`), so the UI can play the blunder in front of you
-- rather than dropping you into a position with no story.
--
-- `solution` alternates solver, defender, solver, ... and therefore always has
-- odd length: a puzzle ends on the solver's move, never on the reply to it.
CREATE TABLE IF NOT EXISTS puzzles (
    id           TEXT PRIMARY KEY,
    source       TEXT NOT NULL,      -- 'game' / 'tablebase'
    game_id      TEXT,               -- lichess game it was mined from
    ply          INTEGER,            -- ply in that game the solver moves at
    setup_fen    TEXT,
    setup_uci    TEXT,
    setup_san    TEXT,
    fen          TEXT NOT NULL,
    solver       TEXT NOT NULL,      -- 'white' / 'black'
    solution     TEXT NOT NULL,      -- JSON list of UCI
    solution_san TEXT NOT NULL,      -- JSON list of SAN, same length
    themes       TEXT NOT NULL,      -- JSON list; also exploded into puzzle_themes
    phase        TEXT NOT NULL,      -- opening / middlegame / endgame
    motive       TEXT NOT NULL,      -- 'win' (punish it) / 'save' (hold it)
    win_before   REAL,               -- solver win% with best play, at `fen`
    win_second   REAL,               -- win% of the second-best move
    unique_gap   REAL,               -- win_before - win_second; the uniqueness test
    tb_category  TEXT,               -- tablebase verdict, when there is one
    nodes        INTEGER,            -- what the confirming pass was run at
    mine         INTEGER NOT NULL DEFAULT 0,  -- from your own game, on your side
    created_at   REAL NOT NULL,
    pos_key      TEXT                -- `fen` without the move counters; see below
);

CREATE INDEX IF NOT EXISTS puzzles_phase  ON puzzles(phase);
CREATE INDEX IF NOT EXISTS puzzles_source ON puzzles(source);
CREATE INDEX IF NOT EXISTS puzzles_game   ON puzzles(game_id, ply);

-- Deliberately **not** unique. Two puzzles can share a position honestly — the
-- same endgame reached with a different solution length, say — and a unique
-- index would make `INSERT OR REPLACE` silently destroy one of them along with
-- the id its solve history hangs off. Deduplication is a decision the
-- generators make (see `puzzles.seen_positions`), not a constraint that
-- deletes evidence behind their back.
CREATE INDEX IF NOT EXISTS puzzles_pos ON puzzles(pos_key);

-- One row per (puzzle, theme). Redundant with `puzzles.themes`, and worth it:
-- the JSON column is what gets displayed, this is what gets filtered and
-- counted, and a theme's firing rate is the number that says whether the
-- detector is any good.
CREATE TABLE IF NOT EXISTS puzzle_themes (
    puzzle_id TEXT NOT NULL REFERENCES puzzles(id) ON DELETE CASCADE,
    theme     TEXT NOT NULL,
    PRIMARY KEY (puzzle_id, theme)
);

CREATE INDEX IF NOT EXISTS puzzle_themes_theme ON puzzle_themes(theme);

-- Solve history. Not derived — this is the only table here that records
-- something that happened rather than something that was computed, so
-- regenerating the corpus must never drop it.
CREATE TABLE IF NOT EXISTS puzzle_attempts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    puzzle_id  TEXT NOT NULL,
    at         REAL NOT NULL,
    solved     INTEGER NOT NULL,   -- 1 = finished with no wrong move and no hint
    hints      INTEGER NOT NULL,
    wrong      INTEGER NOT NULL,   -- how many wrong moves were tried
    ms         INTEGER             -- wall clock, if the page reported it
);

CREATE INDEX IF NOT EXISTS puzzle_attempts_puzzle ON puzzle_attempts(puzzle_id, at);

-- Measured difficulty: the rating of the weakest calibrated engine level that
-- finds the whole solution line (`difficulty.py`, plan §5).
--
-- Its own table rather than columns on `puzzles`, for the reason
-- `puzzle_attempts` has one: `puzzles.save` is INSERT OR REPLACE, so
-- regenerating the corpus at a higher node budget would wipe a column here. A
-- rating is a function of the position and the solution — which is exactly what
-- the puzzle id hashes — so it stays true across a regeneration that reproduces
-- the id, and it is recomputable from scratch if it ever stops being true.
--
-- `elo` is a bound as often as a reading: see `bound`, which is '' when a rung
-- in the middle of the ladder solved it, 'below' when the weakest rung did (the
-- ladder cannot see under its own floor) and 'above' when none did. `ladder`
-- records which calibration the verdict was made against, so a recalibration
-- leaves the old ratings visibly stale instead of silently mixing two scales.
CREATE TABLE IF NOT EXISTS puzzle_ratings (
    puzzle_id TEXT PRIMARY KEY,
    elo       REAL NOT NULL,
    span      REAL NOT NULL,      -- the rung's own +/- from the calibration
    bound     TEXT NOT NULL,      -- '' / 'below' / 'above'
    solved_by INTEGER,            -- node budget of the weakest solving rung
    monotone  INTEGER NOT NULL,   -- did every stronger rung solve it too?
    levels    TEXT NOT NULL,      -- JSON: the full per-rung profile
    method    TEXT NOT NULL,
    ladder    TEXT NOT NULL,      -- fingerprint of the calibration used
    rated_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS puzzle_ratings_elo ON puzzle_ratings(elo);

-- Games from the public Lichess rated dumps that the scraper has already
-- scanned. Not `games`: those are *your* games, and every personal report —
-- review, patterns, repertoire coverage — is a query over that table. Mixing a
-- stranger's game into it would put someone else's blunders in your habit
-- clustering.
--
-- The row is kept whatever the scan concluded, because "scanned and found
-- nothing" is exactly the fact a resumed run needs; `puzzles` is the wrong
-- place to record it, since most games produce none.
CREATE TABLE IF NOT EXISTS mined_games (
    id        TEXT PRIMARY KEY,   -- lichess game id
    month     TEXT,               -- dump file it came from, e.g. '2026-07'
    white_elo INTEGER,
    black_elo INTEGER,
    ply       INTEGER,
    scanned_at REAL NOT NULL,
    nodes     INTEGER NOT NULL,   -- what the scan pass was run at
    kept      INTEGER NOT NULL DEFAULT 0   -- puzzles this game produced
);

CREATE INDEX IF NOT EXISTS mined_games_month ON mined_games(month);

-- Names you have given lines yourself. Keyed by the SAN move sequence rather
-- than by position, unlike `book` and `repertoire`: a name belongs to a line as
-- it is spoken, and every published opening index there has ever been gives two
-- transposing move orders two names. Not derived — this is authored, so a
-- migration may not drop it.
CREATE TABLE IF NOT EXISTS opening_names (
    line TEXT PRIMARY KEY,      -- space-separated SAN, '' is the start
    name TEXT NOT NULL,
    note TEXT,
    at   REAL NOT NULL
);

-- Opening traps: book positions where one or two moves hold and everything
-- else loses. `id` is a content hash of the position and the escapes, so a
-- rescan reproduces the row rather than duplicating it.
--
-- Every column that carries a judgement carries the node count it was made at,
-- for the reason `puzzles.nodes` exists: the screening pass and the confirming
-- pass disagree often enough that a verdict without its budget is not a
-- verdict. `bait` is the one number here that is not an engine opinion at all —
-- it is the share of the field that walked in, counted from `book`.
CREATE TABLE IF NOT EXISTS traps (
    id          TEXT PRIMARY KEY,
    fen         TEXT NOT NULL,      -- 4-field FEN; the side to move is in trouble
    line        TEXT NOT NULL,      -- SAN path from the start, space separated
    ply         INTEGER NOT NULL,
    turn        TEXT NOT NULL,      -- 'white' / 'black': who has to find it
    escapes     TEXT NOT NULL,      -- JSON [{san, uci, win, games, share}]
    losers      TEXT NOT NULL,      -- JSON, same shape, worst-first
    refutation  TEXT NOT NULL,      -- JSON: how the most-played loser is punished
    win_best    REAL NOT NULL,      -- win% for the side to move, best play
    win_next    REAL NOT NULL,      -- win% of the best move that does NOT hold
    severity    REAL NOT NULL,      -- win_best - win_next
    bait        REAL NOT NULL,      -- % of book games here that went wrong
    bait_games  INTEGER NOT NULL,
    games       INTEGER NOT NULL,   -- book games through this position
    nodes       INTEGER NOT NULL,
    multipv     INTEGER NOT NULL,
    found_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS traps_bait ON traps(bait DESC);
CREATE INDEX IF NOT EXISTS traps_turn ON traps(turn, ply);

-- Book positions the trap scan has already judged, kept for the same reason
-- `mined_games` is: "looked at and found nothing" is the fact a resumed scan
-- needs, and it lives nowhere else.
CREATE TABLE IF NOT EXISTS trap_scans (
    fen       TEXT NOT NULL,
    nodes     INTEGER NOT NULL,
    multipv   INTEGER NOT NULL,
    escapes   INTEGER NOT NULL,   -- how many moves held; -1 if unresolved
    severity  REAL,
    kept      INTEGER NOT NULL DEFAULT 0,
    scanned_at REAL NOT NULL,
    PRIMARY KEY (fen, nodes, multipv)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the store, creating or migrating it as needed.

    Opening an up-to-date store takes **no write lock**: `CREATE TABLE IF NOT
    EXISTS` and stamping the schema version both need one, and a long analysis
    run holds the writer for minutes at a time — so `trainer status` in another
    terminal would fail with "database is locked" purely from saying hello.
    The version is therefore read first and nothing is written unless it is
    actually stale.
    """
    db = sqlite3.connect(path or DB_PATH, timeout=60.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    if _current_version(db) != SCHEMA_VERSION:
        # Columns added to existing tables come first: `_SCHEMA` indexes
        # `puzzles(pos_key)`, and an index over a column an older store has not
        # got yet fails the whole script.
        _widen(db)
        db.executescript(_SCHEMA)
        _migrate(db)
    return db


def _widen(db: sqlite3.Connection) -> None:
    """Add columns to tables that already exist, before the schema script runs."""
    _add_column(db, "puzzles", "pos_key", "TEXT")


def _current_version(db: sqlite3.Connection) -> int | None:
    """The stored schema version, or None if this is a fresh database."""
    try:
        found = get_meta(db, "schema_version")
    except sqlite3.OperationalError:
        return None  # no `meta` table yet
    return int(found) if found is not None else None


@contextmanager
def open_db(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    db = connect(path)
    try:
        yield db
    finally:
        db.close()


def win_floor() -> float:
    """`puzzles.WIN_FLOOR`, imported late.

    A migration that hard-coded 75.0 would silently stop agreeing with the
    generator the day the threshold moved, and importing `puzzles` at module
    level would make the storage layer depend on the corpus layer for a
    constant.
    """
    from . import puzzles

    return puzzles.WIN_FLOOR


def _migrate(db: sqlite3.Connection) -> None:
    version = _current_version(db)
    if version is None:
        set_meta(db, "schema_version", str(SCHEMA_VERSION))
        return
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"{DB_PATH} was written by a newer trainer (schema v{version}, "
            f"this one speaks v{SCHEMA_VERSION})"
        )
    # v1 -> v2 added `book`, v2 -> v3 `repertoire`, and v3 -> v4 the three
    # puzzle tables — all of which `_SCHEMA` creates on the way in; nothing
    # else to do. Later migrations may drop and rebuild derived tables
    # (`plies`, `book`, `puzzles`) freely. They may not touch `positions`, nor
    # `repertoire` once it can hold authored lines, nor `puzzle_attempts` — a
    # hand-written note and a solve history are not recomputable from anything.
    # v5 -> v6 added `puzzle_ratings`, which the schema script creates; it is
    # recomputable (it is a measurement of the engine, not of you) and so may be
    # dropped by a later migration if the ladder changes shape.
    if version < 5:
        # v4 -> v5 added `pos_key` (the column itself arrives in `_widen`, above
        # the schema script) and `mined_games`. Backfilling is a string slice of
        # a column that is already there, so it costs nothing.
        for row in db.execute("SELECT id, fen FROM puzzles").fetchall():
            db.execute(
                "UPDATE puzzles SET pos_key = ? WHERE id = ?",
                (" ".join(row["fen"].split()[:4]), row["id"]),
            )
        db.commit()
    if version < 7:
        # v6 -> v7 relabels `motive` from the confirming pass's own evaluation
        # rather than the screening one that picked the candidate. No column
        # changes; this repairs rows that say "one move holds" about a position
        # the deep pass scored as winning (31 of 215 here). Rows with no
        # `win_before` — the tablebase ones — are left alone: their motive comes
        # from an exact category and there is nothing better to derive it from.
        db.execute(
            "UPDATE puzzles SET motive = CASE WHEN win_before >= ? THEN 'win' "
            "ELSE 'save' END WHERE win_before IS NOT NULL",
            (win_floor(),),
        )
        db.commit()
    # v7 -> v8 added `opening_names`, `traps` and `trap_scans`, all of which
    # `_SCHEMA` creates on the way in. `traps` and `trap_scans` are derived from
    # the book plus engine time and may be dropped and rescanned; `opening_names`
    # is authored and may not, for the same reason `repertoire` notes may not.
    set_meta(db, "schema_version", str(SCHEMA_VERSION))


def _add_column(db: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """`ALTER TABLE ... ADD COLUMN`, tolerant of already having run.

    Also tolerant of the table not existing at all: on a fresh store `_widen`
    runs before the schema script has created anything, and there is nothing to
    widen — `PRAGMA table_info` on an absent table returns no rows rather than
    raising, so both cases fall out of the same check.
    """
    existing = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    if existing and column not in existing:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# -- meta ------------------------------------------------------------------


def get_meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    db.commit()


# -- analysis cache --------------------------------------------------------


def cached_analysis(
    db: sqlite3.Connection, fen: str, nodes: int, multipv: int, engine: str
) -> list[dict[str, Any]] | None:
    """The cached lines for exactly this key, or None.

    A deeper cached entry is *not* substituted for a shallower request: the
    whole point of recording the node count is that a judgement is only
    comparable with judgements made at the same budget.
    """
    row = db.execute(
        "SELECT lines FROM positions "
        "WHERE fen = ? AND nodes = ? AND multipv = ? AND engine = ?",
        (fen, nodes, multipv, engine),
    ).fetchone()
    return json.loads(row["lines"]) if row else None


def store_analysis(
    db: sqlite3.Connection,
    fen: str,
    nodes: int,
    multipv: int,
    engine: str,
    lines: list[dict[str, Any]],
    depth: int | None,
    created_at: float,
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO positions "
        "(fen, nodes, multipv, engine, lines, depth, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (fen, nodes, multipv, engine, json.dumps(lines), depth, created_at),
    )


def cache_stats(db: sqlite3.Connection) -> dict[str, Any]:
    row = db.execute(
        "SELECT COUNT(*) AS n, MIN(nodes) AS lo, MAX(nodes) AS hi FROM positions"
    ).fetchone()
    return {"positions": row["n"], "min_nodes": row["lo"], "max_nodes": row["hi"]}
