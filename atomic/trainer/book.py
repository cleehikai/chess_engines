"""An opening book for atomic, built from the Lichess rated dumps.

There is no ECO for atomic and no published book, so "what does everyone else
play here, and how does it score?" is a question nothing currently answers.
This builds that table from the only corpus large enough to answer it: the
monthly `.pgn.zst` dumps, ~330k rated games each.

**Why not just count every prefix.** A naive pass over 12 months keeps every
distinct move-prefix to depth 20 in memory — measured at 2.46M prefixes for a
*single* month, so ~20M for a year, which is several GB of Python tuples for a
book that ends up holding tens of thousands of useful nodes.

Instead the count is *staged*. Counting is monotone — a line played 50 times
implies every one of its prefixes was played at least 50 times — so a node can
only survive the `min_games` threshold if its ancestors did too. Stage 1 counts
to depth 6 over everything and prunes; stage 2 counts to depth 12 but only
along survivors; stage 3 to depth 20. Every surviving count is *exact* (no
sampling, no approximate pruning), and peak memory is bounded by the size of
the surviving tree rather than by the corpus. The cost is re-reading the dumps
once per stage, which the 13s/month parse makes cheap.

**Transpositions merge.** Counting happens on move-prefixes because that needs
no board, but the book is stored keyed by *position*: surviving prefixes are
replayed once each at the end (tens of thousands of boards, not millions) and
rows are aggregated by FEN. Two move orders reaching the same position
therefore pool their statistics, which matters in atomic, where the tree is
narrow and transposes constantly.

Keys are 4-field FENs — piece placement, side to move, castling, en passant —
with the halfmove and fullmove counters dropped, so the same position reached
at different move numbers is one book entry.

**What `score` is not.** It is the observed result of the games that contained
a move, which is not the same as the move's merit: a line played mostly by
1800s outscores one played mostly by 1200s regardless of whether it is any
good. Measured on this account's own positions, the engine's best move and the
field's best-scoring move agree 56% of the time — far above the ~10% a random
pick of the legal moves would give, but not a substitute for the engine. That
is why `elo` is stored and displayed alongside `score`: it is the column that
tells you which of the two you are looking at. Treat the book as evidence about
what humans do and how it tends to go, and the engine as the judge of truth.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.variant

from . import dump

#: How deep the book goes, in plies. Atomic games are short — the median is
#: around 30 ply — so 20 covers the opening and the handover to the middlegame.
DEFAULT_DEPTH = 20

#: A line has to appear this often across the whole corpus to be worth storing.
#: Below ~20 games the win rate is noise: 3 wins from 4 is not 75%.
DEFAULT_MIN_GAMES = 20

#: Where the staged counting stops to prune. Each stage re-reads the dumps.
STAGES = (6, 12, 20)

_RESULT_INDEX = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}


def recent_months(count: int, *, before: str | None = None) -> list[str]:
    """The `count` most recent complete months, newest first, as 'YYYY-MM'.

    `before` (a 'YYYY-MM') pins the starting point for tests; otherwise the
    current month is skipped, because it is still being written to.
    """
    if before is None:
        now = time.gmtime()
        year, month = now.tm_year, now.tm_mon
    else:
        year, month = (int(part) for part in before.split("-"))
    out = []
    for _ in range(count):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        out.append(f"{year:04d}-{month:02d}")
    return out


# -- counting --------------------------------------------------------------


class Counts:
    """Per-node tallies, packed into parallel lists to keep the trie small.

    A dict of small objects would be tidier and roughly triple the memory at
    the sizes this runs at, which is the whole reason for the staged design.
    """

    __slots__ = ("elo_n", "elo_sum", "games", "index", "results")

    def __init__(self) -> None:
        self.index: dict[str, int] = {}
        self.games: list[int] = []
        self.results: list[list[int]] = []  # [white wins, draws, black wins]
        self.elo_sum: list[int] = []
        self.elo_n: list[int] = []

    def bump(self, key: str, result: int, elo: int | None) -> None:
        slot = self.index.get(key)
        if slot is None:
            slot = self.index[key] = len(self.games)
            self.games.append(0)
            self.results.append([0, 0, 0])
            self.elo_sum.append(0)
            self.elo_n.append(0)
        self.games[slot] += 1
        self.results[slot][result] += 1
        if elo is not None:
            self.elo_sum[slot] += elo
            self.elo_n[slot] += 1

    def survivors(self, min_games: int, depth: int | None = None) -> set[str]:
        """Keys played at least `min_games` times, optionally at one depth."""
        out = set()
        for key, slot in self.index.items():
            if self.games[slot] < min_games:
                continue
            if depth is not None and key.count(" ") + 1 != depth:
                continue
            out.add(key)
        return out

    def __len__(self) -> int:
        return len(self.games)


def _game_facts(game: dump.RawGame, min_elo: int | None) -> tuple[int, int | None] | None:
    """The (result index, average Elo) of a game, or None to skip it."""
    result = _RESULT_INDEX.get(game.result)
    if result is None:
        return None  # unfinished ('*')
    white, black = game.elo("White"), game.elo("Black")
    both = [e for e in (white, black) if e is not None]
    elo = sum(both) // len(both) if both else None
    if min_elo is not None and (elo is None or elo < min_elo):
        return None
    return result, elo


def count_stage(
    paths: Iterable[Path],
    *,
    depth: int,
    keep: set[str] | None,
    gate_depth: int,
    min_elo: int | None = None,
    on_month: object = None,
) -> tuple[Counts, int]:
    """Count move-prefixes up to `depth`.

    `keep` restricts the walk: a game only contributes if its prefix at
    `gate_depth` survived the previous stage. Games shorter than `gate_depth`
    were fully counted by the earlier stage and are skipped here, so no node is
    ever double-counted across stages.
    """
    counts = Counts()
    total = 0
    for path in paths:
        for game in dump.read(path):
            facts = _game_facts(game, min_elo)
            if facts is None:
                continue
            result, elo = facts
            moves = game.moves()[:depth]
            if keep is not None:
                if len(moves) <= gate_depth:
                    continue  # already fully counted by an earlier stage
                if " ".join(moves[:gate_depth]) not in keep:
                    continue
            total += 1
            key = ""
            for i, san in enumerate(moves):
                key = san if i == 0 else f"{key} {san}"
                if keep is None or i >= gate_depth:
                    counts.bump(key, result, elo)
        if callable(on_month):
            on_month(path, len(counts))
    return counts, total


@dataclass
class Node:
    """One book line: a move-prefix and how the games containing it went."""

    line: tuple[str, ...]
    games: int
    results: tuple[int, int, int]
    elo: int | None

    @property
    def depth(self) -> int:
        return len(self.line)


def build_tree(
    paths: list[Path],
    *,
    depth: int = DEFAULT_DEPTH,
    min_games: int = DEFAULT_MIN_GAMES,
    min_elo: int | None = None,
    stages: tuple[int, ...] = STAGES,
    on_stage: object = None,
    on_month: object = None,
) -> tuple[list[Node], int]:
    """Count the dumps in stages and return the surviving nodes."""
    stages = tuple(s for s in stages if s < depth) + (depth,)
    surviving: dict[str, Node] = {}
    keep: set[str] | None = None
    gate = 0
    corpus = 0

    for stage in stages:
        counts, seen = count_stage(
            paths,
            depth=stage,
            keep=keep,
            gate_depth=gate,
            min_elo=min_elo,
            on_month=on_month,
        )
        if gate == 0:
            corpus = seen
        for key, slot in counts.index.items():
            games = counts.games[slot]
            if games < min_games:
                continue
            elo_n = counts.elo_n[slot]
            surviving[key] = Node(
                line=tuple(key.split(" ")),
                games=games,
                results=tuple(counts.results[slot]),  # type: ignore[arg-type]
                elo=counts.elo_sum[slot] // elo_n if elo_n else None,
            )
        if callable(on_stage):
            on_stage(stage, len(counts), len(surviving))
        if stage == stages[-1]:
            break
        keep = {k for k in surviving if k.count(" ") + 1 == stage}
        gate = stage
        if not keep:
            break

    return list(surviving.values()), corpus


# -- positions -------------------------------------------------------------


def book_fen(board: chess.Board) -> str:
    """The 4-field FEN used as a book key — no halfmove or fullmove counter."""
    return " ".join(board.fen().split(" ")[:4])


@dataclass
class Entry:
    """One (position, move) row of the finished book."""

    fen: str
    san: str
    uci: str
    ply: int
    games: int = 0
    white_wins: int = 0
    draws: int = 0
    black_wins: int = 0
    elo_sum: int = 0
    elo_n: int = 0

    @property
    def elo(self) -> int | None:
        return self.elo_sum // self.elo_n if self.elo_n else None

    @property
    def score(self) -> float:
        """Score for the side that plays this move, as a percentage."""
        mover_wins = self.white_wins if self.ply % 2 == 0 else self.black_wins
        return 100.0 * (mover_wins + 0.5 * self.draws) / self.games if self.games else 0.0


def to_entries(nodes: Iterable[Node]) -> Iterator[Entry]:
    """Replay surviving lines onto boards and aggregate them by position.

    Lines are walked in depth order with a board cache, so a node of depth *d*
    costs one `push` rather than *d* of them. Illegal lines are dropped rather
    than raised on: the dumps are machine-generated and clean, but a parse slip
    should cost one line, not the build.
    """
    entries: dict[tuple[str, str], Entry] = {}
    boards: dict[tuple[str, ...], chess.variant.AtomicBoard] = {
        (): chess.variant.AtomicBoard()
    }

    for node in sorted(nodes, key=lambda n: n.depth):
        parent = boards.get(node.line[:-1])
        if parent is None:
            continue  # ancestor was pruned or illegal
        san = node.line[-1]
        try:
            move = parent.parse_san(san)
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            continue
        fen = book_fen(parent)
        key = (fen, san)
        entry = entries.get(key)
        if entry is None:
            entry = entries[key] = Entry(
                fen=fen, san=san, uci=move.uci(), ply=node.depth - 1
            )
        entry.games += node.games
        entry.white_wins += node.results[0]
        entry.draws += node.results[1]
        entry.black_wins += node.results[2]
        if node.elo is not None:
            entry.elo_sum += node.elo * node.games
            entry.elo_n += node.games

        child = parent.copy(stack=False)
        child.push(move)
        boards[node.line] = child

    return iter(entries.values())


# -- storage ---------------------------------------------------------------


def replace_book(
    db: sqlite3.Connection, entries: Iterable[Entry], provenance: dict[str, object]
) -> int:
    """Swap in a freshly built book. Atomic: the old one survives a crash."""
    from . import store

    db.execute("DELETE FROM book")
    rows = [
        (
            e.fen, e.san, e.uci, e.ply, e.games,
            e.white_wins, e.draws, e.black_wins, e.elo_sum, e.elo_n,
        )
        for e in entries
    ]
    db.executemany(
        "INSERT OR REPLACE INTO book (fen, san, uci, ply, games, white_wins, "
        "draws, black_wins, elo_sum, elo_n) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    for key, value in provenance.items():
        store.set_meta(db, f"book:{key}", str(value))
    return len(rows)


def _entry(row: sqlite3.Row) -> Entry:
    return Entry(
        fen=row["fen"], san=row["san"], uci=row["uci"], ply=row["ply"],
        games=row["games"], white_wins=row["white_wins"], draws=row["draws"],
        black_wins=row["black_wins"], elo_sum=row["elo_sum"], elo_n=row["elo_n"],
    )


def lookup(db: sqlite3.Connection, fen: str, *, min_games: int = 0) -> list[Entry]:
    """Book moves for a position, most-played first.

    Accepts a full or 4-field FEN — the counters are trimmed here so callers
    can pass whatever they have (the `plies` table stores full FENs).
    """
    key = " ".join(fen.split(" ")[:4])
    rows = db.execute(
        "SELECT * FROM book WHERE fen = ? AND games >= ? ORDER BY games DESC",
        (key, min_games),
    ).fetchall()
    return [_entry(row) for row in rows]


@dataclass
class Reached:
    """One position the book contains, and the line that gets there."""

    line: tuple[str, ...]
    board: chess.variant.AtomicBoard
    entries: list[Entry]

    @property
    def fen(self) -> str:
        return book_fen(self.board)

    @property
    def games(self) -> int:
        return sum(entry.games for entry in self.entries)

    @property
    def ply(self) -> int:
        return len(self.line)


def walk(
    db: sqlite3.Connection,
    *,
    depth: int = DEFAULT_DEPTH,
    min_games: int = 1_000,
    min_share: float = 4.0,
    max_children: int = 6,
) -> Iterator[Reached]:
    """Every position the field actually reaches, breadth-first, deduplicated.

    The book is a table of (position, move) rows and says nothing about which
    positions are *reachable* by lines anyone plays; that is a walk, and three
    consumers want it — the opening index, the trap scanner, and any future
    line browser. Both floors from `repertoire._replies` apply here for the
    same reasons, and positions are visited once however many move orders
    reach them, because the book merges transpositions and a second visit
    would re-report the identical row.
    """
    start = chess.variant.AtomicBoard()
    frontier: list[Reached] = [Reached(line=(), board=start, entries=lookup(db, book_fen(start)))]
    seen: set[str] = set()
    while frontier:
        nxt: list[Reached] = []
        for node in frontier:
            if node.fen in seen or not node.entries:
                continue
            seen.add(node.fen)
            yield node
            if node.ply >= depth:
                continue
            total = node.games
            keep = [
                entry for entry in node.entries
                if entry.games >= min_games and 100.0 * entry.games / total >= min_share
            ]
            for entry in sorted(keep, key=lambda e: -e.games)[:max_children]:
                child = node.board.copy(stack=False)
                try:
                    child.push_san(entry.san)
                except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
                    continue
                nxt.append(
                    Reached(
                        line=(*node.line, entry.san),
                        board=child,
                        entries=lookup(db, book_fen(child)),
                    )
                )
        frontier = nxt


def stats(db: sqlite3.Connection) -> dict[str, object]:
    from . import store

    row = db.execute(
        "SELECT COUNT(*) AS moves, COUNT(DISTINCT fen) AS positions, "
        "MAX(games) AS top FROM book"
    ).fetchone()
    out: dict[str, object] = {
        "moves": row["moves"], "positions": row["positions"], "top": row["top"] or 0,
    }
    for key in ("months", "games", "min_games", "min_elo", "depth", "built_at"):
        out[key] = store.get_meta(db, f"book:{key}")
    return out
