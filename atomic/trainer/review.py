"""Finding the mistakes you make more than once.

A single blunder is a story; the same blunder in five games is a habit, and a
habit is the only thing here worth spending study time on. Three groupings,
each answering a different question:

- **by phase** — where in the game do errors happen? Atomic's median game is
  ~30 ply, so "the opening" is a much larger share of it than in standard
  chess, and the answer decides what to work on at all.
- **by position** — the same FEN reached repeatedly and mishandled repeatedly.
  This is the strongest signal there is: same position, same wrong move, and no
  inference required to connect them.
- **by move** — the same move played at the same ply across *different*
  positions. Catches a move applied by rote whose correctness depends on what
  the opponent did, which position-grouping alone would split into unrelated
  singletons.

Games against Lichess AI are excluded by default: losing to a level-1 bot is
not a habit worth drilling.

`book_context` adds the fourth question — what does everybody *else* do here?
— which decides whether a mistake is an idiosyncrasy or a known trap.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from . import book, winrate

#: Plies at or below this count as "the opening" for grouping purposes. Atomic
#: has no ECO and no published book depth; 12 is where this account's own
#: blunder rate stops being elevated (see `phase_table`), not a borrowed
#: standard-chess convention.
OPENING_PLY = 12


def _base_query(include_ai: bool) -> str:
    return (
        "SELECT p.*, g.id AS gid, g.my_color, g.winner, g.created_at, g.source "
        "FROM plies p JOIN games g ON g.id = p.game_id "
        "WHERE p.mover = g.my_color "
        + ("" if include_ai else "AND (g.source IS NULL OR g.source != 'ai') ")
    )


def my_plies(db: sqlite3.Connection, include_ai: bool = False) -> list[sqlite3.Row]:
    """Every move you played, in an analysed game, with its judgement."""
    return list(db.execute(_base_query(include_ai)))


@dataclass
class Bucket:
    low: int
    high: int
    moves: int = 0
    blunders: int = 0
    lost: float = 0.0

    @property
    def rate(self) -> float:
        return 100.0 * self.blunders / self.moves if self.moves else 0.0

    @property
    def per_move(self) -> float:
        """Average win% thrown away per move played in this phase."""
        return self.lost / self.moves if self.moves else 0.0


def phase_table(
    db: sqlite3.Connection, *, width: int = 10, buckets: int = 5, include_ai: bool = False
) -> list[Bucket]:
    """Blunder rate by ply band — the "where do I lose games" question."""
    out = [Bucket(low=i * width, high=i * width + width - 1) for i in range(buckets)]
    out[-1].high = 10**6  # the last band is open-ended
    for row in my_plies(db, include_ai):
        bucket = out[min(row["ply"] // width, buckets - 1)]
        bucket.moves += 1
        bucket.lost += row["loss"]
        if row["judgement"] == "blunder":
            bucket.blunders += 1
    return out


@dataclass
class PositionCluster:
    fen: str
    ply: int
    best_san: str | None
    rows: list[sqlite3.Row] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def avg_loss(self) -> float:
        return sum(row["loss"] for row in self.rows) / self.count

    @property
    def played(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["san"]] = counts.get(row["san"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def games(self) -> list[str]:
        return [row["gid"] for row in self.rows]


def recurring_positions(
    db: sqlite3.Connection,
    *,
    max_ply: int = OPENING_PLY,
    min_loss: float = winrate.MISTAKE,
    min_count: int = 2,
    include_ai: bool = False,
) -> list[PositionCluster]:
    """The same position, mishandled more than once."""
    clusters: dict[str, PositionCluster] = {}
    for row in my_plies(db, include_ai):
        if row["ply"] > max_ply or row["loss"] < min_loss:
            continue
        cluster = clusters.get(row["fen"])
        if cluster is None:
            cluster = clusters[row["fen"]] = PositionCluster(
                fen=row["fen"], ply=row["ply"], best_san=row["best_san"]
            )
        cluster.rows.append(row)
    found = [c for c in clusters.values() if c.count >= min_count]
    return sorted(found, key=lambda c: (-c.count, -c.avg_loss))


@dataclass
class MoveCluster:
    san: str
    ply: int
    rows: list[sqlite3.Row] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def avg_loss(self) -> float:
        return sum(row["loss"] for row in self.rows) / self.count

    @property
    def positions(self) -> int:
        return len({row["fen"] for row in self.rows})

    @property
    def alternatives(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            key = row["best_san"] or "?"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def repeated_moves(
    db: sqlite3.Connection,
    *,
    max_ply: int = OPENING_PLY,
    min_loss: float = winrate.MISTAKE,
    min_count: int = 2,
    include_ai: bool = False,
) -> list[MoveCluster]:
    """The same move at the same ply, wrong across several different positions."""
    clusters: dict[tuple[str, int], MoveCluster] = {}
    for row in my_plies(db, include_ai):
        if row["ply"] > max_ply or row["loss"] < min_loss:
            continue
        key = (row["san"], row["ply"])
        cluster = clusters.get(key)
        if cluster is None:
            cluster = clusters[key] = MoveCluster(san=row["san"], ply=row["ply"])
        cluster.rows.append(row)
    found = [c for c in clusters.values() if c.count >= min_count]
    return sorted(found, key=lambda c: (-c.count, -c.avg_loss))


@dataclass
class BookContext:
    """How the wider population handles a position you got wrong.

    The engine says what is *best*; the book says what is *usual*. The gap
    between them is the useful part: a bad move that nobody else plays is your
    own invention, while a bad move that half the player base plays is a trap
    worth knowing by name. `share` and `rank` separate those two cases.
    """

    moves: list[book.Entry]
    played: book.Entry | None
    best: book.Entry | None

    @property
    def total(self) -> int:
        return sum(entry.games for entry in self.moves)

    @property
    def share(self) -> float:
        """Percentage of games in which the population plays your move."""
        if self.played is None or not self.total:
            return 0.0
        return 100.0 * self.played.games / self.total

    @property
    def rank(self) -> int | None:
        """1-based popularity rank of your move, or None if it is off-book."""
        for index, entry in enumerate(self.moves, start=1):
            if self.played is not None and entry.san == self.played.san:
                return index
        return None


def book_context(
    db: sqlite3.Connection, fen: str, san: str, best_san: str | None = None
) -> BookContext:
    """Look a position up in the book, tagging the move you played."""
    moves = book.lookup(db, fen)
    by_san = {entry.san: entry for entry in moves}
    return BookContext(
        moves=moves,
        played=by_san.get(san),
        best=by_san.get(best_san) if best_san else None,
    )


def move_number(ply: int) -> str:
    """`3.` for a White move, `3...` for a Black one."""
    return f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'}"
