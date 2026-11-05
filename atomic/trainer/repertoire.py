"""A repertoire: one intended move per position, generated then checked.

Mistake detection names the leak; a repertoire is what plugs it. Atomic makes
this unusually tractable — the median game is ~30 ply and the opening tree is
narrow, so 12 ply of theory covers most of what is decidable before the game
becomes tactics — and unusually awkward: there is no published atomic
repertoire to import and no ECO to hang one on. So the first one is *derived*.

Three steps, in order, because each is only sound given the one before it:

1. **Derive** (`derive`). Walk the book (`book.py`) from the start. At your
   turn take the best-*scoring* move with a real sample behind it; at the
   opponent's turn branch to every reply the field actually plays. The result
   is a trie of positions you will genuinely reach, not a wish list.

2. **Validate** (`validate`). Every derived move is a popularity statistic, and
   the plan's own risk list says so: a book-derived repertoire can teach you
   the dumps' bad habits. So Fairy-Stockfish judges every node by the same
   measure game review uses — win% of the intended move against win% of the
   engine's choice — and anything that leaks more than `FLAG_LOSS` is flagged.
   Disagreements are surfaced, not silently resolved: at ply 1 the engine and
   3.7M humans already disagree about 1.Nh3 versus 1.Nf3, and quietly
   overwriting the field with the engine would hide exactly the cases worth
   looking at. `--fix` promotes the engine's move, but only when asked.

3. **Cover** (`coverage`). Cross the repertoire against your own games: the
   positions you reach often and have no answer for, ranked by how much they
   cost you. Plus `deviations` — positions you *do* have a line for and played
   something else in anyway, which is a different problem with a different fix.

Nodes are keyed on the 4-field FEN, like the book, so transposing lines share a
node and cannot drift into two contradictory answers. The cost of that choice:
validation analyses the canonical `<key> 0 1` completion rather than the exact
FEN a game reached, so those cache entries do not pool with `plies`. Within the
first dozen ply nothing in the evaluation depends on the halfmove clock, and a
single stable key is worth more than the shared cache line.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import chess
import chess.variant

from . import analysis, book, winrate

#: How deep to derive. §6a: 10-14 ply covers the useful part of atomic theory.
DEFAULT_DEPTH = 12

#: A book move needs this many games behind it before it is worth a node.
DEFAULT_MIN_GAMES = 200

#: Opponent replies below this share of the position are not prepared for.
DEFAULT_MIN_SHARE = 3.0

#: Hard cap on replies to one of your moves, most-played first.
DEFAULT_MAX_REPLIES = 6

#: Stop deriving at this many nodes, whatever the depth allows.
DEFAULT_MAX_NODES = 400

#: Win% the intended move may give up against the engine's before it is
#: flagged. Deliberately below `winrate.MISTAKE`: a move you will play in every
#: single game deserves a tighter tolerance than a move played once.
FLAG_LOSS = 10.0

COLORS = ("white", "black")


def _turn_of(color: str) -> chess.Color:
    return chess.WHITE if color == "white" else chess.BLACK


def full_fen(key: str) -> str:
    """Complete a 4-field book key into a FEN the engine will accept."""
    parts = key.split(" ")
    return " ".join(parts[:4]) + " 0 1" if len(parts) == 4 else key


@dataclass
class Node:
    """One position in the repertoire and the move you intend to play in it."""

    color: str
    fen: str
    san: str
    uci: str
    ply: int
    source: str = "derived"
    note: str | None = None
    games: int | None = None
    score: float | None = None
    win_move: float | None = None
    win_best: float | None = None
    best_san: str | None = None
    val_loss: float | None = None
    val_nodes: int | None = None
    val_at: float | None = None

    @property
    def validated(self) -> bool:
        return self.val_loss is not None

    @property
    def flagged(self) -> bool:
        """Does the engine think this move leaks more than it should?"""
        return self.val_loss is not None and self.val_loss >= FLAG_LOSS

    @property
    def agrees(self) -> bool | None:
        """Whether the engine would play the same move. None if unvalidated."""
        return None if self.best_san is None else self.best_san == self.san


# -- derivation ------------------------------------------------------------


def _pick(entries: list[book.Entry], min_games: int) -> book.Entry | None:
    """Your move in a position: best-scoring, among moves with a real sample.

    Not the *most played* one. The field's habits are what you are trying to
    beat, and the two differ often enough to matter — but score is a popularity
    statistic too, which is why nothing here is trusted until `validate` has
    run over it.
    """
    usable = [e for e in entries if e.games >= min_games]
    return max(usable, key=lambda e: e.score) if usable else None


def _replies(
    entries: list[book.Entry], *, min_games: int, min_share: float, cap: int
) -> list[book.Entry]:
    """The opponent moves worth preparing for: common ones, most-played first.

    A reply nobody plays is not preparation, it is a way to make a repertoire
    too big to remember. Both a relative floor (`min_share`) and an absolute
    one (`min_games`) apply, because either alone misbehaves: share alone lets
    a rare position sprout branches off 30 games, and games alone drags every
    sideline of a hugely popular position in.
    """
    total = sum(e.games for e in entries)
    if not total:
        return []
    keep = [
        e
        for e in entries
        if e.games >= min_games and 100.0 * e.games / total >= min_share
    ]
    return sorted(keep, key=lambda e: -e.games)[:cap]


def derive(
    db: sqlite3.Connection,
    color: str,
    *,
    depth: int = DEFAULT_DEPTH,
    min_games: int = DEFAULT_MIN_GAMES,
    min_share: float = DEFAULT_MIN_SHARE,
    max_replies: int = DEFAULT_MAX_REPLIES,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> list[Node]:
    """Grow a repertoire for `color` out of the book, breadth-first.

    Breadth-first rather than depth-first so that `max_nodes` truncates the
    *deepest* lines rather than whichever branch happened to be walked first —
    a repertoire cut short at ply 10 everywhere is usable; one that knows a
    single line to ply 20 and nothing else is not.
    """
    if color not in COLORS:
        raise ValueError(f"color must be one of {COLORS}, got {color!r}")
    turn = _turn_of(color)

    start = chess.variant.AtomicBoard()
    frontier: list[chess.variant.AtomicBoard] = []
    if start.turn == turn:
        frontier.append(start)
    else:
        entries = book.lookup(db, book.book_fen(start))
        for reply in _replies(
            entries, min_games=min_games, min_share=min_share, cap=max_replies
        ):
            child = start.copy(stack=False)
            child.push(chess.Move.from_uci(reply.uci))
            frontier.append(child)

    out: list[Node] = []
    seen: set[str] = set()
    while frontier and len(out) < max_nodes:
        next_frontier: list[chess.variant.AtomicBoard] = []
        for board in frontier:
            key = book.book_fen(board)
            ply = board.ply()
            if key in seen or ply >= depth or len(out) >= max_nodes:
                continue
            seen.add(key)

            mine = _pick(book.lookup(db, key), min_games)
            if mine is None:
                continue  # out of book: nothing to recommend, and honest to say so
            move = chess.Move.from_uci(mine.uci)
            if move not in board.legal_moves:
                continue  # book row disagrees with the board; drop the line
            out.append(
                Node(
                    color=color, fen=key, san=mine.san, uci=mine.uci, ply=ply,
                    games=mine.games, score=mine.score,
                )
            )

            after = board.copy(stack=False)
            after.push(move)
            if after.is_game_over(claim_draw=True) or ply + 2 >= depth:
                continue
            replies = _replies(
                book.lookup(db, book.book_fen(after)),
                min_games=min_games, min_share=min_share, cap=max_replies,
            )
            for reply in replies:
                child = after.copy(stack=False)
                try:
                    child.push(chess.Move.from_uci(reply.uci))
                except (AssertionError, ValueError):
                    continue
                next_frontier.append(child)
        frontier = next_frontier
    return out


# -- storage ---------------------------------------------------------------

_COLUMNS = (
    "color", "fen", "san", "uci", "ply", "source", "note", "games", "score",
    "win_move", "win_best", "best_san", "val_loss", "val_nodes", "val_at",
)


def _node(row: sqlite3.Row) -> Node:
    return Node(**{name: row[name] for name in _COLUMNS})


def save(db: sqlite3.Connection, nodes: Iterable[Node], *, replace: bool = False) -> int:
    """Write nodes, optionally clearing the colour's existing repertoire first.

    Validation is carried over on a plain save: re-deriving with a different
    cutoff should not silently discard hours of engine time for the nodes that
    did not change. `replace` is for the case where it should.
    """
    rows = [tuple(getattr(node, name) for name in _COLUMNS) for node in nodes]
    if replace:
        for color in {row[0] for row in rows}:
            db.execute("DELETE FROM repertoire WHERE color = ?", (color,))
    placeholders = ",".join("?" * len(_COLUMNS))
    db.executemany(
        f"INSERT OR REPLACE INTO repertoire ({','.join(_COLUMNS)}) "
        f"VALUES ({placeholders})",
        rows,
    )
    db.commit()
    return len(rows)


def nodes(db: sqlite3.Connection, color: str | None = None) -> list[Node]:
    sql = f"SELECT {','.join(_COLUMNS)} FROM repertoire"
    params: list[object] = []
    if color:
        sql += " WHERE color = ?"
        params.append(color)
    sql += " ORDER BY ply, fen"
    return [_node(row) for row in db.execute(sql, params)]


def get(db: sqlite3.Connection, color: str, fen: str) -> Node | None:
    key = " ".join(fen.split(" ")[:4])
    row = db.execute(
        f"SELECT {','.join(_COLUMNS)} FROM repertoire WHERE color = ? AND fen = ?",
        (color, key),
    ).fetchone()
    return _node(row) if row else None


def clear(db: sqlite3.Connection, color: str) -> int:
    cur = db.execute("DELETE FROM repertoire WHERE color = ?", (color,))
    db.commit()
    return cur.rowcount


# -- validation ------------------------------------------------------------


def validation_fens(nodes_: Iterable[Node]) -> list[str]:
    """Every position the validator needs analysed: before and after each move.

    Two per node, and the "after" of one node is very often the "before" of a
    reply's node, so the real cost is well under 2x. Returned de-duplicated in
    walk order.

    A move that ends the game contributes only its "before": the position after
    it has no legal moves to search, and `_win_after` reads the result off the
    board instead — which for a repertoire that walks into forced mates is not
    a rare case.
    """
    out: list[str] = []
    for node in nodes_:
        before = full_fen(node.fen)
        board = chess.variant.AtomicBoard(before)
        try:
            move = chess.Move.from_uci(node.uci)
            board.push(move)
        except (AssertionError, ValueError):
            continue
        out.append(before)
        if not board.is_game_over(claim_draw=True):
            out.append(board.fen())
    return list(dict.fromkeys(out))


def validate(
    db: sqlite3.Connection,
    color: str | None = None,
    *,
    nodes_budget: int = analysis.DEFAULT_NODES,
    multipv: int = analysis.DEFAULT_MULTIPV,
    workers: int | None = None,
    model: winrate.WinModel | None = None,
    fix: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Node]:
    """Score every node against Fairy-Stockfish; return the ones it flags.

    The measure is identical to the one game review uses — win% of the engine's
    best against win% after your move, read off the *next* position rather than
    scored separately — so "this repertoire move leaks 14%" and "this move in
    game X leaked 14%" mean the same thing and can sit in the same queue.

    With `fix`, a flagged node's move is replaced by the engine's, its source
    marked `engine`, and the reason recorded in the note. The replacement is
    re-scored from the same analysis, so the stored numbers still describe the
    move actually stored.
    """
    model = model or winrate.load()
    todo = nodes(db, color)
    if not todo:
        return []

    analysis.ensure_analysed(
        db, validation_fens(todo), nodes=nodes_budget, multipv=multipv,
        workers=workers, on_progress=on_progress,
    )

    flagged: list[Node] = []
    updated: list[Node] = []
    for node in todo:
        before = full_fen(node.fen)
        lines = analysis.cached_lines(db, before, nodes_budget, multipv)
        if not lines:
            continue
        board = chess.variant.AtomicBoard(before)
        try:
            move = chess.Move.from_uci(node.uci)
            board.push(move)
        except (AssertionError, ValueError):
            continue

        node.win_best = lines[0].win_pct(model)
        node.best_san = chess.variant.AtomicBoard(before).san(
            chess.Move.from_uci(lines[0].uci)
        )
        node.win_move = _win_after(db, board, model, nodes_budget, multipv, node.win_best)
        node.val_loss = max(0.0, node.win_best - node.win_move)
        node.val_nodes = nodes_budget
        node.val_at = time.time()

        if node.flagged and fix and node.best_san != node.san:
            node.note = (
                f"was {node.san} ({node.games:,} book games, "
                f"{node.score:.0f}%) — leaked {node.val_loss:.0f} win%"
                if node.games
                else f"was {node.san} — leaked {node.val_loss:.0f} win%"
            )
            node.san, node.uci = node.best_san, lines[0].uci
            node.source, node.games, node.score = "engine", None, None
            node.win_move, node.val_loss = node.win_best, 0.0
        elif node.flagged:
            flagged.append(node)
        updated.append(node)

    save(db, updated)
    return flagged


def _win_after(
    db: sqlite3.Connection,
    board: chess.variant.AtomicBoard,
    model: winrate.WinModel,
    nodes_budget: int,
    multipv: int,
    fallback: float,
) -> float:
    """Win% for the side that just moved, read off the resulting position."""
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        if result == "1/2-1/2":
            return 50.0
        # `board.turn` is the opponent, so a decisive result here is ours iff
        # it does not favour the side now to move.
        winner = chess.WHITE if result == "1-0" else chess.BLACK
        return 0.0 if winner == board.turn else 100.0
    after = analysis.cached_lines(db, board.fen(), nodes_budget, multipv)
    return 100.0 - after[0].win_pct(model) if after else fallback


# -- coverage --------------------------------------------------------------


@dataclass
class Gap:
    """A position you keep reaching that the repertoire has no answer for."""

    fen: str
    ply: int
    rows: list[sqlite3.Row] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def avg_loss(self) -> float:
        return sum(row["loss"] for row in self.rows) / self.count if self.rows else 0.0

    @property
    def cost(self) -> float:
        """Total win% this position has cost you — frequency times severity."""
        return sum(row["loss"] for row in self.rows)

    @property
    def played(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["san"]] = counts.get(row["san"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def games(self) -> list[str]:
        return [row["gid"] for row in self.rows]


def _my_opening_plies(
    db: sqlite3.Connection, color: str, max_ply: int, include_ai: bool
) -> Iterator[sqlite3.Row]:
    sql = (
        "SELECT p.*, g.id AS gid FROM plies p JOIN games g ON g.id = p.game_id "
        "WHERE p.mover = g.my_color AND g.my_color = ? AND p.ply < ? "
        + ("" if include_ai else "AND (g.source IS NULL OR g.source != 'ai') ")
    )
    return iter(db.execute(sql, (color, max_ply)))


def coverage(
    db: sqlite3.Connection,
    color: str,
    *,
    max_ply: int = DEFAULT_DEPTH,
    include_ai: bool = False,
) -> tuple[list[Gap], int, int]:
    """Positions you reached with no line for them, worst first.

    Returns (gaps, covered_plies, total_plies) so the caller can report a
    coverage percentage as well as the holes. "Worst" is total win% lost, not
    frequency: a position you reach ten times and handle correctly is not a
    gap worth an evening, and one you reach twice and lose both times is.
    """
    have = {row["fen"] for row in db.execute(
        "SELECT fen FROM repertoire WHERE color = ?", (color,)
    )}
    gaps: dict[str, Gap] = {}
    covered = total = 0
    for row in _my_opening_plies(db, color, max_ply, include_ai):
        total += 1
        key = " ".join(row["fen"].split(" ")[:4])
        if key in have:
            covered += 1
            continue
        gap = gaps.get(key)
        if gap is None:
            gap = gaps[key] = Gap(fen=key, ply=row["ply"])
        gap.rows.append(row)
    ranked = sorted(gaps.values(), key=lambda g: (-g.cost, -g.count))
    return ranked, covered, total


@dataclass
class Deviation:
    """A position the repertoire covers, in which you played something else."""

    node: Node
    rows: list[sqlite3.Row] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def cost(self) -> float:
        return sum(row["loss"] for row in self.rows)

    @property
    def played(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["san"]] = counts.get(row["san"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def deviations(
    db: sqlite3.Connection,
    color: str,
    *,
    max_ply: int = DEFAULT_DEPTH,
    include_ai: bool = False,
) -> list[Deviation]:
    """Where you had a line and did not follow it — a memory problem, not a gap.

    Worth separating from `coverage`: a gap is fixed by deciding on a move, a
    deviation is fixed by drilling the one you already chose.
    """
    by_fen = {node.fen: node for node in nodes(db, color)}
    found: dict[str, Deviation] = {}
    for row in _my_opening_plies(db, color, max_ply, include_ai):
        key = " ".join(row["fen"].split(" ")[:4])
        node = by_fen.get(key)
        if node is None or row["san"] == node.san:
            continue
        entry = found.get(key)
        if entry is None:
            entry = found[key] = Deviation(node=node)
        entry.rows.append(row)
    return sorted(found.values(), key=lambda d: (-d.cost, -d.count))


# -- display ---------------------------------------------------------------


def lines_of(db: sqlite3.Connection, color: str) -> list[tuple[list[str], Node]]:
    """Every repertoire node with the move sequence that reaches it.

    Walked forward from the start rather than reconstructed from the FENs,
    because a position is in general reachable by several move orders and only
    one of them is worth printing: the shortest, which breadth-first gives for
    free.
    """
    by_fen = {node.fen: node for node in nodes(db, color)}
    if not by_fen:
        return []
    turn = _turn_of(color)
    start = chess.variant.AtomicBoard()
    frontier: list[tuple[list[str], chess.variant.AtomicBoard]] = [([], start)]
    out: list[tuple[list[str], Node]] = []
    seen: set[str] = set()

    while frontier:
        nxt: list[tuple[list[str], chess.variant.AtomicBoard]] = []
        for path, board in frontier:
            key = book.book_fen(board)
            if key in seen:
                continue
            seen.add(key)
            if board.turn == turn:
                node = by_fen.get(key)
                if node is None:
                    continue
                out.append((path, node))
                child = board.copy(stack=False)
                child.push(chess.Move.from_uci(node.uci))
                nxt.append(([*path, node.san], child))
            else:
                for entry in book.lookup(db, key):
                    child = board.copy(stack=False)
                    try:
                        child.push(chess.Move.from_uci(entry.uci))
                    except (AssertionError, ValueError):
                        continue
                    if book.book_fen(child) in by_fen:
                        nxt.append(([*path, entry.san], child))
        frontier = nxt
    return out


def stats(db: sqlite3.Connection, color: str) -> dict[str, object]:
    row = db.execute(
        "SELECT COUNT(*) AS n, SUM(val_loss IS NOT NULL) AS validated, "
        "SUM(val_loss >= ?) AS flagged, SUM(san = best_san) AS agreed, "
        "MAX(ply) AS depth FROM repertoire WHERE color = ?",
        (FLAG_LOSS, color),
    ).fetchone()
    return {
        "nodes": row["n"] or 0,
        "validated": row["validated"] or 0,
        "flagged": row["flagged"] or 0,
        "agreed": row["agreed"] or 0,
        "depth": (row["depth"] or -1) + 1,
    }
