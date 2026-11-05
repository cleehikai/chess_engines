"""Opening traps: the places in the book where one move holds and the rest lose.

`generate.py` mines puzzles out of games somebody actually played, so it can
only ever find the mistakes that were made. A trap is the other question: not
"where did this game go wrong" but **"where in the tree is it possible to go
wrong"** — including in lines nobody in the corpus has blundered in yet, which
is exactly the preparation a repertoire is for.

**What qualifies as a trap.** A position, inside book depth, where:

- at most `MAX_ESCAPES` moves keep the balance — "keep" meaning within
  `HOLD_LOSS` win% of the best move;
- the best move that is *not* one of those loses by at least `SEVERITY` win%
  points and leaves the mover at or under `LOST_CEILING`, so the alternative is
  not merely worse but losing;
- the position was tenable to begin with (`win_best >= BALANCE_FLOOR`) — a
  position that is already lost has no trap in it, only a longer defence;
- the escape is not something `M0-greedy` finds by counting material one ply
  deep, which is `generate.is_trivial`'s job and the same filter the puzzle
  corpus uses.

**And how often the field falls in.** `bait` is the share of book games from
this position that played a move which is not an escape. It is the only number
here that is not an engine opinion: the book counted it from 3.7M rated games.
A position where one move holds and 4% of players find it is a different object
from one where 96% do, and only the first is worth memorising. Sorting by
severity finds critical positions; sorting by bait finds traps.

**Which side the trap is for** is just whose turn it is. The same row is a
weapon when the opponent is to move and a pitfall when you are, so it is stored
once and filtered by `turn` rather than being mined twice.

**The funnel**, for the reason `generate.py` has one: a scan at `SCAN_NODES`
over the whole tree, then a confirming pass at `CONFIRM_NODES` on survivors
only, and MultiPV throughout because the whole question is about the *second*
best move. `MULTIPV` bounds what can be seen: a position where six moves hold
is reported as "not a trap" because five is all that was asked for, which is
the right answer for the wrong reason and is recorded as such in `trap_scans`.

Traps are *not* stored as puzzles. A puzzle asks you to find a win that is
there; a trap asks you to avoid a loss that is one move away, and half of them
have no win in them at all. `themes.py` already carries an `opening-trap`
label for the puzzle-shaped ones that come out of real games; this table is the
tree-shaped survey beside it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

import chess
import chess.variant

from . import analysis, book, generate, openings, winrate

#: The screening pass, over every position in the walk. A rung of the
#: calibrated ladder rather than a round number: `lab.cli levels` puts throttled
#: FSF at 400k between its 64k and 1024k rungs, and everything this pass
#: proposes is re-judged at `CONFIRM_NODES` anyway.
SCAN_NODES = 400_000

#: The confirming pass, on survivors only. Five times the screen, which is what
#: it costs to stop a trap being an artefact of a shallow second-best line.
CONFIRM_NODES = 2_000_000

#: How many moves the scan can see. The question is "how many moves hold", so
#: the answer is bounded by this: five is enough to separate one or two from
#: "several", which is the only distinction the definition needs.
MULTIPV = 5

#: Win% a move may give up against the best and still count as holding.
#: Deliberately looser than `repertoire.FLAG_LOSS`: a trap is about moves that
#: *lose*, and calling a 10-point inaccuracy "falling for a trap" would make
#: every position in the book a trap.
HOLD_LOSS = 12.0

#: One or two. The user-facing definition of the whole feature, and the reason
#: it is a constant rather than a hard-coded 1: in atomic a second saving
#: resource is common and a position with exactly two is no easier to play.
MAX_ESCAPES = 2

#: How much worse the best non-escape has to be.
SEVERITY = 25.0

#: ...and how bad it has to leave you. Both are required: a drop from 95% to
#: 65% is 30 points and still winning, which is not a trap.
LOST_CEILING = 32.0

#: A position that is already lost contains no trap.
BALANCE_FLOOR = 38.0

#: Defaults for the walk. Deeper and thinner than the opening index, because a
#: trap ten ply in is still preparation, and traps live in sidelines.
DEFAULT_DEPTH = 12
DEFAULT_MIN_GAMES = 500
DEFAULT_MIN_SHARE = 3.0
DEFAULT_MAX_CHILDREN = 6

#: How far to spell out the punishment after the most-played losing move.
REFUTATION_PLIES = 6


# -- the shape of an answer ------------------------------------------------


@dataclass(frozen=True)
class Move:
    """One move in a trapped position, with everything known about it."""

    san: str
    uci: str
    win: float           # win% for the side to move, after this move
    games: int = 0       # book games that played it
    share: float = 0.0   # ...as a share of the games in this position

    @property
    def played(self) -> bool:
        return self.games > 0


@dataclass
class Trap:
    """A book position where one or two moves hold and the rest lose."""

    fen: str
    line: tuple[str, ...]
    turn: str
    escapes: list[Move]
    losers: list[Move]
    refutation: list[str]            # SAN, starting with the losing move
    win_best: float
    win_next: float
    bait: float
    bait_games: int
    games: int
    nodes: int
    multipv: int = MULTIPV
    found_at: float = field(default_factory=time.time)

    @property
    def ply(self) -> int:
        return len(self.line)

    @property
    def severity(self) -> float:
        return self.win_best - self.win_next

    @property
    def id(self) -> str:
        """Content hash of the position and the moves that hold it.

        Keyed on the escapes rather than on the position alone so that a
        rescan which *changes its mind* about what holds writes a new row
        instead of silently overwriting the old verdict with a different one
        under the same id — the same reason `puzzles.Puzzle.id` hashes the
        solution.
        """
        payload = self.fen + "|" + ",".join(sorted(m.uci for m in self.escapes))
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "fen": self.fen,
            "line": list(self.line),
            "ply": self.ply,
            "turn": self.turn,
            "escapes": [asdict(m) for m in self.escapes],
            "losers": [asdict(m) for m in self.losers],
            "refutation": list(self.refutation),
            "win_best": self.win_best,
            "win_next": self.win_next,
            "severity": self.severity,
            "bait": self.bait,
            "bait_games": self.bait_games,
            "games": self.games,
            "nodes": self.nodes,
            "multipv": self.multipv,
            "found_at": self.found_at,
        }


@dataclass
class Report:
    """What a scan did, in the terms the thresholds are written in."""

    positions: int = 0
    analysed: int = 0
    no_lines: int = 0
    too_many_escapes: int = 0
    not_severe: int = 0
    already_lost: int = 0
    trivial: int = 0
    kept: int = 0

    def line(self) -> str:
        return (
            f"{self.positions} book positions -> {self.kept} traps; rejected "
            f"{self.too_many_escapes} with more than {MAX_ESCAPES} moves that hold, "
            f"{self.not_severe} where the alternatives do not lose, "
            f"{self.already_lost} already lost, {self.trivial} trivial, "
            f"{self.no_lines} unanalysable"
        )


# -- judging one position --------------------------------------------------


def _moves_of(
    lines: Sequence[analysis.Line],
    board: chess.variant.AtomicBoard,
    model: winrate.WinModel,
    entries: Sequence[book.Entry],
) -> list[Move]:
    """Turn engine lines into `Move`s, with the book's counts attached."""
    total = sum(entry.games for entry in entries) or 0
    by_uci = {entry.uci: entry for entry in entries}
    out: list[Move] = []
    for line in lines:
        try:
            san = board.san(chess.Move.from_uci(line.uci))
        except (AssertionError, ValueError):
            continue
        entry = by_uci.get(line.uci)
        out.append(
            Move(
                san=san,
                uci=line.uci,
                win=line.win_pct(model),
                games=entry.games if entry else 0,
                share=100.0 * entry.games / total if entry and total else 0.0,
            )
        )
    return out


def judge(
    moves: Sequence[Move],
    *,
    hold_loss: float = HOLD_LOSS,
    max_escapes: int = MAX_ESCAPES,
    severity: float = SEVERITY,
    lost_ceiling: float = LOST_CEILING,
    balance_floor: float = BALANCE_FLOOR,
) -> tuple[list[Move], list[Move], str]:
    """Split the moves into escapes and losers, and say why it is or is not a trap.

    Returns `(escapes, losers, verdict)`, where verdict is `""` for a trap and
    otherwise names the threshold that rejected it — which is what the report
    counts, and what makes a scan that finds nothing diagnosable rather than
    mysterious.
    """
    if not moves:
        return [], [], "no_lines"
    best = moves[0].win
    escapes = [move for move in moves if move.win >= best - hold_loss]
    losers = [move for move in moves if move not in escapes]
    if best < balance_floor:
        return escapes, losers, "already_lost"
    if len(escapes) > max_escapes:
        return escapes, losers, "too_many_escapes"
    if not losers:
        # Every line the engine was asked for holds, so the boundary is outside
        # what MultiPV could see. Not a trap *as far as this scan can tell*,
        # which is a different fact from "not a trap" and is recorded as one.
        return escapes, losers, "too_many_escapes"
    next_best = losers[0].win
    if best - next_best < severity or next_best > lost_ceiling:
        return escapes, losers, "not_severe"
    return escapes, losers, ""


def _bait(entries: Sequence[book.Entry], escapes: Sequence[Move]) -> tuple[float, int]:
    """How much of the field walks in, counted from the book.

    Every book move that is not an escape counts, including the ones the engine
    never listed: a move too bad to make MultiPV 5 is certainly not one of the
    two that hold.
    """
    total = sum(entry.games for entry in entries)
    if not total:
        return 0.0, 0
    safe = {move.uci for move in escapes}
    walked = sum(entry.games for entry in entries if entry.uci not in safe)
    return 100.0 * walked / total, walked


def _refutation(
    db: sqlite3.Connection,
    board: chess.variant.AtomicBoard,
    loser: Move,
    *,
    nodes: int,
    multipv: int,
    plies: int = REFUTATION_PLIES,
) -> list[str]:
    """The losing move, then how it is punished, in SAN.

    Read out of the confirming pass's own principal variation rather than
    searched again: the PV is the line the engine's judgement of the position
    *is*, and a second search would produce a line that argues for a number
    computed by the first one.
    """
    after = board.copy(stack=False)
    try:
        after.push(chess.Move.from_uci(loser.uci))
    except (AssertionError, ValueError):
        return []
    out = [loser.san]
    lines = analysis.cached_lines(db, after.fen(), nodes, multipv)
    if not lines:
        return out
    for uci in lines[0].pv[:plies]:
        try:
            move = chess.Move.from_uci(uci)
            out.append(after.san(move))
            after.push(move)
        except (AssertionError, ValueError):
            break
    return out


# -- the scan --------------------------------------------------------------


def positions(
    db: sqlite3.Connection,
    *,
    depth: int = DEFAULT_DEPTH,
    min_games: int = DEFAULT_MIN_GAMES,
    min_share: float = DEFAULT_MIN_SHARE,
    max_children: int = DEFAULT_MAX_CHILDREN,
    limit: int | None = None,
) -> list[book.Reached]:
    """The book positions a scan will look at, most-played first."""
    found = list(
        book.walk(
            db,
            depth=depth,
            min_games=min_games,
            min_share=min_share,
            max_children=max_children,
        )
    )
    found = [node for node in found if node.line]   # the start is nobody's trap
    found.sort(key=lambda node: -node.games)
    return found[:limit] if limit else found


def _record_scan(
    db: sqlite3.Connection,
    fen: str,
    nodes: int,
    multipv: int,
    escapes: int,
    severity: float | None,
    kept: bool,
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO trap_scans "
        "(fen, nodes, multipv, escapes, severity, kept, scanned_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (fen, nodes, multipv, escapes, severity, int(kept), time.time()),
    )


def scan(
    db: sqlite3.Connection,
    nodes_: Sequence[book.Reached],
    *,
    model: winrate.WinModel,
    scan_nodes: int = SCAN_NODES,
    confirm_nodes: int = CONFIRM_NODES,
    multipv: int = MULTIPV,
    hold_loss: float = HOLD_LOSS,
    max_escapes: int = MAX_ESCAPES,
    severity: float = SEVERITY,
    workers: int | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[Trap], Report]:
    """Screen every position, then confirm the survivors properly.

    The screen is deliberately *looser* than the definition — half the severity
    and one more escape allowed — because its job is to throw away the 90% of
    the tree where nothing is close, not to decide anything. Everything it
    keeps is re-judged at `confirm_nodes` against the real thresholds, and the
    stored row carries that budget.
    """
    report = Report(positions=len(nodes_))
    if not nodes_:
        return [], report

    if on_progress:
        on_progress("screening", 0, len(nodes_))
    report.analysed += analysis.ensure_analysed(
        db,
        [node.board.fen() for node in nodes_],
        nodes=scan_nodes,
        multipv=multipv,
        workers=workers,
        on_progress=(lambda d, t: on_progress("screening", d, t)) if on_progress else None,
    )

    survivors: list[book.Reached] = []
    for node in nodes_:
        moves = _moves_of(
            analysis.cached_lines(db, node.board.fen(), scan_nodes, multipv),
            node.board, model, node.entries,
        )
        escapes, _, verdict = judge(
            moves,
            hold_loss=hold_loss,
            max_escapes=max_escapes + 1,
            severity=severity / 2,
            lost_ceiling=LOST_CEILING + 15.0,
            balance_floor=BALANCE_FLOOR - 8.0,
        )
        _record_scan(
            db, node.board.fen(), scan_nodes, multipv,
            len(escapes) if moves else -1, None, not verdict,
        )
        if verdict:
            continue
        survivors.append(node)
    db.commit()

    if on_progress:
        on_progress("confirming", 0, len(survivors))
    report.analysed += analysis.ensure_analysed(
        db,
        [node.board.fen() for node in survivors],
        nodes=confirm_nodes,
        multipv=multipv,
        workers=workers,
        on_progress=(lambda d, t: on_progress("confirming", d, t)) if on_progress else None,
    )

    found: list[Trap] = []
    punish: list[tuple[book.Reached, Trap, Move]] = []
    for node in survivors:
        moves = _moves_of(
            analysis.cached_lines(db, node.board.fen(), confirm_nodes, multipv),
            node.board, model, node.entries,
        )
        escapes, losers, verdict = judge(
            moves,
            hold_loss=hold_loss,
            max_escapes=max_escapes,
            severity=severity,
        )
        if verdict:
            setattr(report, verdict, getattr(report, verdict) + 1)
            _record_scan(db, node.board.fen(), confirm_nodes, multipv, len(escapes), None, False)
            continue
        if all(generate.is_trivial(node.board, move.uci) for move in escapes):
            report.trivial += 1
            _record_scan(db, node.board.fen(), confirm_nodes, multipv, len(escapes), None, False)
            continue
        bait, bait_games = _bait(node.entries, escapes)
        # The move the field actually plays into, when there is one — that is
        # the punishment worth spelling out. Failing that, the engine's own
        # best losing try, so a trap nobody has fallen for yet still shows why.
        played = sorted(losers, key=lambda m: (-m.games, m.win))
        trap = Trap(
            fen=node.board.fen(),
            line=node.line,
            turn="white" if node.board.turn == chess.WHITE else "black",
            escapes=escapes,
            losers=played,
            refutation=[],
            win_best=escapes[0].win,
            win_next=losers[0].win,
            bait=bait,
            bait_games=bait_games,
            games=node.games,
            nodes=confirm_nodes,
            multipv=multipv,
        )
        found.append(trap)
        punish.append((node, trap, played[0]))
        _record_scan(db, node.board.fen(), confirm_nodes, multipv, len(escapes), trap.severity, True)
        report.kept += 1
    db.commit()

    if punish:
        if on_progress:
            on_progress("punishing", 0, len(punish))
        after_fens = []
        for node, _, loser in punish:
            probe = node.board.copy(stack=False)
            try:
                probe.push(chess.Move.from_uci(loser.uci))
            except (AssertionError, ValueError):
                continue
            if not probe.is_game_over(claim_draw=True):
                after_fens.append(probe.fen())
        report.analysed += analysis.ensure_analysed(
            db, after_fens, nodes=confirm_nodes, multipv=multipv, workers=workers,
            on_progress=(lambda d, t: on_progress("punishing", d, t)) if on_progress else None,
        )
        for node, trap, loser in punish:
            trap.refutation = _refutation(
                db, node.board, loser, nodes=confirm_nodes, multipv=multipv
            )
    return found, report


def run(
    db: sqlite3.Connection,
    *,
    model: winrate.WinModel | None = None,
    depth: int = DEFAULT_DEPTH,
    min_games: int = DEFAULT_MIN_GAMES,
    min_share: float = DEFAULT_MIN_SHARE,
    max_children: int = DEFAULT_MAX_CHILDREN,
    limit: int | None = None,
    scan_nodes: int = SCAN_NODES,
    confirm_nodes: int = CONFIRM_NODES,
    multipv: int = MULTIPV,
    workers: int | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[int, Report]:
    """Walk the book, scan it, store what qualifies."""
    model = model or winrate.load()
    todo = positions(
        db,
        depth=depth,
        min_games=min_games,
        min_share=min_share,
        max_children=max_children,
        limit=limit,
    )
    found, report = scan(
        db, todo,
        model=model, scan_nodes=scan_nodes, confirm_nodes=confirm_nodes,
        multipv=multipv, workers=workers, on_progress=on_progress,
    )
    return save(db, found), report


# -- storage ---------------------------------------------------------------

_COLUMNS = (
    "id", "fen", "line", "ply", "turn", "escapes", "losers", "refutation",
    "win_best", "win_next", "severity", "bait", "bait_games", "games",
    "nodes", "multipv", "found_at",
)


def save(db: sqlite3.Connection, found: Iterable[Trap]) -> int:
    rows = [
        (
            trap.id, trap.fen, " ".join(trap.line), trap.ply, trap.turn,
            json.dumps([asdict(m) for m in trap.escapes]),
            json.dumps([asdict(m) for m in trap.losers]),
            json.dumps(trap.refutation),
            trap.win_best, trap.win_next, trap.severity, trap.bait,
            trap.bait_games, trap.games, trap.nodes, trap.multipv, trap.found_at,
        )
        for trap in found
    ]
    db.executemany(
        f"INSERT OR REPLACE INTO traps ({','.join(_COLUMNS)}) "
        f"VALUES ({','.join('?' * len(_COLUMNS))})",
        rows,
    )
    db.commit()
    return len(rows)


def _row(row: sqlite3.Row) -> Trap:
    return Trap(
        fen=row["fen"],
        line=tuple(row["line"].split()) if row["line"] else (),
        turn=row["turn"],
        escapes=[Move(**m) for m in json.loads(row["escapes"])],
        losers=[Move(**m) for m in json.loads(row["losers"])],
        refutation=json.loads(row["refutation"]),
        win_best=row["win_best"],
        win_next=row["win_next"],
        bait=row["bait"],
        bait_games=row["bait_games"],
        games=row["games"],
        nodes=row["nodes"],
        multipv=row["multipv"],
        found_at=row["found_at"],
    )


ORDERS = {
    "bait": "bait DESC, games DESC",
    "severity": "severity DESC, bait DESC",
    "popular": "games DESC",
    "early": "ply ASC, games DESC",
}


def listing(
    db: sqlite3.Connection,
    *,
    turn: str | None = None,
    max_ply: int | None = None,
    min_bait: float = 0.0,
    order: str = "bait",
    limit: int = 100,
) -> list[Trap]:
    sql = "SELECT * FROM traps WHERE bait >= ?"
    params: list[object] = [min_bait]
    if turn in ("white", "black"):
        sql += " AND turn = ?"
        params.append(turn)
    if max_ply is not None:
        sql += " AND ply <= ?"
        params.append(max_ply)
    sql += f" ORDER BY {ORDERS.get(order, ORDERS['bait'])} LIMIT ?"
    params.append(limit)
    return [_row(row) for row in db.execute(sql, params)]


def get(db: sqlite3.Connection, trap_id: str) -> Trap | None:
    row = db.execute("SELECT * FROM traps WHERE id = ?", (trap_id,)).fetchone()
    return _row(row) if row else None


def stats(db: sqlite3.Connection) -> dict[str, object]:
    row = db.execute(
        "SELECT COUNT(*) AS n, AVG(bait) AS bait, MAX(severity) AS worst, "
        "SUM(turn = 'white') AS white, SUM(turn = 'black') AS black FROM traps"
    ).fetchone()
    scanned = db.execute("SELECT COUNT(*) AS n FROM trap_scans").fetchone()
    return {
        "traps": row["n"] or 0,
        "avg_bait": row["bait"] or 0.0,
        "worst": row["worst"] or 0.0,
        "white": row["white"] or 0,
        "black": row["black"] or 0,
        "scanned": scanned["n"] or 0,
    }


def described(db: sqlite3.Connection, trap: Trap) -> dict[str, object]:
    """A trap plus the name of the line it sits in — what the UI draws."""
    return {**trap.as_dict(), "name": openings.name_line(db, trap.line).as_dict()}


__all__ = [
    "Move",
    "Report",
    "Trap",
    "judge",
    "listing",
    "positions",
    "run",
    "save",
    "scan",
    "stats",
]
