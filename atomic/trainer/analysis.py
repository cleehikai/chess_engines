"""The analysis service: Fairy-Stockfish over a game, cached and resumable.

One deep pass, MultiPV, over **every** ply of a game — not a cheap scan
followed by a deep re-check. At ~1M nodes/ply and a 21-ply median atomic game
that is 30-60s of engine time per game, which is the budget this project chose
to spend, and it buys two things a two-stage funnel does not:

- the **uniqueness gap** (best move vs second best) is known at every ply, not
  only at the plies where something already went wrong — that is the test a
  position has to pass to become a puzzle, so puzzle mining becomes a query
  over the cache rather than a second pass over the games;
- the cache becomes the asset. Re-reviewing a game, generating puzzles from it,
  or validating a repertoire line that transposes into it are all free
  afterwards.

Resumability is at position granularity: every completed analysis is written to
the cache as its future lands, so an interrupted run loses at most the
positions still in flight. Games are marked analysed only once every ply of
them has been judged.

Parallelism is one Fairy-Stockfish process per worker thread. Threads rather
than processes because the work happens inside those subprocesses — the GIL is
irrelevant and nothing has to be pickled.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Self

import chess
import chess.engine
import chess.variant

from lab import fsf

from . import store, winrate

#: Identifies what produced a cache entry. Change it if the engine or net
#: changes, so old judgements are never silently mixed with new ones.
ENGINE_ID = "fsf14-nnue-atomic"

DEFAULT_NODES = 1_000_000
DEFAULT_MULTIPV = 2


def default_workers() -> int:
    """Leave the machine usable while a long import runs."""
    return max(1, (os.cpu_count() or 4) - 2)


@dataclass(frozen=True)
class Line:
    """One MultiPV line, scored from the moving side's point of view.

    `pv` is the whole principal variation, and it is stored rather than thrown
    away because the explore board wants to show what the engine intends, not
    just its first move. Entries written before it existed have an empty `pv`
    and are still perfectly good — `uci` is the field every judgement uses, and
    it is carried separately so nothing has to reach into a list to find it.
    """

    uci: str
    cp: int | None
    mate: int | None
    depth: int | None
    pv: tuple[str, ...] = ()

    def win_pct(self, model: winrate.WinModel) -> float:
        return model.score_win_pct(self.cp, self.mate)

    @property
    def rank(self) -> tuple[int, int]:
        """Sort key, best first when sorted descending.

        Mates beat evaluations, a shorter mate beats a longer one, and a mate
        *against* is worse than any evaluation — with a longer one preferred,
        since it gives the opponent more chances to go wrong.
        """
        if self.mate is not None:
            return (2, -self.mate) if self.mate > 0 else (0, -self.mate)
        return (1, self.cp or 0)

    @classmethod
    def from_info(cls, info: chess.engine.InfoDict) -> Line | None:
        pv = info.get("pv")
        if not pv:
            return None
        score = info["score"].relative
        return cls(
            uci=pv[0].uci(),
            cp=score.score(),
            mate=score.mate(),
            depth=info.get("depth"),
            pv=tuple(move.uci() for move in pv[:12]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "uci": self.uci, "cp": self.cp, "mate": self.mate,
            "depth": self.depth, "pv": list(self.pv),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Line:
        return cls(
            uci=raw["uci"], cp=raw.get("cp"), mate=raw.get("mate"),
            depth=raw.get("depth"), pv=tuple(raw.get("pv") or ()),
        )


class EnginePool:
    """One Fairy-Stockfish session per worker thread, opened on first use."""

    def __init__(self, nodes: int, hash_mb: int = 128) -> None:
        self._cfg = fsf.FsfConfig(nodes=nodes, threads=1, hash_mb=hash_mb, use_nnue=True)
        self._local = threading.local()
        self._all: list[Any] = []
        self._lock = threading.Lock()

    @property
    def limit(self) -> chess.engine.Limit:
        return self._cfg.limit

    def engine(self):
        got = getattr(self._local, "engine", None)
        if got is None:
            ctx = fsf.open_engine(self._cfg)
            got = ctx.__enter__()
            self._local.engine = got
            with self._lock:
                self._all.append(ctx)
        return got

    def close(self) -> None:
        with self._lock:
            for ctx in self._all:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:  # noqa: BLE001, S110 - shutting down; nothing to do
                    pass
            self._all.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def best_first(lines: list[Line]) -> list[Line]:
    """Order MultiPV lines by score, best first.

    Fairy-Stockfish's MultiPV output is *not* reliably sorted under a node
    limit: the budget can run out partway through an iteration, leaving PV1
    holding a value from the current depth and PV2 one from the previous, so
    the reported second line sometimes scores higher than the first. Measured
    here on real positions, "best minus second best" came out negative by a
    couple of win% points, which is impossible by construction and would poison
    the uniqueness test that decides whether a position becomes a puzzle.

    Applied on read as well as on write, so entries already in the cache are
    fixed without re-analysing them.
    """
    return sorted(lines, key=lambda line: line.rank, reverse=True)


def _read_infos(infos: Any) -> tuple[list[Line], int | None]:
    if isinstance(infos, dict):  # python-chess returns a bare dict for multipv=1
        infos = [infos]
    lines = [line for line in (Line.from_info(i) for i in infos) if line is not None]
    depth = max((line.depth or 0 for line in lines), default=None)
    return best_first(lines), depth


def analyse_fen(pool: EnginePool, fen: str, multipv: int) -> tuple[list[Line], int | None]:
    """Analyse one position. Scores come back from the side to move's view."""
    board = chess.variant.AtomicBoard(fen)
    return _read_infos(pool.engine().analyse(board, pool.limit, multipv=multipv))


class LiveEngines:
    """A small pool of warm sessions, for analysis somebody is waiting on.

    `EnginePool` is built for batch work: it is opened around a run and torn
    down with it, and its sessions are thread-local — exactly wrong under a
    `ThreadingHTTPServer`, which hands every request a fresh thread and so
    would pay a process launch per request and throw the hash away with it.

    This one is a module-level free list, so the explore board's ladder of
    deepening searches reuses one process and, more to the point, one
    *transposition table*: the 400k-node pass over a position lands far sooner
    when the 120k-node pass over the same position filled the table first.

    Two sessions rather than one. A search cannot be cancelled once it is
    running, so when the cursor moves on, the abandoned search is still
    burning; with a single session the answer somebody is now waiting for
    would queue behind it. Not more than two, because each is a real process
    holding real hash and the board asks for one search at a time.
    """

    def __init__(self, size: int = 2, hash_mb: int = 256) -> None:
        self._cfg = fsf.FsfConfig(threads=1, hash_mb=hash_mb, use_nnue=True)
        self._free: list[tuple[Any, Any]] = []
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(size)

    @contextlib.contextmanager
    def session(self) -> Iterator[Any]:
        self._slots.acquire()
        entry = None
        try:
            with self._lock:
                entry = self._free.pop() if self._free else None
            if entry is None:
                ctx = fsf.open_engine(self._cfg)
                entry = (ctx, ctx.__enter__())
            try:
                yield entry[1]
            except BaseException:
                # A session that raised may be a dead process; putting it back
                # would hand the fault to the next request instead of this one.
                with contextlib.suppress(Exception):
                    entry[0].__exit__(None, None, None)
                entry = None
                raise
            with self._lock:
                self._free.append(entry)
        finally:
            self._slots.release()

    def close(self) -> None:
        """Whoever owns the process must call this before it exits.

        python-chess runs each engine's event loop on a *non-daemon* thread,
        and the interpreter joins those before it runs `atexit` handlers — so
        a pool left open hangs the process at shutdown, and registering the
        cleanup with `atexit` does not help: by then it is already waiting.
        Measured here, Ctrl-C on the server never returned.
        """
        with self._lock:
            entries, self._free = self._free, []
        for ctx, _ in entries:
            with contextlib.suppress(Exception):
                ctx.__exit__(None, None, None)


#: The interactive engine. Batch runs keep using `EnginePool`; this exists so
#: the explore board does not launch a process per keystroke.
LIVE = LiveEngines()


def evaluate_now(
    db: sqlite3.Connection, fen: str, *, nodes: int, multipv: int
) -> list[Line]:
    """One position, analysed on a warm session and cached, best line first.

    The same cache `ensure_analysed` writes: a position somebody looked at
    interactively is paid for once, and a later deep pass may find it already
    bought.
    """
    raw = store.cached_analysis(db, fen, nodes, multipv, ENGINE_ID)
    if raw is not None:
        return best_first([Line.from_dict(item) for item in raw])
    board = chess.variant.AtomicBoard(fen)
    with LIVE.session() as engine:
        lines, depth = _read_infos(
            engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv)
        )
    store.store_analysis(
        db, fen, nodes, multipv, ENGINE_ID,
        [line.as_dict() for line in lines], depth, time.time(),
    )
    db.commit()
    return lines


def cached_lines(
    db: sqlite3.Connection, fen: str, nodes: int, multipv: int
) -> list[Line]:
    """Whatever the cache holds for this position, best line first."""
    raw = store.cached_analysis(db, fen, nodes, multipv, ENGINE_ID)
    return best_first([Line.from_dict(item) for item in (raw or [])])


def ensure_analysed(
    db: sqlite3.Connection,
    fens: Iterable[str],
    *,
    nodes: int = DEFAULT_NODES,
    multipv: int = DEFAULT_MULTIPV,
    workers: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Analyse whichever of `fens` are not cached yet; return how many were run.

    The unit of work here is a *position*, not a game, which is what makes
    repertoire validation nearly free: every node of a repertoire line is a
    position some reviewed game may already have paid for, and two repertoire
    lines that transpose pay once between them. Results are written as they
    land, so an interrupted run keeps everything it has already bought.
    """
    todo = [
        fen
        for fen in dict.fromkeys(fens)
        if store.cached_analysis(db, fen, nodes, multipv, ENGINE_ID) is None
    ]
    if not todo:
        return 0
    workers = workers or default_workers()
    done = 0
    with EnginePool(nodes) as pool, ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_fen, pool, fen, multipv): fen for fen in todo}
        for future in as_completed(futures):
            lines, depth = future.result()
            store.store_analysis(
                db, futures[future], nodes, multipv, ENGINE_ID,
                [line.as_dict() for line in lines], depth, time.time(),
            )
            done += 1
            if done % 25 == 0:
                db.commit()
            if on_progress:
                on_progress(done, len(todo))
    db.commit()
    return done


# -- game-level ------------------------------------------------------------


@dataclass
class Judged:
    """One judged ply, ready to be written to `plies`."""

    ply: int
    fen: str
    mover: str
    uci: str
    san: str
    win_before: float
    win_after: float
    loss: float
    judgement: str
    best_uci: str | None
    best_san: str | None
    win_best: float | None
    win_second: float | None
    unique_gap: float | None


def _terminal_win_pct(board: chess.variant.AtomicBoard, mover: chess.Color) -> float | None:
    """Win% for `mover` if `board` is already over, else None.

    Only genuine over-the-board endings count. A resignation or a flag fall
    leaves a perfectly ordinary position behind, and judging the last move by
    the *game's* result rather than the position's would blame a player for
    losing on time.
    """
    if not board.is_game_over(claim_draw=True):
        return None
    result = board.result(claim_draw=True)
    if result == "1/2-1/2":
        return 50.0
    winner = chess.WHITE if result == "1-0" else chess.BLACK
    return 100.0 if winner == mover else 0.0


def positions_of(moves: Sequence[str]) -> list[tuple[str, chess.Move, str, chess.Color]]:
    """Replay SAN moves, yielding (fen_before, move, san, mover) per ply."""
    board = chess.variant.AtomicBoard()
    out = []
    for san in moves:
        try:
            move = board.parse_san(san)
        except ValueError as exc:
            raise ValueError(f"illegal SAN {san!r} at ply {len(out)}: {exc}") from None
        out.append((board.fen(), move, san, board.turn))
        board.push(move)
    return out


def judge_game(
    db: sqlite3.Connection,
    pool: EnginePool,
    moves: Sequence[str],
    model: winrate.WinModel,
    nodes: int,
    multipv: int,
    executor: ThreadPoolExecutor,
) -> list[Judged]:
    """Analyse every position of a game and judge every move played.

    A move's cost is `win%(position before, best play) - win%(position after)`.
    The second term is read off the *next* position's analysis rather than
    analysed separately, so an N-move game costs N+1 analyses instead of 2N.
    """
    plies = positions_of(moves)
    board = chess.variant.AtomicBoard()
    for _fen, move, _san, _turn in plies:
        board.push(move)
    fens = [fen for fen, _, _, _ in plies]
    final_fen = board.fen()
    final_terminal = board.is_game_over(claim_draw=True)
    if not final_terminal:
        fens.append(final_fen)

    # Cache first, engine only for what is missing; the futures write straight
    # through so an interrupted run keeps everything already paid for.
    todo = [
        fen
        for fen in dict.fromkeys(fens)
        if store.cached_analysis(db, fen, nodes, multipv, ENGINE_ID) is None
    ]
    if todo:
        futures = {
            fen: executor.submit(analyse_fen, pool, fen, multipv) for fen in todo
        }
        for fen, future in futures.items():
            lines, depth = future.result()
            store.store_analysis(
                db, fen, nodes, multipv, ENGINE_ID,
                [line.as_dict() for line in lines], depth, time.time(),
            )
        db.commit()

    def lines_for(fen: str) -> list[Line]:
        return cached_lines(db, fen, nodes, multipv)

    judged: list[Judged] = []
    replay = chess.variant.AtomicBoard()
    for index, (fen, move, san, mover) in enumerate(plies):
        before = lines_for(fen)
        if not before:
            replay.push(move)
            continue  # no legal moves to analyse; nothing to judge

        win_best = before[0].win_pct(model)
        win_second = before[1].win_pct(model) if len(before) > 1 else None
        best_uci = before[0].uci
        best_san = replay.san(chess.Move.from_uci(best_uci))
        replay.push(move)

        is_last = index == len(plies) - 1
        if is_last and final_terminal:
            win_after = _terminal_win_pct(replay, mover)
        else:
            after_fen = fens[index + 1] if index + 1 < len(plies) else final_fen
            after_lines = lines_for(after_fen)
            # The next position is scored for the *opponent*, so flip it.
            win_after = (
                100.0 - after_lines[0].win_pct(model) if after_lines else win_best
            )
        assert win_after is not None

        loss = max(0.0, win_best - win_after)
        judged.append(
            Judged(
                ply=index,
                fen=fen,
                mover="white" if mover == chess.WHITE else "black",
                uci=move.uci(),
                san=san,
                win_before=win_best,
                win_after=win_after,
                loss=loss,
                judgement=winrate.classify(loss),
                best_uci=best_uci,
                best_san=best_san,
                win_best=win_best,
                win_second=win_second,
                unique_gap=None if win_second is None else win_best - win_second,
            )
        )
    return judged


def write_plies(db: sqlite3.Connection, game_id: str, judged: Iterable[Judged], nodes: int) -> None:
    db.execute("DELETE FROM plies WHERE game_id = ?", (game_id,))
    db.executemany(
        "INSERT INTO plies (game_id, ply, fen, mover, uci, san, win_before, "
        "win_after, loss, judgement, best_uci, best_san, win_best, win_second, "
        "unique_gap, nodes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                game_id, j.ply, j.fen, j.mover, j.uci, j.san, j.win_before,
                j.win_after, j.loss, j.judgement, j.best_uci, j.best_san,
                j.win_best, j.win_second, j.unique_gap, nodes,
            )
            for j in judged
        ],
    )


def pending_games(db: sqlite3.Connection, nodes: int, account: str | None = None) -> list[sqlite3.Row]:
    """Games not yet analysed to at least `nodes`."""
    sql = (
        "SELECT * FROM games WHERE (analysed_nodes IS NULL OR analysed_nodes < ?) "
        "AND ply > 0"
    )
    params: list[Any] = [nodes]
    if account:
        sql += " AND account = ?"
        params.append(account)
    sql += " ORDER BY created_at DESC"
    return list(db.execute(sql, params))


def analyse_games(
    db: sqlite3.Connection,
    games: Sequence[sqlite3.Row],
    *,
    nodes: int = DEFAULT_NODES,
    multipv: int = DEFAULT_MULTIPV,
    workers: int | None = None,
    model: winrate.WinModel | None = None,
    on_game: Callable[[int, sqlite3.Row, list[Judged]], None] | None = None,
) -> int:
    """Deep-analyse `games`, writing judgements and marking each one done."""
    model = model or winrate.load()
    workers = workers or default_workers()
    done = 0
    with EnginePool(nodes) as pool, ThreadPoolExecutor(max_workers=workers) as executor:
        for index, game in enumerate(games):
            moves = (game["moves"] or "").split()
            if not moves:
                continue
            judged = judge_game(db, pool, moves, model, nodes, multipv, executor)
            write_plies(db, game["id"], judged, nodes)
            db.execute(
                "UPDATE games SET analysed_at = ?, analysed_nodes = ? WHERE id = ?",
                (time.time(), nodes, game["id"]),
            )
            db.commit()
            done += 1
            if on_game:
                on_game(index, game, judged)
    return done
