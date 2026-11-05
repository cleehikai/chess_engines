"""Long-running work, started from the browser and watched while it runs.

Generating puzzles takes minutes to hours. That is fine on a command line and
useless in a page, so the three generators — mine your games, scrape the
Lichess dumps, probe the tablebase — run here instead: one background thread,
progress readable over HTTP, cancellable.

**Only one at a time.** Every job saturates Fairy-Stockfish across every core
the machine has; two of them would not go twice as fast, they would go half as
fast each and fight over the same SQLite writer. So the registry refuses to
start a second job while one is running, and says so rather than queueing —
a queue would let a mis-click commit the machine to an hour of work invisibly.

**Its own connection.** A job opens its own `store` connection on its own
thread, because SQLite connections are not shared across threads and because
the job holds the write lock for long stretches. The UI's per-request
connections are unaffected: WAL means a reader never blocks behind the writer,
which is the property that makes watching a running job possible at all.

**Progress is advisory.** The counters below are for a human watching a bar
move. Nothing reads them back, and a job whose progress callback never fires
is still doing its work — so a generator that cannot report cheaply is not
obliged to invent a number.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import difficulty, endgames, generate, mine, puzzles, store, traps, winrate

#: What the UI may start, and what each one is called there.
KINDS = {
    "generate": "Mine my own games",
    "mine": "Scrape rated Lichess games",
    "endgames": "Tablebase endgames",
    "rate": "Measure puzzle difficulty",
    "traps": "Scan the book for traps",
}


@dataclass
class Job:
    """One background run, and everything the page needs to draw it."""

    id: str
    kind: str
    params: dict[str, Any]
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    stage: str = "starting"
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def running(self) -> bool:
        return self.finished_at is None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": KINDS.get(self.kind, self.kind),
            "params": self.params,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "running": self.running,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "cancelled": self.cancelled,
            "elapsed": (self.finished_at or time.time()) - self.started_at,
        }


class Registry:
    """The one job that may be running, plus what the recent ones did."""

    #: How many finished jobs to remember. They exist so the page can show
    #: "that run found 14 puzzles" after the fact; a session's worth is plenty.
    HISTORY = 12

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._current: str | None = None
        self._counter = itertools.count(1)

    # -- reading -----------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def listing(self) -> dict[str, Any]:
        with self._lock:
            jobs = [self._jobs[key].public() for key in reversed(self._order)]
        return {"jobs": jobs, "running": any(job["running"] for job in jobs),
                "kinds": KINDS}

    # -- writing -----------------------------------------------------------

    def start(self, kind: str, params: dict[str, Any]) -> Job:
        if kind not in KINDS:
            raise ValueError(f"no such job: {kind}")
        with self._lock:
            current = self._jobs.get(self._current or "")
            if current is not None and current.running:
                raise ValueError(
                    f"{KINDS[current.kind]} is already running — "
                    "wait for it or cancel it first"
                )
            job = Job(id=f"j{next(self._counter)}", kind=kind, params=params)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._current = job.id
            for stale in self._order[: -self.HISTORY]:
                self._jobs.pop(stale, None)
            self._order = self._order[-self.HISTORY:]
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. It stops at its next checkpoint, not instantly."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.running:
                return False
            job.cancelled = True
            job.message = "stopping after the current batch…"
            return True

    # -- the thread --------------------------------------------------------

    def _run(self, job: Job) -> None:
        def progress(stage: str, done: int, total: int) -> None:
            job.stage, job.done, job.total = stage, done, total

        try:
            with store.open_db() as db:
                job.result = RUNNERS[job.kind](db, job, progress)
            job.stage = "cancelled" if job.cancelled else "done"
        except Exception as exc:  # noqa: BLE001 - a job must not kill the server
            job.error = f"{type(exc).__name__}: {exc}"
            job.stage = "failed"
            traceback.print_exc()
        finally:
            job.finished_at = time.time()


Progress = Callable[[str, int, int], None]


def _run_generate(db, job: Job, progress: Progress) -> dict[str, Any]:
    """Mine puzzles from your own already-analysed games. Free until stage 2."""
    written, report = generate.run(
        db,
        model=winrate.load(),
        account=job.params.get("account") or None,
        game_id=job.params.get("game") or None,
        nodes=int(job.params.get("nodes") or generate.CONFIRM_NODES),
        limit=int(job.params["limit"]) if job.params.get("limit") else None,
        saves=job.params.get("saves", True),
        on_progress=progress,
    )
    return {"stored": written, "summary": report.line(), **vars(report)}


def _run_mine(db, job: Job, progress: Progress) -> dict[str, Any]:
    """Scrape strangers' rated games out of the Lichess monthly dumps."""
    months = job.params.get("months") or None
    written, report = mine.run(
        db,
        months=months,
        games=int(job.params.get("games") or 200),
        min_elo=int(job.params.get("min_elo") or mine.MIN_ELO),
        scan_nodes=int(job.params.get("scan_nodes") or mine.SCAN_NODES),
        confirm_nodes=int(job.params.get("nodes") or generate.CONFIRM_NODES),
        saves=job.params.get("saves", True),
        cancelled=lambda: job.cancelled,
        on_progress=progress,
    )
    return {
        "stored": written,
        "summary": report.line(),
        "games_scanned": report.games_scanned,
        "games_read": report.games_read,
    }


def _run_endgames(db, job: Job, progress: Progress) -> dict[str, Any]:
    """Exact puzzles from the tablebase. Rate limited, so it reports positions."""
    def relay(seen: int, report: endgames.Report) -> None:
        progress("probing", seen, int(job.params.get("samples") or 200))

    written, report = endgames.run(
        db,
        source=job.params.get("source") or "walk",
        samples=int(job.params.get("samples") or 200),
        limit=int(job.params["limit"]) if job.params.get("limit") else None,
        on_progress=relay,
    )
    return {"stored": written, "summary": report.line()}


def _run_rate(db, job: Job, progress: Progress) -> dict[str, Any]:
    """Run the corpus up the calibrated engine ladder (see `difficulty.py`).

    The only job here that touches no external service and no Fairy-Stockfish:
    it is the Rust core, in this process, and it is interruptible between
    puzzles — so cancelling costs at most one puzzle's worth of work.
    """
    rated, report = difficulty.run(
        db,
        spec=puzzles.Filter(
            theme=job.params.get("theme") or None,
            phase=job.params.get("phase") or None,
            source=job.params.get("source") or None,
            limit=int(job.params["limit"]) if job.params.get("limit") else 1_000_000,
            order="recent",
        ),
        force=bool(job.params.get("force")),
        cancelled=lambda: job.cancelled,
        on_progress=progress,
    )
    return {
        "stored": rated,
        "summary": report.line(),
        "floor": report.floor,
        "unsolved": report.unsolved,
        "non_monotone": report.non_monotone,
        "bands": report.bands,
    }


def _run_traps(db, job: Job, progress: Progress) -> dict[str, Any]:
    """Walk the opening book looking for positions where one move holds.

    Unlike every other job here this one spends its engine time on positions
    nobody has played *yet* — the book says the line is popular, not that
    anyone went wrong in it — so its cost is set by the size of the tree rather
    than by how much history there is to scrape.
    """
    written, report = traps.run(
        db,
        model=winrate.load(),
        depth=int(job.params.get("depth") or traps.DEFAULT_DEPTH),
        min_games=int(job.params.get("min_games") or traps.DEFAULT_MIN_GAMES),
        limit=int(job.params["limit"]) if job.params.get("limit") else None,
        scan_nodes=int(job.params.get("scan_nodes") or traps.SCAN_NODES),
        confirm_nodes=int(job.params.get("nodes") or traps.CONFIRM_NODES),
        on_progress=progress,
    )
    return {"stored": written, "summary": report.line(), **vars(report)}


RUNNERS: dict[str, Callable[[Any, Job, Progress], dict[str, Any]]] = {
    "generate": _run_generate,
    "mine": _run_mine,
    "endgames": _run_endgames,
    "rate": _run_rate,
    "traps": _run_traps,
}


__all__ = ["KINDS", "Job", "Registry"]
