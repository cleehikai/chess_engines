"""The trainer's browser UI, served by the stdlib.

    python -m trainer.cli serve

`lab/server.py` is the play server and stays exactly as it is. This one mounts
everything the trainer knows — puzzles, game review, patterns, the book, the
repertoire — and reuses lab's board plumbing rather than reimplementing it,
which is what `docs/trainer-plan.md` §2 asks for. The imports below reach into
lab's private names on purpose: `_state` *is* the referee, and a second copy of
it in this file would be a second atomic rules implementation to keep in step.

**The referee rule, extended to the answer.** lab's server owns legality so the
page cannot invent a move; this one also owns the *solution* so the page cannot
reveal one. `/api/puzzle/<id>` returns the position, the themes and the number
of moves to find — never the moves. Checking a move, taking a hint and giving
up are all round trips. The cost is a request per move, on localhost, which is
nothing; the benefit is that a puzzle cannot be solved with the devtools open.

**No sessions.** Every endpoint takes the position or the move index it is
talking about. A reload cannot desynchronise the server because the server is
not tracking anything, which is the same property lab's play server has and for
the same reason.

**One connection per request.** SQLite with WAL, opened and closed around each
handler, so a long `puzzles generate` running in another terminal does not lock
the UI out and the UI never holds a write lock while a user reads a page.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import sqlite3
import threading
import webbrowser
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import chess
import chess.engine
import chess.variant

from lab import fsf

# lab's own internals, reused deliberately — see the module docstring.
from lab.server import UI_DIR, EngineHolder, _board_from, _levels, _state

from . import (
    analysis,
    book,
    difficulty,
    explain,
    generate,
    jobs,
    mine,
    openings,
    puzzles,
    repertoire,
    review,
    store,
    themes,
    traps,
    winrate,
)

#: Evaluation budget for the "why was my move wrong" button. Deliberately much
#: cheaper than the review pass: this is a hint, it happens while someone is
#: waiting, and its answer goes into the same cache the deep passes read.
FEEDBACK_NODES = 200_000

#: Budget per search for "why this move". An explanation is four or five
#: searches (the position, the move asked about, two null moves), so each one
#: is kept to what lands in well under a second on the atomic net. Each is a
#: cache key, so asking again is free.
EXPLAIN_NODES = 300_000
FEEDBACK_MULTIPV = 3

#: Fairy-Stockfish as an *opponent*, in seconds per move. Time rather than
#: nodes, which is the reverse of every other engine setting in this project
#: and is right here: node budgets exist so a measurement reproduces on another
#: machine, and nobody is measuring anything by playing a game against the top
#: rung. What is wanted is "as strong as this box can be in a reasonable wait",
#: which is a clock, all the cores, and the NNUE net.
#:
#: These levels are deliberately *not* given Elo numbers. The lab's ladder is
#: anchored by pinning throttled FSF at 64k nodes to 2000; full-strength FSF
#: with the atomic net searches ~1.24 Mnps and is the gold standard the whole
#: ladder is measured against, so it sits above the top of it by construction
#: and a number here would be invented rather than measured.
FSF_LEVELS = (
    ("Fairy-Stockfish — 0.2s/move", 0.2),
    ("Fairy-Stockfish — 1s/move", 1.0),
    ("Fairy-Stockfish — 3s/move, max strength", 3.0),
)

#: Ceiling on what the page may ask for per move, so a hand-written request
#: cannot park the one opponent process on a ten-minute search.
MAX_MOVETIME = 10.0

_ID = re.compile(r"^[0-9a-f]{6,32}$")


def _opponents() -> list[dict[str, Any]]:
    """Everything the play tab may be set to, weakest first.

    Two engines, and the difference matters enough to say in the UI: the
    lab's own M1 alpha-beta at a calibrated node budget — which is what the
    Elo ladder is made of and therefore the only side of this list with
    meaningful ratings — and Fairy-Stockfish + `atomic.nnue` unthrottled,
    which is the gold standard and is stronger than anything the ladder can
    express.
    """
    out = [
        {
            "id": f"m1-{level['nodes']}",
            "engine": "m1",
            "label": level["label"],
            "detail": level["elo"],
            "nodes": level["nodes"],
            "movetime": None,
        }
        for level in _levels()
    ]
    out += [
        {
            "id": f"fsf-{seconds}",
            "engine": "fsf",
            "label": label,
            "detail": "unthrottled, all cores",
            "nodes": None,
            "movetime": seconds,
        }
        for label, seconds in FSF_LEVELS
    ]
    return out


class FsfOpponent:
    """One warm Fairy-Stockfish session that plays, rather than judges.

    Separate from `analysis.LIVE` on purpose. Those sessions are configured
    for analysis — one thread each, so several positions can be judged at
    once — and an opponent wants the opposite: every core on the one move
    somebody is waiting for. Sharing the pool would also mean a 3-second
    opponent search sitting in front of the explore board's next request.

    Opened on first use, so a session that never touches the play tab never
    launches the binary, and re-opened if the process dies under it: a
    crashed engine should cost the move it was thinking about, not the rest
    of the game.
    """

    def __init__(self, hash_mb: int = 512) -> None:
        self._lock = threading.Lock()
        self._threads = max(1, (os.cpu_count() or 2) - 1)
        self._hash_mb = hash_mb
        self._ctx: Any = None
        self._engine: Any = None

    @property
    def label(self) -> str:
        return f"Fairy-Stockfish 14 + atomic NNUE ({self._threads} threads)"

    def _session(self) -> Any:
        if self._engine is None:
            cfg = fsf.FsfConfig(
                threads=self._threads, hash_mb=self._hash_mb, use_nnue=True
            )
            self._ctx = fsf.open_engine(cfg)
            self._engine = self._ctx.__enter__()
        return self._engine

    def _drop(self) -> None:
        ctx, self._ctx, self._engine = self._ctx, None, None
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001, S110 - it is already broken
                pass

    def choose(self, board: chess.variant.AtomicBoard, movetime: float) -> chess.Move:
        limit = chess.engine.Limit(time=max(0.05, min(movetime, MAX_MOVETIME)))
        with self._lock:
            try:
                result = self._session().play(board, limit)
            except chess.engine.EngineError:
                # A protocol-level complaint about this position; the session
                # is still good and the caller gets a 400.
                raise
            except (chess.engine.EngineTerminatedError, BrokenPipeError, OSError):
                self._drop()
                result = self._session().play(board, limit)
        if result.move is None:
            raise ValueError("the engine returned no move")
        return result.move

    def close(self) -> None:
        with self._lock:
            self._drop()


class Api:
    """Everything the UI can ask for, as plain functions over a connection.

    Kept apart from the HTTP handler so each endpoint can be called directly
    from a test or a REPL without standing a server up.
    """

    def __init__(self) -> None:
        self.engine = EngineHolder()   # already serialises itself behind a lock
        self.fsf = FsfOpponent()
        self.jobs = jobs.Registry()

    # -- summary -----------------------------------------------------------

    def summary(self, db: sqlite3.Connection) -> dict[str, Any]:
        games = db.execute(
            "SELECT COUNT(*) AS n, SUM(analysed_nodes IS NOT NULL) AS analysed "
            "FROM games"
        ).fetchone()
        judged = {
            row["judgement"]: row["n"]
            for row in db.execute(
                "SELECT judgement, COUNT(*) AS n FROM plies p "
                "JOIN games g ON g.id = p.game_id WHERE p.mover = g.my_color "
                "GROUP BY judgement"
            )
        }
        model = winrate.load()
        return {
            "games": games["n"] or 0,
            "analysed": games["analysed"] or 0,
            "judged": judged,
            "puzzles": puzzles.stats(db),
            "book": book.stats(db),
            "repertoire": {
                color: repertoire.stats(db, color) for color in repertoire.COLORS
            },
            "phases": [
                {
                    "low": bucket.low,
                    "high": None if bucket.high > 10**5 else bucket.high,
                    "moves": bucket.moves,
                    "blunders": bucket.blunders,
                    "rate": bucket.rate,
                    "per_move": bucket.per_move,
                }
                for bucket in review.phase_table(db)
            ],
            "winrate": {
                "fitted": model.fitted,
                "describe": model.describe(),
                "even_cp": model.even_cp,
            },
            "themes": {
                "catalog": themes.CATALOG,
                "counts": puzzles.theme_counts(db),
            },
            "traps": traps.stats(db),
            "difficulty": difficulty.summary(db),
            "levels": _levels(),
            "opponents": _opponents(),
            "engine": self.engine.name,
            "fsf": self.fsf.label,
            "scrape": mine.stats(db),
            "jobs": jobs.KINDS,
        }

    # -- puzzles -----------------------------------------------------------

    def puzzle_list(self, db: sqlite3.Connection, query: dict[str, list[str]]) -> dict:
        def one(name: str) -> str | None:
            value = (query.get(name) or [""])[0].strip()
            return value or None

        def number(name: str) -> float | None:
            value = one(name)
            return float(value) if value is not None else None

        spec = puzzles.Filter(
            min_rating=number("min_rating"),
            max_rating=number("max_rating"),
            theme=one("theme"),
            phase=one("phase"),
            source=one("source"),
            motive=one("motive"),
            game_id=one("game"),
            mine=one("mine") == "1",
            unsolved=one("unsolved") == "1",
            unseen=one("unseen") == "1",
            failed=one("failed") == "1",
            limit=min(int(one("limit") or 60), 500),
            order=one("order") or "recent",
        )
        found = puzzles.search(db, spec)
        seen: dict[str, bool] = {}
        for row in db.execute(
            "SELECT puzzle_id, MAX(solved) AS solved FROM puzzle_attempts "
            "GROUP BY puzzle_id"
        ):
            seen[row["puzzle_id"]] = bool(row["solved"])
        return {
            "total": puzzles.count(db, spec),
            "puzzles": [
                {
                    **puzzle.public(),
                    "solved": seen.get(puzzle.id, False),
                    "seen": puzzle.id in seen,
                }
                for puzzle in found
            ],
        }

    def puzzle(self, db: sqlite3.Connection, puzzle_id: str) -> dict:
        puzzle = puzzles.get(db, puzzle_id)
        if puzzle is None:
            raise KeyError(f"no puzzle {puzzle_id}")
        payload = puzzle.public()
        payload["state"] = _state(chess.variant.AtomicBoard(puzzle.fen))
        if puzzle.setup_fen and puzzle.setup_uci:
            # Give the page the position *before* the error and the move that
            # made it, so it can play the blunder in front of you.
            board, move = _board_from(puzzle.setup_fen, puzzle.setup_uci)
            payload["setup_state"] = _state(board, move)
            payload["before_state"] = _state(
                chess.variant.AtomicBoard(puzzle.setup_fen)
            )
        verdict = difficulty.load(db, puzzle_id)
        payload["difficulty"] = verdict.public() if verdict else None
        attempts = puzzles.history(db, puzzle_id)
        payload["attempts"] = len(attempts)
        payload["solved_before"] = any(row["solved"] for row in attempts)
        payload["history"] = self._history(db, puzzle)
        return payload

    @staticmethod
    def _history(db: sqlite3.Connection, puzzle: puzzles.Puzzle) -> list[dict[str, Any]]:
        """The game up to the puzzle, so ← can walk back past the blunder.

        "How did this position arise" is a real question and the answer is
        already in the store for a puzzle mined from a game you imported.
        Positions are sent light — no legal-move table — because this is a
        replay, not a board you may move on; asking to move from one of these
        nodes costs an ordinary `/api/position` round trip and gets the full
        referee's answer like everything else.

        Scraped and tablebase puzzles have no history here: the dump's movetext
        is not stored (`mine.py` explains why a stranger's game does not go in
        `games`), and a tablebase position never had a game in front of it.
        """
        if not puzzle.game_id or puzzle.ply is None:
            return []
        row = db.execute(
            "SELECT moves FROM games WHERE id = ?", (puzzle.game_id,)
        ).fetchone()
        if row is None:
            return []
        # `ply` is the ply the *solver* moves at, so the history runs up to the
        # move before it — which is the opponent's error, already shown as the
        # setup move. One before that is where the walk-back should stop.
        board = chess.variant.AtomicBoard()
        out = [{"fen": board.fen(), "san": None, "uci": None, "ply": -1,
                "last": None, "blast": []}]
        for index, san in enumerate((row["moves"] or "").split()):
            if index >= puzzle.ply - 1:
                break
            move = board.push_san(san)
            state = _state(board, move)
            out.append({
                "fen": state["fen"], "san": san, "uci": move.uci(), "ply": index,
                "last": state.get("last"), "blast": state.get("blast", []),
            })
        return out

    def puzzle_move(self, db: sqlite3.Connection, payload: dict) -> dict:
        """Check one move against the solution and answer with the reply.

        The page sends how far into the line it believes it is; the server
        replays from the stored solution rather than trusting a position sent
        back to it, so a tampered FEN cannot walk the line forward.
        """
        puzzle = puzzles.get(db, payload["id"])
        if puzzle is None:
            raise KeyError(f"no puzzle {payload['id']}")
        index = int(payload.get("index", 0))
        if index % 2 or index >= len(puzzle.solution):
            raise ValueError(f"index {index} is not a solver move in this puzzle")
        uci = str(payload["move"])

        board = chess.variant.AtomicBoard(puzzle.fen)
        for played in puzzle.solution[:index]:
            board.push(chess.Move.from_uci(played))

        if uci != puzzle.solution[index]:
            return {
                "correct": False,
                "index": index,
                "played": self._wrong_move(db, board, uci, puzzle),
            }

        board.push(chess.Move.from_uci(uci))
        result: dict[str, Any] = {
            "correct": True,
            "index": index,
            "san": puzzle.solution_san[index],
            "state": _state(board, chess.Move.from_uci(uci)),
            "reply": None,
            "done": index + 1 >= len(puzzle.solution),
        }
        if not result["done"]:
            reply = puzzle.solution[index + 1]
            board.push(chess.Move.from_uci(reply))
            result["reply"] = {
                "uci": reply,
                "san": puzzle.solution_san[index + 1],
                "state": _state(board, chess.Move.from_uci(reply)),
            }
        if result["done"]:
            result["solution"] = puzzle.reveal()
        return result

    def _wrong_move(
        self,
        db: sqlite3.Connection,
        board: chess.variant.AtomicBoard,
        uci: str,
        puzzle: puzzles.Puzzle,
    ) -> dict[str, Any]:
        """Say what the wrong move actually costs, not just that it is wrong.

        Read from the cache first — the confirming pass stored MultiPV lines
        for this exact position — and only reach for the engine when the move
        played is not among them, which is the common case for a genuinely bad
        move.
        """
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci}")
        san = board.san(move)
        model = winrate.load()
        info: dict[str, Any] = {"uci": uci, "san": san, "loss": None, "win": None}

        for budget, multipv in (
            (puzzle.nodes or 0, generate.CONFIRM_MULTIPV),
            (FEEDBACK_NODES, FEEDBACK_MULTIPV),
        ):
            if not budget:
                continue
            for line in analysis.cached_lines(db, board.fen(), budget, multipv):
                if line.uci == uci:
                    info["win"] = line.win_pct(model)
                    break
            if info["win"] is not None:
                break

        if info["win"] is None:
            # Score the position *after* the move: one analysis, and it is the
            # honest number — what the opponent gets, flipped.
            after = board.copy(stack=False)
            after.push(move)
            if after.is_game_over(claim_draw=True):
                result = after.result(claim_draw=True)
                info["win"] = (
                    50.0 if result == "1/2-1/2"
                    else 100.0 if (result == "1-0") == (board.turn == chess.WHITE)
                    else 0.0
                )
            else:
                analysis.ensure_analysed(
                    db, [after.fen()], nodes=FEEDBACK_NODES,
                    multipv=FEEDBACK_MULTIPV, workers=1,
                )
                lines = analysis.cached_lines(
                    db, after.fen(), FEEDBACK_NODES, FEEDBACK_MULTIPV
                )
                if lines:
                    info["win"] = 100.0 - lines[0].win_pct(model)
        if info["win"] is not None and puzzle.win_before is not None:
            info["loss"] = max(0.0, puzzle.win_before - info["win"])
        return info

    def puzzle_hint(self, db: sqlite3.Connection, payload: dict) -> dict:
        """Two levels: which piece, then which move. Never more than asked."""
        puzzle = puzzles.get(db, payload["id"])
        if puzzle is None:
            raise KeyError(f"no puzzle {payload['id']}")
        index = int(payload.get("index", 0))
        level = int(payload.get("level", 1))
        if index >= len(puzzle.solution):
            raise ValueError("the line is already finished")
        uci = puzzle.solution[index]
        if level <= 1:
            return {"level": 1, "from": uci[:2]}
        return {"level": 2, "from": uci[:2], "to": uci[2:4], "uci": uci}

    def puzzle_solution(self, db: sqlite3.Connection, puzzle_id: str) -> dict:
        puzzle = puzzles.get(db, puzzle_id)
        if puzzle is None:
            raise KeyError(f"no puzzle {puzzle_id}")
        states = []
        board = chess.variant.AtomicBoard(puzzle.fen)
        for uci in puzzle.solution:
            move = chess.Move.from_uci(uci)
            board.push(move)
            states.append(_state(board, move))
        return {**puzzle.reveal(), "states": states}

    def puzzle_attempt(self, db: sqlite3.Connection, payload: dict) -> dict:
        puzzles.record_attempt(
            db,
            str(payload["id"]),
            solved=bool(payload.get("solved")),
            hints=int(payload.get("hints", 0)),
            wrong=int(payload.get("wrong", 0)),
            ms=payload.get("ms"),
        )
        return {"ok": True}

    # -- games and review --------------------------------------------------

    def games(self, db: sqlite3.Connection, query: dict[str, list[str]]) -> dict:
        limit = min(int((query.get("limit") or ["100"])[0]), 1000)
        rows = db.execute(
            "SELECT g.*, "
            "(SELECT COUNT(*) FROM plies p WHERE p.game_id = g.id "
            " AND p.mover = g.my_color AND p.judgement = 'blunder') AS blunders, "
            "(SELECT COUNT(*) FROM puzzles z WHERE z.game_id = g.id) AS puzzles "
            "FROM games g ORDER BY g.created_at DESC LIMIT ?",
            (limit,),
        )
        return {
            "games": [
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "white": row["white"],
                    "black": row["black"],
                    "white_rating": row["white_rating"],
                    "black_rating": row["black_rating"],
                    "my_color": row["my_color"],
                    "winner": row["winner"],
                    "speed": row["speed"],
                    "status": row["status"],
                    "source": row["source"],
                    "rated": bool(row["rated"]),
                    "ply": row["ply"],
                    "analysed": row["analysed_nodes"] is not None,
                    "blunders": row["blunders"],
                    "puzzles": row["puzzles"],
                }
                for row in rows
            ]
        }

    def game(self, db: sqlite3.Connection, game_id: str) -> dict:
        row = db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
        if row is None:
            raise KeyError(f"no game {game_id}")
        plies = [
            {
                "ply": ply["ply"],
                "fen": ply["fen"],
                "mover": ply["mover"],
                "uci": ply["uci"],
                "san": ply["san"],
                "win_before": ply["win_before"],
                "win_after": ply["win_after"],
                "loss": ply["loss"],
                "judgement": ply["judgement"],
                "best_uci": ply["best_uci"],
                "best_san": ply["best_san"],
                "unique_gap": ply["unique_gap"],
            }
            for ply in db.execute(
                "SELECT * FROM plies WHERE game_id = ? ORDER BY ply", (game_id,)
            )
        ]
        found = db.execute(
            "SELECT id, ply, phase, motive, themes FROM puzzles WHERE game_id = ?",
            (game_id,),
        )
        # The final position has no ply row of its own; the page needs it to
        # show the end of the game.
        board = chess.variant.AtomicBoard()
        for san in (row["moves"] or "").split():
            board.push_san(san)
        return {
            "game": {
                "id": row["id"],
                "white": row["white"],
                "black": row["black"],
                "white_rating": row["white_rating"],
                "black_rating": row["black_rating"],
                "my_color": row["my_color"],
                "winner": row["winner"],
                "status": row["status"],
                "speed": row["speed"],
                "created_at": row["created_at"],
                "analysed": row["analysed_nodes"] is not None,
                "nodes": row["analysed_nodes"],
                "moves": (row["moves"] or "").split(),
            },
            "plies": plies,
            "final_fen": board.fen(),
            "puzzles": [
                {
                    "id": p["id"],
                    "ply": p["ply"],
                    "phase": p["phase"],
                    "motive": p["motive"],
                    "themes": json.loads(p["themes"]),
                }
                for p in found
            ],
        }

    def patterns(self, db: sqlite3.Connection, query: dict[str, list[str]]) -> dict:
        max_ply = int((query.get("max_ply") or [str(review.OPENING_PLY)])[0])
        min_count = int((query.get("min_count") or ["2"])[0])

        def with_book(fen: str, san: str, best: str | None) -> dict | None:
            context = review.book_context(db, fen, san, best)
            if not context.moves:
                return None
            return {
                "share": context.share,
                "rank": context.rank,
                "played": None if context.played is None else {
                    "san": context.played.san,
                    "games": context.played.games,
                    "score": context.played.score,
                },
                "best": None if context.best is None else {
                    "san": context.best.san,
                    "games": context.best.games,
                    "score": context.best.score,
                },
                "field": [
                    {"san": e.san, "games": e.games, "score": e.score}
                    for e in context.moves[:4]
                ],
            }

        positions = [
            {
                "fen": cluster.fen,
                "ply": cluster.ply,
                "count": cluster.count,
                "avg_loss": cluster.avg_loss,
                "best_san": cluster.best_san,
                "played": cluster.played,
                "games": cluster.games,
                "book": {
                    san: with_book(cluster.fen, san, cluster.best_san)
                    for san in cluster.played
                },
            }
            for cluster in review.recurring_positions(
                db, max_ply=max_ply, min_count=min_count
            )
        ]
        moves = [
            {
                "san": cluster.san,
                "ply": cluster.ply,
                "count": cluster.count,
                "positions": cluster.positions,
                "avg_loss": cluster.avg_loss,
                "alternatives": cluster.alternatives,
                "fens": sorted({row["fen"] for row in cluster.rows}),
            }
            for cluster in review.repeated_moves(
                db, max_ply=max_ply, min_count=min_count
            )
            if cluster.positions > 1
        ]
        return {"positions": positions, "moves": moves}

    # -- book and repertoire ----------------------------------------------

    def book_line(self, db: sqlite3.Connection, query: dict[str, list[str]]) -> dict:
        moves = [m for m in (query.get("moves") or [""])[0].split(",") if m]
        board = chess.variant.AtomicBoard()
        for san in moves:
            board.push_san(san)
        key = book.book_fen(board)
        entries = book.lookup(db, key)
        total = sum(entry.games for entry in entries)
        seeds, yours = openings.curated(), openings.user_names(db)
        return {
            "moves": moves,
            "name": openings.name_line(db, moves, names=seeds, yours=yours).as_dict(),
            # Every move on offer is named too, so the table reads as an index
            # of what each move *is* rather than a list of squares.
            "names": {
                entry.san: openings.name_line(
                    db, [*moves, entry.san], names=seeds, yours=yours
                ).short
                for entry in entries
            },
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "state": _state(board),
            "total": total,
            "entries": [
                {
                    "san": entry.san,
                    "uci": entry.uci,
                    "games": entry.games,
                    "share": 100.0 * entry.games / total if total else 0.0,
                    "score": entry.score,
                    "elo": entry.elo,
                    "white_wins": entry.white_wins,
                    "draws": entry.draws,
                    "black_wins": entry.black_wins,
                }
                for entry in entries
            ],
            "stats": book.stats(db),
        }

    def repertoire_view(self, db: sqlite3.Connection, color: str) -> dict:
        lines = repertoire.lines_of(db, color)
        gaps, covered, total = repertoire.coverage(db, color)
        drifted = repertoire.deviations(db, color)
        # Named once here, from the *whole* line including the intended move:
        # what you are memorising is "the Kapsarov after 2...Nh6", and a name
        # for the position without the move you play in it is half a name.
        seeds, yours = openings.curated(), openings.user_names(db)
        return {
            "color": color,
            "stats": repertoire.stats(db, color),
            "lines": [
                {
                    "path": path,
                    "name": openings.name_line(
                        db, [*path, node.san], names=seeds, yours=yours
                    ).as_dict(),
                    "parent": openings.name_line(
                        db, path, names=seeds, yours=yours
                    ).as_dict(),
                    "fen": repertoire.full_fen(node.fen),
                    "san": node.san,
                    "uci": node.uci,
                    "ply": node.ply,
                    "games": node.games,
                    "score": node.score,
                    "validated": node.validated,
                    "flagged": node.flagged,
                    "agrees": node.agrees,
                    "best_san": node.best_san,
                    "win_move": node.win_move,
                    "win_best": node.win_best,
                    "val_loss": node.val_loss,
                }
                for path, node in lines
            ],
            "coverage": {"covered": covered, "total": total},
            "gaps": [
                {
                    "fen": gap.fen,
                    "ply": gap.ply,
                    "count": gap.count,
                    "cost": gap.cost,
                    "played": gap.played,
                    "games": gap.games,
                }
                for gap in gaps[:20]
            ],
            "deviations": [
                {
                    "fen": repertoire.full_fen(dev.node.fen),
                    "ply": dev.node.ply,
                    "intended": dev.node.san,
                    "played": dev.played,
                    "count": dev.count,
                    "cost": dev.cost,
                }
                for dev in drifted[:20]
            ],
        }

    # -- openings and traps ------------------------------------------------

    def opening_index(self, db: sqlite3.Connection, query: dict[str, list[str]]) -> dict:
        """The named opening tree, as deep as the page asked for.

        Depth and the two floors are the page's to choose because they are the
        only controls that matter here: an index nobody can fit on a screen is
        as useless as no index, and how much of the tree fits is a question
        about the reader rather than about the book.
        """
        def number(name: str, fallback: float) -> float:
            raw = (query.get(name) or [""])[0]
            try:
                return float(raw) if raw else fallback
            except ValueError:
                return fallback

        depth = int(min(max(number("depth", 6), 1), openings.DEFAULT_DEPTH))
        return {
            "tree": [
                node.as_dict()
                for node in openings.tree(
                    db,
                    depth=depth,
                    min_games=int(number("min_games", 5_000)),
                    min_share=number("min_share", 8.0),
                )
            ],
            "yours": [
                {"line": line, "name": name, "note": note}
                for line, (name, note) in sorted(openings.user_names(db).items())
            ],
            "stats": book.stats(db),
        }

    def opening_name(self, db: sqlite3.Connection, payload: dict) -> dict:
        """Name a line, clear a name, or just ask what a line is called."""
        line = str(payload.get("line", "")).split()
        if "name" in payload:
            openings.rename(
                db, " ".join(line), payload.get("name"), str(payload.get("note", ""))
            )
        return openings.name_line(db, line).as_dict()

    def trap_list(self, db: sqlite3.Connection, query: dict[str, list[str]]) -> dict:
        def one(name: str) -> str | None:
            value = (query.get(name) or [""])[0].strip()
            return value or None

        turn = one("turn")
        max_ply = one("max_ply")
        found = traps.listing(
            db,
            turn=turn if turn in ("white", "black") else None,
            max_ply=int(max_ply) if max_ply else None,
            min_bait=float(one("min_bait") or 0.0),
            order=one("order") or "bait",
            limit=min(int(one("limit") or 60), 300),
        )
        seeds, yours = openings.curated(), openings.user_names(db)
        return {
            "traps": [
                {
                    **trap.as_dict(),
                    "name": openings.name_line(
                        db, trap.line, names=seeds, yours=yours
                    ).as_dict(),
                }
                for trap in found
            ],
            "stats": traps.stats(db),
        }

    def trap(self, db: sqlite3.Connection, trap_id: str) -> dict:
        """One trap, with the board states the page needs to walk into it.

        The line is replayed here rather than in the page for the same reason
        every other position is: the server is the referee, and a page that
        replayed SAN itself would be a second atomic rules implementation.
        """
        found = traps.get(db, trap_id)
        if found is None:
            raise KeyError(f"no trap {trap_id}")
        board = chess.variant.AtomicBoard()
        history = []
        for san in found.line:
            move = board.parse_san(san)
            board.push(move)
            history.append({"san": san, "uci": move.uci(), "state": _state(board, move)})
        payload = {
            **found.as_dict(),
            "name": openings.name_line(db, found.line).as_dict(),
            "history": history,
            "state": _state(board),
            "book": [
                {"san": e.san, "uci": e.uci, "games": e.games, "score": e.score}
                for e in book.lookup(db, book.book_fen(board))
            ],
        }
        # The punishment, replayed from the trapped position so the page can
        # step through it rather than reading a string of SAN.
        line = []
        probe = board.copy(stack=False)
        for san in found.refutation:
            try:
                move = probe.parse_san(san)
            except (ValueError, AssertionError):
                break
            probe.push(move)
            line.append({"san": san, "uci": move.uci(), "state": _state(probe, move)})
        payload["punishment"] = line
        return payload

    # -- board -------------------------------------------------------------

    def position(self, payload: dict) -> dict:
        board = chess.variant.AtomicBoard(payload["fen"])
        return _state(board)

    def move(self, payload: dict) -> dict:
        # SAN has to be read off the position *before* the move, so it is taken
        # here rather than left to the page — which has no way to compute it.
        before = chess.variant.AtomicBoard(payload["fen"])
        san = before.san(chess.Move.from_uci(payload["move"]))
        board, move = _board_from(payload["fen"], payload["move"])
        return {**_state(board, move), "san": san}

    def engine_move(self, payload: dict) -> dict:
        """One reply from whichever opponent the page is set to.

        `engine` defaults to the lab's M1, which is what this endpoint used to
        be the only way to reach, so an older page keeps working unchanged.
        """
        board = chess.variant.AtomicBoard(payload["fen"])
        if board.is_game_over(claim_draw=True):
            return _state(board)
        if str(payload.get("engine", "m1")) == "fsf":
            move = self.fsf.choose(board, float(payload.get("movetime", 1.0)))
        else:
            move = self.engine.choose(board, int(payload.get("nodes", 64_000)))
        san = board.san(move)
        board.push(move)
        return {**_state(board, move), "san": san}

    def evaluate(self, db: sqlite3.Connection, payload: dict) -> dict:
        """An engine opinion on an arbitrary position.

        This is what the explore board runs on once a puzzle is over: several
        lines, each with the variation spelled out in SAN so the panel reads
        like an analysis board rather than a list of first moves. The result
        goes into the same cache the deep passes read, so exploring a position
        is not wasted work — a later generation run may find it already paid
        for.

        The page calls this repeatedly for one position at a rising `nodes`,
        so it is answered on a warm session (`analysis.LIVE`) rather than a
        process opened and closed around the search: the launch is a third of
        a cheap search's cost, and throwing the transposition table away
        between rungs would make each rung pay for the last one again.
        """
        fen = str(payload["fen"])
        nodes = min(int(payload.get("nodes", FEEDBACK_NODES)), 4_000_000)
        multipv = min(max(int(payload.get("multipv", FEEDBACK_MULTIPV)), 1), 5)
        board = chess.variant.AtomicBoard(fen)
        if board.is_game_over(claim_draw=True):
            return {
                "over": True, "fen": fen, "lines": [],
                "turn": "white" if board.turn == chess.WHITE else "black",
                "result": board.result(claim_draw=True),
            }
        model = winrate.load()
        out = []
        for line in analysis.evaluate_now(db, fen, nodes=nodes, multipv=multipv):
            move = chess.Move.from_uci(line.uci)
            if move not in board.legal_moves:
                continue
            out.append(
                {
                    "uci": line.uci,
                    "san": board.san(move),
                    "cp": line.cp,
                    "mate": line.mate,
                    "depth": line.depth,
                    "win": line.win_pct(model),
                    **self._variation(board, line.pv),
                }
            )
        return {
            "over": False,
            "fen": fen,
            "nodes": nodes,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "lines": out,
        }

    def explain(self, db: sqlite3.Connection, payload: dict) -> dict:
        """Why the engine plays its move here, or why `move` is worse.

        Every search goes through `analysis.evaluate_now`, so it is answered on
        a warm session and lands in the same cache as everything else.
        """
        fen = str(payload["fen"])
        nodes = min(int(payload.get("nodes", EXPLAIN_NODES)), 2_000_000)

        def search(position: str, multipv: int) -> list[analysis.Line]:
            return analysis.evaluate_now(db, position, nodes=nodes, multipv=multipv)

        result = explain.explain(
            fen, search,
            move=payload.get("move") or None,
            versus=payload.get("versus") or None,
            model=winrate.load(),
        )
        result["nodes"] = nodes
        return result

    @staticmethod
    def _variation(
        board: chess.variant.AtomicBoard, pv: Sequence[str]
    ) -> dict[str, list[str]]:
        """A principal variation as SAN and UCI, stopping where it stops parsing.

        Both, because the page shows one and plays the other: clicking the
        third move of a variation walks the explore board into it, and doing
        that from SAN would mean the page working out which move a name refers
        to — which is exactly the judgement the referee rule keeps server-side.

        Cache entries written before `Line.pv` existed have no variation, and a
        truncated PV from a node-limited search can end mid-air. Neither is
        worth an error: show what parses and stop.
        """
        replay = board.copy(stack=False)
        san: list[str] = []
        uci: list[str] = []
        for step in pv:
            try:
                move = chess.Move.from_uci(step)
            except ValueError:
                break
            if move not in replay.legal_moves:
                break
            san.append(replay.san(move))
            uci.append(step)
            replay.push(move)
        return {"pv": san, "pv_uci": uci}

    # -- improvement -------------------------------------------------------

    def improvement(self, db: sqlite3.Connection) -> dict:
        report = puzzles.improvement(db)
        report["catalog"] = themes.CATALOG
        return report

    # -- jobs --------------------------------------------------------------

    def job_start(self, payload: dict) -> dict:
        kind = str(payload.get("kind", ""))
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise TypeError("params must be an object")
        return self.jobs.start(kind, params).public()

    def job_list(self, db: sqlite3.Connection) -> dict:
        listing = self.jobs.listing()
        listing["corpus"] = puzzles.stats(db)
        listing["scrape"] = mine.stats(db)
        listing["analysed_games"] = db.execute(
            "SELECT COUNT(*) AS n FROM games WHERE analysed_nodes IS NOT NULL"
        ).fetchone()["n"]
        return listing


# -- HTTP ------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    api: Api  # injected in `serve`

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        pass

    # -- plumbing ----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _static(self, name: str) -> None:
        """Serve one file out of `ui/`, and nothing outside it."""
        path = (UI_DIR / name).resolve()
        if not path.is_file() or UI_DIR.resolve() not in path.parents:
            self._json({"error": "not found"}, 404)
            return
        kind, _ = mimetypes.guess_type(path.name)
        self._send(200, path.read_bytes(), f"{kind or 'text/plain'}; charset=utf-8")

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route, query = parsed.path, parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                self._static("trainer.html")
            elif not route.startswith("/api/"):
                self._static(route.lstrip("/"))
            else:
                self._json(self._get_api(route, query))
        except KeyError as exc:
            self._json({"error": str(exc)}, 404)
        except (ValueError, sqlite3.Error) as exc:
            self._json({"error": str(exc)}, 400)

    def _get_api(self, route: str, query: dict[str, list[str]]) -> Any:
        api = self.api
        with store.open_db() as db:
            if route == "/api/summary":
                return api.summary(db)
            if route == "/api/puzzles":
                return api.puzzle_list(db, query)
            if route.startswith("/api/puzzle/"):
                return api.puzzle(db, self._id(route.rsplit("/", 1)[-1]))
            if route == "/api/games":
                return api.games(db, query)
            if route.startswith("/api/game/"):
                return api.game(db, route.rsplit("/", 1)[-1])
            if route == "/api/patterns":
                return api.patterns(db, query)
            if route == "/api/book":
                return api.book_line(db, query)
            if route == "/api/openings":
                return api.opening_index(db, query)
            if route == "/api/traps":
                return api.trap_list(db, query)
            if route.startswith("/api/trap/"):
                return api.trap(db, self._id(route.rsplit("/", 1)[-1]))
            if route == "/api/repertoire":
                color = (query.get("color") or ["white"])[0]
                if color not in repertoire.COLORS:
                    raise ValueError(f"no such colour: {color}")
                return api.repertoire_view(db, color)
            if route == "/api/improvement":
                return api.improvement(db)
            if route == "/api/jobs":
                return api.job_list(db)
            if route.startswith("/api/job/"):
                job = api.jobs.get(route.rsplit("/", 1)[-1])
                if job is None:
                    raise KeyError("no such job")
                return job.public()
            if route == "/api/themes":
                return {
                    "catalog": themes.CATALOG,
                    "firing": [
                        {
                            "theme": f.theme,
                            "count": f.count,
                            "rate": f.rate,
                            "verdict": f.verdict,
                        }
                        for f in themes.firing_rates(db)
                    ],
                }
        raise KeyError("not found")

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            self._json(self._post_api(route, payload))
        except KeyError as exc:
            self._json({"error": str(exc)}, 404)
        except (
            ValueError, TypeError, AssertionError, sqlite3.Error,
            # A missing binary or a dead engine is an answer, not a crash: the
            # play tab can say "Fairy-Stockfish is not installed" and stay up.
            fsf.FsfUnavailable, chess.engine.EngineError,
        ) as exc:
            self._json({"error": str(exc)}, 400)

    def _post_api(self, route: str, payload: dict) -> Any:
        api = self.api
        # Board endpoints need no database at all.
        if route == "/api/new":
            api.engine.new_game()
            return _state(chess.variant.AtomicBoard())
        if route == "/api/position":
            return api.position(payload)
        if route == "/api/move":
            return api.move(payload)
        if route == "/api/engine":
            return api.engine_move(payload)
        # Jobs live in the process, not the database.
        if route == "/api/jobs/start":
            return api.job_start(payload)
        if route == "/api/jobs/cancel":
            return {"cancelled": api.jobs.cancel(str(payload.get("id", "")))}

        with store.open_db() as db:
            if route == "/api/puzzle/move":
                return api.puzzle_move(db, payload)
            if route == "/api/puzzle/hint":
                return api.puzzle_hint(db, payload)
            if route == "/api/puzzle/solution":
                return api.puzzle_solution(db, self._id(str(payload["id"])))
            if route == "/api/puzzle/attempt":
                return api.puzzle_attempt(db, payload)
            if route == "/api/evaluate":
                return api.evaluate(db, payload)
            if route == "/api/explain":
                return api.explain(db, payload)
            if route == "/api/opening/name":
                return api.opening_name(db, payload)
        raise KeyError("not found")

    @staticmethod
    def _id(value: str) -> str:
        if not _ID.match(value):
            raise ValueError(f"not a puzzle id: {value!r}")
        return value


def serve(host: str = "127.0.0.1", port: int = 8771, open_browser: bool = True) -> int:
    Handler.api = Api()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"

    with store.open_db() as db:
        info = puzzles.stats(db)
        games = db.execute("SELECT COUNT(*) AS n FROM games").fetchone()["n"]
    print(f"atomic trainer on {url}")
    print(f"  {games} games, {info['total']} puzzles "
          f"({info['from_games']} mined, {info['from_tablebase']} exact)")
    if not winrate.load().fitted:
        print("  WARNING: the cp->win% curve is unfitted; every judgement is a guess")
    print("\nCtrl-C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
        analysis.LIVE.close()   # or Ctrl-C never returns; see LiveEngines.close
        Handler.api.fsf.close()   # same reason: a live engine holds the exit
    return 0
