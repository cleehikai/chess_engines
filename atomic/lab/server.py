"""A browser UI for playing the atomic engine.

    python -m lab.cli play

Stdlib only — `http.server` plus python-chess, which the lab already depends
on. A local research tool should not need a web framework, and adding one would
make `requirements.txt` a worse description of what the project actually needs.

The server is the referee: it owns the board, validates every move against
`chess.variant.AtomicBoard`, and computes explosion squares. The page never
decides what is legal. That keeps the one thing that is genuinely hard about
atomic — the rules — in the implementation that has been perft-checked against
Fairy-Stockfish, rather than reimplemented in JavaScript where it would drift.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.variant

from . import rating
from .methods import EVAL_TERMS, AlphaBetaMethod, Budget
from .rules import explosion_squares, game_result

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

#: Node budgets offered in the UI. Labels are deliberately vague where the
#: rating is not measured — see `_levels()`.
LEVELS = (
    ("Casual", 2_000),
    ("Club", 16_000),
    ("Strong", 64_000),
    ("Full strength", 1_024_000),
)


class EngineHolder:
    """One engine instance, guarded by a lock.

    `atomic_core.Engine` owns a transposition table and is not safe to call
    from two requests at once. A lock is simpler than a pool and the UI is
    single-user by construction.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._method = AlphaBetaMethod()

    @property
    def name(self) -> str:
        return self._method.name

    @property
    def terms(self) -> tuple[str, ...]:
        return self._method.terms

    def new_game(self) -> None:
        with self._lock:
            self._method.reset()

    def choose(self, board: chess.variant.AtomicBoard, nodes: int) -> chess.Move:
        with self._lock:
            return self._method.choose(board, Budget(nodes=nodes))


@lru_cache(maxsize=1)
def _levels() -> list[dict]:
    """The difficulty levels, annotated with a measured rating where one exists.

    A level whose rating has not been measured says so rather than guessing.
    Inventing a number here would quietly undermine the one thing this project
    is careful about.
    """
    measured = rating.latest_ratings()
    out = []
    for label, nodes in LEVELS:
        row = measured.get(("M1-alphabeta", f"{nodes}n"))
        out.append(
            {
                "label": label,
                "nodes": nodes,
                "elo": rating.describe(row) if row else "not yet measured",
                "measured": row is not None,
            }
        )
    return out


def _state(board: chess.variant.AtomicBoard, last: chess.Move | None = None) -> dict:
    """Everything the page needs to render a position."""
    blast = []
    if last is not None:
        # `board` already has `last` played, so the explosion has to be
        # recomputed from the position before it.
        before = board.copy()
        before.pop()
        blast = [chess.square_name(sq) for sq in explosion_squares(before, last)]

    legal: dict[str, list[str]] = {}
    for move in board.legal_moves:
        legal.setdefault(chess.square_name(move.from_square), []).append(move.uci())

    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "legal": legal,
        "check": board.is_check(),
        "over": board.is_game_over(claim_draw=True),
        "result": game_result(board),
        "last": last.uci() if last else None,
        "blast": blast,
        "ply": len(board.move_stack),
    }


def _board_from(fen: str, last_uci: str | None) -> tuple[chess.variant.AtomicBoard, chess.Move | None]:
    """Rebuild a board, replaying the last move so explosions can be shown.

    The page sends back the FEN it is displaying rather than a game id, so the
    server holds no per-client state and a reload cannot desynchronise it.
    """
    board = chess.variant.AtomicBoard(fen)
    if not last_uci:
        return board, None
    move = chess.Move.from_uci(last_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {last_uci} in {fen}")
    board.push(move)
    return board, move


class Handler(BaseHTTPRequestHandler):
    engine: EngineHolder  # injected in `serve`

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        pass  # the default logs every request to stderr; too noisy for a UI

    # -- plumbing ----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._json(
                {
                    "engine": self.engine.name,
                    "terms": list(self.engine.terms),
                    "all_terms": list(EVAL_TERMS),
                    "levels": _levels(),
                    "calibrated": rating.is_calibrated(),
                }
            )
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/new":
                self.engine.new_game()
                self._json(_state(chess.variant.AtomicBoard()))
            elif self.path == "/api/move":
                board, move = _board_from(payload["fen"], payload["move"])
                self._json(_state(board, move))
            elif self.path == "/api/engine":
                board = chess.variant.AtomicBoard(payload["fen"])
                if board.is_game_over(claim_draw=True):
                    self._json(_state(board))
                    return
                move = self.engine.choose(board, int(payload.get("nodes", 64_000)))
                board.push(move)
                self._json(_state(board, move))
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, KeyError) as exc:
            self._json({"error": str(exc)}, 400)


def serve(host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True) -> int:
    Handler.engine = EngineHolder()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"

    levels = _levels()
    print(f"atomic lab UI on {url}")
    print(f"  engine: {Handler.engine.name}  [{', '.join(Handler.engine.terms)}]")
    for level in levels:
        print(f"    {level['label']:<16} {level['nodes']:>9,} nodes   {level['elo']}")
    print("\nCtrl-C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0
