"""Lichess atomic tablebase client — exact endgame ground truth, for free.

The Chalmers thesis's clearest documented weakness was mid/endgame play: atomic
games have a median length of 30 ply, so self-play data is overwhelmingly
openings and the endgame never gets trained. Lichess serves atomic tablebases
with DTZ and per-move win/draw/loss categories, which is exactly the missing
supervision — and no published atomic engine appears to use it.

Two uses:
  1. **Training targets.** Label endgame positions with exact outcomes instead
     of noisy self-play results.
  2. **Play-time probing.** Resolve leaves exactly once material is low enough.

Everything is cached in SQLite, because the polite request rate is the binding
constraint and every position we probe twice is a request wasted.
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import chess.variant

from . import config

#: Be a good citizen — this is a free public service run by a nonprofit.
MIN_INTERVAL = 0.25  # seconds between live requests
USER_AGENT = "atomic-lab/0.1 (research; contact: cleehikai@gmail.com)"


@dataclass(frozen=True)
class Probe:
    """A tablebase answer for one position."""

    category: str            # "win" / "loss" / "draw" / "cursed-win" / "blessed-loss" / "unknown"
    dtz: int | None
    moves: list[dict[str, Any]]

    @property
    def known(self) -> bool:
        return self.category not in ("unknown", "")

    @property
    def score(self) -> float | None:
        """Outcome from the side to move: 1.0 win, 0.5 draw, 0.0 loss."""
        return {
            "win": 1.0,
            "cursed-win": 0.5,   # win in theory, drawn by the 50-move rule
            "draw": 0.5,
            "blessed-loss": 0.5,
            "loss": 0.0,
        }.get(self.category)

    def best_move(self) -> str | None:
        """UCI of the move that keeps the best category, shortest DTZ first."""
        if not self.moves:
            return None
        rank = {"loss": 0, "blessed-loss": 1, "draw": 2, "cursed-win": 3, "win": 4}
        # A move's category is reported from the *opponent's* view, so the move
        # we want is the one that leaves them in the worst category.
        best = min(
            self.moves,
            key=lambda m: (rank.get(m.get("category", "unknown"), 5),
                           abs(m.get("dtz") or 10**6)),
        )
        return best.get("uci")


class Tablebase:
    """Cached client for the Lichess atomic tablebase."""

    def __init__(self, cache_path=None, offline: bool = False) -> None:
        self.offline = offline
        self._last_request = 0.0
        self._db = sqlite3.connect(cache_path or config.TABLEBASE_CACHE)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS probes (fen TEXT PRIMARY KEY, json TEXT)"
        )
        self._db.commit()

    # -- public ------------------------------------------------------------

    def probe(self, board: chess.variant.AtomicBoard) -> Probe | None:
        """Look up `board`. Returns None if out of tablebase range or offline."""
        if self.piece_count(board) > 6:
            return None  # atomic tables only go to 6 pieces
        fen = board.fen()
        cached = self._cached(fen)
        if cached is not None:
            return self._to_probe(cached)
        if self.offline:
            return None
        raw = self._fetch(fen)
        if raw is None:
            return None
        self._store(fen, raw)
        return self._to_probe(raw)

    @staticmethod
    def piece_count(board: chess.Board) -> int:
        return bin(board.occupied).count("1")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Tablebase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _cached(self, fen: str) -> dict | None:
        row = self._db.execute(
            "SELECT json FROM probes WHERE fen = ?", (fen,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def _store(self, fen: str, raw: dict) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO probes (fen, json) VALUES (?, ?)",
            (fen, json.dumps(raw)),
        )
        self._db.commit()

    def _fetch(self, fen: str) -> dict | None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        url = f"{config.TABLEBASE_URL}?{urllib.parse.urlencode({'fen': fen})}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None      # position not in the tables
            raise
        finally:
            self._last_request = time.monotonic()
        return raw

    @staticmethod
    def _to_probe(raw: dict) -> Probe:
        return Probe(
            category=raw.get("category", "unknown"),
            dtz=raw.get("dtz"),
            moves=raw.get("moves", []) or [],
        )


if __name__ == "__main__":
    # Smoke test: a lone queen mates on the edge in atomic.
    board = chess.variant.AtomicBoard("4k3/8/8/8/8/8/8/QK6 w - - 0 1")
    with Tablebase() as tb:
        probe = tb.probe(board)
    print(board.fen())
    print(f"  category={probe.category} dtz={probe.dtz} "
          f"score={probe.score} best={probe.best_move()}")
