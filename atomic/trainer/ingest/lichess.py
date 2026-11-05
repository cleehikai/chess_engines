"""Import a Lichess account's atomic games.

No token needed: a user's finished public games are exported anonymously, and
`perfType=atomic` filters to the variant server-side. An OAuth token only buys
a higher rate limit (60 games/s for your own history against 20 anonymous) and
access to ongoing or private games, so it stays optional. The token is read
from `ATOMIC_LICHESS_TOKEN` if set, or from `data/lichess_token` (one line,
git-ignored) if that file exists. Both are ignored when absent.

Imports are incremental. The store remembers the newest `createdAt` seen for an
account and passes it as `since`, so the second run of the day downloads the
day's games rather than the whole history.

Chess.com has no equivalent: its published-data API's `rules` enum is
`chess, chess960, bughouse, kingofthehill, threecheck, crazyhouse` — atomic is
not in it, and its variants platform has no public API at all. See
`docs/trainer-plan.md` §1.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from lab import config

from .. import store

API = "https://lichess.org/api/games/user/{user}"
USER_AGENT = "atomic-trainer/0.1 (research; contact: cleehikai@gmail.com)"

#: Optional OAuth token. Put it in this file rather than in a shell history or a
#: chat message; `atomic/.gitignore` keeps the file out of the repository.
TOKEN_FILE = config.DATA / "lichess_token"

#: Statuses where nobody actually moved. Kept out of the store: they are not
#: games, and an "analysed" 0-ply game is just noise in every later query.
EMPTY_STATUSES = {"noStart"}


class LichessError(RuntimeError):
    pass


def token() -> str | None:
    """The optional Lichess OAuth token, from the environment or the file."""
    value = os.environ.get("ATOMIC_LICHESS_TOKEN", "").strip()
    if value:
        return value
    try:
        value = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/x-ndjson"}
    value = token()
    if value:
        headers["Authorization"] = f"Bearer {value}"
    return urllib.request.Request(url, headers=headers)


def stream_games(
    user: str, since: int | None = None, max_games: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield atomic games for `user`, newest first, as parsed NDJSON."""
    params: dict[str, Any] = {"perfType": "atomic", "moves": "true", "sort": "dateDesc"}
    if since is not None:
        params["since"] = since
    if max_games is not None:
        params["max"] = max_games
    url = f"{API.format(user=urllib.parse.quote(user))}?{urllib.parse.urlencode(params)}"

    for attempt in range(5):
        try:
            with urllib.request.urlopen(_request(url), timeout=120) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise LichessError(f"no such Lichess user: {user}") from None
            if exc.code == 429:
                # Lichess asks for a full minute after a 429. Honour it.
                wait = 60
                print(f"  rate limited; waiting {wait}s (attempt {attempt + 1}/5)")
                time.sleep(wait)
                continue
            raise LichessError(f"lichess returned HTTP {exc.code}") from None
    raise LichessError("gave up after repeated rate limiting")


def _row(game: dict[str, Any], account: str) -> tuple | None:
    """Flatten one API game into a `games` row, or None if it is not a game."""
    if game.get("status") in EMPTY_STATUSES:
        return None
    moves = game.get("moves") or ""
    players = game.get("players", {})

    def side(colour: str) -> dict[str, Any]:
        return players.get(colour, {}) or {}

    def name(colour: str) -> str:
        player = side(colour)
        user = player.get("user") or {}
        if user:
            return user.get("name", user.get("id", "?"))
        level = player.get("aiLevel")
        return f"lichess AI level {level}" if level else "anonymous"

    def user_id(colour: str) -> str | None:
        return ((side(colour).get("user") or {}).get("id") or "").lower() or None

    lower = account.lower()
    my_color = (
        "white" if user_id("white") == lower
        else "black" if user_id("black") == lower
        else None
    )

    return (
        game["id"],
        "lichess",
        lower,
        my_color,
        1 if game.get("rated") else 0,
        game.get("speed"),
        game.get("source"),
        game.get("createdAt"),
        name("white"),
        name("black"),
        side("white").get("rating"),
        side("black").get("rating"),
        game.get("winner"),
        game.get("status"),
        moves,
        len(moves.split()),
        json.dumps(game, separators=(",", ":")),
    )


def import_account(
    db: sqlite3.Connection,
    user: str,
    *,
    full: bool = False,
    max_games: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Fetch and store `user`'s atomic games. Incremental unless `full`."""
    account = user.lower()
    since = None
    if not full:
        row = db.execute(
            "SELECT MAX(created_at) AS newest FROM games WHERE account = ?", (account,)
        ).fetchone()
        if row and row["newest"]:
            since = int(row["newest"]) + 1

    seen = added = skipped = 0
    for game in stream_games(user, since=since, max_games=max_games):
        seen += 1
        row = _row(game, account)
        if row is None:
            skipped += 1
            continue
        cursor = db.execute(
            "INSERT INTO games (id, site, account, my_color, rated, speed, source, "
            "created_at, white, black, white_rating, black_rating, winner, status, "
            "moves, ply, raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO NOTHING",
            row,
        )
        added += cursor.rowcount or 0
        if on_progress and seen % 25 == 0:
            on_progress(seen)
    db.commit()
    store.set_meta(db, f"last_import:{account}", str(int(time.time())))
    return {"seen": seen, "added": added, "skipped": skipped}
