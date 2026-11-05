"""Streaming reader for the Lichess atomic database dumps.

`database.lichess.org` publishes 28.9M rated atomic games as monthly
`.pgn.zst` files (3.69 GB in total, ~44 MB per recent month). Two things here
need them: fitting the cp->win% curve, and building the opening book. Neither
needs the whole archive, and neither needs a decompressed copy on disk — the
files stream, so a month is read at ~40 MB of network and no disk footprint
beyond the compressed cache.

The parser is deliberately not `chess.pgn`: it splits games into (headers,
movetext) without touching a board, so a run that keeps 3% of games after
filtering pays board costs on 3% of games rather than all of them. On a 44 MB
month that is the difference between a minute and an hour.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from lab import config

BASE = "https://database.lichess.org/atomic"
DUMPS = config.DATA / "dumps"

USER_AGENT = "atomic-trainer/0.1 (research; contact: cleehikai@gmail.com)"

_HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]$')
#: Strip comments ({...}, which carry %eval and %clk), NAGs, and move numbers.
_COMMENT = re.compile(r"\{[^}]*\}")
_NUMBER = re.compile(r"\d+\.(\.\.)?")
_NAG = re.compile(r"\$\d+")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


class RawGame(NamedTuple):
    headers: dict[str, str]
    movetext: str

    def moves(self) -> list[str]:
        """SAN tokens, with comments, NAGs, numbers and the result removed."""
        text = _COMMENT.sub(" ", self.movetext)
        text = _NAG.sub(" ", _NUMBER.sub(" ", text))
        return [token for token in text.split() if token not in _RESULTS]

    @property
    def result(self) -> str:
        return self.headers.get("Result", "*")

    def elo(self, colour: str) -> int | None:
        try:
            return int(self.headers[f"{colour}Elo"])
        except (KeyError, ValueError):
            return None


def url_for(month: str) -> str:
    return f"{BASE}/lichess_db_atomic_rated_{month}.pgn.zst"


def fetch(month: str, cache_dir: Path | None = None) -> Path:
    """Download one month's dump if it is not already cached."""
    cache_dir = cache_dir or DUMPS
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"lichess_db_atomic_rated_{month}.pgn.zst"
    if path.exists() and path.stat().st_size > 0:
        return path
    request = urllib.request.Request(url_for(month), headers={"User-Agent": USER_AGENT})
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            seen += len(chunk)
            if total:
                print(f"  downloading {month}: {seen / total:6.1%}", end="\r")
    tmp.replace(path)
    print(f"  downloaded {month}: {path.stat().st_size / 1e6:.0f} MB      ")
    return path


def read(path: Path) -> Iterator[RawGame]:
    """Stream games out of a `.pgn.zst` file."""
    import zstandard

    decompressor = zstandard.ZstdDecompressor()
    with path.open("rb") as raw, decompressor.stream_reader(raw) as stream:
        headers: dict[str, str] = {}
        movetext: list[str] = []
        in_moves = False
        for line in _lines(stream):
            if line.startswith("["):
                if in_moves:
                    yield RawGame(headers, " ".join(movetext))
                    headers, movetext, in_moves = {}, [], False
                match = _HEADER.match(line)
                if match:
                    headers[match.group(1)] = match.group(2)
            elif line:
                in_moves = True
                movetext.append(line)
        if headers and movetext:
            yield RawGame(headers, " ".join(movetext))


def _lines(stream) -> Iterator[str]:
    """Decode a binary stream into stripped text lines without loading it all."""
    buffer = b""
    while chunk := stream.read(1 << 20):
        buffer += chunk
        *complete, buffer = buffer.split(b"\n")
        for line in complete:
            yield line.decode("utf-8", "replace").strip()
    if buffer:
        yield buffer.decode("utf-8", "replace").strip()
