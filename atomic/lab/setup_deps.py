"""Fetch the gold standard: Fairy-Stockfish 14 + the atomic NNUE net.

    python -m lab.setup_deps

Idempotent — skips anything already present and correctly sized.
"""

from __future__ import annotations

import sys
import urllib.request

from . import config

FSF_URL = (
    "https://github.com/fairy-stockfish/Fairy-Stockfish/releases/download/"
    "fairy_sf_14/fairy-stockfish_x86-64-bmi2.exe"
)
# atomic-2cf13ff256cc.nnue — +719 Elo over FSF's handcrafted atomic eval.
# Listed at https://fairy-stockfish.github.io/nnue/ (author: belzedar_, 2022-08-24).
NNUE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1bC7T3iDft8Kbuxlu3Vm2fERxk7cOSoDy&export=download&confirm=t"
)

MIN_FSF_BYTES = 1_000_000
MIN_NNUE_BYTES = 10_000_000


def _fetch(url: str, dest, min_bytes: int) -> None:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  ok      {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    print(f"  fetch   {dest.name} <- {url.split('?')[0]}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    size = tmp.stat().st_size
    if size < min_bytes:
        tmp.unlink()
        raise RuntimeError(
            f"{dest.name} download looks wrong ({size:,} bytes, expected "
            f">={min_bytes:,}). Google Drive may be serving an HTML "
            "interstitial; download it by hand from "
            "https://fairy-stockfish.github.io/nnue/"
        )
    tmp.replace(dest)
    print(f"  done    {dest.name} ({size:,} bytes)")


def main() -> int:
    print("Fairy-Stockfish gold standard:")
    _fetch(FSF_URL, config.FSF_EXE, MIN_FSF_BYTES)
    _fetch(NNUE_URL, config.FSF_NNUE, MIN_NNUE_BYTES)

    # Prove it actually runs atomic with the net loaded.
    import chess.variant

    from . import fsf

    cfg = fsf.FsfConfig(nodes=50_000)
    with fsf.open_engine(cfg) as engine:
        info = engine.analyse(chess.variant.AtomicBoard(), cfg.limit)
    print(f"\nverified: startpos eval {info['score'].white()} at depth {info['depth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
