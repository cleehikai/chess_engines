"""Paths and shared constants for the atomic lab."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # .../chess_engines/atomic
BIN = ROOT / "bin"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

FSF_EXE = Path(os.environ.get("ATOMIC_FSF", BIN / "fairy-stockfish.exe"))
FSF_NNUE = Path(os.environ.get("ATOMIC_FSF_NNUE", DATA / "atomic.nnue"))

# Lichess variant tablebase (atomic, up to 6 pieces). Free, no key, be polite.
TABLEBASE_URL = "https://tablebase.lichess.ovh/atomic"
TABLEBASE_CACHE = DATA / "tb_cache.sqlite"

for _d in (BIN, DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
