"""Sidecar metadata for trained NNUE nets.

Each net checkpoint `foo.pt` gets an optional `foo.meta.json` next to it that
records *how it was trained*: how many positions, the Stockfish label depth,
the lambda mix, epochs, and so on. This is deliberately a **sidecar** file, not
data embedded in the .pt, so the inference load path (torch.load with
weights_only=True) stays completely untouched — the shipped engine never has to
know the metadata exists.

The rating tool (tools/rate.py) reads this to log "data quantity/quality -> Elo"
without you having to retype the training args by hand. Nets trained before this
existed simply have no sidecar; the tools degrade gracefully.
"""
from __future__ import annotations

import glob as _glob
import json
import os
from typing import Any


def meta_path(checkpoint: str) -> str:
    """`data/nnue_v2.pt` -> `data/nnue_v2.meta.json`."""
    base, _ext = os.path.splitext(checkpoint)
    return base + ".meta.json"


def count_positions(shards_glob: str) -> int:
    """Total lines across all shards matching the glob (one position per line)."""
    total = 0
    for path in _glob.glob(shards_glob):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                total += sum(1 for _ in fh)
        except OSError:
            pass
    return total


def label_depth_of(shards_glob: str) -> int | None:
    """Best-effort: read the Stockfish label depth if a `<prefix>.info.json`
    was written by gen_data. Returns None if unknown."""
    for path in _glob.glob(shards_glob):
        base = path.split(".shard")[0]
        info = base + ".info.json"
        if os.path.exists(info):
            try:
                with open(info, encoding="utf-8") as fh:
                    return int(json.load(fh).get("depth"))
            except (OSError, ValueError, TypeError):
                return None
    return None


def write(checkpoint: str, data: dict[str, Any]) -> str:
    """Write (merge over any existing) sidecar metadata for a checkpoint."""
    path = meta_path(checkpoint)
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                existing = json.load(fh)
        except (OSError, ValueError):
            existing = {}
    existing.update({k: v for k, v in data.items() if v is not None})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, sort_keys=True)
    return path


def read(checkpoint: str) -> dict[str, Any]:
    """Read sidecar metadata for a checkpoint, or {} if none."""
    path = meta_path(checkpoint)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}
