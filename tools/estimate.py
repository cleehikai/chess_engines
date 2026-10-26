"""Up-front runtime estimates so a long run can be cancelled before it commits.

Each long-running tool (data-gen, train, match, rate) computes a rough ETA from
a measured throughput constant and calls `announce()` *before* doing any real
work. It prints the estimate and — if the run looks long — gives you a short
countdown window to hit Ctrl-C. The throughput constants are calibrated from the
numbers logged in PROGRESS.md / the project memory; they are deliberately rough
(a banner to sanity-check against, not a promise).
"""
from __future__ import annotations

import sys
import time

# Below this many seconds, don't bother with the countdown — just note the ETA.
_COUNTDOWN_THRESHOLD_S = 120.0
# How long to pause (Ctrl-C window) for a run that clears the threshold.
_COUNTDOWN_S = 5


def fmt_duration(seconds: float) -> str:
    """Human-readable duration, e.g. '45s', '3.2 min', '1.8 hr'."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hr"


def announce(label: str, seconds: float, basis: str = "") -> None:
    """Print an ETA banner before a long run; give a Ctrl-C window if it's long.

    label   — what's about to run, e.g. "Rating (96 games)".
    seconds — estimated wall-clock time.
    basis   — one-line note on how the estimate was derived (optional).
    """
    eta = fmt_duration(seconds)
    print(f"\n[estimate] {label}: ~{eta}", flush=True)
    if basis:
        print(f"[estimate] {basis}", flush=True)

    if seconds < _COUNTDOWN_THRESHOLD_S:
        return

    print(
        f"[estimate] This is a long run. Ctrl-C within {_COUNTDOWN_S}s to cancel"
        f" (starting in", end="", flush=True)
    try:
        for i in range(_COUNTDOWN_S, 0, -1):
            print(f" {i}", end="", flush=True)
            time.sleep(1)
        print(" — go).", flush=True)
    except KeyboardInterrupt:
        print("\n[estimate] Cancelled.", flush=True)
        sys.exit(130)
