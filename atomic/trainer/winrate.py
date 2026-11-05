"""Centipawns to win%, for *atomic*, and the move classification built on it.

Lichess judges moves by how much **win%** a move throws away rather than how
many centipawns, because a 300cp drop means something completely different in
an equal position than in a won one. Its conversion is

    Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)

which is a logistic fitted to *standard* chess games. It cannot be reused here,
and this repo already measured the reason: Fairy-Stockfish + the atomic net
evaluates the atomic **starting position** at +250 to +310 cp. Under the
standard curve that reads as White winning ~85% of the time from move zero,
which is nonsense — White scores nothing like that. Every threshold built on
that number would be wrong, and since "was this a blunder" is the input to game
review, puzzle mining, and repertoire validation alike, being wrong here is
being wrong everywhere.

So the curve is fitted to atomic, from real games, by `fit_winrate.py`:

    win% = 100 / (1 + exp(-(a + k * cp)))

Two parameters rather than one. The intercept `a` is not assumed to be zero
because the engine's zero is not necessarily the point where a game is even —
if the atomic net carries a systematic offset, forcing the curve through 50% at
0cp would smear that offset across every judgement instead of exposing it.

The *thresholds* (10 / 20 / 30 win% points for inaccuracy / mistake / blunder)
are kept from Lichess deliberately. Once the scale itself is correct, "you threw
30% of a game" means the same thing in atomic as in standard chess; it is the
conversion, not the threshold, that needed re-deriving.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lab import config

MODEL_PATH: Path = config.DATA / "winrate_atomic.json"

#: Lichess's standard-chess slope. Used only as a labelled fallback so the
#: trainer runs before the fit has been done — never silently.
LICHESS_STANDARD_K = 0.00368208

#: Win% points thrown away. Same as Lichess, applied to the atomic scale.
INACCURACY = 10.0
MISTAKE = 20.0
BLUNDER = 30.0


@dataclass(frozen=True)
class WinModel:
    """cp -> win%, plus where the numbers came from."""

    a: float
    k: float
    fitted: bool
    #: free-form provenance: sample size, node budget, dump month, fit date
    provenance: dict[str, Any]

    def win_pct(self, cp: float) -> float:
        """Win% for the side the score belongs to. Input is a POV score."""
        return 100.0 / (1.0 + math.exp(-(self.a + self.k * cp)))

    def score_win_pct(self, cp: int | None, mate: int | None) -> float:
        """Win% from an engine score, where mate is decisive by definition."""
        if mate is not None:
            return 100.0 if mate > 0 else 0.0
        if cp is None:
            raise ValueError("score has neither cp nor mate")
        return self.win_pct(cp)

    @property
    def even_cp(self) -> float:
        """The evaluation at which the game is actually even.

        Zero for a well-centred engine. Anything else is a measured bias in the
        evaluation, and worth knowing about.
        """
        return -self.a / self.k

    def describe(self) -> str:
        if not self.fitted:
            return (
                f"win% model: UNFITTED fallback (lichess standard-chess k="
                f"{self.k:.8f}). Run `python -m trainer.fit_winrate`."
            )
        n = self.provenance.get("positions", "?")
        return (
            f"win% model: a={self.a:+.5f} k={self.k:.8f} "
            f"(even at {self.even_cp:+.0f}cp, fitted on {n} positions)"
        )


def fallback_model() -> WinModel:
    return WinModel(
        a=0.0,
        k=LICHESS_STANDARD_K,
        fitted=False,
        provenance={"source": "lichess standard chess, not atomic"},
    )


def load(path: Path | None = None) -> WinModel:
    """The fitted atomic model, or a clearly-labelled fallback."""
    path = path or MODEL_PATH
    if not path.exists():
        return fallback_model()
    raw = json.loads(path.read_text())
    return WinModel(
        a=float(raw["a"]),
        k=float(raw["k"]),
        fitted=True,
        provenance=raw.get("provenance", {}),
    )


def save(model: WinModel, path: Path | None = None) -> None:
    (path or MODEL_PATH).write_text(json.dumps(asdict(model), indent=2))


def classify(loss: float) -> str:
    """Name a move by how much win% it threw away."""
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    return "ok"


if __name__ == "__main__":
    model = load()
    print(model.describe())
    print(f"{'cp':>7}  win%")
    for cp in (-1000, -500, -300, -100, 0, 100, 250, 300, 500, 1000):
        print(f"{cp:>7}  {model.win_pct(cp):5.1f}")
