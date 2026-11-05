"""The atomic trainer: game review, puzzles, and opening work.

Sits *on top of* `lab` and never modifies it. `lab` is a measurement workbench
whose value is that its claims are measured; this package is a product built
from the same parts (Fairy-Stockfish, the atomic rules, the calibrated engine
ladder) and must not leak product concerns back into it.

See `docs/trainer-plan.md`.
"""

__all__ = ["analysis", "store", "winrate"]
