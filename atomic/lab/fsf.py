"""Fairy-Stockfish adapter — the gold standard we measure everything against.

FSF with the atomic NNUE net (`atomic-2cf13ff256cc.nnue`, +719 Elo over its own
handcrafted eval) is the strongest publicly available atomic engine. We never
try to beat it at full strength; we *throttle* it to a ladder of node limits and
read off where each of our methods lands on that ladder.

Why node limits rather than `Skill Level` or `UCI_LimitStrength`:
  - Node limits are deterministic and hardware-independent, so a result measured
    on this machine reproduces on any other.
  - `UCI_LimitStrength`/`UCI_Elo` is calibrated for *standard* chess and means
    nothing in atomic.
  - `Skill Level` injects random move-picking noise, which inflates variance.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import chess
import chess.engine

from . import config

#: Throttle ladder, in nodes/move. Anchors for the Elo scale; each rung is
#: roughly a fixed Elo step apart because engine strength is ~logarithmic in
#: nodes (the classic "doubling nodes is worth a constant Elo" rule).
LADDER = (1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000)


class FsfUnavailable(RuntimeError):
    """Raised when the Fairy-Stockfish binary or its NNUE net is missing."""


def check_available() -> None:
    if not config.FSF_EXE.exists():
        raise FsfUnavailable(
            f"Fairy-Stockfish not found at {config.FSF_EXE}. "
            "Run `python -m lab.setup_deps` to download it."
        )


@dataclass(frozen=True)
class FsfConfig:
    """One rung of the ladder, i.e. one concrete opponent."""

    nodes: int = 64_000
    threads: int = 1
    hash_mb: int = 64
    use_nnue: bool = True

    @property
    def name(self) -> str:
        tag = "nnue" if self.use_nnue else "hce"
        return f"fsf-{tag}-{self.nodes//1000}kn"

    @property
    def limit(self) -> chess.engine.Limit:
        return chess.engine.Limit(nodes=self.nodes)


@contextlib.contextmanager
def open_engine(cfg: FsfConfig) -> Iterator[chess.engine.SimpleEngine]:
    """Open a configured Fairy-Stockfish session for atomic play.

    `UCI_Variant` is deliberately *not* set here: python-chess manages it itself
    and derives it from `board.uci_variant`, so it is sent as "atomic" on every
    position that is an `AtomicBoard`. Setting it manually raises EngineError.
    """
    check_available()
    engine = chess.engine.SimpleEngine.popen_uci(str(config.FSF_EXE))
    try:
        options: dict[str, object] = {
            "Threads": cfg.threads,
            "Hash": cfg.hash_mb,
            "Use NNUE": cfg.use_nnue,
        }
        if cfg.use_nnue:
            if not config.FSF_NNUE.exists():
                raise FsfUnavailable(
                    f"Atomic NNUE net not found at {config.FSF_NNUE}. "
                    "Run `python -m lab.setup_deps` to download it."
                )
            options["EvalFile"] = str(config.FSF_NNUE)
        engine.configure(options)
        yield engine
    finally:
        engine.quit()


def analyse(board: chess.Board, cfg: FsfConfig) -> chess.engine.InfoDict:
    """One-shot analysis. Convenience only — reuse a session for many calls."""
    with open_engine(cfg) as engine:
        return engine.analyse(board, cfg.limit)
