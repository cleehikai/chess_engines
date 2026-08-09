"""The `Method` interface — one implementation per engine idea under test.

Everything in this project is a comparison between methods, so a method has to
be cheap to write and impossible to measure unfairly. The contract:

    a method receives a position and a compute budget, and returns a move.

The budget is expressed in the same currency for every method (`Budget`), so a
gauntlet at "64k nodes" means the same thing whether the method is alpha-beta,
MCTS, or a raw policy net. Methods that cannot honour a given currency should
say so via `supports()` rather than silently ignoring it — a method that
quietly overspends its budget produces a meaningless Elo.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import chess
import chess.variant

from .rules import explosion_squares, material_after

# Piece values in centipawns, from Pav (2025) "Inferring Piece Value in Chess
# and Chess Variants", Table 9, uniform-random snapshot over 1.5M Lichess atomic
# games, normalised so pawn = 100. Atomic values are heavily *compressed*
# relative to standard chess and bishops are worth clearly more than knights.
#   knight 1.9P, bishop 2.5P, rook 4.9P, queen 10.2P
ATOMIC_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 190,
    chess.BISHOP: 252,
    chess.ROOK: 486,
    chess.QUEEN: 1018,
    chess.KING: 0,  # the king is the win condition, not material
}


@dataclass(frozen=True)
class Budget:
    """A compute budget. Exactly one field should be set."""

    nodes: int | None = None
    depth: int | None = None
    seconds: float | None = None

    def __post_init__(self) -> None:
        given = [f for f in (self.nodes, self.depth, self.seconds) if f is not None]
        if len(given) != 1:
            raise ValueError("Budget takes exactly one of nodes/depth/seconds")

    @property
    def currency(self) -> str:
        if self.nodes is not None:
            return "nodes"
        if self.depth is not None:
            return "depth"
        return "seconds"

    def __str__(self) -> str:
        if self.nodes is not None:
            return f"{self.nodes}n"
        if self.depth is not None:
            return f"d{self.depth}"
        return f"{self.seconds}s"


@runtime_checkable
class Method(Protocol):
    """An atomic-chess move chooser under test."""

    name: str

    def supports(self, budget: Budget) -> bool:
        """Whether this method can honour `budget`'s currency."""

    def choose(self, board: chess.variant.AtomicBoard, budget: Budget) -> chess.Move:
        """Return a legal move for `board`, spending at most `budget`."""

    def reset(self) -> None:
        """Clear per-game state (transposition table, search tree, ...)."""


class BaseMethod:
    """Convenience base: no state, supports every currency."""

    name = "unnamed"

    def supports(self, budget: Budget) -> bool:  # noqa: ARG002
        return True

    def reset(self) -> None:
        return None


# --------------------------------------------------------------------------
# M0 — the floor. Not a serious engine; it exists so that every part of the
# harness (arena, adjudication, rating, reporting) is exercised end to end
# before any real method is written, and so every later Elo has a known floor
# to sit above.
# --------------------------------------------------------------------------


class RandomMethod(BaseMethod):
    """Uniformly random legal move."""

    name = "M0-random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, board: chess.variant.AtomicBoard, budget: Budget) -> chess.Move:  # noqa: ARG002
        return self._rng.choice(list(board.legal_moves))


class GreedyMaterialMethod(BaseMethod):
    """One-ply greedy on explosion-aware material, with mate detection.

    Deliberately shallow, but it is *atomic*-shallow rather than chess-shallow:
    it scores a capture by the material left standing after the explosion
    resolves, which is the single cheapest piece of variant knowledge there is.
    A standard-chess greedy engine would happily play a queen capture that
    detonates its own rook and bishop alongside it.
    """

    name = "M0-greedy"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, board: chess.variant.AtomicBoard, budget: Budget) -> chess.Move:  # noqa: ARG002
        me = board.turn
        best_score = None
        best_moves: list[chess.Move] = []
        for move in board.legal_moves:
            board.push(move)
            try:
                # Winning now beats any material consideration.
                if board.is_variant_win() or board.is_checkmate():
                    return move
                score = material_after(board, me, ATOMIC_PIECE_VALUES)
                if board.is_variant_loss():
                    score -= 1_000_000
            finally:
                board.pop()
            if best_score is None or score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return self._rng.choice(best_moves)


# --------------------------------------------------------------------------
# M1 — atomic-native handcrafted eval + alpha-beta, running on the Rust core.
#
# The first method that is actually an engine. Its point is not the absolute
# rating but the ablation series: each evaluation term can be switched off
# independently, so "what is blast-radius king safety worth, in Elo" is a
# measurement rather than an opinion.
# --------------------------------------------------------------------------

#: The evaluation terms M1 can switch on. Mirrors `EVAL_TERMS` in the Rust
#: crate; kept here so `lab` can enumerate ablations without importing the
#: extension module (which may not be built yet).
EVAL_TERMS = ("material", "king_blast", "pawn_shield", "mobility", "pst")


class AlphaBetaMethod(BaseMethod):
    """Alpha-beta with an atomic-native handcrafted evaluation.

    Delegates everything expensive to `atomic_core`. The Python side exists
    only to translate between `chess.variant.AtomicBoard` and FENs, which costs
    one FEN round-trip per move — negligible next to a 100k-node search.
    """

    name = "M1-alphabeta"

    def __init__(
        self,
        terms: Sequence[str] | None = None,
        tt_mb: int = 64,
        name: str | None = None,
        quiescence: bool = True,
    ) -> None:
        try:
            import atomic_core
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                "M1 needs the Rust core; run `python -m lab.build_core`"
            ) from exc
        unknown = set(terms or ()) - set(EVAL_TERMS)
        if unknown:
            raise ValueError(
                f"unknown eval term(s) {sorted(unknown)}; known: {', '.join(EVAL_TERMS)}"
            )
        self._terms = tuple(terms) if terms is not None else EVAL_TERMS
        self._quiescence = quiescence
        self._engine = atomic_core.Engine(
            terms=list(self._terms) if terms is not None else None,
            tt_mb=tt_mb,
            quiescence=quiescence,
        )
        if name is not None:
            self.name = name
        elif terms is not None:
            # Name an ablation after what it is *missing*: that is what the
            # experiment is about, and it keeps result files self-describing.
            missing = [t for t in EVAL_TERMS if t not in self._terms]
            if missing:
                self.name = "M1-no-" + "-".join(missing)

    @property
    def terms(self) -> tuple[str, ...]:
        return self._terms

    def supports(self, budget: Budget) -> bool:
        # Seconds are deliberately unsupported: a wall-clock budget would make
        # results depend on what else the machine is doing.
        return budget.nodes is not None or budget.depth is not None

    def choose(self, board: chess.variant.AtomicBoard, budget: Budget) -> chess.Move:
        uci, _score, _depth, _nodes = self._engine.search(
            board.fen(),
            nodes=budget.nodes if budget.nodes is not None else 2**60,
            depth=budget.depth if budget.depth is not None else 64,
        )
        if uci is None:
            raise RuntimeError(f"no move returned for {board.fen()}")
        return chess.Move.from_uci(uci)

    def reset(self) -> None:
        self._engine.clear()


def _ablation(term: str):
    """Build a factory for M1 with exactly one evaluation term removed."""
    remaining = [t for t in EVAL_TERMS if t != term]

    def factory(**kwargs) -> AlphaBetaMethod:
        return AlphaBetaMethod(terms=remaining, **kwargs)

    return factory


REGISTRY: dict[str, object] = {
    RandomMethod.name: RandomMethod,
    GreedyMaterialMethod.name: GreedyMaterialMethod,
    AlphaBetaMethod.name: AlphaBetaMethod,
    # The material-only control: alpha-beta with no atomic knowledge beyond
    # Pav's piece values. Every other M1 number should be read against this.
    "M1-material-only": lambda **kw: AlphaBetaMethod(
        terms=["material"], name="M1-material-only", **kw
    ),
    **{f"M1-no-{term}": _ablation(term) for term in EVAL_TERMS},
    # Search-side ablation. Together with `M1-noqs-no-king_blast` this is a 2x2
    # design: it separates "blast-radius king safety is worthless" from "the
    # quiescence search already covers it", which the eval-only ablation
    # cannot distinguish.
    "M1-noqs": lambda **kw: AlphaBetaMethod(quiescence=False, name="M1-noqs", **kw),
    "M1-noqs-no-king_blast": lambda **kw: AlphaBetaMethod(
        terms=[t for t in EVAL_TERMS if t != "king_blast"],
        quiescence=False,
        name="M1-noqs-no-king_blast",
        **kw,
    ),
}


def build(name: str, **kwargs) -> Method:
    """Instantiate a registered method by name."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown method {name!r}; known: {', '.join(sorted(REGISTRY))}"
        ) from None
    return factory(**kwargs)


__all__ = [
    "ATOMIC_PIECE_VALUES",
    "EVAL_TERMS",
    "AlphaBetaMethod",
    "BaseMethod",
    "Budget",
    "GreedyMaterialMethod",
    "Method",
    "RandomMethod",
    "REGISTRY",
    "build",
    "explosion_squares",
]
