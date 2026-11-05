"""A readable evaluation: named atomic concepts, weighted to agree with the engine.

The explainer (`trainer.explain`) is exact when a short forcing line separates
two moves. When none does, the honest answer so far has been "positional", and
that names the gap without filling it. This module fills it as far as a model
that a person can read is able to.

The model is linear, deliberately. Each concept is a count a human can check on
the board ("own pieces next to your king", "material one capture would win"),
measured as *side to move minus opponent*, and each one gets a weight in
centipawns from a ridge regression against Fairy-Stockfish's own evaluations.
That makes every number it produces decomposable: "the engine prefers Nf3 by
0.6 pawns; of that, 0.3 is development and 0.2 is fewer pieces packed round
your king". A deeper model would fit the engine better and explain nothing.

What it is not: the engine's reason. The net does not compute these concepts,
and a linear model of a neural evaluation is an approximation whose quality has
to be stated every time it speaks. So the fit is measured on held-out positions
(`r2`), shown wherever an attribution is, and the explainer only uses an
attribution when the model agrees with the engine about which move is better.
When it does not, it says that too.

Fitted from positions the engine has already evaluated: the trainer's analysis
cache, which on a machine with analysed games holds Fairy-Stockfish + the
atomic net at a million nodes a ply. Positions where the evaluation is about
tactics, not position, are left out, because those are the ones the
search-based explanation already covers and they would swamp the fit:

- the side to move is in check, or has a capture that wins material;
- the score is a mate, or beyond `CLAMP_CP`.

    py -3.13 -m trainer.cli concepts fit           # from the analysis cache
    py -3.13 -m trainer.cli concepts show          # the weights, and the fit
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.variant
import numpy as np

from lab import config
from lab.methods import ATOMIC_PIECE_VALUES
from lab.rules import explosion_squares

MODEL_PATH = config.DATA / "concepts_atomic.json"

#: Evaluations beyond this are decided positions. Their exact size is noise,
#: and leaving them in lets a few lopsided scores set every weight.
CLAMP_CP = 1000

#: Ridge penalty, in units of the standardised features. Small: the point is
#: to stop correlated counts (material and mobility, say) trading huge weights
#: of opposite sign, not to shrink the real ones.
RIDGE = 1.0

#: Single-capture gains are clamped here. A capture that explodes the king is
#: a won game, and its "value" would otherwise dominate the feature.
GAIN_CLAMP = 1200

CENTRE = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])


@dataclass(frozen=True)
class Concept:
    name: str
    #: The concept's name in a sentence, as "{unit}: White +1, Black -2". Any
    #: "own" or "other" in it is relative to the side the number belongs to.
    unit: str
    #: What it measures, shown when someone asks.
    about: str


CONCEPTS: tuple[Concept, ...] = (
    Concept("pawns", "pawns", "pawns on the board"),
    Concept("knights", "knights", "knights on the board"),
    Concept("bishops", "bishops", "bishops on the board"),
    Concept("rooks", "rooks", "rooks on the board"),
    Concept("queens", "queens", "queens on the board"),
    Concept("loose", "material one capture would win (pawns)",
            "the most material one capture could win, if it were this side's move "
            "(king explosions are counted as fuses instead)"),
    Concept("king_bombs", "pieces beside their own king",
            "non-pawn pieces beside their own king: a capture on any of them explodes it"),
    Concept("king_fuses", "attacked pieces beside their own king",
            "own pieces (pawns included) beside the king that an enemy piece attacks: each is "
            "a capture that would explode it"),
    Concept("king_pressure", "attacks around the other king",
            "attacks on the squares around the enemy king, by pieces that could capture there"),
    Concept("king_pawns", "pawns beside their own king",
            "pawns beside their own king. They survive blasts, but a capture on one still "
            "explodes the king"),
    Concept("clusters", "pieces standing side by side (pairs)",
            "adjacent pairs of non-pawn pieces: one capture takes both"),
    Concept("mobility", "moves available", "pseudo-legal moves"),
    Concept("developed", "minor pieces developed", "knights and bishops off the back rank"),
    Concept("centre", "attacks on the centre", "attacks on d4, e4, d5 and e5"),
    Concept("pawn_advance", "pawn advance (ranks past the 4th)",
            "how far pawns have pushed past the fourth rank"),
    Concept("castling", "castling rights", "castling rights still held"),
)

NAMES = tuple(c.name for c in CONCEPTS)
BY_NAME = {c.name: c for c in CONCEPTS}


# -- features ---------------------------------------------------------------


def _value(piece: chess.Piece | None) -> int:
    return ATOMIC_PIECE_VALUES.get(piece.piece_type, 0) if piece else 0


def capture_gain(board: chess.Board, move: chess.Move) -> int:
    """Net material, for the side making `move`, of the explosion it causes.

    The capturer counted once on its landing square, as in the core's
    `blast_gain`. Exploding the enemy king counts as `GAIN_CLAMP`; exploding
    your own is not a move, and the caller filters those out.

    `best_capture` does not use the king case: a capture that explodes a king
    is what `king_fuses` counts, and letting one fact into two features lets
    the fit give them opposite signs that cancel. Measured here: with both,
    `king_fuses` came out at +2.6 pawns, the wrong way round, propped up by a
    larger negative share of `loose`.
    """
    mover = board.turn
    blast = explosion_squares(board, move) - chess.SquareSet([move.from_square])
    gain = -_value(board.piece_at(move.from_square))
    for square in blast:
        piece = board.piece_at(square)
        if piece is None:
            continue
        if piece.piece_type == chess.KING:
            return GAIN_CLAMP if piece.color != mover else -GAIN_CLAMP
        gain += _value(piece) if piece.color != mover else -_value(piece)
    if board.is_en_passant(move):
        gain += ATOMIC_PIECE_VALUES[chess.PAWN]
    return max(-GAIN_CLAMP, min(GAIN_CLAMP, gain))


def best_capture(board: chess.Board, color: chess.Color) -> int:
    """The most material a single capture by `color` would win, if it were their move.

    Material only. Captures that would explode `color`'s own king are skipped:
    they are not available. Captures that explode the *enemy* king are skipped
    too, because `king_fuses` already counts them (see `capture_gain`). Kings
    cannot capture. Zero when no capture gains anything.
    """
    probe = board.copy(stack=False)
    probe.turn = color
    probe.ep_square = board.ep_square if board.turn == color else None
    king = probe.king(color)
    best = 0
    for move in probe.pseudo_legal_moves:
        if not probe.is_capture(move):
            continue
        if move.from_square == king:
            continue
        blast = explosion_squares(probe, move)
        if king is not None and king in blast:
            continue
        enemy = probe.king(not color)
        if enemy is not None and enemy in blast:
            continue
        best = max(best, capture_gain(probe, move))
    return best


def _side(board: chess.Board, color: chess.Color) -> dict[str, float]:
    """Every concept, counted for one side."""
    out: dict[str, float] = {
        "pawns": len(board.pieces(chess.PAWN, color)),
        "knights": len(board.pieces(chess.KNIGHT, color)),
        "bishops": len(board.pieces(chess.BISHOP, color)),
        "rooks": len(board.pieces(chess.ROOK, color)),
        "queens": len(board.pieces(chess.QUEEN, color)),
    }
    out["loose"] = best_capture(board, color) / 100.0

    own = board.occupied_co[color]
    king = board.king(color)
    enemy_king = board.king(not color)
    non_pawn = own & ~board.pawns & ~board.kings
    if king is not None:
        ring = chess.SquareSet(chess.BB_KING_ATTACKS[king])
        out["king_bombs"] = len(ring & chess.SquareSet(non_pawn))
        out["king_pawns"] = len(ring & board.pieces(chess.PAWN, color))
        # Only this side's own pieces: the enemy cannot capture its own men, and
        # an enemy piece standing beside this king is the opposite of a fuse —
        # it cannot be taken at all without exploding the king doing it.
        fuses = 0
        for square in ring & chess.SquareSet(own):
            attackers = board.attackers(not color, square) - board.pieces(chess.KING, not color)
            if attackers:
                fuses += 1
        out["king_fuses"] = fuses
    else:
        out["king_bombs"] = out["king_pawns"] = out["king_fuses"] = 0

    pressure = 0
    if enemy_king is not None:
        for square in chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king]):
            pressure += len(board.attackers(color, square) - board.pieces(chess.KING, color))
    out["king_pressure"] = pressure

    pieces = list(chess.SquareSet(non_pawn))
    out["clusters"] = sum(
        1 for i, a in enumerate(pieces) for b in pieces[i + 1:]
        if chess.square_distance(a, b) == 1
    )

    probe = board.copy(stack=False)
    probe.turn = color
    probe.ep_square = None
    out["mobility"] = sum(1 for _ in probe.pseudo_legal_moves)

    home = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    minors = board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color)
    out["developed"] = len(minors - chess.SquareSet(home))

    out["centre"] = sum(len(board.attackers(color, sq)) for sq in CENTRE)

    advance = 0
    for square in board.pieces(chess.PAWN, color):
        rank = chess.square_rank(square) if color == chess.WHITE else 7 - chess.square_rank(square)
        advance += max(0, rank - 3)
    out["pawn_advance"] = advance

    rights = 0
    if board.has_kingside_castling_rights(color):
        rights += 1
    if board.has_queenside_castling_rights(color):
        rights += 1
    out["castling"] = rights
    return out


def sides(board: chess.Board, pov: chess.Color) -> tuple[dict[str, float], dict[str, float]]:
    """Each concept for `pov` and for the opponent, unsubtracted."""
    return _side(board, pov), _side(board, not pov)


def features(board: chess.Board, pov: chess.Color) -> np.ndarray:
    """`pov` minus opponent, in `NAMES` order."""
    ours, theirs = sides(board, pov)
    return np.array([ours[n] - theirs[n] for n in NAMES], dtype=float)


# -- the model --------------------------------------------------------------


@dataclass(frozen=True)
class ConceptModel:
    #: Centipawns per unit of (side to move minus opponent), in `names` order.
    weights: tuple[float, ...]
    #: The side-to-move bonus the fit found, in centipawns.
    intercept: float
    names: tuple[str, ...]
    #: Share of the engine's evaluation variance explained, on held-out positions.
    r2: float
    #: How often the model and the engine agree on which of two positions is
    #: better, on held-out pairs. The number that matters for explanations.
    pairwise: float
    samples: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def weight(self, name: str) -> float:
        return self.weights[self.names.index(name)]

    def predict(self, board: chess.Board) -> float:
        """Centipawns for the side to move, by this model's terms."""
        x = features(board, board.turn)
        return float(self.intercept + np.dot(np.array(self.weights), x))

    def describe(self) -> str:
        engine = self.provenance.get("engine", "?")
        return (
            f"concept model: {len(self.names)} concepts, fitted on {self.samples} "
            f"positions labelled by {engine}; held-out R² {self.r2:.2f}, "
            f"picks the better of two positions {100 * self.pairwise:.0f}% of the time"
        )


def load(path: Path | None = None) -> ConceptModel | None:
    """The fitted model, or None if there is none (or it is for other concepts)."""
    path = path or MODEL_PATH
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    names = tuple(raw["names"])
    if names != NAMES:
        return None  # fitted for a different concept set; refit rather than misread
    return ConceptModel(
        weights=tuple(float(w) for w in raw["weights"]),
        intercept=float(raw["intercept"]),
        names=names,
        r2=float(raw["r2"]),
        pairwise=float(raw["pairwise"]),
        samples=int(raw["samples"]),
        provenance=raw.get("provenance", {}),
    )


def save(model: ConceptModel, path: Path | None = None) -> None:
    (path or MODEL_PATH).write_text(json.dumps(asdict(model), indent=2) + "\n")


# -- fitting ----------------------------------------------------------------


def is_positional(board: chess.variant.AtomicBoard, cp: int | None, mate: int | None) -> bool:
    """Whether a labelled position says something about position, not tactics."""
    if mate is not None or cp is None or abs(cp) > CLAMP_CP:
        return False
    if board.is_check() or board.is_game_over(claim_draw=True):
        return False
    if any(_explodes_enemy_king(board, move) for move in board.legal_moves):
        return False
    return best_capture(board, board.turn) <= 0


def _explodes_enemy_king(board: chess.Board, move: chess.Move) -> bool:
    king = board.king(not board.turn)
    return king is not None and board.is_capture(move) and king in explosion_squares(board, move)


def fit(
    samples: Sequence[tuple[str, int]],
    *,
    holdout: float = 0.2,
    seed: int = 0,
    provenance: dict[str, Any] | None = None,
) -> ConceptModel:
    """Ridge regression of the engine's score on concept differences.

    `samples` are `(fen, cp)` with cp for the side to move. Split by position
    into train and held-out sets *before* fitting, so the reported R² and pair
    accuracy are what the model does on positions it has not seen.
    """
    rows: list[np.ndarray] = []
    ys: list[float] = []
    for fen, cp in samples:
        board = chess.variant.AtomicBoard(fen)
        rows.append(features(board, board.turn))
        ys.append(float(max(-CLAMP_CP, min(CLAMP_CP, cp))))
    if len(rows) < 50:
        raise ValueError(f"only {len(rows)} usable positions; need at least 50 to fit")
    x = np.vstack(rows)
    y = np.array(ys)

    order = list(range(len(y)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * (1 - holdout))
    train, test = np.array(order[:cut]), np.array(order[cut:])

    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale == 0] = 1.0
    z = (x[train] - mean) / scale
    design = np.hstack([np.ones((len(train), 1)), z])
    penalty = RIDGE * np.eye(design.shape[1])
    penalty[0, 0] = 0.0  # the intercept is not shrunk
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y[train])

    weights = beta[1:] / scale
    intercept = float(beta[0] - np.dot(weights, mean))

    predicted = intercept + x[test] @ weights
    residual = float(np.sum((y[test] - predicted) ** 2))
    total = float(np.sum((y[test] - y[test].mean()) ** 2)) or 1.0
    r2 = 1.0 - residual / total
    pairwise = _pair_accuracy(y[test], predicted, seed)

    return ConceptModel(
        weights=tuple(round(float(w), 2) for w in weights),
        intercept=round(intercept, 2),
        names=NAMES,
        r2=round(r2, 4),
        pairwise=round(pairwise, 4),
        samples=len(y),
        provenance=provenance or {},
    )


def _pair_accuracy(truth: np.ndarray, guess: np.ndarray, seed: int, pairs: int = 20_000) -> float:
    """How often the model orders two random held-out positions as the engine does.

    Pairs the engine scores within 30cp of each other are skipped: their order
    is noise, and counting them would score the model on coin flips.
    """
    rng = np.random.default_rng(seed)
    n = len(truth)
    if n < 2:
        return 0.0
    a = rng.integers(0, n, pairs)
    b = rng.integers(0, n, pairs)
    keep = np.abs(truth[a] - truth[b]) >= 30
    if not keep.any():
        return 0.0
    agree = np.sign(truth[a] - truth[b]) == np.sign(guess[a] - guess[b])
    return float(agree[keep].mean())


def samples_from_cache(db, engine: str, limit: int | None = None) -> list[tuple[str, int]]:
    """Positional `(fen, cp)` pairs from every position the trainer has analysed."""
    out: list[tuple[str, int]] = []
    query = "SELECT fen, lines FROM positions WHERE engine = ? ORDER BY nodes DESC"
    seen: set[str] = set()
    for row in db.execute(query, (engine,)):
        fen = row["fen"]
        if fen in seen:
            continue
        seen.add(fen)
        lines = json.loads(row["lines"])
        if not lines:
            continue
        top = max(lines, key=_rank)
        try:
            board = chess.variant.AtomicBoard(fen)
        except ValueError:
            continue
        if not board.is_valid():
            continue  # null-move probes from the explainer live in the cache too
        if is_positional(board, top.get("cp"), top.get("mate")):
            out.append((fen, int(top["cp"])))
        if limit and len(out) >= limit:
            break
    return out


def _rank(line: dict[str, Any]) -> tuple[int, int]:
    mate, cp = line.get("mate"), line.get("cp")
    if mate is not None:
        return (2, -mate) if mate > 0 else (0, -mate)
    return (1, cp or 0)


def samples_from_play(
    search: Callable[[str], tuple[int | None, int | None, str | None]],
    *,
    games: int,
    plies: int = 40,
    explore: float = 0.25,
    seed: int = 0,
    on_progress: Callable[[int], None] | None = None,
) -> list[tuple[str, int]]:
    """Positional `(fen, cp)` pairs from games the engine plays against itself.

    For a machine with no analysis cache yet. `search(fen)` returns
    `(cp, mate, best_uci)`. With probability `explore` a random legal move is
    played instead of the engine's, which is what keeps the sample from being
    sixty copies of the same opening.
    """
    rng = random.Random(seed)
    out: list[tuple[str, int]] = []
    for game in range(games):
        board = chess.variant.AtomicBoard()
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            cp, mate, best = search(board.fen())
            if is_positional(board, cp, mate):
                out.append((board.fen(), int(cp)))
            moves = list(board.legal_moves)
            if best is None or rng.random() < explore:
                move = rng.choice(moves)
            else:
                move = chess.Move.from_uci(best)
            board.push(move)
        if on_progress:
            on_progress(game + 1)
    return out


# -- attribution ------------------------------------------------------------


@dataclass(frozen=True)
class Contribution:
    name: str
    #: Concept value in each position, for the side we are explaining.
    ours: tuple[float, float]
    theirs: tuple[float, float]
    #: Centipawns this concept adds to position A over position B.
    cp: float

    def sentence(self, side: str, other: str) -> str:
        """`moves available: White +11, Black -4 (+1.6)`.

        Each side's change from position B to position A, named, then what
        the concept is worth in pawns. Naming both sides rather than "you" and
        "the opponent" is what keeps "attacks around the other king" readable.
        """
        concept = BY_NAME[self.name]
        changes = []
        for who, (now, then) in ((side, self.ours), (other, self.theirs)):
            delta = now - then
            if abs(delta) >= 0.05:
                amount = f"{delta:+.1f}" if concept.name == "loose" else f"{delta:+.0f}"
                changes.append(f"{who} {amount.replace('-', '−')}")
        change = ", ".join(changes) or "no change"
        return f"{concept.unit}: {change} ({self.cp / 100:+.1f})".replace("(-", "(−")


def attribute(
    model: ConceptModel,
    a: chess.Board,
    b: chess.Board,
    pov: chess.Color,
) -> tuple[float, list[Contribution]]:
    """Why the model prefers position A to B for `pov`, concept by concept.

    Both positions are scored for `pov` whoever is to move in them, so two
    lines of different lengths can still be compared. Returns the model's
    total difference in centipawns and the per-concept contributions,
    largest first.
    """
    a_ours, a_theirs = sides(a, pov)
    b_ours, b_theirs = sides(b, pov)
    weights = dict(zip(model.names, model.weights, strict=True))
    out: list[Contribution] = []
    for name in model.names:
        delta = (a_ours[name] - a_theirs[name]) - (b_ours[name] - b_theirs[name])
        cp = weights[name] * delta
        out.append(Contribution(
            name=name,
            ours=(a_ours[name], b_ours[name]),
            theirs=(a_theirs[name], b_theirs[name]),
            cp=cp,
        ))
    out.sort(key=lambda c: -abs(c.cp))
    return float(sum(c.cp for c in out)), out


def table(model: ConceptModel) -> Iterable[str]:
    """The weights as a human reads them: largest first, in pawns per unit."""
    rows = sorted(zip(model.names, model.weights, strict=True), key=lambda nw: -abs(nw[1]))
    for name, weight in rows:
        yield f"  {weight / 100:+6.2f}  per {BY_NAME[name].unit:<44} {BY_NAME[name].about}"
