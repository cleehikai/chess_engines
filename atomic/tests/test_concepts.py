"""The concept model: features count what they claim, the fit recovers known
weights, and the explainer only uses an attribution that agrees with the engine."""

from __future__ import annotations

import random

import chess
import chess.variant
import numpy as np
import pytest

from trainer import analysis, concepts, explain, winrate

MODEL = winrate.WinModel(a=0.0, k=0.004, fitted=True, provenance={})


def side(fen: str, color: chess.Color = chess.WHITE) -> dict[str, float]:
    return concepts.sides(chess.variant.AtomicBoard(fen), color)[0]


# -- features ---------------------------------------------------------------


def test_start_position_is_symmetric():
    x = concepts.features(chess.variant.AtomicBoard(), chess.WHITE)
    assert np.allclose(x, 0.0)


def test_king_bombs_count_non_pawns_beside_the_king():
    # White king e1 with queen d1 and bishop f1 beside it, pawns on d2-f2.
    counts = side("4k3/8/8/8/8/8/3PPP2/3QKB2 w - - 0 1")
    assert counts["king_bombs"] == 2
    assert counts["king_pawns"] == 3


def test_a_fuse_is_an_occupied_square_beside_the_king_that_the_enemy_attacks():
    # The black rook on d8 attacks d1; capturing there explodes e1.
    counts = side("3rk3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert counts["king_fuses"] == 1
    # The enemy king cannot capture, so its attacks are not fuses.
    counts = side("8/8/8/8/8/8/3k4/3QK3 w - - 0 1")
    assert counts["king_fuses"] == 0


def test_loose_material_is_the_best_capture_net_of_the_blast():
    # White Bxd5 explodes the black knight on d5 and the rook on c6, and the
    # bishop dies too.
    fen = "4k3/8/2r5/3n4/8/8/6B1/4K3 w - - 0 1"
    gain = concepts.best_capture(chess.variant.AtomicBoard(fen), chess.WHITE)
    values = concepts.ATOMIC_PIECE_VALUES
    assert gain == values[chess.KNIGHT] + values[chess.ROOK] - values[chess.BISHOP]


def test_single_capture_gains_are_clamped():
    fen = "4k3/8/2r5/3q4/8/8/6B1/4K3 w - - 0 1"
    assert concepts.best_capture(chess.variant.AtomicBoard(fen), chess.WHITE) == concepts.GAIN_CLAMP


def test_a_capture_that_would_explode_your_own_king_is_not_loose_material():
    # The white king on e1 is next to d2, so capturing the knight there would
    # explode White's own king.
    fen = "4k3/8/8/8/8/8/3n4/3QK3 w - - 0 1"
    assert concepts.best_capture(chess.variant.AtomicBoard(fen), chess.WHITE) == 0


def test_clusters_count_adjacent_pairs():
    counts = side("4k3/8/8/8/8/2NB4/8/4K3 w - - 0 1")
    assert counts["clusters"] == 1


def test_positional_filter_drops_tactics():
    tactical = chess.variant.AtomicBoard("4k3/8/2r5/3q4/8/8/6B1/4K3 w - - 0 1")
    assert not concepts.is_positional(tactical, 0, None)
    quiet = chess.variant.AtomicBoard()
    assert concepts.is_positional(quiet, 50, None)
    assert not concepts.is_positional(quiet, None, 3)
    assert not concepts.is_positional(quiet, 5000, None)


# -- fitting -------------------------------------------------------------------


def _random_positions(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.variant.AtomicBoard()
        for _ in range(rng.randint(4, 30)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            out.append(board.fen())
    return out


def test_fit_recovers_a_planted_linear_evaluation():
    """Label positions with a known concept evaluation; the fit should find it."""
    truth = {name: 0.0 for name in concepts.NAMES}
    truth.update({"developed": 30.0, "king_bombs": -40.0, "mobility": 3.0, "queens": 300.0})
    samples = []
    for fen in _random_positions(600, seed=1):
        board = chess.variant.AtomicBoard(fen)
        x = concepts.features(board, board.turn)
        cp = 10 + sum(truth[n] * v for n, v in zip(concepts.NAMES, x, strict=True))
        samples.append((fen, int(round(cp))))
    model = concepts.fit(samples, seed=2)
    assert model.r2 > 0.95
    assert model.pairwise > 0.95
    for name in ("developed", "king_bombs", "mobility"):
        assert model.weight(name) == pytest.approx(truth[name], abs=6)


def test_fit_refuses_too_few_positions():
    with pytest.raises(ValueError):
        concepts.fit([(chess.STARTING_FEN, 0)] * 10)


def test_save_and_load_round_trip(tmp_path):
    model = concepts.ConceptModel(
        weights=tuple(float(i) for i in range(len(concepts.NAMES))), intercept=5.0,
        names=concepts.NAMES, r2=0.4, pairwise=0.7, samples=100,
        provenance={"engine": "test"},
    )
    path = tmp_path / "c.json"
    concepts.save(model, path)
    assert concepts.load(path) == model


# -- attribution in the explainer ---------------------------------------------


def _planted(**weights: float) -> concepts.ConceptModel:
    w = {name: 0.0 for name in concepts.NAMES}
    w.update(weights)
    return concepts.ConceptModel(
        weights=tuple(w[n] for n in concepts.NAMES), intercept=0.0, names=concepts.NAMES,
        r2=0.5, pairwise=0.75, samples=1000, provenance={"engine": "test"},
    )


def test_attribution_names_the_concept_that_separates_two_moves():
    # 1.Nf3 develops a knight; 1.a3 does not. After either, Black replies a6.
    start = chess.variant.AtomicBoard()
    a, b = start.copy(), start.copy()
    for san in ("Nf3", "a6"):
        a.push_san(san)
    for san in ("a3", "a6"):
        b.push_san(san)
    total, parts = concepts.attribute(_planted(developed=40.0), a, b, chess.WHITE)
    assert total == pytest.approx(40.0)
    assert parts[0].name == "developed"
    assert parts[0].sentence("White", "Black") == "minor pieces developed: White +1 (+0.4)"


def line(*ucis: str, cp: int | None = 0, mate: int | None = None) -> analysis.Line:
    return analysis.Line(uci=ucis[0], cp=cp, mate=mate, depth=10, pv=tuple(ucis))


def _search(table):
    return lambda fen, multipv: table.get(fen, [])[:multipv]


def test_positional_gap_gets_a_concept_reason_when_the_model_agrees():
    fen = chess.STARTING_FEN
    search = _search({fen: [line("g1f3", "a7a6", cp=200), line("a2a3", "a7a6", cp=20)]})
    # 1.5 pawns of a 1.8-pawn preference: enough to be offered as the reason.
    result = explain.explain(fen, search, model=MODEL, concept_model=_planted(developed=150.0))
    reasons = result["alternative"]["reasons"]
    assert [r["kind"] for r in reasons] == ["positional", "concepts"]
    assert reasons[1]["agrees"] and reasons[1]["covered"]
    assert "minor pieces developed: White +1" in reasons[1]["text"]
    assert reasons[1]["contributions"][0]["name"] == "developed"


def test_a_model_that_disagrees_with_the_engine_says_so():
    fen = chess.STARTING_FEN
    search = _search({fen: [line("g1f3", "a7a6", cp=200), line("a2a3", "a7a6", cp=20)]})
    # A model that *dislikes* development prefers a3, against the engine.
    result = explain.explain(fen, search, model=MODEL, concept_model=_planted(developed=-40.0))
    concept = result["alternative"]["reasons"][1]
    assert not concept["agrees"]
    assert "outside these concepts" in concept["text"]


def test_concepts_that_cover_a_sliver_are_offered_as_a_pointer_not_an_answer():
    fen = chess.STARTING_FEN
    search = _search({fen: [line("g1f3", "a7a6", cp=200), line("a2a3", "a7a6", cp=20)]})
    # Agrees in direction, but 0.2 pawns of a 1.8-pawn preference.
    result = explain.explain(fen, search, model=MODEL, concept_model=_planted(developed=20.0))
    concept = result["alternative"]["reasons"][1]
    assert concept["agrees"] and not concept["covered"]
    assert "do not measure" in concept["text"]


def test_no_concept_model_means_no_concept_reason():
    fen = chess.STARTING_FEN
    search = _search({fen: [line("g1f3", "a7a6", cp=200), line("a2a3", "a7a6", cp=20)]})
    result = explain.explain(fen, search, model=MODEL, concept_model=None)
    assert [r["kind"] for r in result["alternative"]["reasons"]] == ["positional"]
