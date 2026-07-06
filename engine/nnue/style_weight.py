"""Per-position style weighting for NNUE fine-tuning.

Given a board and a StyleProfile, return a training weight in ~[0.3, 3.0] that
is HIGH for positions characteristic of the style and LOW otherwise. Fine-tuning
with these weights nudges the net to be most accurate on the kinds of positions
the style seeks -- so at play time it evaluates them with more resolution and
tends to steer toward them -- WITHOUT changing the eval targets themselves.
Winning stays the objective; this only reweights attention.

Reuses the hand-crafted feature extractor (engine/evaluate.py) so the notion of
"open", "king attack", etc. matches the play-time shaping in style/shaping.py.
"""
from __future__ import annotations

import chess

from engine.evaluate import evaluate
from style.profiles import StyleProfile

_W_LO, _W_HI = 0.3, 3.0


def position_style_weight(board: chess.Board, profile: StyleProfile) -> float:
    """Weight in [_W_LO, _W_HI]. 1.0 = neutral. Higher = more on-style."""
    _, f = evaluate(board)

    # Orient own/enemy to side-to-move (mirror of style/shaping.py).
    if board.turn == chess.WHITE:
        own_mat, enemy_mat = f.material_white, f.material_black
        own_ks, enemy_ks = f.king_safety_white, f.king_safety_black
        own_mob, enemy_mob = f.mobility_white, f.mobility_black
    else:
        own_mat, enemy_mat = f.material_black, f.material_white
        own_ks, enemy_ks = f.king_safety_black, f.king_safety_white
        own_mob, enemy_mob = f.mobility_black, f.mobility_white

    score = 0.0  # accumulate "on-styleness" in arbitrary units

    if profile.material_risk > 0:
        # Sacrificer-like: positions where side-to-move invested material but is
        # active are highly characteristic.
        invested = max(0, enemy_mat - own_mat)
        active = max(0, own_mob - enemy_mob)
        score += profile.material_risk * (invested * 0.02 + active * 0.05)

    if profile.king_attack:
        # Positions where the enemy king is exposed.
        score += profile.king_attack * max(0, -enemy_ks) * 0.03

    if profile.king_safety:
        # Defender-like: positions where own king safety is in question (there is
        # something to defend) are characteristic.
        score += profile.king_safety * max(0, -own_ks) * 0.03

    if profile.openness > 0:
        score += profile.openness * max(0.0, f.openness - 0.4) * 3.0
    elif profile.openness < 0:
        score += (-profile.openness) * max(0.0, 0.6 - f.openness) * 3.0

    if profile.mobility:
        score += profile.mobility * max(0, own_mob - enemy_mob) * 0.03

    if profile.simplify > 0:
        # Grinder: simplified (low-phase) positions are on-style.
        score += profile.simplify * (1.0 - f.phase) * 1.5

    # Map score -> multiplier around 1.0, clamped.
    weight = 1.0 + score
    return max(_W_LO, min(_W_HI, weight))
