"""Play-time style shaping.

`make_shaped_eval` wraps the base evaluation with a bounded style bonus derived
from a StyleProfile. It works on both the hand-crafted eval (Phase 1/2) and,
later, the NNUE eval, because it only needs the (score, EvalFeatures) contract.

The cardinal safety rule: winning dominates. The style bonus is:
  * capped at `profile.max_shaping_pawns` * 100 centipawns, and
  * scaled DOWN as |base_score| grows (i.e. as the position becomes decided),
so style only breaks ties among roughly equal moves. It can never turn a
winning move into a losing one or reward a genuinely losing sacrifice.
"""
from __future__ import annotations

import math
from typing import Callable

import chess

from engine.evaluate import EvalFeatures
from style.profiles import StyleProfile

# Base eval returns (score_cp_from_stm, features).
BaseEvalFn = Callable[[chess.Board], "tuple[int, EvalFeatures]"]


def make_shaped_eval(base_eval: BaseEvalFn, profile: StyleProfile) -> Callable[[chess.Board], int]:
    """Return an eval fn (board -> centipawns from side-to-move) with style
    shaping applied. If the profile does no shaping, returns the base score
    directly (zero overhead, exact-strength baseline)."""
    cap = int(round(profile.max_shaping_pawns * 100))

    if cap <= 0 and profile.contempt == 0:
        def plain(board: chess.Board) -> int:
            return base_eval(board)[0]
        return plain

    def shaped(board: chess.Board) -> int:
        base_score, feats = base_eval(board)

        # Contempt: adjust how we value near-draw positions. Positive contempt
        # nudges dead-equal scores toward "slightly bad" so the engine avoids
        # draws; negative makes draws acceptable (defending). Only bites near 0.
        score = base_score
        if profile.contempt:
            # side-to-move perspective: subtract contempt when near equal.
            damp = _decidedness_damp(base_score)
            score -= int(profile.contempt * damp)

        if cap <= 0:
            return score

        raw = _style_bonus(board, feats, profile)
        # Clamp to the cap, then scale down as the position becomes decided so
        # style never overrides a real advantage/disadvantage.
        raw = max(-cap, min(cap, raw))
        raw = int(raw * _decidedness_damp(base_score))
        return score + raw

    return shaped


def _decidedness_damp(base_score: int) -> float:
    """Return a factor in (0, 1] that shrinks as |base_score| grows.

    Near 0 (unclear position) -> ~1.0: style matters most.
    Large |score| (position decided) -> ~0: style is muted.
    Uses a smooth logistic-ish falloff; ~0.5 at ~150cp.
    """
    return 1.0 / (1.0 + (abs(base_score) / 150.0) ** 2)


def _style_bonus(board: chess.Board, feats: EvalFeatures,
                 profile: StyleProfile) -> int:
    """Compute the (uncapped) style bonus in centipawns, from the side-to-move
    perspective."""
    white_to_move = board.turn == chess.WHITE

    # Orient features to the side to move: "own" vs "enemy".
    if white_to_move:
        own_mat, enemy_mat = feats.material_white, feats.material_black
        own_ks, enemy_ks = feats.king_safety_white, feats.king_safety_black
        own_mob, enemy_mob = feats.mobility_white, feats.mobility_black
    else:
        own_mat, enemy_mat = feats.material_black, feats.material_white
        own_ks, enemy_ks = feats.king_safety_black, feats.king_safety_white
        own_mob, enemy_mob = feats.mobility_black, feats.mobility_white

    bonus = 0.0

    # --- king_attack: reward pressure on the enemy king (lower enemy safety) --
    if profile.king_attack:
        # enemy_ks is higher when the enemy king is safe; we reward it being low.
        bonus += profile.king_attack * (-enemy_ks) * 0.25

    # --- king_safety: reward our own king being safe -------------------------
    if profile.king_safety:
        bonus += profile.king_safety * own_ks * 0.25

    # --- material_risk: encourage/discourage investing material for activity --
    if profile.material_risk:
        material_diff = own_mat - enemy_mat  # negative if we've invested
        activity = (own_mob - enemy_mob) - (enemy_ks * 0.1)
        if material_diff < 0:
            # We are down material; reward it only in proportion to our activity
            # and only up to a modest amount (sound-ish sacrifice, not a blunder).
            invested = min(-material_diff, 300)  # cap "sac" credit at ~3 pawns
            sac_soundness = _sigmoid(activity / 20.0)  # 0..1
            bonus += profile.material_risk * invested * 0.15 * sac_soundness
        else:
            # material_risk < 0 (Defender) also penalizes needless exposure.
            if profile.material_risk < 0:
                bonus += profile.material_risk * (-min(activity, 0)) * 0.1

    # --- openness: prefer open (>0) or closed (<0) positions ------------------
    if profile.openness:
        # feats.openness is 0..1; center at 0.5 so it can push both ways.
        bonus += profile.openness * (feats.openness - 0.5) * 120.0

    # --- mobility: extra weight on piece activity ----------------------------
    if profile.mobility:
        bonus += profile.mobility * (own_mob - enemy_mob) * 1.5

    # --- simplify: prefer trading down (>0) toward endgames -------------------
    if profile.simplify:
        # feats.phase: 1.0 opening -> 0.0 bare kings. Lower phase = simplified.
        # Only reward simplification when we are not worse (checked via caller's
        # decidedness damp; here just express the preference).
        bonus += profile.simplify * (1.0 - feats.phase) * 80.0

    return int(round(bonus))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
