"""Style profiles: configurable, *separate* playing personalities.

Each profile is a bounded set of weights that the shaping layer adds ON TOP of
the real evaluation. The cardinal rule (baked into shaping.py): winning stays
the dominant objective. Style bonuses are capped as a fraction of a pawn and
scaled down as the position sharpens, so they only break ties among roughly
equal moves and never justify a losing sacrifice.

Weights are interpreted by style/shaping.py. All are dimensionless multipliers
(centipawns per unit of the corresponding feature) unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StyleProfile:
    name: str
    description: str

    # --- Evaluation nudges (centipawn weights on positional features) --------
    # Positive material_risk => tolerate/encourage investing material for
    # activity; the shaping layer converts this into a bounded bonus for lines
    # where we are slightly down material but very active.
    material_risk: float = 0.0
    king_attack: float = 0.0        # reward pressure on the ENEMY king
    king_safety: float = 0.0        # reward safety of OWN king
    openness: float = 0.0           # >0 prefers open positions, <0 prefers closed
    mobility: float = 0.0           # extra weight on piece activity
    simplify: float = 0.0           # >0 prefers trading down (endgame grinding)

    # --- Search behaviour ----------------------------------------------------
    # How hard style is allowed to push, as a fraction of a pawn (100cp). This
    # is the hard cap enforced in shaping.py. Keep small (<= 0.6) to preserve
    # strength.
    max_shaping_pawns: float = 0.35
    # Contempt: score draws as slightly bad for us (positive) => avoid draws,
    # or slightly good (negative) => accept draws when defending.
    contempt: int = 0

    tags: tuple[str, ...] = field(default_factory=tuple)


# --- Preset profiles ----------------------------------------------------------
# Each is deliberately distinct. Tune these values in Phase 2 against real games.

_BALANCED = StyleProfile(
    name="balanced",
    description="No style shaping. Maximum strength baseline / control group.",
    max_shaping_pawns=0.0,
    tags=("neutral",),
)

_SACRIFICER = StyleProfile(
    name="sacrificer",
    description="Seeks sound-ish sacrifices for initiative and king attack. "
                "Never rewards simply losing material.",
    material_risk=0.9,
    king_attack=0.8,
    openness=0.3,
    mobility=0.4,
    max_shaping_pawns=0.55,
    contempt=12,           # dislikes draws; wants sharp decisive play
    tags=("aggressive", "tactical"),
)

_ATTACKER = StyleProfile(
    name="attacker",
    description="Piles pressure on the enemy king with pieces and pawn storms, "
                "without necessarily investing material.",
    king_attack=1.0,
    mobility=0.5,
    openness=0.2,
    max_shaping_pawns=0.45,
    contempt=8,
    tags=("aggressive",),
)

_DEFENDER = StyleProfile(
    name="defender",
    description="Solid and prophylactic: prioritizes own king safety, absorbs "
                "pressure, and counterattacks. Risk-averse when ahead.",
    king_safety=1.0,
    material_risk=-0.4,    # dislikes speculative material investment
    openness=-0.2,
    max_shaping_pawns=0.4,
    contempt=-8,           # content to hold draws when under pressure
    tags=("defensive", "solid"),
)

_OPEN_GAME = StyleProfile(
    name="open",
    description="Steers toward open files, diagonals, and piece activity in "
                "tactically rich positions.",
    openness=1.0,
    mobility=0.6,
    king_attack=0.3,
    max_shaping_pawns=0.4,
    tags=("open", "tactical"),
)

_CLOSED_GAME = StyleProfile(
    name="closed",
    description="Prefers pawn chains, space, and long maneuvering. Avoids early "
                "liquidation.",
    openness=-1.0,
    king_safety=0.4,
    simplify=-0.3,
    max_shaping_pawns=0.4,
    tags=("closed", "positional"),
)

_ENDGAME_GRINDER = StyleProfile(
    name="grinder",
    description="Simplifies into favorable endings and wins on technique.",
    simplify=0.8,
    king_safety=0.3,
    max_shaping_pawns=0.35,
    contempt=15,           # avoids draws, tries to convert small edges
    tags=("positional", "technical"),
)


PROFILES: dict[str, StyleProfile] = {
    p.name: p for p in (
        _BALANCED,
        _SACRIFICER,
        _ATTACKER,
        _DEFENDER,
        _OPEN_GAME,
        _CLOSED_GAME,
        _ENDGAME_GRINDER,
    )
}

DEFAULT_PROFILE = "balanced"
STYLE_NAMES = list(PROFILES)
