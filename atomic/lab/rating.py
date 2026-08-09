"""Put every method on one Elo scale, with honest error bars.

Two steps, deliberately separated:

1. **Calibrate the ladder** (`calibrate`). The Fairy-Stockfish rungs play each
   other; we fit their Elo differences by maximum likelihood and pin the scale
   by fixing the reference rung. This happens once and is cached in
   `results/ladder.json`. Without it, "Elo" would be an unbacked number.

2. **Rate a method** (`rate`). Fit the one unknown rating that best explains the
   method's scores against the now-known rungs, and report a likelihood-profile
   confidence interval.

The scale is anchored by `REFERENCE_RUNG` / `REFERENCE_ELO`, an arbitrary but
fixed choice. Only *differences* on this scale are meaningful; do not compare
these numbers to Lichess or CCRL ratings. That is the point — the project asks
"which method is better", and a self-consistent internal scale answers it with
far less noise than any external anchor could.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from . import config

#: The rung whose rating defines the origin of the scale, and the value it gets.
REFERENCE_RUNG = 64_000
REFERENCE_ELO = 2000.0

LADDER_FILE = config.RESULTS / "ladder.json"


def expected_score(rating: float, opponent: float) -> float:
    """Logistic Elo expectation."""
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


@dataclass(frozen=True)
class Opponent:
    """A rated opponent and the score achieved against it."""

    name: str
    rating: float
    score: float   # points scored by the player being rated
    games: int


@dataclass(frozen=True)
class Rating:
    elo: float
    low: float
    high: float
    games: int
    score: float
    bounded: str = ""   # "below"/"above" when the result is a shutout

    def __str__(self) -> str:
        if self.bounded == "above":
            return f"> {self.low:.0f} Elo (won every game, {self.games} games)"
        if self.bounded == "below":
            return f"< {self.high:.0f} Elo (lost every game, {self.games} games)"
        span = (self.high - self.low) / 2
        return f"{self.elo:.0f} +/- {span:.0f} Elo ({self.score}/{self.games})"


def _log_likelihood(rating: float, opponents: Sequence[Opponent]) -> float:
    """Log-likelihood of the observed scores, treating them as binomial.

    Draws count as half a win against half a game, the standard Elo treatment.
    """
    total = 0.0
    for opp in opponents:
        p = min(max(expected_score(rating, opp.rating), 1e-12), 1 - 1e-12)
        wins, losses = opp.score, opp.games - opp.score
        total += wins * math.log(p) + losses * math.log(1 - p)
    return total


def _mle(opponents: Sequence[Opponent], lo: float = -2000.0, hi: float = 6000.0) -> float:
    """Maximum-likelihood rating by ternary search (the likelihood is unimodal)."""
    for _ in range(200):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if _log_likelihood(m1, opponents) < _log_likelihood(m2, opponents):
            lo = m1
        else:
            hi = m2
    return (lo + hi) / 2


def _profile_interval(
    opponents: Sequence[Opponent], best: float, drop: float = 1.92
) -> tuple[float, float]:
    """Likelihood-profile interval: where log-likelihood falls `drop` below peak.

    `drop = 1.92` is the 95% level for one parameter (half a chi-square
    quantile). Profile intervals are used rather than a normal approximation
    because gauntlet samples are small and often lopsided, where the normal
    approximation is badly wrong.
    """
    peak = _log_likelihood(best, opponents)
    target = peak - drop

    def bisect(lo: float, hi: float) -> float:
        for _ in range(200):
            mid = (lo + hi) / 2
            if _log_likelihood(mid, opponents) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return bisect(best, best - 4000.0), bisect(best, best + 4000.0)


def rate(opponents: Sequence[Opponent]) -> Rating:
    """Fit one rating from scores against rated opponents."""
    total_games = sum(o.games for o in opponents)
    total_score = sum(o.score for o in opponents)
    if total_games == 0:
        raise ValueError("no games to rate")

    strongest = max(o.rating for o in opponents)
    weakest = min(o.rating for o in opponents)

    if total_score == 0:
        # Lost everything: only an upper bound is identified. Use the
        # conventional "as if it had scored half a game" correction.
        padded = [
            Opponent(o.name, o.rating, o.score + (0.5 if o.rating == weakest else 0), o.games)
            for o in opponents
        ]
        best = _mle(padded)
        return Rating(best, best, best, total_games, total_score, bounded="below")
    if total_score == total_games:
        padded = [
            Opponent(
                o.name, o.rating, o.score - (0.5 if o.rating == strongest else 0), o.games
            )
            for o in opponents
        ]
        best = _mle(padded)
        return Rating(best, best, best, total_games, total_score, bounded="above")

    best = _mle(opponents)
    low, high = _profile_interval(opponents, best)
    return Rating(best, low, high, total_games, total_score)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def load_ladder() -> dict[int, float]:
    """Cached rung ratings, or a theoretical prior if not yet calibrated.

    The prior comes from the standard doubling law: each doubling of nodes is
    worth a roughly constant Elo. 70 Elo/doubling is a conservative figure for a
    modern engine at these node counts. Calibrating replaces it with measurement
    — always calibrate before quoting a number you care about.
    """
    if LADDER_FILE.exists():
        raw = json.loads(LADDER_FILE.read_text(encoding="utf-8"))
        return {int(k): float(v) for k, v in raw["ratings"].items()}

    from . import fsf

    elo_per_doubling = 70.0
    return {
        rung: REFERENCE_ELO
        + elo_per_doubling * math.log2(rung / REFERENCE_RUNG)
        for rung in fsf.LADDER
    }


def save_ladder(ratings: dict[int, float], note: str = "") -> None:
    LADDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LADDER_FILE.write_text(
        json.dumps(
            {
                "reference_rung": REFERENCE_RUNG,
                "reference_elo": REFERENCE_ELO,
                "note": note,
                "ratings": {str(k): v for k, v in sorted(ratings.items())},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def is_calibrated() -> bool:
    return LADDER_FILE.exists()


def rate_from_matches(matches, ladder: dict[int, float] | None = None) -> Rating:
    """Turn a list of `arena.MatchResult` against FSF rungs into one Rating."""
    ladder = ladder or load_ladder()
    opponents: list[Opponent] = []
    for match in matches:
        rung = _rung_of(match.player_b)
        if rung is None or rung not in ladder:
            continue
        opponents.append(
            Opponent(match.player_b, ladder[rung], match.score_a, len(match.games))
        )
    if not opponents:
        raise ValueError("no matches against known ladder rungs")
    return rate(opponents)


def _rung_of(player_name: str) -> int | None:
    """Recover the node count from an `FsfConfig.name` like 'fsf-nnue-64kn'."""
    if not player_name.startswith("fsf-"):
        return None
    tail = player_name.rsplit("-", 1)[-1]
    if not tail.endswith("kn"):
        return None
    try:
        return int(tail[:-2]) * 1000
    except ValueError:
        return None
