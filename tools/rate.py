"""Realistic Elo rating for an NNUE net, anchored to Stockfish.

`tools/match.py` tells you "net A beats net B" but gives no *absolute* number.
This tool pegs your net against Stockfish 18 with its strength capped to known
Elo levels (UCI_LimitStrength + UCI_Elo, range 1320-3190), then solves for the
net's rating from the match scores. The level where your net scores ~50% is its
rating; we interpolate between the bracketing levels and attach a confidence
interval from the score's standard error.

Every rating is appended to `data/ratings.csv` together with the net's training
metadata (positions, label depth, lambda, epochs — read from the .meta.json
sidecar written by train.py) and the time control, so you can watch
"more/better data -> higher Elo" over your v1..vN nets.

    # Rate a net (auto time control 0.2s/move, ladder around a first guess)
    python -m tools.rate --nnue data/nnue_v2.pt --version v2 --games 24

    # Wider/faster ladder, longer thinking time
    python -m tools.rate --nnue data/nnue.pt --version v1 \
        --games 40 --movetime 0.3 --levels 1320,1500,1700,1900,2100

Notes on realism:
- The Elo is only meaningful *at this time control* — it is recorded with every
  row. Change movetime and the number moves.
- Stockfish's UCI_Elo is itself approximate (±100ish, blitz-skewed) but
  *consistent*, which is what you need to track progress.
- Use enough games (>=24) or the CI will be too wide to trust a small gain.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import chess
import chess.engine

from engine.board import Engine
from engine.nnue import meta

SF_PATH = os.path.join(os.path.dirname(__file__), "stockfish.exe")
SF_LOW, SF_HIGH = 1320, 3190  # Stockfish UCI_Elo bounds (SF18)
RATINGS_CSV = os.path.join("data", "ratings.csv")


# --- one game: your net vs Stockfish-at-a-fixed-Elo ------------------------

def play_vs_sf(net: Engine, sf: chess.engine.SimpleEngine, sf_elo: int,
               net_is_white: bool, movetime: float,
               max_plies: int = 200) -> float:
    """Return the net's score for one game: 1.0 win, 0.5 draw, 0.0 loss."""
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(sf_elo)})
    board = chess.Board()
    limit = chess.engine.Limit(time=movetime)
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        net_to_move = (board.turn == chess.WHITE) == net_is_white
        if net_to_move:
            net.board = board.copy()
            info = net.best_move(depth=8, movetime=movetime)
            move = info.best_move
        else:
            move = sf.play(board, limit).move
        if move is None:
            break
        board.push(move)

    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return 0.5
    net_won = (o.winner == chess.WHITE) == net_is_white
    return 1.0 if net_won else 0.0


def match_vs_sf(net_ckpt: str, style: str, sf: chess.engine.SimpleEngine,
                sf_elo: int, games: int, movetime: float) -> tuple[float, int]:
    """Play `games` color-balanced games vs SF at `sf_elo`.
    Returns (net_score_fraction, games_played)."""
    total = 0.0
    for g in range(games):
        net = Engine(style=style, nnue_checkpoint=net_ckpt)  # fresh TT each game
        net_white = (g % 2 == 0)
        s = play_vs_sf(net, sf, sf_elo, net_white, movetime)
        total += s
        res = {1.0: "win", 0.5: "draw", 0.0: "loss"}[s]
        side = "W" if net_white else "B"
        print(f"    vs SF{sf_elo}  game {g+1}/{games} ({side}): {res}"
              f"   running {total}/{g+1}", flush=True)
    return total / games, games


# --- score <-> Elo-difference math -----------------------------------------

def elo_diff_from_score(score: float, n: int) -> tuple[float, float, float]:
    """Given a score fraction vs an opponent, return
    (elo_diff, lo, hi) where elo_diff>0 means the net is stronger, and
    [lo, hi] is a ~95% CI on the diff. Clamps extreme scores so the log is finite.
    """
    p = min(max(score, 0.5 / (n + 1)), 1 - 0.5 / (n + 1))  # keep off 0/1
    diff = -400.0 * math.log10(1.0 / p - 1.0)
    # Standard error of the mean score, then map score±1.96se to Elo.
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    lo_p = min(max(p - 1.96 * se, 1e-4), 1 - 1e-4)
    hi_p = min(max(p + 1.96 * se, 1e-4), 1 - 1e-4)
    lo = -400.0 * math.log10(1.0 / lo_p - 1.0)
    hi = -400.0 * math.log10(1.0 / hi_p - 1.0)
    return diff, lo, hi


# --- driver ----------------------------------------------------------------

def rate(net_ckpt: str, style: str, levels: list[int], games: int,
         movetime: float) -> dict:
    if not os.path.exists(SF_PATH):
        raise SystemExit(f"Stockfish not found at {SF_PATH}")

    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    try:
        results = []  # (sf_elo, net_score, n)
        for lvl in levels:
            lvl = max(SF_LOW, min(SF_HIGH, lvl))
            print(f"  --- Stockfish @ {lvl} Elo ---", flush=True)
            score, n = match_vs_sf(net_ckpt, style, sf, lvl, games, movetime)
            diff, _lo, _hi = elo_diff_from_score(score, n)
            est = lvl + diff
            print(f"  SF{lvl}: net scored {score*100:.0f}%  "
                  f"=> net ~ {est:.0f} Elo", flush=True)
            results.append((lvl, score, n))
    finally:
        sf.quit()

    # Combined estimate: average of (level + elo_diff) across levels, weighted
    # toward the levels closest to 50% (most informative), and a CI from the
    # single most-balanced level.
    ests, weights = [], []
    for lvl, score, n in results:
        diff, _lo, _hi = elo_diff_from_score(score, n)
        w = 1.0 - abs(score - 0.5) * 2.0 * 0.8   # near-50% weighted highest
        ests.append(lvl + diff)
        weights.append(max(w, 0.2))
    elo = sum(e * w for e, w in zip(ests, weights)) / sum(weights)

    # CI from the most balanced level (closest to 50%).
    best = min(results, key=lambda r: abs(r[1] - 0.5))
    b_lvl, b_score, b_n = best
    diff, lo, hi = elo_diff_from_score(b_score, b_n)
    ci = (abs(hi - lo)) / 2.0

    return {
        "elo": round(elo),
        "ci": round(ci),
        "per_level": [(l, s) for (l, s, _n) in results],
        "games_per_level": games,
    }


def append_log(row: dict) -> None:
    os.makedirs(os.path.dirname(RATINGS_CSV) or ".", exist_ok=True)
    new = not os.path.exists(RATINGS_CSV)
    cols = ["date", "version", "net", "elo", "ci", "movetime_s",
            "games_per_level", "levels", "per_level_scores",
            "positions", "label_depth", "lambda", "epochs", "best_val", "style"]
    with open(RATINGS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnue", required=True, help="net checkpoint to rate")
    ap.add_argument("--version", default="",
                    help="version tag for the ratings log, e.g. v2")
    ap.add_argument("--style", default="balanced")
    ap.add_argument("--games", type=int, default=24,
                    help="games per Stockfish level (>=24 recommended)")
    ap.add_argument("--movetime", type=float, default=0.2,
                    help="seconds/move for BOTH sides (the time control)")
    ap.add_argument("--levels", default="1320,1500,1700,1900",
                    help="comma-separated Stockfish Elo levels to test")
    ap.add_argument("--no-log", action="store_true",
                    help="don't append to data/ratings.csv")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    print(f"Rating {args.nnue} (style={args.style}) at {args.movetime*1000:.0f}"
          f"ms/move vs Stockfish levels {levels}\n", flush=True)

    t0 = time.perf_counter()
    r = rate(args.nnue, args.style, levels, args.games, args.movetime)
    dt = time.perf_counter() - t0

    md = meta.read(args.nnue)
    print("\n" + "=" * 56)
    print(f"  {args.version or os.path.basename(args.nnue)}: "
          f"{r['elo']} Elo  (±{r['ci']}, {args.movetime*1000:.0f}ms/move)")
    if md:
        print(f"  trained on {md.get('positions','?')} positions, "
              f"label depth {md.get('label_depth','?')}, "
              f"lambda {md.get('lambda','?')}, epochs {md.get('epochs','?')}")
    print(f"  per level: " +
          "  ".join(f"SF{l}:{s*100:.0f}%" for l, s in r["per_level"]))
    print(f"  ({dt/60:.1f} min)")
    print("=" * 56)

    if not args.no_log:
        append_log({
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "version": args.version,
            "net": args.nnue,
            "elo": r["elo"],
            "ci": r["ci"],
            "movetime_s": args.movetime,
            "games_per_level": r["games_per_level"],
            "levels": ";".join(str(l) for l, _ in r["per_level"]),
            "per_level_scores": ";".join(f"{s:.2f}" for _, s in r["per_level"]),
            "positions": md.get("positions", ""),
            "label_depth": md.get("label_depth", ""),
            "lambda": md.get("lambda", ""),
            "epochs": md.get("epochs", ""),
            "best_val": md.get("best_val", ""),
            "style": args.style,
        })
        print(f"logged -> {RATINGS_CSV}")


if __name__ == "__main__":
    main()
