"""Fit the atomic cp -> win% curve, from real games.

    python -m trainer.fit_winrate --month 2026-07 --positions 12000

The experiment: sample positions from rated atomic games, evaluate each with
Fairy-Stockfish + the atomic net at a fixed node count, and regress the
evaluation against **what actually happened in that game**. The result is the
conversion every judgement in the trainer depends on.

Protocol, and why each part is not optional:

- **Positions are sampled from many games, few per game.** The statistical unit
  is the game result, not the position: twenty positions from one game carry
  roughly one game's worth of information, and treating them as twenty
  independent observations would shrink the confidence interval by a factor of
  4.5 for free. Default is 3 positions from each of ~4000 games.
- **Time forfeits and abandoned games are excluded** (`Termination: Normal`
  only). The result of a flag fall says nothing about the position, and this
  curve is a statement about positions.
- **Both players inside a rating band.** Win probability given an evaluation is
  a function of who is playing it; a curve fitted on 900-rated games would not
  describe the user's. Default band brackets the account this was built for.
- **Two parameters, not one.** `win% = 100/(1 + exp(-(a + k*cp)))` — the
  intercept is fitted rather than assumed, because if the atomic net carries a
  systematic offset (this repo has already measured that it scores the *start
  position* at +250 to +310cp) then forcing the curve through 50% at 0cp would
  bury that bias inside every judgement instead of reporting it.
- **Reported against the standard-chess curve it replaces**, in log-loss, so
  "the standard constant is wrong for atomic" is a measurement rather than an
  assertion.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import chess
import chess.variant
import numpy as np

from lab import config

from . import analysis, dump, store, winrate

#: Evaluations are clamped before fitting. Beyond this the game is decided and
#: the exact number is engine noise; leaving it unclamped lets a handful of
#: +40000 mate scores drag the slope.
CLAMP_CP = 1500

RESULTS = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


class Sample(NamedTuple):
    fen: str
    #: Result from the point of view of the side to move in `fen`.
    y: float
    game_id: str


def sample_positions(
    month: str,
    *,
    positions: int,
    per_game: int,
    band: tuple[int, int],
    seed: int,
    min_ply: int = 10,
) -> list[Sample]:
    """Reservoir-sample games from a month's dump, then plies from each game.

    Reservoir rather than "take the first N": games are ordered by date within
    a month and the population of players changes over it, so the first N are
    not a sample of the month.
    """
    rng = random.Random(seed)
    # Oversample: short games yield fewer than `per_game` positions and some
    # movetext fails to parse, so the reservoir has to be larger than the
    # arithmetic suggests or the run comes back short.
    wanted_games = math.ceil(positions / per_game * 1.35)
    path = dump.fetch(month)
    reservoir: list[dump.RawGame] = []
    seen = 0
    started = time.time()

    for game in dump.read(path):
        headers = game.headers
        if headers.get("Termination") != "Normal":
            continue
        if headers.get("Result") not in RESULTS:
            continue
        white, black = game.elo("White"), game.elo("Black")
        if white is None or black is None:
            continue
        if not (band[0] <= white <= band[1] and band[0] <= black <= band[1]):
            continue
        seen += 1
        if len(reservoir) < wanted_games:
            reservoir.append(game)
        else:
            index = rng.randrange(seen)
            if index < wanted_games:
                reservoir[index] = game
        if seen % 20000 == 0:
            print(f"  scanned {seen:,} eligible games ({time.time() - started:.0f}s)", end="\r")

    print(f"  {seen:,} eligible games in {month}; sampled {len(reservoir)}          ")

    out: list[Sample] = []
    for game in reservoir:
        moves = game.moves()
        if len(moves) < min_ply:
            continue
        # Skip the opening book-ish first plies and the final couple, where the
        # position is either identical across thousands of games or already over.
        low, high = 4, len(moves) - 2
        if high <= low:
            continue
        choices = rng.sample(range(low, high), min(per_game, high - low))
        board = chess.variant.AtomicBoard()
        white_score = RESULTS[game.headers["Result"]]
        wanted = set(choices)
        try:
            for ply, san in enumerate(moves):
                if ply in wanted:
                    y = white_score if board.turn == chess.WHITE else 1.0 - white_score
                    out.append(Sample(board.fen(), y, game.headers.get("Site", "?")))
                board.push_san(san)
        except ValueError:
            continue  # malformed movetext; drop the game rather than guess
    rng.shuffle(out)
    return out[:positions]


def evaluate(samples: list[Sample], nodes: int, workers: int) -> list[tuple[float, float]]:
    """Evaluate each sampled position, returning (cp, y) pairs.

    Results go through the shared analysis cache, so a re-run — or a later
    puzzle pass over the same positions — costs nothing.
    """
    out: list[tuple[float, float]] = []
    done = 0
    started = time.time()
    with store.open_db() as db, analysis.EnginePool(nodes) as pool, ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        cached = {}
        todo: list[Sample] = []
        queued: set[str] = set()
        for sample in samples:
            if sample.fen in cached or sample.fen in queued:
                continue  # sampled twice: openings repeat, and games transpose
            lines = store.cached_analysis(db, sample.fen, nodes, 1, analysis.ENGINE_ID)
            if lines:
                cached[sample.fen] = [analysis.Line.from_dict(item) for item in lines]
            else:
                todo.append(sample)
                queued.add(sample.fen)
        print(f"  {len(cached)} positions already cached, {len(todo)} to evaluate")

        futures = {
            executor.submit(analysis.analyse_fen, pool, sample.fen, 1): sample
            for sample in todo
        }
        for future, sample in futures.items():
            lines, depth = future.result()
            store.store_analysis(
                db, sample.fen, nodes, 1, analysis.ENGINE_ID,
                [line.as_dict() for line in lines], depth, time.time(),
            )
            cached[sample.fen] = lines
            done += 1
            if done % 200 == 0:
                db.commit()
                rate = done / max(time.time() - started, 1e-9)
                print(
                    f"  evaluated {done}/{len(todo)}  {rate:.1f}/s  "
                    f"~{(len(todo) - done) / max(rate, 1e-9) / 60:.0f}m left",
                    end="\r",
                )
        db.commit()

    for sample in samples:
        lines = cached.get(sample.fen)
        if not lines:
            continue
        line = lines[0]
        if line.mate is not None:
            cp = CLAMP_CP if line.mate > 0 else -CLAMP_CP
        elif line.cp is None:
            continue
        else:
            cp = max(-CLAMP_CP, min(CLAMP_CP, line.cp))
        out.append((float(cp), sample.y))
    print(f"  evaluated {len(out)} positions in {(time.time() - started) / 60:.1f} minutes")
    return out


def fit(pairs: list[tuple[float, float]], iterations: int = 50) -> tuple[float, float]:
    """MLE for `p = sigmoid(a + k*cp)` by IRLS. Returns (a, k)."""
    cp = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    # Fit on cp/1000 so the two columns are on comparable scales, then rescale.
    x = np.column_stack([np.ones_like(cp), cp / 1000.0])
    beta = np.zeros(2)
    for _ in range(iterations):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        z = eta + (y - p) / w
        xtw = x.T * w
        try:
            beta_next = np.linalg.solve(xtw @ x, xtw @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_next - beta)) < 1e-10:
            beta = beta_next
            break
        beta = beta_next
    return float(beta[0]), float(beta[1] / 1000.0)


def log_loss(pairs: list[tuple[float, float]], a: float, k: float) -> float:
    cp = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    p = 1.0 / (1.0 + np.exp(-(a + k * cp)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(pairs: list[tuple[float, float]], a: float, k: float, buckets: int = 10) -> list[dict]:
    """Observed vs predicted score, by evaluation bucket. The honesty check."""
    ordered = sorted(pairs)
    size = max(1, len(ordered) // buckets)
    rows = []
    for start in range(0, len(ordered), size):
        chunk = ordered[start : start + size]
        if len(chunk) < size // 2:
            break
        cps = [c for c, _ in chunk]
        observed = sum(y for _, y in chunk) / len(chunk)
        predicted = sum(1.0 / (1.0 + math.exp(-(a + k * c))) for c in cps) / len(chunk)
        rows.append(
            {
                "cp_low": min(cps),
                "cp_high": max(cps),
                "n": len(chunk),
                "observed": observed * 100,
                "predicted": predicted * 100,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2026-07", help="dump month, YYYY-MM")
    parser.add_argument("--positions", type=int, default=12_000)
    parser.add_argument("--per-game", type=int, default=3)
    parser.add_argument("--band", type=int, nargs=2, default=(1400, 2200))
    parser.add_argument("--nodes", type=int, default=analysis.DEFAULT_NODES)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dry-run", action="store_true", help="sample only, no engine")
    args = parser.parse_args(argv)

    workers = args.workers or analysis.default_workers()
    print(
        f"fitting the atomic win% curve: {args.positions} positions, "
        f"{args.per_game}/game, Elo {args.band[0]}-{args.band[1]}, "
        f"{args.nodes:,} nodes, {workers} workers"
    )
    samples = sample_positions(
        args.month,
        positions=args.positions,
        per_game=args.per_game,
        band=tuple(args.band),
        seed=args.seed,
    )
    print(f"  {len(samples)} positions sampled")
    if args.dry_run:
        return 0

    pairs = evaluate(samples, args.nodes, workers)
    if len(pairs) < 500:
        raise SystemExit(f"only {len(pairs)} usable positions; refusing to fit")

    a, k = fit(pairs)
    fitted_loss = log_loss(pairs, a, k)
    standard_loss = log_loss(pairs, 0.0, winrate.LICHESS_STANDARD_K)
    # Robustness: does the curve change if the decided positions are dropped?
    near = [p for p in pairs if abs(p[0]) <= 800]
    a_near, k_near = fit(near)

    print()
    print(f"  fitted:   a={a:+.5f}  k={k:.8f}   -> even at {-a / k:+.0f} cp")
    print(f"  |cp|<800: a={a_near:+.5f}  k={k_near:.8f}   -> even at {-a_near / k_near:+.0f} cp")
    print(f"  log-loss: {fitted_loss:.4f} fitted vs {standard_loss:.4f} lichess-standard")
    print(f"            ({(standard_loss - fitted_loss) / standard_loss:+.1%} better)")
    print()
    print("  calibration (observed vs predicted score by eval bucket):")
    print(f"    {'cp range':>16}  {'n':>5}  {'observed':>8}  {'predicted':>9}")
    rows = calibration(pairs, a, k)
    for row in rows:
        print(
            f"    {row['cp_low']:>7.0f}..{row['cp_high']:<7.0f} {row['n']:>5}  "
            f"{row['observed']:>7.1f}%  {row['predicted']:>8.1f}%"
        )

    model = winrate.WinModel(
        a=a,
        k=k,
        fitted=True,
        provenance={
            "month": args.month,
            "positions": len(pairs),
            "per_game": args.per_game,
            "band": list(args.band),
            "nodes": args.nodes,
            "engine": analysis.ENGINE_ID,
            "clamp_cp": CLAMP_CP,
            "seed": args.seed,
            "even_cp": -a / k,
            "log_loss": fitted_loss,
            "log_loss_lichess_standard": standard_loss,
            "a_near": a_near,
            "k_near": k_near,
            "fitted_at": time.strftime("%Y-%m-%d"),
        },
    )
    winrate.save(model)
    out = config.RESULTS / "winrate_fit.json"
    out.write_text(
        json.dumps(
            {"model": {"a": a, "k": k}, "provenance": model.provenance, "calibration": rows},
            indent=2,
        )
    )
    print(f"\n  saved {winrate.MODEL_PATH.name} and {out.name}")
    print("  now: python -m trainer.cli rejudge   (free — re-scores from the cache)")

    print("\n  what this changes, in the numbers this repo already published:")
    for cp in (0, 100, 250, 308, 500, 1000):
        print(
            f"    {cp:>5} cp  ->  {model.win_pct(cp):5.1f}% atomic   "
            f"(standard curve said {winrate.fallback_model().win_pct(cp):5.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
