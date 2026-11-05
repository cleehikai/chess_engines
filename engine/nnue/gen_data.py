"""Generate NNUE training data: (position, stockfish_eval, game_result).

Positions are produced by lightly-randomized self-play so the dataset covers a
broad, realistic distribution (not just opening theory). Each position is
labeled by Stockfish at a shallow fixed depth (the standard NNUE recipe uses
depth ~9 / a few thousand nodes).

Output format: a simple line-oriented text file, one sample per line:
    <fen>\t<eval_cp>\t<result>
where eval_cp is white-relative centipawns (clamped) and result in {1,0,-1}
(white win/draw/loss) once the generating game finishes. This is easy to inspect
and load; we convert to tensors in train.py.

Parallelized across CPU cores with multiprocessing: each worker runs its own
Stockfish process and appends to its own shard file.

Usage:
    python -m engine.nnue.gen_data --games 500 --workers 8 --depth 9 \
        --out data/train
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time

import chess
import chess.engine
import chess.polyglot

from tools.estimate import announce

SF_PATH = os.path.abspath(os.path.join("tools", "stockfish.exe"))
EVAL_CLAMP = 2000  # centipawns; mates/huge evals clamped to keep targets sane


def _sf_eval_cp(engine: chess.engine.SimpleEngine, board: chess.Board,
                depth: int) -> tuple[int, chess.Move | None]:
    """White-relative centipawn eval (clamped) AND Stockfish's best move.

    The best move lets the caller filter out *tactical* positions (where the top
    move is a capture): those targets are dominated by the swing of an immediate
    exchange and teach the net exchange arithmetic rather than positional
    understanding. Labeling only quiet positions is the standard NNUE recipe and
    improves label quality more per position than raising depth alone.
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    cp = score.score(mate_score=EVAL_CLAMP)
    cp = max(-EVAL_CLAMP, min(EVAL_CLAMP, cp))
    pv = info.get("pv")
    best = pv[0] if pv else None
    return cp, best


def _play_one_game(engine: chess.engine.SimpleEngine, depth: int,
                   rng: random.Random, seen: set[int], max_plies: int = 200):
    """Play a lightly-randomized game; yield (fen, eval_cp) for each quiet
    position. `seen` is a shared set of zobrist hashes for dedup across the
    worker's games. Returns (samples, result)."""
    board = chess.Board()
    samples: list[tuple[str, int]] = []
    # A few random plies ONLY for opening diversity (kept small so early
    # positions stay realistic). Every quiet position is labeled -- including
    # the opening -- so the net learns development and tempo, not just midgames.
    opening_random = rng.randint(1, 4)

    for ply in range(max_plies):
        if board.is_game_over():
            break

        # Reuse Stockfish's analysis for both the label AND the game move, so we
        # don't pay for two searches per ply.
        cp, best = _sf_eval_cp(engine, board, depth)

        # Quiet-position filter: label only positions that are not in check and
        # whose best move is not a capture/promotion. Dedup by zobrist hash so
        # near-duplicate positions (common in low-randomness self-play) don't
        # dominate the dataset and inflate the count without adding signal.
        quiet = (not board.is_check() and best is not None
                 and not board.is_capture(best) and not best.promotion)
        if quiet:
            key = chess.polyglot.zobrist_hash(board)
            if key not in seen:
                seen.add(key)
                samples.append((board.fen(), cp))

        if ply < opening_random:
            move = rng.choice(list(board.legal_moves))
        elif rng.random() < 0.05:
            # Occasionally inject a random move mid-game for state diversity.
            move = rng.choice(list(board.legal_moves))
        else:
            move = best or rng.choice(list(board.legal_moves))
        board.push(move)

    outcome = board.outcome()
    if outcome is None:
        result_val = 0
    elif outcome.winner is None:
        result_val = 0
    else:
        result_val = 1 if outcome.winner == chess.WHITE else -1
    return samples, result_val


def _worker(args) -> str:
    wid, n_games, depth, seed, out_prefix = args
    rng = random.Random(seed)
    shard = f"{out_prefix}.shard{wid}.tsv"
    engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    engine.configure({"Threads": 1, "Hash": 64})
    written = 0
    seen: set[int] = set()  # zobrist hashes for cross-game dedup in this worker
    try:
        with open(shard, "w", encoding="utf-8") as f:
            for g in range(n_games):
                samples, result = _play_one_game(engine, depth, rng, seen)
                for fen, cp in samples:
                    f.write(f"{fen}\t{cp}\t{result}\n")
                    written += 1
                if (g + 1) % 5 == 0:
                    f.flush()
                    print(f"[worker {wid}] game {g+1}/{n_games} "
                          f"({written} samples)", flush=True)
    finally:
        engine.quit()
    # Return count separately from the path -- Windows paths contain ':' (drive
    # letter), so packing them into one "path:count" string and splitting on ':'
    # mis-parses. Return a tuple instead.
    return (shard, written)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="total games")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--depth", type=int, default=12,
                    help="Stockfish label depth (higher = stronger teacher = "
                         "stronger net; 12-16 recommended, was 9)")
    ap.add_argument("--out", default=os.path.join("data", "train"),
                    help="output path prefix")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not os.path.exists(SF_PATH):
        raise SystemExit(f"Stockfish not found at {SF_PATH}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    per_worker = max(1, args.games // args.workers)
    tasks = [
        (wid, per_worker, args.depth, args.seed + wid * 7919, args.out)
        for wid in range(args.workers)
    ]
    n_games = per_worker * args.workers
    print(f"Generating ~{n_games} games across "
          f"{args.workers} workers, Stockfish depth {args.depth}...")

    # Calibration (measured on this machine, 6 physical / 12 logical cores).
    # The estimate models AGGREGATE throughput (games/hr for the whole pool),
    # NOT per-worker cost -- because those move in opposite directions and only
    # the aggregate matters for wall-clock. Each worker slows down as you add
    # more (hyperthreading contention past 6 physical cores), but total
    # throughput still RISES monotonically with worker count. Measured games/hr
    # at depth 10 (48-game head-to-head batches):
    #     4w=5850  6w=5904  8w=6609  12w=7432   -> more workers = faster overall.
    # (An earlier version modeled per-worker contention and produced a
    #  non-monotonic ETA where 6w looked slower than 4w -- an interpolation
    #  artifact, since real throughput is monotonic. Fixed here.)
    #
    # Depth: aggregate cost grows geometrically, ~1.65 per ply (measured solo
    # s/game d9=1.12, d10=1.87, d12=5.04 -> (5.04/1.12)**(1/3) ~= 1.65).
    #
    # Reproduces the real 40k-game / d10 / 10-worker run (~4.8 hr) to +~18%,
    # erring LONG so the ETA never under-promises.
    DEPTH_BRANCHING = 1.65
    # (workers, measured games/hr) at depth 10. Interpolated/extrapolated below.
    _GAMES_PER_HR_D10 = [(4, 5850), (6, 5904), (8, 6609), (12, 7432)]

    def _games_per_hr_d10(w: int) -> float:
        pts = _GAMES_PER_HR_D10
        if w <= pts[0][0]:
            return pts[0][1]
        if w >= pts[-1][0]:
            return pts[-1][1]  # clamp: no gain modeled past max measured
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= w <= x1:
                return y0 + (y1 - y0) * (w - x0) / (x1 - x0)
        return pts[-1][1]

    # Aggregate seconds per game at depth 10 for this worker count, then scale
    # by the geometric depth factor (relative to the d10 measurement point).
    agg_s_per_game_d10 = 3600.0 / _games_per_hr_d10(args.workers)
    depth_factor = DEPTH_BRANCHING ** (args.depth - 10)
    est_s = n_games * agg_s_per_game_d10 * depth_factor
    announce(
        f"Data-gen (~{n_games} games, depth {args.depth})", est_s,
        f"~{_games_per_hr_d10(args.workers):.0f} games/hr @ d10 x {args.workers}w"
        f" x {DEPTH_BRANCHING}^(d-10)={depth_factor:.2f} depth factor")

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, tasks)
    dt = time.perf_counter() - t0

    total = sum(count for _shard, count in results)
    print(f"Done: {total} samples in {dt:.0f}s "
          f"({total/max(dt,1):.0f} samples/s). Shards:")
    for shard, count in results:
        print(f"   {shard}: {count}")


if __name__ == "__main__":
    main()
