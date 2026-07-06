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

SF_PATH = os.path.abspath(os.path.join("tools", "stockfish.exe"))
EVAL_CLAMP = 2000  # centipawns; mates/huge evals clamped to keep targets sane


def _sf_eval_cp(engine: chess.engine.SimpleEngine, board: chess.Board,
                depth: int) -> int:
    """White-relative centipawn eval, clamped. Mate scores mapped to +/-CLAMP."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()
    cp = score.score(mate_score=EVAL_CLAMP)
    return max(-EVAL_CLAMP, min(EVAL_CLAMP, cp))


def _play_one_game(engine: chess.engine.SimpleEngine, depth: int,
                   rng: random.Random, max_plies: int = 200):
    """Play a lightly-randomized game; yield (board_copy, eval_cp) for each
    quiet-ish position. Returns the final result string."""
    board = chess.Board()
    samples: list[tuple[str, int]] = []
    # A few random plies ONLY for opening diversity (kept small so early
    # positions stay realistic). Every non-check position is labeled -- including
    # the opening -- so the net learns development and tempo, not just midgames.
    opening_random = rng.randint(1, 4)

    for ply in range(max_plies):
        if board.is_game_over():
            break

        # Label the current position (skip in-check positions -> noisy targets).
        if not board.is_check():
            cp = _sf_eval_cp(engine, board, depth)
            samples.append((board.fen(), cp))

        if ply < opening_random:
            move = rng.choice(list(board.legal_moves))
        else:
            # Occasionally inject a random move mid-game for state diversity, but
            # mostly follow Stockfish so games and evals stay realistic.
            if rng.random() < 0.05:
                move = rng.choice(list(board.legal_moves))
            else:
                result = engine.play(board, chess.engine.Limit(depth=depth))
                move = result.move or rng.choice(list(board.legal_moves))
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
    try:
        with open(shard, "w", encoding="utf-8") as f:
            for g in range(n_games):
                samples, result = _play_one_game(engine, depth, rng)
                for fen, cp in samples:
                    f.write(f"{fen}\t{cp}\t{result}\n")
                    written += 1
                if (g + 1) % 5 == 0:
                    f.flush()
                    print(f"[worker {wid}] game {g+1}/{n_games} "
                          f"({written} samples)", flush=True)
    finally:
        engine.quit()
    return f"{shard}:{written}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="total games")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--depth", type=int, default=9, help="Stockfish label depth")
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
    print(f"Generating ~{per_worker*args.workers} games across "
          f"{args.workers} workers, Stockfish depth {args.depth}...")

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, tasks)
    dt = time.perf_counter() - t0

    total = sum(int(r.split(":")[1]) for r in results)
    print(f"Done: {total} samples in {dt:.0f}s "
          f"({total/max(dt,1):.0f} samples/s). Shards:")
    for r in results:
        print("  ", r)


if __name__ == "__main__":
    main()
