"""Quick engine-vs-engine match harness for strength checks.

Plays a small match between two Engine configurations at equal time-per-move
(the fair way to compare engines) and prints the score. Used to check whether a
new NNUE net actually beats the hand-crafted eval before we ship it.

Usage:
    python -m tools.match --nnue data/nnue.pt --games 6 --movetime 0.2
    python -m tools.match --a-nnue data/a.pt --b-nnue data/b.pt   # net vs net
"""
from __future__ import annotations

import argparse

import chess

from engine.board import Engine
from tools.estimate import announce


def _mk(style: str, nnue: str | None) -> Engine:
    return Engine(style=style, nnue_checkpoint=nnue)


def play(white: Engine, black: Engine, movetime: float,
         max_plies: int = 160) -> int:
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        eng = white if board.turn == chess.WHITE else black
        eng.board = board.copy()
        info = eng.best_move(depth=8, movetime=movetime)
        if info.best_move is None:
            break
        board.push(info.best_move)
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return 0
    return 1 if o.winner == chess.WHITE else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--movetime", type=float, default=0.2)
    ap.add_argument("--style", default="balanced")
    # A vs B. Default: A = NNUE (from --nnue), B = hand-crafted.
    ap.add_argument("--nnue", default=None, help="A-side NNUE checkpoint")
    ap.add_argument("--a-nnue", default=None)
    ap.add_argument("--b-nnue", default=None)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    a_ckpt = args.a_nnue or args.nnue
    b_ckpt = args.b_nnue
    a_name = args.a_name if (args.a_nnue or args.b_nnue) else "nnue"
    b_name = args.b_name if (args.a_nnue or args.b_nnue) else "hand"

    # A move by one side costs ~movetime; a typical game is ~80 plies, so
    # ~80 * movetime per game (one side thinks per ply).
    est_s = args.games * 80 * args.movetime
    announce(
        f"Match ({args.games} games)", est_s,
        f"{args.games} games x ~80 plies x {args.movetime*1000:.0f}ms/move")

    score = {a_name: 0, b_name: 0, "draw": 0}
    for g in range(args.games):
        A = _mk(args.style, a_ckpt)
        B = _mk(args.style, b_ckpt)
        if g % 2 == 0:
            r = play(A, B, args.movetime)
            who = a_name if r == 1 else (b_name if r == -1 else "draw")
        else:
            r = play(B, A, args.movetime)
            who = b_name if r == 1 else (a_name if r == -1 else "draw")
        score[who] += 1
        print(f"game {g+1}: {who}", flush=True)
    print(f"RESULT ({args.movetime*1000:.0f}ms/move): {score}")


if __name__ == "__main__":
    main()
