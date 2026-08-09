"""Perft: the correctness gate for any atomic move generator.

Reference counts come from Fairy-Stockfish, which the Chalmers thesis also used
as its ground truth. Any new board implementation (the Rust core, or anything
else) must reproduce these exactly before a single Elo it produces is worth
reading.

    python -m lab.perft --generate      # rebuild the reference file from FSF
    python -m lab.perft                 # check python-chess against it
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import chess
import chess.variant

from . import config, fsf

REFERENCE = config.ROOT / "tests" / "perft_positions.json"

#: Positions chosen to exercise the rules that atomic gets wrong most often.
POSITIONS: dict[str, tuple[str, int]] = {
    "startpos": (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5),
    "kiwipete-ish": (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4),
    "adjacent-kings": (
        # Legal only in atomic: kings touching, neither in check.
        "8/8/8/3kK3/8/8/8/8 w - - 0 1", 6),
    "en-passant-blast": (
        # En passant explodes around the landing square, not the victim square.
        "rnbqkbnr/1pp1pppp/8/p2pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", 4),
    "pinned-but-regicidal": (
        # The white knight is absolutely pinned (zero legal moves in standard
        # chess) yet Nxd6 is legal here: the blast kills the black king first.
        "4r3/8/2kb4/8/4N3/8/8/4K3 w - - 0 1", 5),
    "own-king-in-blast": (
        # Rxd6 is illegal: it would destroy White's own king on c5.
        "4k3/3R4/3n4/2K5/8/8/8/8 w - - 0 1", 5),
    "castling-through-blast": (
        # Explosions destroy castling rights when they touch king or rook.
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", 4),
    "promotion-race": (
        "8/PPPk4/8/8/8/8/4pppp/4K3 w - - 0 1", 5),
}


class FsfPerft:
    """Keeps one Fairy-Stockfish process alive to answer `go perft` queries.

    A fresh process per query does not work: Stockfish treats stdin EOF as
    `quit` and abandons the search, returning garbage.
    """

    def __init__(self) -> None:
        fsf.check_available()
        self.proc = subprocess.Popen(
            [str(config.FSF_EXE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name UCI_Variant value atomic")

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        assert self.proc.stdout is not None
        lines = []
        for line in self.proc.stdout:
            lines.append(line.rstrip())
            if line.startswith(token) or token in line:
                return lines
        raise RuntimeError(f"Fairy-Stockfish closed before emitting {token!r}")

    def perft(self, fen: str, depth: int) -> int:
        self._send(f"position fen {fen}")
        self._send(f"go perft {depth}")
        lines = self._read_until("Nodes searched")
        for line in reversed(lines):
            if line.startswith("Nodes searched"):
                return int(line.split(":")[1].strip())
        raise RuntimeError("no perft total in output")

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> FsfPerft:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def python_chess_perft(board: chess.variant.AtomicBoard, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += python_chess_perft(board, depth - 1)
        board.pop()
    return total


def generate() -> int:
    """Rebuild the reference file from Fairy-Stockfish."""
    out: dict[str, dict] = {}
    with FsfPerft() as engine:
        for name, (fen, max_depth) in POSITIONS.items():
            counts = {}
            print(f"{name}:")
            for depth in range(1, max_depth + 1):
                start = time.perf_counter()
                counts[str(depth)] = engine.perft(fen, depth)
                print(f"  depth {depth}: {counts[str(depth)]:>12,}"
                      f"  ({time.perf_counter()-start:.2f}s)")
            out[name] = {"fen": fen, "perft": counts}
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(
        json.dumps({"source": "Fairy-Stockfish 14, UCI_Variant=atomic",
                    "positions": out}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {REFERENCE}")
    return 0


def check(max_depth: int = 4, backend: str = "python-chess") -> int:
    """Verify a backend against the reference counts."""
    if not REFERENCE.exists():
        print("no reference file; run `python -m lab.perft --generate` first")
        return 1
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))

    if backend == "rust":
        try:
            import atomic_core
        except ImportError:
            print("atomic_core not built; run `python -m lab.build_core`")
            return 1
        run = lambda fen, d: atomic_core.perft(fen, d)  # noqa: E731
    else:
        run = lambda fen, d: python_chess_perft(  # noqa: E731
            chess.variant.AtomicBoard(fen), d)

    failures = 0
    for name, entry in data["positions"].items():
        for depth_s, expected in entry["perft"].items():
            depth = int(depth_s)
            if depth > max_depth:
                continue
            start = time.perf_counter()
            got = run(entry["fen"], depth)
            elapsed = time.perf_counter() - start
            ok = got == expected
            failures += not ok
            rate = got / elapsed / 1e6 if elapsed > 0 else 0
            print(f"{'ok  ' if ok else 'FAIL'} {name:<24} d{depth} "
                  f"{got:>12,} {'' if ok else f'(expected {expected:,})'}"
                  f"  {rate:6.2f} Mnps")
    print(f"\n{backend}: {'all match' if not failures else f'{failures} MISMATCHES'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true",
                        help="rebuild reference counts from Fairy-Stockfish")
    parser.add_argument("--backend", default="python-chess",
                        choices=["python-chess", "rust"])
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args(argv)
    return generate() if args.generate else check(args.max_depth, args.backend)


if __name__ == "__main__":
    sys.exit(main())
