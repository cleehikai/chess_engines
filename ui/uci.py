"""UCI protocol adapter (Phase 2).

Makes the engine speak enough UCI to load in Cutechess-cli, Arena, Nibbler, or
Banksia -- for strength testing (gauntlets vs Stockfish) and engine-vs-engine
style matches. The playing style is exposed as a UCI combo option `Style`, and
search strength via `Depth` / `movetime`.

Run standalone:
    python -m ui.uci
then type UCI commands, or point a GUI at:  python -m ui.uci
"""
from __future__ import annotations

import os
import sys

import chess

from engine.board import Engine
from style.profiles import STYLE_NAMES, DEFAULT_PROFILE

ENGINE_NAME = "StyleEngine 0.1"
AUTHOR = "caleb"


class UCIAdapter:
    def __init__(self) -> None:
        # Optional NNUE checkpoint via env var (so Cutechess configs can set it).
        ckpt = os.environ.get("STYLE_ENGINE_NNUE") or None
        if ckpt and not os.path.exists(ckpt):
            ckpt = None
        self._nnue_path = ckpt
        self.engine = Engine(style=DEFAULT_PROFILE, nnue_checkpoint=ckpt)
        self.depth = 6            # default fixed depth if no time control given
        self.running = True

    def loop(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self.handle(line)
            if not self.running:
                break

    def handle(self, line: str) -> None:
        parts = line.split()
        cmd = parts[0]
        if cmd == "uci":
            self._identify()
        elif cmd == "isready":
            self._send("readyok")
        elif cmd == "ucinewgame":
            self.engine.new_game()
        elif cmd == "setoption":
            self._setoption(parts)
        elif cmd == "position":
            self._position(parts)
        elif cmd == "go":
            self._go(parts)
        elif cmd in ("quit", "stop"):
            if cmd == "quit":
                self.running = False

    # -- handlers -------------------------------------------------------------
    def _identify(self) -> None:
        self._send(f"id name {ENGINE_NAME}")
        self._send(f"id author {AUTHOR}")
        # Style combo option.
        default = self.engine.style
        var = " ".join(f"var {s}" for s in STYLE_NAMES)
        self._send(f"option name Style type combo default {default} {var}")
        self._send("option name Depth type spin default 6 min 1 max 20")
        cur = self._nnue_path or ""
        self._send(f"option name EvalFile type string default {cur}")
        self._send("uciok")

    def _setoption(self, parts: list[str]) -> None:
        # Format: setoption name <Name> value <Value>
        if "name" not in parts or "value" not in parts:
            return
        name = parts[parts.index("name") + 1]
        value = parts[parts.index("value") + 1]
        if name == "Style" and value in STYLE_NAMES:
            self.engine.set_style(value)
        elif name == "Depth":
            try:
                self.depth = max(1, int(value))
            except ValueError:
                pass
        elif name == "EvalFile":
            path = value if value and os.path.exists(value) else None
            style = self.engine.style
            self.engine = Engine(style=style, nnue_checkpoint=path)
            self._nnue_path = path

    def _position(self, parts: list[str]) -> None:
        if "startpos" in parts:
            self.engine.new_game()
            moves_idx = parts.index("moves") + 1 if "moves" in parts else len(parts)
        elif "fen" in parts:
            fen_idx = parts.index("fen") + 1
            if "moves" in parts:
                moves_idx = parts.index("moves")
                fen = " ".join(parts[fen_idx:moves_idx])
                moves_idx += 1
            else:
                fen = " ".join(parts[fen_idx:])
                moves_idx = len(parts)
            self.engine.set_fen(fen)
        else:
            return
        for uci in parts[moves_idx:]:
            self.engine.push_uci(uci)

    def _go(self, parts: list[str]) -> None:
        movetime = None
        depth = self.depth
        # Parse a subset of go params relevant to us.
        if "movetime" in parts:
            movetime = int(parts[parts.index("movetime") + 1]) / 1000.0
        if "depth" in parts:
            depth = int(parts[parts.index("depth") + 1])
        # Basic time-control support: budget a slice of remaining time.
        for tag in ("wtime", "btime"):
            if tag in parts:
                stm_white = self.engine.board.turn == chess.WHITE
                if (tag == "wtime") == stm_white:
                    remaining = int(parts[parts.index(tag) + 1]) / 1000.0
                    movetime = max(0.05, min(movetime or remaining, remaining / 30.0))

        info = self.engine.best_move(depth=depth, movetime=movetime)
        if info.best_move is None:
            # No legal move (mate/stalemate); UCI expects (none) or a move.
            self._send("bestmove 0000")
            return
        pv = " ".join(m.uci() for m in info.pv)
        self._send(
            f"info depth {info.depth} score cp {info.score} "
            f"nodes {info.nodes} time {int(info.elapsed*1000)} pv {pv}"
        )
        self._send(f"bestmove {info.best_move.uci()}")

    # -- io -------------------------------------------------------------------
    def _send(self, s: str) -> None:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()


def main() -> None:
    UCIAdapter().loop()


if __name__ == "__main__":
    main()
