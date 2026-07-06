"""Watch a strength match play out live on a board (the visual counterpart to
tools/match.py).

Runs the same engine-vs-engine matchup as `tools.match` but renders each game on
a pygame board so you can SEE the games instead of only reading the score. The
engines think in a background thread; the main thread animates the position and
running score.

Usage (mirrors tools/match.py):
    python -m tools.watch --nnue data/nnue.pt --games 6 --movetime 0.3
    python -m tools.watch --a-nnue data/old.pt --b-nnue data/nnue.pt \
        --a-name old --b-name new --games 10
    python -m tools.watch --a-style attacker --b-style defender   # style vs style

Controls: Space = pause/resume, Right = step one move while paused, Esc = quit.
"""
from __future__ import annotations

import argparse
import threading
import time

import chess
import pygame

from engine.board import Engine
from ui.render import (BoardRenderer, draw_evalbar, load_font,
                       BG, PANEL_BG, TEXT, MUTED)

SQUARE = 72
BOARD_PX = SQUARE * 8
EVALBAR_W = 24
MARGIN = 16
PANEL_W = 260
WIDTH = MARGIN + EVALBAR_W + MARGIN + BOARD_PX + MARGIN + PANEL_W + MARGIN
HEIGHT = MARGIN + BOARD_PX + MARGIN


class MatchWatcher:
    def __init__(self, a_cfg, b_cfg, a_name, b_name, games, movetime, depth):
        self.a_cfg, self.b_cfg = a_cfg, b_cfg          # (style, nnue) tuples
        self.a_name, self.b_name = a_name, b_name
        self.games, self.movetime, self.depth = games, movetime, depth

        self.board = chess.Board()
        self.last_move: chess.Move | None = None
        self.eval_cp = 0
        self.score = {a_name: 0, b_name: 0, "draw": 0}
        self.game_idx = 0
        self.a_is_white = True          # who has White this game
        self.status = "Starting…"
        self.paused = False
        self._step = False              # one-move step request while paused

        self._move_ready: chess.Move | None = None
        self._board_ready: chess.Board | None = None   # position AFTER the move
        self._eval_ready = 0
        self._thinking = False
        self._done = False
        self._lock = threading.Lock()

    # -- engine driving (background thread) ----------------------------------
    def _mk(self, cfg) -> Engine:
        style, nnue = cfg
        return Engine(style=style, nnue_checkpoint=nnue)

    def _think(self) -> None:
        """Compute the side-to-move's move in a worker; hand it back to the UI."""
        board_copy = self.board.copy()
        white_cfg = self.a_cfg if self.a_is_white else self.b_cfg
        black_cfg = self.b_cfg if self.a_is_white else self.a_cfg
        cfg = white_cfg if board_copy.turn == chess.WHITE else black_cfg
        eng = self._mk(cfg)
        eng.board = board_copy
        info = eng.best_move(depth=self.depth, movetime=self.movetime)
        cp = info.score if board_copy.turn == chess.WHITE else -info.score
        result_board = board_copy
        if info.best_move is not None:
            result_board = board_copy.copy()
            result_board.push(info.best_move)
        with self._lock:
            self._move_ready = info.best_move
            self._board_ready = result_board   # authoritative next position
            self._eval_ready = cp
            self._thinking = False

    def _start_think(self) -> None:
        self._thinking = True
        who = self._side_name(self.board.turn)
        self.status = f"Game {self.game_idx+1}/{self.games} — {who} to move"
        threading.Thread(target=self._think, daemon=True).start()

    def _side_name(self, turn: bool) -> str:
        white_name = self.a_name if self.a_is_white else self.b_name
        black_name = self.b_name if self.a_is_white else self.a_name
        return white_name if turn == chess.WHITE else black_name

    def _poll(self) -> None:
        if not self._thinking:
            return
        with self._lock:
            mv = self._move_ready
            new_board = self._board_ready
            if mv is None or new_board is None:
                return
            self._move_ready = None
            self._board_ready = None
            self.eval_cp = self._eval_ready
            self._thinking = False
        # Adopt the worker's resulting board directly (avoids re-pushing a move
        # onto a board that may have advanced — the source of a pseudo-legal
        # push error under races).
        self.board = new_board
        self.last_move = mv
        if self.board.is_game_over(claim_draw=True) or len(self.board.move_stack) >= 160:
            self._end_game()

    def _end_game(self) -> None:
        o = self.board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            self.score["draw"] += 1
            res = "draw"
        else:
            white_won = o.winner == chess.WHITE
            winner_is_a = (white_won == self.a_is_white)
            res = self.a_name if winner_is_a else self.b_name
            self.score[res] += 1
        self.status = f"Game {self.game_idx+1} → {res}"
        self.game_idx += 1
        if self.game_idx >= self.games:
            self._done = True
            self.status = f"Done. {self._score_str()}"
        else:
            # Next game; swap colors so both sides play White equally.
            self.board = chess.Board()
            self.last_move = None
            self.eval_cp = 0
            self.a_is_white = (self.game_idx % 2 == 0)
            self._next_at = time.perf_counter() + 1.2   # brief pause between games

    def _score_str(self) -> str:
        return (f"{self.a_name} {self.score[self.a_name]}  "
                f"{self.b_name} {self.score[self.b_name]}  "
                f"draw {self.score['draw']}")

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Match Viewer")
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        renderer = BoardRenderer(SQUARE)
        ui_font = load_font(20, mono=True)
        small = load_font(15, mono=True)
        clock = pygame.time.Clock()
        self._next_at = time.perf_counter()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT:
                        self._step = True

            now = time.perf_counter()
            gate = (not self.paused) or self._step
            if (not self._done and not self._thinking and gate
                    and now >= getattr(self, "_next_at", now)):
                self._start_think()
                self._step = False
            self._poll()

            # draw
            screen.fill(BG)
            bx = MARGIN + EVALBAR_W + MARGIN
            renderer.draw(screen, self.board, bx, MARGIN, self.last_move)
            draw_evalbar(screen, MARGIN, MARGIN, EVALBAR_W, BOARD_PX,
                         self.eval_cp, small)
            self._draw_panel(screen, ui_font, small)
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def _draw_panel(self, screen, ui_font, small) -> None:
        px = MARGIN + EVALBAR_W + MARGIN + BOARD_PX + MARGIN
        pygame.draw.rect(screen, PANEL_BG, (px, MARGIN, PANEL_W, BOARD_PX))
        x, y = px + 14, MARGIN + 14
        for line, font, col in (
            ("MATCH VIEWER", ui_font, TEXT),
            ("", small, MUTED),
            (f"{self.a_name}  vs  {self.b_name}", ui_font, TEXT),
            (f"White: {self._side_name(chess.WHITE)}", small, MUTED),
            ("", small, MUTED),
            ("Score", small, MUTED),
            (f"  {self.a_name}: {self.score[self.a_name]}", ui_font, TEXT),
            (f"  {self.b_name}: {self.score[self.b_name]}", ui_font, TEXT),
            (f"  draw: {self.score['draw']}", ui_font, TEXT),
            ("", small, MUTED),
            (f"eval {self.eval_cp/100:+.2f}", small, MUTED),
            ("", small, MUTED),
            (self.status, small, TEXT),
            ("", small, MUTED),
            ("PAUSED" if self.paused else "", small, (240, 200, 120)),
            ("Space pause · → step · Esc quit", small, MUTED),
        ):
            surf = font.render(line, True, col)
            screen.blit(surf, (x, y))
            y += surf.get_height() + 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch a strength match live.")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--movetime", type=float, default=0.3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--style", default="balanced", help="style for both sides")
    ap.add_argument("--a-style", default=None, help="A-side style (overrides --style)")
    ap.add_argument("--b-style", default=None, help="B-side style (overrides --style)")
    ap.add_argument("--nnue", default=None, help="A-side NNUE (B = hand eval)")
    ap.add_argument("--a-nnue", default=None)
    ap.add_argument("--b-nnue", default=None)
    ap.add_argument("--a-name", default=None)
    ap.add_argument("--b-name", default=None)
    args = ap.parse_args()

    a_nnue = args.a_nnue or args.nnue
    b_nnue = args.b_nnue
    a_style = args.a_style or args.style
    b_style = args.b_style or args.style
    # Sensible default names: reflect what actually differs.
    a_name = args.a_name or (a_style if (args.a_style or args.b_style) else "nnue")
    b_name = args.b_name or (b_style if (args.a_style or args.b_style)
                             else ("hand" if b_nnue is None else "B"))
    if a_name == b_name:
        a_name, b_name = a_name + "-A", b_name + "-B"

    watcher = MatchWatcher((a_style, a_nnue), (b_style, b_nnue),
                           a_name, b_name, args.games, args.movetime, args.depth)
    watcher.run()


if __name__ == "__main__":
    main()
