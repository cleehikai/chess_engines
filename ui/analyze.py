"""Game analyzer: step through a game and see engine judgments.

Load a PGN (file or pasted), then navigate move-by-move (← →, Home/End). For
each position the analyzer shows:
  * the engine eval and its best move / line,
  * blunder / mistake / inaccuracy flags (chess.com-style, from eval swings),
  * on demand (press S), what each of the 7 styles would play here.

The eval pass runs in a BACKGROUND thread using the NNUE net at a higher depth
(accurate but slow on CPU), so navigation stays responsive — positions fill in
their verdicts as the pass reaches them. Per-style suggestions are computed
on demand for the current position only (7 quick searches).

Usage:
    python -m ui.analyze game.pgn                       # analyze a PGN file
    python -m ui.analyze game.pgn --nnue data/nnue.pt   # (nnue is the default)
    python -m ui.analyze --nnue data/nnue.pt            # paste a PGN in-window
"""
from __future__ import annotations

import argparse
import io
import threading
import time
from dataclasses import dataclass, field

import chess
import chess.pgn
import pygame

from engine.board import Engine
from style.profiles import STYLE_NAMES
from ui.render import (BoardRenderer, draw_evalbar, load_font,
                       BG, PANEL_BG, TEXT, MUTED)

SQUARE = 72
BOARD_PX = SQUARE * 8
EVALBAR_W = 24
MARGIN = 16
PANEL_W = 340
WIDTH = MARGIN + EVALBAR_W + MARGIN + BOARD_PX + MARGIN + PANEL_W + MARGIN
HEIGHT = MARGIN + BOARD_PX + MARGIN

# Eval-swing thresholds (centipawns, from the moving side's POV) for tags.
BLUNDER, MISTAKE, INACCURACY = 300, 150, 75


@dataclass
class Ply:
    """One position in the game plus its analysis (filled in by the bg pass)."""
    board: chess.Board                 # position BEFORE `move` is played
    move: chess.Move | None            # the move actually played (None at end)
    san: str = ""
    analyzed: bool = False
    eval_cp: int = 0                   # White-POV eval of this position
    best_move: chess.Move | None = None
    best_san: str = ""
    tag: str = ""                      # "", inaccuracy, mistake, blunder (for `move`)


class Analyzer:
    def __init__(self, plies: list[Ply], nnue: str | None, depth: int):
        self.plies = plies
        self.nnue = nnue
        self.depth = depth
        self.idx = 0                   # current ply being viewed
        self.flip = False

        # On-demand per-style suggestions for the current position.
        self.style_moves: dict[str, str] = {}   # style -> "e4 (+0.3)"
        self.style_pos_idx: int | None = None    # which ply style_moves is for
        self._style_computing = False

        self._engine = Engine(nnue_checkpoint=nnue)   # shared analysis engine
        self._analysis_done = False
        self._lock = threading.Lock()
        self.status = "Analyzing… (navigate freely; verdicts fill in)"

    # -- background analysis pass --------------------------------------------
    def start_analysis(self) -> None:
        threading.Thread(target=self._analyze_all, daemon=True).start()

    def _analyze_all(self) -> None:
        """Evaluate every position with the NNUE engine, then derive move tags
        from consecutive eval swings. Fills plies in order."""
        for p in self.plies:
            info = self._engine._searcher.search(p.board, depth=self.depth)
            cp = info.score if p.board.turn == chess.WHITE else -info.score
            best_san = ""
            if info.best_move is not None and p.board.is_legal(info.best_move):
                best_san = p.board.san(info.best_move)
            with self._lock:
                p.eval_cp = cp
                p.best_move = info.best_move
                p.best_san = best_san
                p.analyzed = True
        # Tag played moves by how far they fell short of the eval BEFORE the move
        # (from the mover's perspective).
        with self._lock:
            for i in range(len(self.plies) - 1):
                cur, nxt = self.plies[i], self.plies[i + 1]
                if cur.move is None:
                    continue
                mover_white = cur.board.turn == chess.WHITE
                before = cur.eval_cp if mover_white else -cur.eval_cp
                after = nxt.eval_cp if mover_white else -nxt.eval_cp
                loss = before - after       # positive = the move lost eval
                if loss >= BLUNDER:
                    cur.tag = "blunder"
                elif loss >= MISTAKE:
                    cur.tag = "mistake"
                elif loss >= INACCURACY:
                    cur.tag = "inaccuracy"
            self._analysis_done = True
            self.status = "Analysis complete."

    # -- on-demand per-style suggestions -------------------------------------
    def request_style_compare(self) -> None:
        if self._style_computing:
            return
        idx = self.idx
        self._style_computing = True
        self.style_pos_idx = idx
        self.style_moves = {s: "…" for s in STYLE_NAMES}
        threading.Thread(target=self._compute_styles, args=(idx,),
                         daemon=True).start()

    def _compute_styles(self, idx: int) -> None:
        board = self.plies[idx].board
        for style in STYLE_NAMES:
            eng = Engine(style=style, nnue_checkpoint=self.nnue)
            eng.board = board.copy()
            info = eng.best_move(depth=max(3, self.depth - 2))
            txt = "-"
            if info.best_move is not None and board.is_legal(info.best_move):
                cp = info.score if board.turn == chess.WHITE else -info.score
                txt = f"{board.san(info.best_move)} ({cp/100:+.1f})"
            with self._lock:
                self.style_moves[style] = txt
        self._style_computing = False

    # -- navigation -----------------------------------------------------------
    def goto(self, i: int) -> None:
        self.idx = max(0, min(len(self.plies) - 1, i))
        # Invalidate style compare if we moved off its position.
        if self.style_pos_idx != self.idx:
            self.style_moves = {}
            self.style_pos_idx = None

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Game Analyzer")
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        renderer = BoardRenderer(SQUARE, flip=self.flip)
        ui_font = load_font(20, mono=True)
        small = load_font(15, mono=True)
        clock = pygame.time.Clock()
        self.start_analysis()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                        self.goto(self.idx + 1)
                    elif event.key in (pygame.K_LEFT, pygame.K_UP):
                        self.goto(self.idx - 1)
                    elif event.key == pygame.K_HOME:
                        self.goto(0)
                    elif event.key == pygame.K_END:
                        self.goto(len(self.plies) - 1)
                    elif event.key == pygame.K_f:
                        self.flip = not self.flip
                        renderer.flip = self.flip
                    elif event.key == pygame.K_s:
                        self.request_style_compare()

            screen.fill(BG)
            bx = MARGIN + EVALBAR_W + MARGIN
            p = self.plies[self.idx]
            renderer.draw(screen, p.board, bx, MARGIN,
                          self.plies[self.idx - 1].move if self.idx > 0 else None)
            with self._lock:
                eval_cp = p.eval_cp if p.analyzed else 0
            draw_evalbar(screen, MARGIN, MARGIN, EVALBAR_W, BOARD_PX, eval_cp, small)
            self._draw_panel(screen, ui_font, small)
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def _draw_panel(self, screen, ui_font, small) -> None:
        px = MARGIN + EVALBAR_W + MARGIN + BOARD_PX + MARGIN
        pygame.draw.rect(screen, PANEL_BG, (px, MARGIN, PANEL_W, BOARD_PX))
        x, y = px + 14, MARGIN + 14

        def line(s, font=small, col=TEXT, dy=None):
            nonlocal y
            surf = font.render(s, True, col)
            screen.blit(surf, (x, y))
            y += (dy if dy is not None else surf.get_height() + 2)

        p = self.plies[self.idx]
        movenum = self.idx // 2 + 1
        turn = "…" if self.idx % 2 else ""
        line("GAME ANALYZER", ui_font)
        line(f"Ply {self.idx}/{len(self.plies)-1}", small, MUTED)
        y += 6

        # What move led here + its tag.
        if self.idx > 0:
            prev = self.plies[self.idx - 1]
            tag_col = {"blunder": (240, 90, 90), "mistake": (240, 160, 90),
                       "inaccuracy": (230, 210, 110)}.get(prev.tag, TEXT)
            mark = {"blunder": "??", "mistake": "?", "inaccuracy": "?!"}.get(prev.tag, "")
            line(f"Move played: {prev.san} {mark}", ui_font, tag_col)
            if prev.tag:
                line(f"  → {prev.tag.upper()}", small, tag_col)
        y += 6

        with self._lock:
            analyzed = p.analyzed
            eval_cp, best_san = p.eval_cp, p.best_san
        if analyzed:
            line(f"Eval: {eval_cp/100:+.2f}  (White POV)", ui_font)
            line(f"Best here: {best_san or '-'}", small, MUTED)
        else:
            line("Eval: analyzing…", small, MUTED)
        y += 10

        # Style comparison (on demand).
        line("Styles (press S)", small, MUTED)
        if self.style_pos_idx == self.idx and self.style_moves:
            for s in STYLE_NAMES:
                line(f"  {s:10s} {self.style_moves.get(s, '…')}", small, TEXT)
        elif self.style_pos_idx == self.idx and self._style_computing:
            line("  computing…", small, MUTED)
        else:
            line("  (press S for this position)", small, MUTED)
        y += 8

        line("← → step · Home/End · F flip · S styles",
             small, MUTED)
        line(self.status, small, MUTED)


# -- loading ------------------------------------------------------------------
def _plies_from_game(game: chess.pgn.Game) -> list[Ply]:
    board = game.board()
    plies: list[Ply] = []
    for move in game.mainline_moves():
        san = board.san(move)
        plies.append(Ply(board=board.copy(), move=move, san=san))
        board.push(move)
    # Final position (no move to play from it).
    plies.append(Ply(board=board.copy(), move=None, san=""))
    return plies


def _load_pgn_text(text: str) -> list[Ply]:
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise ValueError("no game found in PGN")
    return _plies_from_game(game)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a game.")
    ap.add_argument("pgn", nargs="?", help="path to a .pgn file")
    ap.add_argument("--nnue", default="data/nnue.pt",
                    help="NNUE checkpoint for analysis (default: data/nnue.pt)")
    ap.add_argument("--depth", type=int, default=6, help="analysis search depth")
    args = ap.parse_args()

    if args.pgn:
        with open(args.pgn, encoding="utf-8") as fh:
            plies = _load_pgn_text(fh.read())
    else:
        # No file: read a PGN from stdin so the window has something to show.
        import sys
        print("Paste a PGN, then Ctrl+Z Enter (Windows) / Ctrl+D (Unix):")
        plies = _load_pgn_text(sys.stdin.read())

    import os
    nnue = args.nnue if args.nnue and os.path.exists(args.nnue) else None
    Analyzer(plies, nnue=nnue, depth=args.depth).run()


if __name__ == "__main__":
    main()
