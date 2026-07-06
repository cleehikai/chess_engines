"""Shared, stateless board rendering used by the interactive GUI, the match
viewer (tools/watch.py), and the game analyzer (ui/analyze.py).

This is deliberately minimal: given a pygame surface, a chess.Board, and a
geometry, it draws squares, last-move highlight, and pieces. Interaction, panels,
clocks, and engine wiring live in the callers. Colors/glyphs are shared with the
main GUI so everything looks identical.
"""
from __future__ import annotations

import chess
import pygame

# Palette + glyphs (kept in sync with ui/gui.py on purpose).
LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
LASTMOVE = (205, 210, 106)
BG = (30, 30, 34)
PANEL_BG = (44, 44, 50)
TEXT = (230, 230, 232)
MUTED = (150, 150, 156)
WHITE_PIECE = (248, 248, 250)
WHITE_OUTLINE = (24, 24, 28)
BLACK_PIECE = (20, 20, 22)

PIECE_UNICODE = {
    (chess.PAWN, True): "♙", (chess.PAWN, False): "♟",
    (chess.KNIGHT, True): "♘", (chess.KNIGHT, False): "♞",
    (chess.BISHOP, True): "♗", (chess.BISHOP, False): "♝",
    (chess.ROOK, True): "♖", (chess.ROOK, False): "♜",
    (chess.QUEEN, True): "♕", (chess.QUEEN, False): "♛",
    (chess.KING, True): "♔", (chess.KING, False): "♚",
}


def load_font(size: int, mono: bool = False):
    """Prefer a font that renders chess Unicode glyphs; fall back gracefully."""
    candidates = (["segoeuisymbol", "dejavusans", "arialunicodems"]
                  if not mono else ["consolas", "dejavusansmono", "couriernew"])
    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)


class BoardRenderer:
    """Draws a board at a fixed square size and top-left origin."""

    def __init__(self, square: int, flip: bool = False):
        self.square = square
        self.flip = flip  # True => Black at bottom
        self.piece_font = load_font(int(square * 0.80))

    def resize(self, square: int) -> None:
        self.square = square
        self.piece_font = load_font(int(square * 0.80))

    def _topleft(self, square: int, ox: int, oy: int) -> tuple[int, int]:
        file_, rank = chess.square_file(square), chess.square_rank(square)
        if self.flip:
            col, row = 7 - file_, rank
        else:
            col, row = file_, 7 - rank
        return ox + col * self.square, oy + row * self.square

    def draw(self, screen, board: chess.Board, ox: int, oy: int,
             last_move: chess.Move | None = None) -> None:
        sq = self.square
        for square in chess.SQUARES:
            x, y = self._topleft(square, ox, oy)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = LIGHT if light else DARK
            if last_move and square in (last_move.from_square, last_move.to_square):
                color = LASTMOVE
            pygame.draw.rect(screen, color, (x, y, sq, sq))
        for square, piece in board.piece_map().items():
            glyph = PIECE_UNICODE[(piece.piece_type, piece.color)]
            x, y = self._topleft(square, ox, oy)
            center = (x + sq // 2, y + sq // 2)
            if piece.color:
                self._blit_outlined(screen, glyph, center)
            else:
                surf = self.piece_font.render(glyph, True, BLACK_PIECE)
                screen.blit(surf, surf.get_rect(center=center))

    def _blit_outlined(self, screen, glyph: str, center: tuple[int, int]) -> None:
        off = max(1, self.square // 40)
        outline = self.piece_font.render(glyph, True, WHITE_OUTLINE)
        for dx, dy in ((-off, 0), (off, 0), (0, -off), (0, off),
                       (-off, -off), (off, -off), (-off, off), (off, off)):
            screen.blit(outline, outline.get_rect(center=(center[0] + dx, center[1] + dy)))
        fill = self.piece_font.render(glyph, True, WHITE_PIECE)
        screen.blit(fill, fill.get_rect(center=center))


def draw_evalbar(screen, x: int, y: int, w: int, h: int, eval_cp: int,
                 font) -> None:
    """Vertical eval bar (White fraction from bottom), White-POV centipawns."""
    pygame.draw.rect(screen, (20, 20, 22), (x, y, w, h))
    cp = max(-1000, min(1000, eval_cp))
    frac = 0.5 + 0.5 * (cp / 1000.0)
    white_h = int(h * frac)
    pygame.draw.rect(screen, (245, 245, 245), (x, y + h - white_h, w, white_h))
    label = font.render(f"{eval_cp/100:+.1f}", True, (120, 120, 120))
    screen.blit(label, (x - 2, y + h + 2))
