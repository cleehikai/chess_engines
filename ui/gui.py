"""Standalone pygame GUI (Phase 1).

Click-to-move board against the engine, with:
  * a live eval bar,
  * a style selector (cycles through the profiles),
  * a strength control (search depth),
  * optional chess clocks (common time controls),
  * premoves (chess.com style),
  * fullscreen toggle,
  * move list / status text.

Pieces are drawn with Unicode glyphs so no image assets are needed. Runs fully
on CPU. Launch via `python run.py`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import chess
import pygame

from engine.board import Engine
from style.profiles import STYLE_NAMES, PROFILES

# --- Fixed layout constants (used to size the board within the window) --------
EVALBAR_W = 28
PANEL_W = 300
MARGIN = 16
# The default windowed size targets a 72px square; fullscreen recomputes.
DEFAULT_SQUARE = 72
DEFAULT_BOARD_PX = DEFAULT_SQUARE * 8
WIDTH = MARGIN + EVALBAR_W + MARGIN + DEFAULT_BOARD_PX + MARGIN + PANEL_W + MARGIN
HEIGHT = MARGIN + DEFAULT_BOARD_PX + MARGIN

LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
HILITE = (246, 246, 105)
LASTMOVE = (205, 210, 106)
PREMOVE = (235, 97, 80)          # chess.com-ish red for premoves
BG = (30, 30, 34)
PANEL_BG = (44, 44, 50)
TEXT = (230, 230, 232)
MUTED = (150, 150, 156)
WHITE_PIECE = (248, 248, 250)
WHITE_OUTLINE = (24, 24, 28)     # dark outline so white pieces read on light squares
BLACK_PIECE = (20, 20, 22)

PIECE_UNICODE = {
    (chess.PAWN, True): "♙", (chess.PAWN, False): "♟",
    (chess.KNIGHT, True): "♘", (chess.KNIGHT, False): "♞",
    (chess.BISHOP, True): "♗", (chess.BISHOP, False): "♝",
    (chess.ROOK, True): "♖", (chess.ROOK, False): "♜",
    (chess.QUEEN, True): "♕", (chess.QUEEN, False): "♛",
    (chess.KING, True): "♔", (chess.KING, False): "♚",
}


@dataclass
class TimeControl:
    """A per-side clock: `base` seconds plus `increment` seconds per move."""
    name: str
    base: float          # initial time per side (seconds)
    increment: float     # added after each move (seconds)


# Common time controls, plus an untimed default. Presets are cycled in the panel.
TIME_CONTROLS: list[TimeControl | None] = [
    None,                                    # untimed
    TimeControl("1+0 Bullet", 60, 0),
    TimeControl("2+1 Bullet", 120, 1),
    TimeControl("3+0 Blitz", 180, 0),
    TimeControl("3+2 Blitz", 180, 2),
    TimeControl("5+0 Blitz", 300, 0),
    TimeControl("5+3 Blitz", 300, 3),
    TimeControl("10+0 Rapid", 600, 0),
    TimeControl("15+10 Rapid", 900, 10),
]


class ChessGUI:
    def __init__(self, engine: Engine, human_is_white: bool = True,
                 depth: int = 4, time_control: TimeControl | None = None,
                 both_sides: bool = False):
        self.engine = engine
        self.human_white = human_is_white
        self.depth = depth
        # Pass-and-play: the human controls BOTH colors; the engine never auto-
        # moves. Toggleable live from the panel. Premoves are disabled in this
        # mode (there is no opponent reply to premove against).
        self.both_sides = both_sides

        # Paste box for setting up a position from a FEN or PGN. When active,
        # keystrokes are captured into `self.paste_text` until Enter/Esc.
        self.paste_active = False
        self.paste_text = ""
        self.paste_error = ""
        # Replay state: when a PGN is loaded, its moves are queued here and
        # stepped through (auto-play), then control returns to the human.
        self._replay_moves: list[chess.Move] = []
        self._replay_next = 0.0

        self.selected: int | None = None
        self.legal_targets: set[int] = set()
        self.last_move: chess.Move | None = None
        self.status = "Your move" if human_is_white else "Engine thinking..."
        self.eval_cp = 0

        # Polish state: move history (SAN) and the engine's last plan (PV + eval).
        self.san_history: list[str] = []
        self.engine_pv_san: list[str] = []
        self.engine_depth = 0
        self.engine_nps = 0

        # Premoves: a QUEUE of human moves (chess.com style), each played in
        # turn as it becomes legal after the engine replies. While queuing, a
        # virtual board tracks the position after the already-queued premoves so
        # the next premove's piece and legal targets are computed correctly.
        # Shown with red highlights; a double-click anywhere clears the queue.
        self.premoves: list[chess.Move] = []
        self._last_click_time = 0.0

        # Clocks. `time_control is None` means untimed. Times are in seconds.
        self.tc = time_control
        self.white_time = time_control.base if time_control else 0.0
        self.black_time = time_control.base if time_control else 0.0
        self._clock_running = time_control is not None
        self._last_tick = time.perf_counter()
        self.flagged: bool | None = None   # which color ran out (True=White)
        self._tc_index = (TIME_CONTROLS.index(time_control)
                          if time_control in TIME_CONTROLS else 0)

        # Display geometry, recomputed each frame from the window size.
        self.fullscreen = False
        self.square = DEFAULT_SQUARE
        self.board_px = DEFAULT_BOARD_PX
        self.board_x = MARGIN + EVALBAR_W + MARGIN
        self.board_y = MARGIN
        self.evalbar_x = MARGIN
        self.panel_x = self.board_x + self.board_px + MARGIN

        self._thinking = False
        self._engine_result: chess.Move | None = None
        self._engine_info = None
        self._lock = threading.Lock()

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Style Chess Engine")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self._recompute_layout()
        clock = pygame.time.Clock()

        if not self.human_white:
            self._start_engine_move()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode(
                        (max(event.w, 480), max(event.h, 360)), pygame.RESIZABLE)
                    self._recompute_layout()
                elif event.type == pygame.KEYDOWN:
                    if self.paste_active:
                        self._handle_paste_key(event)
                    elif event.key == pygame.K_F11 or (
                            event.key == pygame.K_RETURN and
                            event.mod & pygame.KMOD_ALT):
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_ESCAPE and self.fullscreen:
                        self._toggle_fullscreen()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._on_click(event.pos)

            self._tick_clock()
            self._poll_engine()
            self._step_replay()
            self._draw()
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    # -- window / layout ------------------------------------------------------
    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            info = pygame.display.Info()
            self.screen = pygame.display.set_mode(
                (info.current_w, info.current_h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self._recompute_layout()

    def _recompute_layout(self) -> None:
        """Size the board to fill the available height; center it, keep the
        eval bar to its left and the panel to its right."""
        w, h = self.screen.get_size()
        # Board is a square limited by height and by width minus the fixed chrome.
        avail_w = w - (MARGIN + EVALBAR_W + MARGIN + MARGIN + PANEL_W + MARGIN)
        avail_h = h - 2 * MARGIN
        board_px = max(8 * 24, min(avail_w, avail_h))
        self.square = board_px // 8
        self.board_px = self.square * 8
        self.evalbar_x = MARGIN
        self.board_x = MARGIN + EVALBAR_W + MARGIN
        # Center vertically.
        self.board_y = max(MARGIN, (h - self.board_px) // 2)
        self.panel_x = self.board_x + self.board_px + MARGIN

        # Fonts scale with the square size.
        self.piece_font = _load_font(int(self.square * 0.80))
        self.ui_font = _load_font(20, mono=True)
        self.small_font = _load_font(16, mono=True)

    # -- interaction ----------------------------------------------------------
    def _on_click(self, pos: tuple[int, int]) -> None:
        x, y = pos
        now = time.perf_counter()
        double_click = (now - self._last_click_time) < 0.35
        self._last_click_time = now

        # Panel buttons first.
        if x >= self.panel_x:
            self._on_panel_click(x, y)
            return

        # A double-click clears the whole premove queue (chess.com behaviour).
        if double_click and self.premoves:
            self.premoves = []
            self.selected = None
            self.legal_targets = set()
            return

        sq = self._square_at(x, y)
        if sq is None:
            return
        board = self.engine.board

        # While the engine is thinking, clicks build a PREMOVE for our side.
        # (Not in pass-and-play, where there is no engine reply to premove.)
        if self._thinking and not self.both_sides:
            self._handle_premove_click(sq)
            return

        piece = board.piece_at(sq)
        if self.selected is None:
            if piece is not None and piece.color == board.turn:
                self.selected = sq
                self.legal_targets = {m.to_square for m in board.legal_moves
                                      if m.from_square == sq}
        else:
            move = self._build_move(self.selected, sq)
            if move is not None and board.is_legal(move):
                self._make_human_move(move)
            self.selected = None
            self.legal_targets = set()

    def _premove_board(self) -> chess.Board:
        """A virtual board = current position with all queued premoves applied.
        Used so a new premove sees the pieces where the earlier premoves will
        have left them. Premoves are pushed pseudo-legally (turn is forced to the
        human's color each step) so we can queue replies to the engine's move
        that isn't on the board yet."""
        b = self.engine.board.copy(stack=False)
        human_color = chess.WHITE if self.human_white else chess.BLACK
        for mv in self.premoves:
            b.turn = human_color
            # Only apply if it's a pseudo-legal move for our piece; skip if not
            # (the highlight logic already restricts to reachable squares).
            if b.piece_at(mv.from_square) and mv in b.pseudo_legal_moves:
                b.push(mv)
            else:
                b.turn = human_color
        b.turn = human_color
        return b

    def _premove_targets(self, frm: int) -> set[int]:
        """Squares the piece on `frm` could reach on the virtual premove board,
        ignoring whose turn it is (pseudo-legal). This replaces the old 'every
        square' highlight so premove targets match the piece's real movement."""
        vb = self._premove_board()
        vb.turn = chess.WHITE if self.human_white else chess.BLACK
        return {m.to_square for m in vb.pseudo_legal_moves if m.from_square == frm}

    def _handle_premove_click(self, sq: int) -> None:
        """Build a premove from two clicks on our own pieces/targets, appended to
        the queue. A premove is pseudo-validated against the virtual board (right
        piece, reachable square) — each is only *played* if it is legal when our
        turn actually comes."""
        vb = self._premove_board()
        human_color = chess.WHITE if self.human_white else chess.BLACK
        if self.selected is None:
            piece = vb.piece_at(sq)
            if piece is not None and piece.color == human_color:
                self.selected = sq
                # Show the piece's reachable destinations (pseudo-legal).
                self.legal_targets = self._premove_targets(sq)
        else:
            move = self._build_move_on(vb, self.selected, sq)
            if (move is not None and move.from_square != move.to_square
                    and sq in self._premove_targets(self.selected)):
                self.premoves.append(move)
            self.selected = None
            self.legal_targets = set()

    def _on_panel_click(self, x: int, y: int) -> None:
        # Button hit-testing mirrors the layout in _draw_panel.
        if self._style_btn_rect.collidepoint(x, y):
            self._cycle_style()
        elif self._depth_up_rect.collidepoint(x, y):
            self.depth = min(8, self.depth + 1)
        elif self._depth_dn_rect.collidepoint(x, y):
            self.depth = max(1, self.depth - 1)
        elif self._tc_btn_rect.collidepoint(x, y):
            self._cycle_time_control()
        elif self._both_btn_rect.collidepoint(x, y):
            self._toggle_both_sides()
        elif self._setup_btn_rect.collidepoint(x, y):
            self._open_paste()
        elif self._analyze_btn_rect.collidepoint(x, y):
            self._analyze_game()
        elif self._new_btn_rect.collidepoint(x, y):
            self._new_game()

    def _analyze_game(self) -> None:
        """Export the game played so far to a PGN and open it in the analyzer
        (separate window/process so play is unaffected)."""
        board = self.engine.board
        if not board.move_stack:
            self.status = "No moves to analyze yet"
            return
        import os
        import subprocess
        import sys
        import chess.pgn
        game = chess.pgn.Game()
        node = game
        replay = chess.Board()
        for mv in board.move_stack:
            node = node.add_variation(mv)
            replay.push(mv)
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "last_game.pgn")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(game))
        try:
            subprocess.Popen([sys.executable, "-m", "ui.analyze", path,
                              "--nnue", "data/nnue.pt"])
            self.status = "Analyzer launched (new window)"
        except Exception as e:
            self.status = f"Could not launch analyzer: {e}"

    def _toggle_both_sides(self) -> None:
        self.both_sides = not self.both_sides
        if self.both_sides:
            self.premoves = []
            self.status = ("White to move" if self.engine.board.turn == chess.WHITE
                           else "Black to move")
        else:
            self.status = "Your move"
            # If it's the engine's turn now, let it move.
            if self.engine.board.turn != (chess.WHITE if self.human_white
                                          else chess.BLACK):
                self._start_engine_move()

    def _cycle_style(self) -> None:
        cur = self.engine.style
        i = (STYLE_NAMES.index(cur) + 1) % len(STYLE_NAMES)
        self.engine.set_style(STYLE_NAMES[i])

    def _cycle_time_control(self) -> None:
        # Changing the time control only takes effect on a new game.
        self._tc_index = (self._tc_index + 1) % len(TIME_CONTROLS)
        self._pending_tc = TIME_CONTROLS[self._tc_index]
        self.status = (f"{self._pending_tc.name} - press New Game"
                       if self._pending_tc else "Untimed - press New Game")

    def _new_game(self) -> None:
        if self._thinking:
            return
        self.engine.new_game()
        self.selected = None
        self.legal_targets = set()
        self.last_move = None
        self.premoves = []
        self.eval_cp = 0
        self.san_history = []
        self.engine_pv_san = []
        self.engine_depth = 0
        self.engine_nps = 0
        # Apply any pending time-control choice.
        self.tc = getattr(self, "_pending_tc", self.tc)
        self.white_time = self.tc.base if self.tc else 0.0
        self.black_time = self.tc.base if self.tc else 0.0
        self._clock_running = self.tc is not None
        self._last_tick = time.perf_counter()
        self.flagged = None
        self._replay_moves = []
        self.paste_active = False
        self.paste_error = ""
        if self.both_sides:
            self.status = "White to move"
        else:
            self.status = "Your move" if self.human_white else "Engine thinking..."
            if not self.human_white:
                self._start_engine_move()

    # -- position setup: paste a FEN or PGN ----------------------------------
    def _open_paste(self) -> None:
        self.paste_active = True
        self.paste_text = ""
        self.paste_error = ""
        self.status = "Paste FEN or PGN, then Enter (Esc to cancel)"

    def _handle_paste_key(self, event) -> None:
        if event.key == pygame.K_ESCAPE:
            self.paste_active = False
            self.paste_error = ""
            self.status = "Your move"
        elif event.key == pygame.K_RETURN:
            self._apply_paste(self.paste_text.strip())
        elif event.key == pygame.K_BACKSPACE:
            self.paste_text = self.paste_text[:-1]
        elif event.key == pygame.K_v and (event.mod & pygame.KMOD_CTRL):
            try:
                import pygame.scrap as scrap  # optional clipboard
                scrap.init()
                data = scrap.get(pygame.SCRAP_TEXT)
                if data:
                    self.paste_text += data.decode("utf-8", "ignore").replace("\x00", "")
            except Exception:
                pass
        elif event.unicode and event.unicode.isprintable():
            self.paste_text += event.unicode

    def _apply_paste(self, text: str) -> None:
        """Auto-detect FEN vs PGN and set up the board. A FEN is a single line
        with 6 space-separated fields and slashes; anything else is tried as a
        PGN (which then auto-replays)."""
        if not text:
            self.paste_active = False
            return
        if self._looks_like_fen(text):
            try:
                self.engine.set_fen(text)
                self._reset_view_after_setup(f"Position set from FEN")
            except Exception as e:
                self.paste_error = f"Bad FEN: {e}"
            return
        # Try PGN.
        try:
            import io
            import chess.pgn
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                raise ValueError("no game found")
            moves = list(game.mainline_moves())
            if not moves:
                raise ValueError("no moves in PGN")
            self.engine.new_game()
            # Honor a non-standard start position in the PGN header if present.
            fen = game.headers.get("FEN")
            if fen:
                self.engine.set_fen(fen)
            self._reset_view_after_setup(f"Replaying {len(moves)} moves…")
            self._replay_moves = moves
            self._replay_next = time.perf_counter()
        except Exception as e:
            self.paste_error = f"Not a valid FEN or PGN: {e}"

    @staticmethod
    def _looks_like_fen(text: str) -> bool:
        parts = text.split()
        return (len(parts) >= 4 and "/" in parts[0]
                and parts[1] in ("w", "b") and "\n" not in text.strip())

    def _reset_view_after_setup(self, status: str) -> None:
        self.paste_active = False
        self.paste_error = ""
        self.selected = None
        self.legal_targets = set()
        self.premoves = []
        self.last_move = None
        self.san_history = []
        self.engine_pv_san = []
        self._thinking = False
        self._clock_running = self.tc is not None
        self._last_tick = time.perf_counter()
        self.status = status

    def _step_replay(self) -> None:
        """Auto-play queued PGN moves one every ~0.4s, then hand control back."""
        if not self._replay_moves or self._thinking or self.paste_active:
            return
        now = time.perf_counter()
        if now < self._replay_next:
            return
        move = self._replay_moves.pop(0)
        board = self.engine.board
        if board.is_legal(move):
            self.san_history.append(board.san(move))
            self.engine.push(move)
            self.last_move = move
        self._replay_next = now + 0.4
        if not self._replay_moves:
            self.status = ("White to move" if board.turn == chess.WHITE
                           else "Black to move")

    def _build_move(self, frm: int, to: int) -> chess.Move | None:
        return self._build_move_on(self.engine.board, frm, to)

    def _build_move_on(self, board: chess.Board, frm: int,
                       to: int) -> chess.Move | None:
        piece = board.piece_at(frm)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
            return chess.Move(frm, to, promotion=chess.QUEEN)  # auto-queen
        return chess.Move(frm, to)

    def _make_human_move(self, move: chess.Move) -> None:
        # Record SAN before pushing (SAN needs the pre-move position).
        self.san_history.append(self.engine.board.san(move))
        self.engine.push(move)
        self.last_move = move
        mover_was_white = not self.engine.board.turn  # side that just moved
        self._add_increment(white=mover_was_white)
        if self.engine.board.is_game_over():
            self._clock_running = False
            self.status = _result_text(self.engine.board)
            return
        if self.both_sides:
            # Pass-and-play: no engine reply; the human moves the other side.
            self.status = ("White to move" if self.engine.board.turn == chess.WHITE
                           else "Black to move")
        else:
            self._start_engine_move()

    # -- clock ----------------------------------------------------------------
    def _tick_clock(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if not self._clock_running or self.tc is None:
            return
        if self.engine.board.turn == chess.WHITE:
            self.white_time -= dt
            if self.white_time <= 0:
                self.white_time = 0
                self._flag(chess.WHITE)
        else:
            self.black_time -= dt
            if self.black_time <= 0:
                self.black_time = 0
                self._flag(chess.BLACK)

    def _flag(self, color: bool) -> None:
        self._clock_running = False
        self.flagged = color
        winner = "Black" if color == chess.WHITE else "White"
        self.status = f"Flag - {winner} wins on time"

    def _add_increment(self, white: bool) -> None:
        if self.tc is None or self.tc.increment <= 0:
            return
        if white:
            self.white_time += self.tc.increment
        else:
            self.black_time += self.tc.increment

    def _engine_time_budget(self) -> float | None:
        """How long the engine may think this move. In a timed game it spends a
        fraction of its remaining time, so when low it moves fast (sacrificing
        depth) rather than flagging. Untimed => None (depth-limited only)."""
        if self.tc is None:
            return None
        remaining = self.white_time if not self.human_white else self.black_time
        # Classic time-budget heuristic: ~1/25 of remaining time plus most of
        # the increment, with a hard safety cap so it never risks flagging.
        budget = remaining / 25.0 + 0.8 * self.tc.increment
        budget = max(0.02, min(budget, remaining - 0.15))
        return max(0.02, budget)

    # -- engine threading -----------------------------------------------------
    def _start_engine_move(self) -> None:
        self._thinking = True
        self.status = f"Engine ({self.engine.style}) thinking..."
        board_copy = self.engine.board.copy()
        depth = self.depth
        movetime = self._engine_time_budget()

        def work():
            info = self.engine._searcher.search(board_copy, depth=depth,
                                                movetime=movetime)
            # Render the PV as SAN from the searched position (for the readout).
            pv_san = _pv_to_san(board_copy, info.pv)
            nps = int(info.nodes / info.elapsed) if info.elapsed > 0 else 0
            with self._lock:
                self._engine_result = info.best_move
                self._engine_info = (pv_san, info.depth, nps)
                self.eval_cp = info.score if board_copy.turn == chess.WHITE else -info.score

        threading.Thread(target=work, daemon=True).start()

    def _poll_engine(self) -> None:
        if not self._thinking:
            return
        with self._lock:
            move = self._engine_result
            if move is None:
                return
            self._engine_result = None
            info = self._engine_info
            self._engine_info = None
        self._thinking = False
        if info is not None:
            self.engine_pv_san, self.engine_depth, self.engine_nps = info
        if move is not None:
            engine_is_white = not self.human_white
            self.san_history.append(self.engine.board.san(move))
            self.engine.push(move)
            self.last_move = move
            self._add_increment(white=engine_is_white)
        if self.engine.board.is_game_over():
            self._clock_running = False
            self.status = _result_text(self.engine.board)
        else:
            self.status = "Your move"
            self._try_play_premove()

    def _try_play_premove(self) -> None:
        """After the engine moves, play the FRONT of the premove queue if it's
        now legal. If it isn't (the position changed), discard the whole queue —
        a broken first premove invalidates the ones chained behind it."""
        if not self.premoves:
            return
        move = self.premoves.pop(0)
        board = self.engine.board
        # Re-derive promotion in case the queued move lacked it.
        built = self._build_move(move.from_square, move.to_square)
        candidate = built if built is not None else move
        if board.is_legal(candidate):
            self._make_human_move(candidate)
            # The move we just played was ours; the rest of the queue waits for
            # the engine's next reply (handled by the normal poll->premove flow).
        else:
            self.premoves = []

    # -- geometry -------------------------------------------------------------
    def _square_at(self, x: int, y: int) -> int | None:
        if not (self.board_x <= x < self.board_x + self.board_px and
                self.board_y <= y < self.board_y + self.board_px):
            return None
        col = (x - self.board_x) // self.square
        row = (y - self.board_y) // self.square
        # Orient so the human's side is at the bottom.
        if self.human_white:
            file_, rank = col, 7 - row
        else:
            file_, rank = 7 - col, row
        return chess.square(file_, rank)

    def _square_topleft(self, square: int) -> tuple[int, int]:
        file_ = chess.square_file(square)
        rank = chess.square_rank(square)
        if self.human_white:
            col, row = file_, 7 - rank
        else:
            col, row = 7 - file_, rank
        return self.board_x + col * self.square, self.board_y + row * self.square

    # -- drawing --------------------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_board()
        self._draw_pieces()
        self._draw_premove()
        self._draw_evalbar()
        self._draw_panel()
        if self.paste_active:
            self._draw_paste_overlay()

    def _draw_paste_overlay(self) -> None:
        """A modal text box over the board for pasting a FEN or PGN."""
        w, h = self.screen.get_size()
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        self.screen.blit(dim, (0, 0))
        bw, bh = min(w - 80, 700), 200
        bx, by = (w - bw) // 2, (h - bh) // 2
        pygame.draw.rect(self.screen, PANEL_BG, (bx, by, bw, bh), border_radius=8)
        pygame.draw.rect(self.screen, (90, 90, 100), (bx, by, bw, bh), 2, border_radius=8)
        self._text("Paste a FEN or PGN  (Ctrl+V to paste, Enter to load, Esc to cancel)",
                   bx + 16, by + 14, self.small_font, TEXT)
        # Text field.
        fx, fy, fw, fh = bx + 16, by + 46, bw - 32, bh - 100
        pygame.draw.rect(self.screen, (26, 26, 30), (fx, fy, fw, fh), border_radius=6)
        shown = self.paste_text[-400:] if len(self.paste_text) > 400 else self.paste_text
        self._wrapped(shown + "▏", fx + 8, fy + 8, self.small_font, TEXT, fw - 16)
        if self.paste_error:
            self._text(self.paste_error, bx + 16, by + bh - 30, self.small_font,
                       (240, 120, 110))

    def _draw_board(self) -> None:
        board = self.engine.board
        sq = self.square
        for square in chess.SQUARES:
            x, y = self._square_topleft(square)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = LIGHT if light else DARK
            if self.last_move and square in (self.last_move.from_square,
                                             self.last_move.to_square):
                color = LASTMOVE
            if square == self.selected:
                color = HILITE
            pygame.draw.rect(self.screen, color, (x, y, sq, sq))
            if square in self.legal_targets and not self._thinking:
                cx, cy = x + sq // 2, y + sq // 2
                r = max(6, sq // 7) if board.piece_at(square) is None else sq // 2 - 4
                width = 0 if board.piece_at(square) is None else 4
                pygame.draw.circle(self.screen, (60, 60, 60), (cx, cy), r, width)

    def _draw_pieces(self) -> None:
        board = self.engine.board
        for square, piece in board.piece_map().items():
            glyph = PIECE_UNICODE[(piece.piece_type, piece.color)]
            x, y = self._square_topleft(square)
            center = (x + self.square // 2, y + self.square // 2)
            if piece.color:  # White: draw a dark outline behind the glyph.
                self._blit_glyph_outlined(glyph, center, WHITE_PIECE, WHITE_OUTLINE)
            else:
                surf = self.piece_font.render(glyph, True, BLACK_PIECE)
                self.screen.blit(surf, surf.get_rect(center=center))

    def _blit_glyph_outlined(self, glyph: str, center: tuple[int, int],
                             fill, outline) -> None:
        """Render `glyph` with a 1-2px outline by stamping the outline color at
        offsets, then the fill on top — makes white pieces legible on light
        squares."""
        off = max(1, self.square // 40)
        outline_surf = self.piece_font.render(glyph, True, outline)
        for dx, dy in ((-off, 0), (off, 0), (0, -off), (0, off),
                       (-off, -off), (off, -off), (-off, off), (off, off)):
            rect = outline_surf.get_rect(center=(center[0] + dx, center[1] + dy))
            self.screen.blit(outline_surf, rect)
        fill_surf = self.piece_font.render(glyph, True, fill)
        self.screen.blit(fill_surf, fill_surf.get_rect(center=center))

    def _draw_premove(self) -> None:
        """Red highlight on every queued premove's from/to squares + a red dot
        on each destination (chess.com style). Later premoves in the queue are
        drawn slightly more transparent so ordering reads visually."""
        if not self.premoves:
            return
        sq = self.square
        for i, pm in enumerate(self.premoves):
            alpha = max(60, 120 - i * 20)
            for square in (pm.from_square, pm.to_square):
                x, y = self._square_topleft(square)
                overlay = pygame.Surface((sq, sq), pygame.SRCALPHA)
                overlay.fill((*PREMOVE, alpha))
                self.screen.blit(overlay, (x, y))
            x, y = self._square_topleft(pm.to_square)
            cx, cy = x + sq // 2, y + sq // 2
            pygame.draw.circle(self.screen, PREMOVE, (cx, cy), max(7, sq // 6))

    def _draw_evalbar(self) -> None:
        x, y = self.evalbar_x, self.board_y
        h = self.board_px
        pygame.draw.rect(self.screen, (20, 20, 22), (x, y, EVALBAR_W, h))
        # Map eval (cp, White POV) to a 0..1 white fraction via a soft clamp.
        cp = max(-1000, min(1000, self.eval_cp))
        frac = 0.5 + 0.5 * (cp / 1000.0)
        white_h = int(h * frac)
        pygame.draw.rect(self.screen, (245, 245, 245),
                         (x, y + h - white_h, EVALBAR_W, white_h))
        # Eval number.
        txt = f"{self.eval_cp/100:+.1f}"
        label = self.small_font.render(txt, True, (120, 120, 120))
        self.screen.blit(label, (x - 2, y + h + 2))

    def _draw_panel(self) -> None:
        px = self.panel_x
        py = self.board_y
        pw = PANEL_W
        pygame.draw.rect(self.screen, PANEL_BG, (px, py, pw, self.board_px))
        pad = 14
        cx = px + pad
        y = py + pad

        y = self._text("STYLE ENGINE", cx, y, self.ui_font, TEXT); y += 6

        # Clocks (only when timed).
        if self.tc is not None:
            y = self._draw_clocks(cx, y, pw - 2 * pad); y += 6

        # Style button.
        prof = PROFILES[self.engine.style]
        self._style_btn_rect = pygame.Rect(cx, y, pw - 2 * pad, 40)
        pygame.draw.rect(self.screen, (66, 66, 74), self._style_btn_rect, border_radius=6)
        self._text(f"Style: {self.engine.style}", cx + 10, y + 10, self.ui_font, TEXT)
        y += 46
        # Wrapped description.
        y = self._wrapped(prof.description, cx, y, self.small_font, MUTED, pw - 2 * pad)
        y += 6

        # Depth control.
        self._text(f"Depth: {self.depth}", cx, y + 6, self.ui_font, TEXT)
        self._depth_dn_rect = pygame.Rect(cx + 120, y, 34, 32)
        self._depth_up_rect = pygame.Rect(cx + 160, y, 34, 32)
        pygame.draw.rect(self.screen, (66, 66, 74), self._depth_dn_rect, border_radius=6)
        pygame.draw.rect(self.screen, (66, 66, 74), self._depth_up_rect, border_radius=6)
        self._text("-", cx + 132, y + 4, self.ui_font, TEXT)
        self._text("+", cx + 172, y + 4, self.ui_font, TEXT)
        y += 44

        # Time control selector (applies on New Game).
        tc_label = self.tc.name if self.tc else "Untimed"
        pending = getattr(self, "_pending_tc", self.tc)
        pending_label = pending.name if pending else "Untimed"
        self._tc_btn_rect = pygame.Rect(cx, y, pw - 2 * pad, 32)
        pygame.draw.rect(self.screen, (66, 66, 74), self._tc_btn_rect, border_radius=6)
        show = pending_label if pending_label != tc_label else tc_label
        self._text(f"Time: {show}", cx + 10, y + 6, self.small_font, TEXT)
        y += 40

        # Pass-and-play toggle + position setup, side by side.
        half = (pw - 2 * pad - 8) // 2
        self._both_btn_rect = pygame.Rect(cx, y, half, 32)
        both_bg = (60, 84, 60) if self.both_sides else (66, 66, 74)
        pygame.draw.rect(self.screen, both_bg, self._both_btn_rect, border_radius=6)
        self._text("Both sides" if self.both_sides else "1-player",
                   cx + 8, y + 6, self.small_font, TEXT)
        self._setup_btn_rect = pygame.Rect(cx + half + 8, y, half, 32)
        pygame.draw.rect(self.screen, (66, 66, 74), self._setup_btn_rect, border_radius=6)
        self._text("Setup (FEN/PGN)", cx + half + 14, y + 6, self.small_font, TEXT)
        y += 40

        # New game + Analyze, side by side.
        half = (pw - 2 * pad - 8) // 2
        self._new_btn_rect = pygame.Rect(cx, y, half, 34)
        pygame.draw.rect(self.screen, (80, 60, 60), self._new_btn_rect, border_radius=6)
        self._text("New Game", cx + 8, y + 8, self.small_font, TEXT)
        self._analyze_btn_rect = pygame.Rect(cx + half + 8, y, half, 34)
        pygame.draw.rect(self.screen, (60, 66, 80), self._analyze_btn_rect, border_radius=6)
        self._text("Analyze game", cx + half + 14, y + 8, self.small_font, TEXT)
        y += 44

        # Engine's plan (principal variation) + eval + speed.
        eng = "NNUE" if self.engine.uses_nnue else "hand eval"
        self._text(f"Engine's plan  ({eng})", cx, y, self.small_font, MUTED)
        y += 20
        if self.engine_pv_san:
            plan = " ".join(self.engine_pv_san[:6])
            y = self._wrapped(plan, cx, y, self.small_font, TEXT, pw - 2 * pad)
            meta = f"eval {self.eval_cp/100:+.2f}  d{self.engine_depth}"
            if self.engine_nps:
                meta += f"  {self.engine_nps//1000}k nps"
            y = self._text(meta, cx, y + 2, self.small_font, MUTED)
        else:
            y = self._text("(waiting for engine)", cx, y, self.small_font, MUTED)
        y += 10

        # Move list (last several moves, paired by move number).
        self._text("Moves", cx, y, self.small_font, MUTED)
        y += 20
        self._draw_move_list(cx, y, pw - 2 * pad,
                             bottom=py + self.board_px - 44)

        # Status at the bottom.
        self._wrapped(self.status, cx, py + self.board_px - 34, self.small_font,
                      TEXT, pw - 2 * pad)

    def _draw_clocks(self, x: int, y: int, w: int) -> int:
        """Two clock chips: opponent (engine) on top, human below, matching how
        the board is oriented."""
        top_is_white = not self.human_white   # opponent on top
        top_time = self.white_time if top_is_white else self.black_time
        bot_time = self.black_time if top_is_white else self.white_time
        top_active = self.engine.board.turn == (chess.WHITE if top_is_white else chess.BLACK)
        for label, secs, active in (("Engine", top_time, top_active and self._clock_running),
                                    ("You", bot_time, (not top_active) and self._clock_running)):
            chip = pygame.Rect(x, y, w, 30)
            bg = (70, 90, 70) if active else (58, 58, 64)
            pygame.draw.rect(self.screen, bg, chip, border_radius=6)
            self._text(label, x + 10, y + 6, self.small_font, MUTED)
            tstr = _fmt_clock(secs)
            tcol = (240, 120, 110) if secs <= 10 else TEXT
            surf = self.ui_font.render(tstr, True, tcol)
            self.screen.blit(surf, surf.get_rect(midright=(x + w - 10, y + 15)))
            y += 34
        return y

    def _draw_move_list(self, x: int, y: int, w: int, bottom: int) -> None:
        """Render the game's SAN moves as 'N. white black' rows, showing the
        most recent rows that fit above `bottom`."""
        line_h = self.small_font.get_height() + 2
        rows: list[str] = []
        h = self.san_history
        for i in range(0, len(h), 2):
            num = i // 2 + 1
            white = h[i]
            black = h[i + 1] if i + 1 < len(h) else ""
            rows.append(f"{num:>2}. {white:7s} {black}")
        max_rows = max(1, (bottom - y) // line_h)
        for row in rows[-max_rows:]:
            y = self._text(row, x, y, self.small_font, TEXT)

    # -- text helpers ---------------------------------------------------------
    def _text(self, s: str, x: int, y: int, font, color) -> int:
        surf = font.render(s, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height()

    def _wrapped(self, s: str, x: int, y: int, font, color, max_w: int) -> int:
        words = s.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if font.size(test)[0] > max_w and line:
                y = self._text(line, x, y, font, color)
                line = w
            else:
                line = test
        if line:
            y = self._text(line, x, y, font, color)
        return y


def _pv_to_san(board: chess.Board, pv: list[chess.Move]) -> list[str]:
    """Convert a principal variation (list of moves) to SAN strings."""
    out: list[str] = []
    b = board.copy(stack=False)
    for mv in pv:
        if not b.is_legal(mv):
            break
        out.append(b.san(mv))
        b.push(mv)
    return out


def _fmt_clock(secs: float) -> str:
    secs = max(0.0, secs)
    if secs >= 60:
        m, s = divmod(int(secs), 60)
        return f"{m}:{s:02d}"
    # Under a minute, show tenths so bullet time pressure is visible.
    return f"{secs:04.1f}"


def _load_font(size: int, mono: bool = False):
    """Prefer a font that renders chess Unicode glyphs; fall back gracefully."""
    candidates = (["segoeuisymbol", "dejavusans", "arialunicodems"]
                  if not mono else ["consolas", "dejavusansmono", "couriernew"])
    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)


def _result_text(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate - {winner} wins"
    if board.is_stalemate():
        return "Draw - stalemate"
    if board.is_insufficient_material():
        return "Draw - insufficient material"
    if board.can_claim_threefold_repetition():
        return "Draw - repetition"
    return "Game over"
