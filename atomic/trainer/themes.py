"""What a puzzle is *about*, in atomic terms.

Standard tactical vocabulary mostly does not survive contact with this variant.
A fork assumes the forking piece collects one of the two targets next move, but
in atomic the collector dies with its victim; a pin assumes the pinned piece
may not move, but here it may, gladly, if moving it explodes the enemy king. So
the taxonomy is rebuilt from the rules rather than inherited: each theme below
is a *detector over the solution line*, written against `lab.rules`, and each
one is a hypothesis in the same sense as the evaluation terms in
`docs/method-catalog.md`.

They are meant to be falsified. `firing_rates` reports how often each one fires
over the corpus, and the honest reading of that table is:

- a theme that never fires describes nothing and should be deleted;
- a theme that fires on nearly everything ("this puzzle contains a capture")
  separates nothing and should be narrowed.

Detectors run over the whole line, not just the first move, because the point
of an atomic puzzle is usually the *reason* the obvious refutation fails —
which is a fact about the defender's reply, two plies in.

One deliberate omission: no detector looks at the evaluation. A theme is a
statement about the position, and evaluation-derived labels ("winning",
"crushing") would just be the win% column wearing a hat.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import chess
import chess.variant

from lab.rules import explosion_squares

#: Every theme this module can produce, with the one-line description the UI
#: shows. Order is the order they are displayed in.
CATALOG: dict[str, str] = {
    "king-blast": "the winning move detonates on the enemy king",
    "regicidal": "legal only because it explodes the king — a pin or a check ignored",
    "blast-fork": "one capture destroys two or more enemy pieces",
    "sacrifice": "hands over material to light a fuse that cannot be put out",
    "quiet": "the move that wins is not a capture and not a check",
    "defensive-diffusion": "steps a piece out of a blast radius before it goes off",
    "own-king-in-blast": "the defence is that they cannot capture without dying",
    "connected-kings": "the kings stand adjacent and neither can be captured",
    "pawn-immunity": "a pawn survives an explosion that clears everything else",
    "ep-blast": "en passant, with the blast centred on the landing square",
    "no-capture-fortress": "held because no capture on the board is playable",
    "promotion": "a pawn promotes in the solution line",
    "tb-conversion": "exact tablebase ground truth: one move holds the result",
    "opening-trap": "inside book depth, off a line the field actually plays",
    "long": "four solver moves or more",
}

#: Where a puzzle sits in the game. Mechanical, and deliberately not inherited
#: from standard chess: atomic's median game is 30 ply, so an "opening" that
#: ran to move 15 would cover most of it.
OPENING_PLY = 16
ENDGAME_PIECES = 8

#: A line of four solver moves or more is `long`. Atomic lines are short —
#: the whole game has a 30-ply median — so this is not the standard-chess
#: notion of a long combination.
LONG_LINE = 4


def phase_of(board: chess.Board, ply: int | None) -> str:
    """opening / middlegame / endgame, by ply and by how much is left."""
    if bin(board.occupied).count("1") <= ENDGAME_PIECES:
        return "endgame"
    if ply is not None and ply <= OPENING_PLY:
        return "opening"
    return "middlegame"


# -- the walk --------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One move of the solution, with the position it was played in.

    `solver` marks the moves the human has to find; the others are the
    defender's best tries, which are part of the puzzle's content even though
    nobody is asked to play them.
    """

    board: chess.variant.AtomicBoard   # position BEFORE the move
    move: chess.Move
    solver: bool

    @property
    def blast(self) -> chess.SquareSet:
        return explosion_squares(self.board, self.move)

    @property
    def mover(self) -> chess.Color:
        return self.board.turn


def walk(fen: str, solution: Sequence[str]) -> Iterator[Step]:
    """Replay a solution line, yielding a `Step` per move.

    Raises `ValueError` on an illegal line, which is a generator bug rather
    than a user error and should not be quietly labelled.
    """
    board = chess.variant.AtomicBoard(fen)
    for index, uci in enumerate(solution):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal solution move {uci} at index {index} in {fen}")
        yield Step(board=board.copy(stack=False), move=move, solver=index % 2 == 0)
        board.push(move)


def final_board(fen: str, solution: Sequence[str]) -> chess.variant.AtomicBoard:
    board = chess.variant.AtomicBoard(fen)
    for uci in solution:
        board.push(chess.Move.from_uci(uci))
    return board


# -- helpers the detectors share -------------------------------------------

#: Material in the units `lab.methods` measured for atomic, coarse on purpose:
#: `sacrifice` only needs to know that something substantial was handed over.
_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 10,
    chess.KING: 0,
}


def _material(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece_type, value in _VALUE.items():
        total += value * len(board.pieces(piece_type, color))
        total -= value * len(board.pieces(piece_type, not color))
    return total


def _victims(step: Step) -> list[chess.Piece]:
    """Enemy pieces destroyed by this move, the direct victim included."""
    return [
        piece
        for square in step.blast
        if (piece := step.board.piece_at(square)) is not None
        and piece.color != step.mover
    ]


def _suicidal_captures(board: chess.variant.AtomicBoard) -> list[chess.Move]:
    """Captures the side to move would like to have, and cannot: they die too.

    `AtomicBoard.legal_moves` has already filtered these out, so they have to
    be recovered from the pseudo-legal set. This is the mechanism behind half
    of atomic's defensive ideas — a piece is protected not because it is
    defended but because taking it is fatal — and no standard-chess concept
    corresponds to it.
    """
    king = board.king(board.turn)
    if king is None:
        return []
    out = []
    for move in board.pseudo_legal_moves:
        if not board.is_capture(move):
            continue
        if king in explosion_squares(board, move):
            out.append(move)
    return out


def _kings_adjacent(board: chess.Board) -> bool:
    white, black = board.king(chess.WHITE), board.king(chess.BLACK)
    if white is None or black is None:
        return False
    return bool(chess.BB_KING_ATTACKS[white] & chess.BB_SQUARES[black])


def _gives_check(step: Step) -> bool:
    after = step.board.copy(stack=False)
    after.push(step.move)
    return after.is_check()


# -- the detectors ---------------------------------------------------------
#
# Each takes the full list of steps and returns True if the theme applies.
# Solver steps and defender steps are separated by `step.solver`, because
# "the solver sacrificed" and "the defender had to sacrifice" are different
# statements and conflating them would make every theme fire twice as often.


def _solver_steps(steps: Sequence[Step]) -> list[Step]:
    return [step for step in steps if step.solver]


def _king_blast(steps: Sequence[Step]) -> bool:
    return any(
        step.board.king(not step.mover) in step.blast for step in _solver_steps(steps)
    )


def _regicidal(steps: Sequence[Step]) -> bool:
    """A move that standard-chess legality would forbid, and atomic permits.

    Two forms, both of which come down to the same rule: exploding the enemy
    king ends the game immediately, so nothing else about the position — being
    in check, having the piece pinned — gets a chance to matter.
    """
    for step in _solver_steps(steps):
        if step.board.king(not step.mover) not in step.blast:
            continue
        if step.board.is_check():
            return True  # ignored a check to win first
        if step.board.is_pinned(step.mover, step.move.from_square):
            return True  # moved a pinned piece; the pin never resolves
    return False


def _blast_fork(steps: Sequence[Step]) -> bool:
    """One explosion takes two or more enemy pieces — atomic's real fork.

    Pawns are excluded from the count. They are immune to blast damage, so a
    pawn in the radius is not a casualty; only the directly captured one can
    be, and counting it would fire this on every ordinary pawn capture.
    """
    for step in _solver_steps(steps):
        killed = [p for p in _victims(step) if p.piece_type != chess.PAWN]
        if len(killed) >= 2:
            return True
    return False


def _sacrifice(steps: Sequence[Step]) -> bool:
    """Material goes down and stays down, and the line wins anyway.

    Measured across the solver's first move and the defender's answer to it,
    which is the only honest window: a capture that looks like a sacrifice and
    is recaptured for free is not one, and in atomic the recapture is often
    impossible, which is exactly the idea being named.
    """
    solver_moves = _solver_steps(steps)
    if not solver_moves:
        return False
    first = solver_moves[0]
    mover = first.mover
    before = _material(first.board, mover)
    reply = steps[1] if len(steps) > 1 else None
    after_board = final_board(
        first.board.fen(),
        [first.move.uci()] + ([reply.move.uci()] if reply else []),
    )
    return _material(after_board, mover) <= before - 2


def _quiet(steps: Sequence[Step]) -> bool:
    """Captures are on the board, and the move that works is not one of them.

    Worth a name because it is the shape this account's own errors take: 50 of
    56 blunders had a quiet move as the refutation. In a variant where every
    threat is a capture, declining all of them is the whole difficulty.

    The "captures are available" clause is not decoration. Without it this
    fired on 73% of the corpus — most endgame positions have no capture to
    make, so "the answer is a quiet move" said nothing about them. A theme
    that describes three quarters of everything is not a theme.
    """
    solver_moves = _solver_steps(steps)
    if not solver_moves:
        return False
    first = solver_moves[0]
    if first.board.is_capture(first.move) or _gives_check(first):
        return False
    return any(first.board.is_capture(move) for move in first.board.legal_moves)


def _defensive_diffusion(steps: Sequence[Step]) -> bool:
    """A quiet solver move that walks a piece out of a live blast radius.

    The test is concrete rather than positional: the square the piece left is
    inside the explosion of some capture the opponent could actually play, and
    the square it arrived on is not.
    """
    for step in _solver_steps(steps):
        if step.board.is_capture(step.move):
            continue
        enemy = step.board.copy(stack=False)
        enemy.turn = not step.mover
        radius = chess.SquareSet()
        for move in enemy.pseudo_legal_moves:
            if enemy.is_capture(move):
                radius |= explosion_squares(enemy, move)
        if step.move.from_square in radius and step.move.to_square not in radius:
            return True
    return False


def _own_king_in_blast(steps: Sequence[Step], last: chess.variant.AtomicBoard) -> bool:
    """The defender is held together by captures they are not allowed to make.

    Checked in every position where the *defender* is on move, the final one
    included — the line usually ends precisely because the recapture they want
    would take their own king with it, and that position has no move attached
    to it to hang the test on.
    """
    boards = [step.board for step in steps if not step.solver]
    if not last.is_game_over(claim_draw=True):
        boards.append(last)
    return any(_suicidal_captures(board) for board in boards)


def _connected_kings(steps: Sequence[Step], last: chess.Board) -> bool:
    return any(_kings_adjacent(step.board) for step in steps) or _kings_adjacent(last)


def _pawn_immunity(steps: Sequence[Step]) -> bool:
    """A pawn stands in the fire and walks out of it.

    Only counts when the pawn is *adjacent* to the detonation rather than on
    it: a pawn on the landing square is captured directly and dies like
    anything else, so its survival is not the rule being exploited.
    """
    for step in _solver_steps(steps):
        blast = step.blast
        if not blast:
            continue
        neighbours = chess.SquareSet(chess.BB_KING_ATTACKS[step.move.to_square])
        for square in neighbours:
            piece = step.board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN and square not in blast:
                return True
    return False


def _ep_blast(steps: Sequence[Step]) -> bool:
    return any(step.board.is_en_passant(step.move) for step in _solver_steps(steps))


def _no_capture_fortress(steps: Sequence[Step], last: chess.variant.AtomicBoard) -> bool:
    """The fuse goes out: captures were available, and by the end there are none.

    In atomic that is a real structural fact rather than a curiosity — with no
    capture available there is no way to start an explosion, and a position
    with no explosions in it cannot be lost quickly.

    The "were available" clause matters for the same reason as in `quiet`.
    Testing only the final position fired on 60% of the corpus, because a
    four-piece endgame usually has no captures in it and never did; that is a
    fact about small endgames, not about the puzzle.
    """
    if last.is_game_over(claim_draw=True):
        return False
    if any(last.is_capture(move) for move in last.legal_moves):
        return False
    return any(
        step.board.is_capture(move)
        for step in steps
        for move in step.board.legal_moves
    )


def _promotion(steps: Sequence[Step]) -> bool:
    return any(step.move.promotion for step in _solver_steps(steps))


# -- the entry point -------------------------------------------------------


def detect(
    fen: str,
    solution: Sequence[str],
    *,
    source: str = "game",
    in_book: bool = False,
    ply: int | None = None,
) -> list[str]:
    """Every theme that fires on this puzzle, in catalog order.

    `source` and `in_book` are facts about provenance rather than about the
    position, so they are passed in rather than sniffed: `tb-conversion` means
    "a tablebase said so", which no amount of staring at the board reveals.
    """
    steps = list(walk(fen, solution))
    last = final_board(fen, solution)
    fired = {
        "king-blast": _king_blast(steps),
        "regicidal": _regicidal(steps),
        "blast-fork": _blast_fork(steps),
        "sacrifice": _sacrifice(steps),
        "quiet": _quiet(steps),
        "defensive-diffusion": _defensive_diffusion(steps),
        "own-king-in-blast": _own_king_in_blast(steps, last),
        "connected-kings": _connected_kings(steps, last),
        "pawn-immunity": _pawn_immunity(steps),
        "ep-blast": _ep_blast(steps),
        "no-capture-fortress": _no_capture_fortress(steps, last),
        "promotion": _promotion(steps),
        "tb-conversion": source == "tablebase",
        "opening-trap": in_book and (ply is None or ply <= OPENING_PLY),
        "long": len(_solver_steps(steps)) >= LONG_LINE,
    }
    return [theme for theme in CATALOG if fired.get(theme)]


# -- validation ------------------------------------------------------------


@dataclass
class Firing:
    """How often one theme fires, and therefore whether it is worth having."""

    theme: str
    count: int
    total: int

    @property
    def rate(self) -> float:
        return 100.0 * self.count / self.total if self.total else 0.0

    #: Below this many puzzles, a firing rate says more about the corpus than
    #: about the detector, and the report should decline to judge rather than
    #: condemn a theme on four examples.
    MIN_CORPUS = 40

    #: Themes that record where a puzzle came from rather than what is in it.
    #: A breadth test is meaningless for these: `tb-conversion` firing on 62%
    #: of the corpus means 62% of the corpus came from the tablebase, which is
    #: a fact about the generators and not a defect in a detector.
    PROVENANCE = frozenset({"tb-conversion", "opening-trap"})

    @property
    def verdict(self) -> str:
        """The judgement `docs/trainer-plan.md` §4 asks for, applied.

        Thresholds are conventions, not measurements, and are named as such:
        a theme carrying under 2% of the corpus is not distinguishing puzzles
        from each other, and one carrying over 60% is barely doing so either.

        One systematic caveat the numbers cannot show: the generator rejects
        anything `M0-greedy` solves, and in atomic that removes almost every
        one-move king blast. So `king-blast` reading zero is a fact about the
        triviality filter, not evidence that the detector is broken.
        """
        if self.total < self.MIN_CORPUS:
            return "too few puzzles to judge" if self.count == 0 else "fires"
        if self.theme in self.PROVENANCE:
            return "provenance — breadth is not a defect"
        if self.count == 0:
            return "never fires — delete it"
        if self.rate < 2.0:
            return "rare"
        if self.rate > 60.0:
            return "too broad — narrow it"
        return "ok"


def firing_rates(db: sqlite3.Connection) -> list[Firing]:
    """The firing rate of every theme in the catalog over the stored corpus."""
    total = db.execute("SELECT COUNT(*) AS n FROM puzzles").fetchone()["n"]
    counts = {
        row["theme"]: row["n"]
        for row in db.execute(
            "SELECT theme, COUNT(*) AS n FROM puzzle_themes GROUP BY theme"
        )
    }
    return [
        Firing(theme=theme, count=counts.get(theme, 0), total=total)
        for theme in CATALOG
    ]


def describe(themes: Iterable[str]) -> str:
    return ", ".join(themes) or "untagged"
