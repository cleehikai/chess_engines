"""Play atomic games between two players and record the results.

A "player" here is either a `Method` (something we are testing) or a rung of
the Fairy-Stockfish ladder (something we measure against). Both are wrapped in
the same `Player` interface so a gauntlet, a round robin, and a head-to-head
are all the same code path.

Fairness rules baked in, because getting these wrong silently corrupts every
Elo the project ever produces:

- **Colour balance.** Games are played in pairs from the same opening, once
  with each colour. In atomic White's first-move advantage is roughly *three
  times* the standard-chess figure (Pav 2025, Table 9), so an odd number of
  games or an unbalanced colour split is not a small bias — it is the dominant
  term.
- **Opening diversity.** Both players start from the same randomly chosen
  short opening line, otherwise every game is a replay of the same forcing
  theory and the sample size is a lie.
- **Ply cap.** Atomic games are short (median 30 ply); a game past `max_plies`
  is a harness bug or a shuffling draw, and is scored as a draw either way.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess
import chess.engine
import chess.variant

from . import fsf
from .methods import Budget, Method
from .rules import game_result


class Player(Protocol):
    name: str

    def move(self, board: chess.variant.AtomicBoard) -> chess.Move: ...
    def new_game(self) -> None: ...


class MethodPlayer:
    """Wraps a `Method` at a fixed budget."""

    def __init__(self, method: Method, budget: Budget) -> None:
        if not method.supports(budget):
            raise ValueError(f"{method.name} cannot honour a {budget.currency} budget")
        self._method = method
        self._budget = budget
        self.name = f"{method.name}@{budget}"

    def move(self, board: chess.variant.AtomicBoard) -> chess.Move:
        return self._method.choose(board, self._budget)

    def new_game(self) -> None:
        self._method.reset()


class FsfPlayer:
    """Wraps a live Fairy-Stockfish session at one rung of the ladder."""

    def __init__(self, engine: chess.engine.SimpleEngine, cfg: fsf.FsfConfig) -> None:
        self._engine = engine
        self._cfg = cfg
        self.name = cfg.name

    def move(self, board: chess.variant.AtomicBoard) -> chess.Move:
        result = self._engine.play(board, self._cfg.limit, game=id(board))
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move for {board.fen()}")
        return result.move

    def new_game(self) -> None:
        return None


# --------------------------------------------------------------------------


@dataclass
class GameRecord:
    white: str
    black: str
    result: str           # "1-0" / "0-1" / "1/2-1/2"
    plies: int
    termination: str      # "explosion" / "checkmate" / "draw" / "ply-cap" / "error"
    opening: str          # UCI moves of the shared opening line
    seconds: float
    moves: str = ""       # full game in UCI, space separated

    def score_for(self, name: str) -> float | None:
        """1.0 win / 0.5 draw / 0.0 loss for `name`, or None if not a player."""
        if self.result == "1/2-1/2":
            return 0.5 if name in (self.white, self.black) else None
        winner = self.white if self.result == "1-0" else self.black
        loser = self.black if self.result == "1-0" else self.white
        if name == winner:
            return 1.0
        if name == loser:
            return 0.0
        return None


@dataclass
class MatchResult:
    player_a: str
    player_b: str
    games: list[GameRecord] = field(default_factory=list)

    @property
    def score_a(self) -> float:
        return sum(g.score_for(self.player_a) or 0.0 for g in self.games)

    @property
    def wins_draws_losses(self) -> tuple[int, int, int]:
        w = d = ll = 0
        for g in self.games:
            s = g.score_for(self.player_a)
            w += s == 1.0
            d += s == 0.5
            ll += s == 0.0
        return w, d, ll

    def __str__(self) -> str:
        w, d, ll = self.wins_draws_losses
        n = len(self.games)
        pct = 100 * self.score_a / n if n else 0.0
        return (
            f"{self.player_a} vs {self.player_b}: "
            f"+{w} ={d} -{ll}  ({self.score_a}/{n}, {pct:.1f}%)"
        )

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for g in self.games:
                fh.write(json.dumps(dataclasses.asdict(g)) + "\n")


# --------------------------------------------------------------------------


def random_openings(count: int, plies: int = 4, seed: int = 0) -> list[list[chess.Move]]:
    """Short random-but-legal opening lines, used as balanced game starts.

    Random rather than book because atomic has no reliable opening book, and
    because a book would bias the sample toward whatever theory the book author
    believed. Lines that end the game early are rejected.
    """
    rng = random.Random(seed)
    openings: list[list[chess.Move]] = []
    guard = 0
    while len(openings) < count and guard < count * 100:
        guard += 1
        board = chess.variant.AtomicBoard()
        line: list[chess.Move] = []
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            board.push(move)
            line.append(move)
        if len(line) == plies and not board.is_game_over():
            openings.append(line)
    return openings


def play_game(
    white: Player,
    black: Player,
    opening: Sequence[chess.Move] = (),
    max_plies: int = 400,
) -> GameRecord:
    """Play one game to completion and return the record."""
    board = chess.variant.AtomicBoard()
    white.new_game()
    black.new_game()
    for move in opening:
        board.push(move)

    start = time.perf_counter()
    termination = "ply-cap"
    result = "1/2-1/2"
    while board.ply() < max_plies:
        outcome = game_result(board)
        if outcome is not None:
            result = outcome
            if board.is_variant_end():
                termination = "explosion"
            elif board.is_checkmate():
                termination = "checkmate"
            else:
                termination = "draw"
            break
        player = white if board.turn == chess.WHITE else black
        try:
            move = player.move(board)
        except Exception as exc:  # a crashed method forfeits, it does not
            termination = f"error:{type(exc).__name__}"  # abort the gauntlet
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
        if move not in board.legal_moves:
            termination = "error:illegal-move"
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
        board.push(move)

    return GameRecord(
        white=white.name,
        black=black.name,
        result=result,
        plies=board.ply(),
        termination=termination,
        opening=" ".join(m.uci() for m in opening),
        seconds=round(time.perf_counter() - start, 3),
        moves=" ".join(m.uci() for m in board.move_stack),
    )


def play_match(
    player_a: Player,
    player_b: Player,
    games: int,
    opening_plies: int = 4,
    seed: int = 0,
    max_plies: int = 400,
    progress: bool = True,
) -> MatchResult:
    """Play `games` games, colour-balanced in pairs over shared openings."""
    if games % 2:
        games += 1  # colour balance is not optional
    openings = random_openings(games // 2, plies=opening_plies, seed=seed)
    match = MatchResult(player_a.name, player_b.name)
    for i, opening in enumerate(openings):
        for white, black in ((player_a, player_b), (player_b, player_a)):
            record = play_game(white, black, opening, max_plies=max_plies)
            match.games.append(record)
            if progress:
                print(f"  [{len(match.games):>3}/{games}] {record.result} "
                      f"{record.termination:<12} {record.plies:>3} ply  "
                      f"{white.name} vs {black.name}")
    return match


def gauntlet(
    method: Method,
    budget: Budget,
    rungs: Sequence[int] = fsf.LADDER,
    games_per_rung: int = 20,
    seed: int = 0,
    log: Path | None = None,
) -> list[MatchResult]:
    """Play `method` against each rung of the Fairy-Stockfish ladder."""
    player = MethodPlayer(method, budget)
    results: list[MatchResult] = []
    for rung in rungs:
        cfg = fsf.FsfConfig(nodes=rung)
        with fsf.open_engine(cfg) as engine:
            print(f"\n{player.name}  vs  {cfg.name}")
            match = play_match(
                player, FsfPlayer(engine, cfg), games_per_rung, seed=seed
            )
        print(f"  => {match}")
        results.append(match)
        if log is not None:
            match.to_jsonl(log)
        if match.score_a == 0:
            print("  (shut out — skipping the stronger rungs)")
            break
    return results
