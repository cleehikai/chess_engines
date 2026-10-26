"""High-level engine facade used by the GUI and the UCI adapter.

Holds the current position, a Searcher, and the active style profile. The eval
function handed to the searcher is the style-shaped eval, so switching style is
just swapping the profile and rebuilding the eval closure.
"""
from __future__ import annotations

from typing import Optional

import chess

from engine.evaluate import (
    EvalFeatures,
    cheap_features,
    evaluate as base_evaluate,
)
from engine.search import Searcher, SearchInfo
from style.profiles import StyleProfile, PROFILES, DEFAULT_PROFILE
from style.shaping import make_shaped_eval


class Engine:
    def __init__(self, style: str = DEFAULT_PROFILE,
                 nnue_checkpoint: Optional[str] = None):
        """If `nnue_checkpoint` is given, the NNUE net provides the position
        score while the hand-crafted evaluator supplies the positional features
        the style layer needs. Otherwise the hand-crafted eval is used for both.
        """
        self.board = chess.Board()
        self._profile: StyleProfile = PROFILES[style]
        self._nnue = None
        if nnue_checkpoint:
            # FastNNUE (Numba-JIT, fused bitboard eval) is ~8x faster than the
            # NumPy NNUEEvaluator and lets the search go deeper at equal time.
            # Falls back to the NumPy path if Numba is unavailable.
            try:
                from engine.nnue.fast_infer import FastNNUE
                self._nnue = FastNNUE(nnue_checkpoint)
            except Exception:
                from engine.nnue.infer import NNUEEvaluator
                self._nnue = NNUEEvaluator(nnue_checkpoint)
        self._searcher = Searcher(eval_fn=self._build_eval())

    @property
    def uses_nnue(self) -> bool:
        return self._nnue is not None

    # -- style ----------------------------------------------------------------
    @property
    def style(self) -> str:
        return self._profile.name

    def set_style(self, name: str) -> None:
        if name not in PROFILES:
            raise ValueError(f"unknown style {name!r}; choices: {list(PROFILES)}")
        self._profile = PROFILES[name]
        self._searcher = Searcher(eval_fn=self._build_eval())

    def _build_eval(self):
        if self._nnue is None:
            return make_shaped_eval(base_evaluate, self._profile)
        nnue = self._nnue

        # When the profile does no shaping (e.g. `balanced`), the style layer
        # never touches EvalFeatures, so computing them per node is pure waste.
        # Previously we called the full hand `evaluate()` on EVERY NNUE node just
        # to obtain features that were then discarded -- profiling showed that
        # hand eval was ~66% of the NNUE search's time. Hand the searcher the raw
        # NNUE score directly in that case; when shaping IS active, supply only
        # the cheap bitboard features (no PST loop) the style layer reads.
        if self._profile.max_shaping_pawns <= 0 and self._profile.contempt == 0:
            return nnue.eval_cp

        def base_eval_with_feats(board: chess.Board):
            return nnue.eval_cp(board), cheap_features(board)

        return make_shaped_eval(base_eval_with_feats, self._profile)

    # -- position -------------------------------------------------------------
    def new_game(self) -> None:
        self.board.reset()

    def set_fen(self, fen: str) -> None:
        self.board.set_fen(fen)

    def push_uci(self, uci: str) -> None:
        self.board.push_uci(uci)

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    # -- search ---------------------------------------------------------------
    def best_move(self, *, depth: int = 4,
                  movetime: float | None = None) -> SearchInfo:
        return self._searcher.search(self.board, depth=depth, movetime=movetime)

    def play_best(self, *, depth: int = 4,
                  movetime: float | None = None) -> Optional[chess.Move]:
        info = self.best_move(depth=depth, movetime=movetime)
        if info.best_move is not None:
            self.board.push(info.best_move)
        return info.best_move
