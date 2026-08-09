"""Rule-level tests for the atomic edge cases engines get wrong.

Run: python -m pytest atomic/tests -q   (from the repo root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import chess.variant
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import perft as perft_mod  # noqa: E402
from lab.methods import ATOMIC_PIECE_VALUES, GreedyMaterialMethod, Budget  # noqa: E402
from lab.rules import explosion_squares, explodes_king, is_quiet  # noqa: E402


def sq(name: str) -> int:
    return chess.parse_square(name)


class TestExplosion:
    def test_non_capture_explodes_nothing(self):
        board = chess.variant.AtomicBoard()
        assert not explosion_squares(board, chess.Move.from_uci("e2e4"))

    def test_capture_takes_capturer_and_victim(self):
        board = chess.variant.AtomicBoard(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        blown = explosion_squares(board, chess.Move.from_uci("e4d5"))
        assert sq("d5") in blown, "landing square holds capturer and victim"

    def test_pawns_survive_adjacent_blast(self):
        # White knight captures on d5; the black pawn on c6 is adjacent but
        # pawns are immune to blast damage.
        board = chess.variant.AtomicBoard(
            "rnbqkbnr/pp2pppp/2p5/3p4/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 3"
        )
        blown = explosion_squares(board, chess.Move.from_uci("c3d5"))
        assert sq("c6") not in blown

    def test_matches_engine_after_push(self):
        """explosion_squares must predict exactly what the board does."""
        board = chess.variant.AtomicBoard(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        move = chess.Move.from_uci("e4d5")
        predicted = set(explosion_squares(board, move))
        before = {s for s in chess.SQUARES if board.piece_at(s)}
        board.push(move)
        after = {s for s in chess.SQUARES if board.piece_at(s)}
        removed = before - after
        # The capturer left its origin square by moving, not by exploding.
        assert removed - {move.from_square} == predicted - {move.from_square}


class TestAtomicLegality:
    def test_kings_may_be_adjacent(self):
        board = chess.variant.AtomicBoard("8/8/8/3kK3/8/8/8/8 w - - 0 1")
        assert not board.is_check()
        assert board.legal_moves.count() == 7, "8 neighbours minus the occupied one"

    def test_king_cannot_capture(self):
        board = chess.variant.AtomicBoard("8/8/8/3qK3/8/8/8/8 w - - 0 1")
        assert chess.Move.from_uci("e5d5") not in board.legal_moves

    def test_pinned_piece_may_move_if_it_explodes_the_king(self):
        """The rule most implementations get wrong.

        White's knight on e4 is absolutely pinned to its king by the rook on e8
        — in standard chess it has *zero* legal moves. In atomic, Nxd6 is legal
        because the explosion destroys the black king on c6, which ends the game
        before the discovered check can matter.
        """
        fen = "4r3/8/2kb4/8/4N3/8/8/4K3 w - - 0 1"
        knight = chess.E4

        standard = chess.Board(fen)
        assert [m for m in standard.legal_moves if m.from_square == knight] == []

        board = chess.variant.AtomicBoard(fen)
        atomic_moves = [m.uci() for m in board.legal_moves if m.from_square == knight]
        assert atomic_moves == ["e4d6"]

        move = chess.Move.from_uci("e4d6")
        assert explodes_king(board, move, chess.BLACK)
        board.push(move)
        assert board.is_variant_end()
        assert board.result() == "1-0"


class TestWinConditions:
    def test_exploding_the_enemy_king_wins(self):
        """Rxd7 never touches e8, but the blast reaches it."""
        board = chess.variant.AtomicBoard("4k3/3n4/8/8/8/8/8/3RK3 w - - 0 1")
        move = chess.Move.from_uci("d1d7")
        assert move in board.legal_moves
        assert explodes_king(board, move, chess.BLACK)
        board.push(move)
        assert board.is_variant_end()
        assert board.result() == "1-0"

    def test_own_king_explosion_is_illegal(self):
        """A capture that would blow up your own king is not a legal move."""
        # White rook on d7 could capture the knight on d6, but White's king on
        # c5 is adjacent to d6 and would be destroyed too.
        board = chess.variant.AtomicBoard("4k3/3R4/3n4/2K5/8/8/8/8 w - - 0 1")
        assert chess.Move.from_uci("d7d6") not in board.legal_moves


class TestQuiescence:
    def test_startpos_is_quiet(self):
        assert is_quiet(chess.variant.AtomicBoard())

    def test_position_with_a_capture_is_not_quiet(self):
        board = chess.variant.AtomicBoard(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        assert not is_quiet(board)


class TestPieceValues:
    def test_bishops_outvalue_knights(self):
        """Pav (2025): bishop 2.5 pawns vs knight 1.9 in atomic."""
        assert ATOMIC_PIECE_VALUES[chess.BISHOP] > ATOMIC_PIECE_VALUES[chess.KNIGHT]

    def test_values_are_compressed_versus_standard_chess(self):
        """A knight is worth well under 3 pawns in atomic."""
        ratio = ATOMIC_PIECE_VALUES[chess.KNIGHT] / ATOMIC_PIECE_VALUES[chess.PAWN]
        assert 1.5 < ratio < 2.5


class TestMethods:
    def test_greedy_returns_a_legal_move(self):
        board = chess.variant.AtomicBoard()
        move = GreedyMaterialMethod(seed=1).choose(board, Budget(nodes=100))
        assert move in board.legal_moves

    def test_greedy_leaves_the_board_untouched(self):
        board = chess.variant.AtomicBoard()
        before = board.fen()
        GreedyMaterialMethod(seed=1).choose(board, Budget(nodes=100))
        assert board.fen() == before
        assert not board.move_stack


REFERENCE = Path(__file__).parent / "perft_positions.json"


@pytest.mark.skipif(not REFERENCE.exists(), reason="run `python -m lab.perft --generate`")
@pytest.mark.parametrize("name", list(json.loads(REFERENCE.read_text())["positions"]))
def test_perft_matches_fairy_stockfish(name: str):
    """python-chess must agree with Fairy-Stockfish on atomic move generation."""
    entry = json.loads(REFERENCE.read_text())["positions"][name]
    board = chess.variant.AtomicBoard(entry["fen"])
    for depth_s, expected in entry["perft"].items():
        depth = int(depth_s)
        if depth > 3:  # keep the suite fast; deeper checks live in lab.perft
            continue
        assert perft_mod.python_chess_perft(board, depth) == expected, (
            f"{name} depth {depth}"
        )


atomic_core = pytest.importorskip(
    "atomic_core", reason="run `python -m lab.build_core`"
)


class TestM1Search:
    """The Rust searcher's atomic-specific behaviour.

    The value of these is that the search takes shortcuts no standard-chess
    engine can take — returning a move without searching it because its blast
    radius contains the enemy king — and a shortcut that is subtly wrong would
    silently produce illegal moves rather than weak ones.
    """

    def test_finds_the_pinned_knight_regicide(self):
        """The knight is absolutely pinned, yet Nxd6 wins on the spot."""
        fen = "4r3/8/2kb4/8/4N3/8/8/4K3 w - - 0 1"
        engine = atomic_core.Engine()
        move, score, _depth, nodes = engine.search(fen, 10_000)
        assert move == "e4d6"
        assert score > 29_000, "a king-exploding capture must score as a win"
        assert nodes == 0, "an immediate win should not need to be searched"

    def test_never_plays_a_win_that_also_kills_its_own_king(self):
        """Rxd6 catches the black king on e7 — and the white king on c5.

        The single most dangerous shortcut in the searcher: 'this capture
        explodes their king, return it immediately'. It is only sound because
        `legal_moves()` has already discarded self-destructive captures.
        """
        fen = "8/4k3/3n4/2K5/8/8/8/3R4 w - - 0 1"
        legal = {m.uci() for m in chess.variant.AtomicBoard(fen).legal_moves}
        assert "d1d6" not in legal
        move, _score, _depth, _nodes = atomic_core.Engine().search(fen, 20_000)
        assert move in legal

    def test_search_returns_only_legal_moves(self):
        """Play a full self-game with python-chess validating every move."""
        engine = atomic_core.Engine()
        board = chess.variant.AtomicBoard()
        for _ in range(60):
            if board.is_game_over():
                break
            uci, _score, _depth, _nodes = engine.search(board.fen(), 4_000)
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, f"illegal {uci} in {board.fen()}"
            board.push(move)

    def test_blast_gain_prices_the_whole_radius(self):
        """A capture is not worth its victim: the capturer always dies too.

        Here Rxd6 wins a knight (190) but loses the rook (486) and, because it
        is adjacent to the landing square, the white king. Standard MVV-LVA
        would rate this capture positively; the atomic version must not.
        """
        engine = atomic_core.Engine()
        gain = engine.blast_gain("4k3/3R4/3n4/8/8/8/8/4K3 w - - 0 1", "d7d6")
        assert gain == 190 - 486

    def test_unknown_eval_term_is_rejected(self):
        """A typo in an ablation would silently run a different experiment."""
        with pytest.raises(ValueError):
            atomic_core.Engine(terms=["king_saftey"])

    def test_ablation_terms_match_the_python_registry(self):
        from lab.methods import EVAL_TERMS

        assert tuple(atomic_core.EVAL_TERMS) == EVAL_TERMS
        assert tuple(atomic_core.Engine().terms) == EVAL_TERMS


class TestUiServer:
    """The play-UI's state payload.

    The page trusts the server for legality and for what exploded, so these
    are the two things worth pinning down.
    """

    def test_state_lists_legal_moves_by_origin_square(self):
        from lab import server

        state = server._state(chess.variant.AtomicBoard())
        assert state["turn"] == "white"
        assert not state["over"]
        assert sorted(state["legal"]["e2"]) == ["e2e3", "e2e4"]
        # 8 pawns + 2 knights can move from the starting position.
        assert len(state["legal"]) == 10

    def test_state_reports_the_explosion(self):
        """exd5 destroys both pawns; nothing else is adjacent and non-pawn."""
        from lab import server

        board = chess.variant.AtomicBoard()
        for uci in ("e2e4", "d7d5"):
            board.push(chess.Move.from_uci(uci))
        board, move = server._board_from(board.fen(), "e4d5")
        state = server._state(board, move)
        assert state["blast"] == ["d5"]
        assert state["last"] == "e4d5"
        assert "d5" not in server._state(board)["blast"], "blast is per-move, not sticky"

    def test_illegal_moves_are_refused(self):
        from lab import server

        with pytest.raises(ValueError):
            server._board_from(chess.variant.AtomicBoard().fen(), "a1a8")

    def test_levels_never_invent_a_rating(self):
        """A difficulty level with no measurement must say so, not guess."""
        from lab import server

        server._levels.cache_clear()
        for level in server._levels():
            assert level["measured"] == (level["elo"] != "not yet measured")
        server._levels.cache_clear()
