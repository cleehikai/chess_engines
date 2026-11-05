"""The explainer: rules facts it states, and the reasoning it builds from search.

Most of these run on a scripted search, so the logic (what counts as a threat,
when a line is forcing, what the summary leads with) is pinned down without an
engine. One test at the end runs the real Fairy-Stockfish when it is
installed, because a scripted search cannot show that real PVs survive the
walk.
"""

from __future__ import annotations

import chess
import chess.variant
import pytest

from lab import config, fsf
from trainer import analysis, explain, winrate

MODEL = winrate.WinModel(a=0.0, k=0.004, fitted=True, provenance={})


def line(*ucis: str, cp: int | None = 0, mate: int | None = None) -> analysis.Line:
    return analysis.Line(uci=ucis[0], cp=None if mate is not None else cp,
                         mate=mate, depth=10, pv=tuple(ucis))


class Scripted:
    """A search that answers from a table keyed on FEN, and records its calls."""

    def __init__(self, table: dict[str, list[analysis.Line]]) -> None:
        self.table = table
        self.calls: list[str] = []

    def __call__(self, fen: str, multipv: int) -> list[analysis.Line]:
        self.calls.append(fen)
        return self.table.get(fen, [])[:multipv]


def fen_after(*sans: str, start: str | None = None) -> str:
    board = chess.variant.AtomicBoard(start) if start else chess.variant.AtomicBoard()
    for san in sans:
        board.push_san(san)
    return board.fen()


def null_fen(fen: str) -> str:
    board = explain.null_board(chess.variant.AtomicBoard(fen))
    assert board is not None
    return board.fen()


# -- rules facts -------------------------------------------------------------


def test_capturer_is_counted_once_when_it_starts_inside_the_blast():
    # Qd8xc7: the queen's own square is adjacent to c7. It dies on c7, and
    # counting it again on d8 would report a queen lost twice.
    board = chess.variant.AtomicBoard("rnbqkbnr/ppNppppp/8/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1")
    ply = explain.describe_ply(board, chess.Move.from_uci("d8c7"))
    pieces = sorted((d["square"], d["piece"]) for d in ply["destroyed"])
    assert pieces == [("b8", "n"), ("c7", "N"), ("c7", "q"), ("c8", "b")]
    assert "d8" not in ply["blast"]
    values = explain.ATOMIC_PIECE_VALUES
    expected = values[chess.KNIGHT] - (
        values[chess.QUEEN] + values[chess.KNIGHT] + values[chess.BISHOP]
    )
    assert ply["delta"] == expected
    assert "the capturing queen dies with it" in ply["text"]


def test_pawns_in_the_radius_survive_and_are_not_reported():
    board = chess.variant.AtomicBoard("rnbqkbnr/ppNppppp/8/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1")
    ply = explain.describe_ply(board, chess.Move.from_uci("d8c7"))
    assert not any(d["square"] in ("b7", "d7") for d in ply["destroyed"])


def test_king_explosion_is_named_and_ends_the_line():
    # After 1.Nf3 f6?? 2.Ne5 ... the f7 square is next to e8.
    board = chess.variant.AtomicBoard("rnbqkbnr/ppppp1pp/8/4Np2/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    ply = explain.describe_ply(board, chess.Move.from_uci("e5d7"))
    assert ply["king"] == "black"
    assert "explodes Black's king on e8" in ply["text"]


def test_self_blast_note_names_own_pieces_but_not_the_capturer():
    # Bxe5 with White's own knight on d4 next to e5: the bishop dies as the
    # capturer, the knight dies as a neighbour, and only the knight is news.
    board = chess.variant.AtomicBoard("4k3/8/8/4n3/3N4/8/6B1/4K3 w - - 0 1")
    notes = explain.notes(board, chess.Move.from_uci("g2e4"))
    assert notes == []  # not a capture
    board = chess.variant.AtomicBoard("4k3/8/8/4n3/3N4/2B5/8/4K3 w - - 0 1")
    kinds = {n["kind"]: n for n in explain.notes(board, chess.Move.from_uci("d4e6"))}
    assert "self-blast" not in kinds
    board = chess.variant.AtomicBoard("4k3/8/8/4n3/3N4/8/8/B3K3 w - - 0 1")
    found = explain.notes(board, chess.Move.from_uci("a1e5"))
    assert [n["kind"] for n in found] == ["self-blast"]
    assert "knight (d4)" in found[0]["text"]
    assert "bishop" not in found[0]["text"]


def test_touching_kings_are_flagged():
    board = chess.variant.AtomicBoard("8/8/8/3k4/8/3K4/8/8 w - - 0 1")
    kinds = [n["kind"] for n in explain.notes(board, chess.Move.from_uci("d3d4"))]
    assert kinds == ["connected-kings"]


def test_null_board_refuses_a_position_in_check():
    board = chess.variant.AtomicBoard("4k3/8/8/8/8/8/4q3/R3K3 w - - 0 1")
    assert board.is_check()
    assert explain.null_board(board) is None


# -- proof lines --------------------------------------------------------------


def test_proof_cuts_at_the_king_explosion():
    fen = fen_after("Nf3", "h6")
    plies, forcing = explain.proof(fen, ["f3e5", "a7a6", "e5f7", "a6a5", "a2a3"])
    assert forcing
    assert [p["san"] for p in plies] == ["Ne5", "a6", "Nxf7#"]


def test_a_line_with_no_material_is_positional():
    plies, forcing = explain.proof(chess.STARTING_FEN, ["g1f3", "g8f6", "b1c3", "b8c6"])
    assert not forcing
    assert len(plies) == 2


def test_a_mate_longer_than_the_window_is_followed_to_the_end():
    # The real FSF line from the stress run: mate in 5, explosion on ply 11.
    fen = "rnbqkbnr/ppppp1pp/5p2/8/8/4PN2/PPPP1PPP/RNBQKB1R w KQkq - 0 2"
    pv = ["f3e5", "f6e5", "d1h5", "g7g6", "h5d5", "d7d6", "d5f7", "e8d7", "f7e7"]
    short, _ = explain.proof(fen, pv)
    long, forcing = explain.proof(fen, pv, mate=5)
    assert not short[-1]["king"]
    assert forcing and long[-1]["king"] == "black"


def test_line_text_numbers_moves_from_the_position():
    fen = fen_after("Nf3")
    plies = explain.walk(fen, ["f7f6", "e2e3", "e7e6"])
    assert explain._line_text(plies) == "1...f6 2.e3 e6"
    assert explain._line_text(plies, numbered=False) == "f6 e3 e6"


# -- the explanation --------------------------------------------------------


def test_a_defence_is_explained_by_the_threat_it_stops():
    fen = fen_after("Nf3")  # Black to move; Ne5 and Nxf7 is the threat
    after_f6 = fen_after("Nf3", "f6")
    search = Scripted({
        fen: [line("f7f6", "e2e3", cp=-50), line("h7h6", "f3e5", "a7a6", "e5f7", mate=-2)],
        null_fen(fen): [line("f3e5", "a7a6", "e5f7", mate=2)],
        null_fen(after_f6): [line("e7e6", "e2e3", cp=10)],
    })
    result = explain.explain(fen, search, model=MODEL)
    kinds = [r["kind"] for r in result["reasons"]]
    assert kinds[0] == "parries"
    assert "Nxf7#" in result["reasons"][0]["text"]
    assert "stops a threat" in result["summary"]
    # Why not h6: the threat lands, and the explanation says it was ignored.
    alternative = result["alternative"]
    assert alternative["san"] == "h6"
    assert alternative["reasons"][0]["kind"] == "refuted"
    assert "does nothing about the threat" in alternative["reasons"][0]["text"]


def test_a_move_the_user_chose_is_scored_by_searching_after_it():
    fen = fen_after("Nf3")
    after_a6 = fen_after("Nf3", "a6")
    search = Scripted({
        fen: [line("f7f6", "e2e3", cp=-50)],
        after_a6: [line("f3e5", "b7b6", "e5f7", mate=2)],
        null_fen(fen): [line("f3e5", "a7a6", "e5f7", mate=2)],
    })
    result = explain.explain(fen, search, move="a7a6", model=MODEL)
    assert after_a6 in search.calls
    alternative = result["alternative"]
    assert alternative["loss"] > 30
    assert alternative["verdict"] == "blunder"
    assert "Nxf7#" in alternative["reasons"][0]["text"]
    assert result["summary"].startswith("a6 costs")


def test_a_preference_with_no_forcing_line_says_so():
    fen = chess.STARTING_FEN
    search = Scripted({
        fen: [line("g1f3", "g8f6", cp=200), line("e2e4", "g8f6", "b1c3", cp=60)],
    })
    result = explain.explain(fen, search, model=MODEL)
    why = result["alternative"]["reasons"][0]
    assert why["kind"] == "positional"
    assert "no short forcing line" in why["text"]


def test_close_moves_are_not_called_worse():
    fen = chess.STARTING_FEN
    search = Scripted({fen: [line("g1f3", cp=100), line("e2e3", cp=98)]})
    result = explain.explain(fen, search, model=MODEL)
    assert result["alternative"]["reasons"][0]["kind"] == "close"


def test_an_immediate_win_needs_no_other_reason():
    fen = "rnbqkbnr/ppppp1pp/8/4Np2/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
    search = Scripted({fen: [line("e5d7", mate=1)]})
    result = explain.explain(fen, search, model=MODEL)
    assert [r["kind"] for r in result["reasons"]] == ["wins"]
    # No threat search was spent once the answer was known.
    assert search.calls == [fen]


def test_illegal_moves_are_rejected():
    search = Scripted({chess.STARTING_FEN: [line("g1f3")]})
    with pytest.raises(ValueError):
        explain.explain(chess.STARTING_FEN, search, move="e2e5", model=MODEL)


# -- the real engine ---------------------------------------------------------


@pytest.mark.skipif(not config.FSF_EXE.exists(), reason="Fairy-Stockfish not installed")
def test_real_engine_explains_the_f7_defence():
    """After 1.Nf3, Black has to cover f7 against Ne5 and Nxf7.

    Run on FSF's handcrafted atomic eval so it needs only the binary. That
    eval is far weaker than the net, but this threat is two moves deep and
    either one sees it.
    """
    with fsf.open_engine(fsf.FsfConfig(use_nnue=False, hash_mb=32)) as engine:
        def search(fen: str, multipv: int) -> list[analysis.Line]:
            infos = engine.analyse(
                chess.variant.AtomicBoard(fen),
                chess.engine.Limit(nodes=200_000), multipv=multipv,
            )
            return analysis._read_infos(infos)[0]

        result = explain.explain(fen_after("Nf3"), search, move="h7h6")

    assert result["alternative"]["verdict"] == "blunder"
    text = " ".join(r["text"] for r in result["alternative"]["reasons"])
    assert "explodes Black's king" in text
