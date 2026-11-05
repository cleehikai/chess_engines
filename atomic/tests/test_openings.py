"""Opening names and opening traps — the parts that decide, not the engine.

Both modules ask Fairy-Stockfish for numbers and then make claims out of them.
The numbers are the engine's problem; the claims are this file's. So every test
here hands the deciding function a table of win percentages or a small book and
checks the verdict, which is the half that can be quietly wrong forever.

The naming tests are worth more than they look. "Counterattack" is a *claim*
about the initiative, and the whole point of `openings.classify` is that the
claim is measured rather than asserted — so the tests are mostly pairs: the
same move in two different fields, coming out with two different names.
"""

from __future__ import annotations

import chess
import chess.variant
import pytest

from trainer import book, openings, store, traps


def _at(sans):
    """The board after a SAN line, for classifying the move played next in it."""
    board = chess.variant.AtomicBoard()
    for san in sans:
        board.push_san(san)
    return board


def _book_from_lines(db, lines):
    """Build a book from {(san, ...): (games, white, draws, black)}.

    Each key is a full move sequence from the start; its last move is the book
    entry and everything before it locates the position. Copied in shape from
    `test_trainer.py`'s fixture of the same name — the two files test different
    modules and sharing a helper across them would couple them for six lines.
    """
    entries = []
    for sans, (games, w, d, b) in lines.items():
        board = chess.variant.AtomicBoard()
        for san in sans[:-1]:
            board.push_san(san)
        move = board.parse_san(sans[-1])
        entries.append(
            book.Entry(
                fen=book.book_fen(board), san=sans[-1], uci=move.uci(),
                ply=board.ply(), games=games, white_wins=w, draws=d, black_wins=b,
            )
        )
    book.replace_book(db, entries, {"games": 0})


# -- the naming grammar ----------------------------------------------------


def test_blacks_first_reply_is_a_defence():
    assert openings.kind_of(_at(["Nf3"]), "f6") == "Defence"
    assert openings.kind_of(_at([]), "Nf3") == "Variation"


def test_a_queen_sortie_into_their_half_is_an_attack():
    assert openings.kind_of(_at(["e4", "e5"]), "Qh5") == "Attack"
    # ...and one that stays at home is not. 2.Qf3 attacks nothing and is not in
    # Black's half, so calling it an Attack would empty the word out.
    assert openings.kind_of(_at(["e3", "e6"]), "Qf3") == "Variation"


def test_a_white_central_thrust_is_not_by_itself_an_attack():
    """Space is not an attack; if it were, half the tree would be named one."""
    assert openings.kind_of(_at(["Nf3", "f6", "Nc3", "c6"]), "e4") == "Variation"


def test_an_atomic_pawn_offer_is_judged_through_the_explosion_rules():
    """1...e5 looks like a gambit and is not: Nxe5 destroys the knight too."""
    char = openings.character(_at(["Nf3"]), "e5")
    assert not char.gambit
    assert char.thrust


def test_a_counterattack_has_to_gain_ground_to_be_called_one():
    """The suffix is a claim about the initiative, so it is measured.

    One move, one position, two different fields: one where Black's score goes
    up after it and one where it goes down. Only the first is a Counterattack;
    the second is Black's answer to the first move and nothing more.
    """
    board = _at(["e4"])
    assert openings.classify(
        board, openings.character(board, "d5", gain=+12.0), ply=1
    ) == "Counterattack"
    assert openings.classify(
        board, openings.character(board, "d5", gain=-12.0), ply=1
    ) == "Defence"


def test_without_book_statistics_nothing_is_called_a_counterattack(tmp_path):
    """A missing measurement makes a weaker claim, never a guessed one."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        named = openings.name_line(db, ["e4", "d5"])   # the book is empty
    assert "Counterattack" not in named.full
    assert named.full.endswith("d5 Defence")


def test_the_measured_counterattack_comes_out_of_a_real_book(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            # Black scores 30% across White's first move...
            ("e4",): (1000, 700, 0, 300),
            # ...and 70% after this reply, which is what makes it a counter.
            ("e4", "d5"): (500, 150, 0, 350),
            ("e4", "e6"): (500, 400, 0, 100),
        })
        counter = openings.name_line(db, ["e4", "d5"])
        quiet = openings.name_line(db, ["e4", "e6"])
    assert counter.kind == "Counterattack"
    assert quiet.kind == "Defence"
    assert any("against the parent line" in item for item in counter.evidence)


# -- whose name it is ------------------------------------------------------


def test_a_curated_name_wins_over_the_derived_one_and_says_where_it_is_from(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        named = openings.name_line(db, ["Nf3", "f6", "Nc3"])
    assert named.full.endswith("Kapsarov Attack")
    assert named.source == "community"


def test_your_own_name_beats_every_other_source(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        openings.rename(db, "Nf3 f6 Nc3", "The Thing I Always Lose To", "note")
        named = openings.name_line(db, ["Nf3", "f6", "Nc3"])
        assert named.full == "The Thing I Always Lose To"
        assert named.source == "user"
        openings.rename(db, "Nf3 f6 Nc3", None)
        assert openings.name_line(db, ["Nf3", "f6", "Nc3"]).source == "community"


def test_two_lines_ending_in_the_same_move_are_named_differently(tmp_path):
    """The defect this exists for: `e4 d5` and `e4 Nf6 Bc4 d5` were one name."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        short = openings.name_line(db, ["e4", "d5"])
        long = openings.name_line(db, ["e4", "Nf6", "Bc4", "d5"])
    assert short.full != long.full
    assert short.family == long.family == "King's Pawn Opening"


def test_a_note_belongs_to_the_line_it_was_written_about(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        family = openings.name_line(db, ["Nf3", "f6"])
        child = openings.name_line(db, ["Nf3", "f6", "e4"])
    assert family.note and family.aliases
    assert not child.note and not child.aliases
    assert child.family_note == family.note


def test_the_full_name_never_grows_a_second_colon(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        named = openings.name_line(db, ["Nf3", "f6", "Nc3", "Nh6", "g4"])
    assert named.full.count(":") == 1


def test_every_curated_name_declares_where_it_came_from():
    for entry in openings.curated().values():
        assert entry.source in ("standard", "community", "descriptive")
        assert entry.name


# -- what qualifies as a trap ----------------------------------------------


def _moves(*wins):
    return [
        traps.Move(san=f"m{i}", uci=f"a{i}a{i}", win=win, games=10)
        for i, win in enumerate(wins)
    ]


def test_one_move_holding_and_the_rest_losing_is_a_trap():
    escapes, losers, verdict = traps.judge(_moves(55.0, 20.0, 5.0))
    assert verdict == ""
    assert [m.win for m in escapes] == [55.0]
    assert [m.win for m in losers] == [20.0, 5.0]


def test_two_moves_holding_is_still_a_trap_and_three_is_not():
    assert traps.judge(_moves(55.0, 50.0, 10.0))[2] == ""
    assert traps.judge(_moves(55.0, 50.0, 46.0, 10.0))[2] == "too_many_escapes"


def test_a_second_best_that_is_merely_worse_is_not_a_trap():
    """A 30-point drop that leaves you winning is not walking into anything."""
    assert traps.judge(_moves(95.0, 65.0, 60.0))[2] == "not_severe"
    # ...and neither is a drop too small to be decisive.
    assert traps.judge(_moves(55.0, 40.0, 5.0))[2] == "not_severe"


def test_a_position_that_is_already_lost_holds_no_trap():
    assert traps.judge(_moves(20.0, 2.0, 1.0))[2] == "already_lost"


def test_every_line_holding_means_the_boundary_was_never_seen():
    """MultiPV bounds what can be judged, and that is not a verdict of no."""
    assert traps.judge(_moves(55.0, 54.0))[2] == "too_many_escapes"


def test_bait_counts_every_book_move_that_is_not_an_escape():
    """Including the ones too bad for the engine to have listed at all."""
    entries = [
        book.Entry(fen="F", san="hold", uci="a1a2", ply=0, games=100),
        book.Entry(fen="F", san="lose", uci="b1b2", ply=0, games=200),
        book.Entry(fen="F", san="awful", uci="c1c2", ply=0, games=700),
    ]
    share, count = traps._bait(entries, [traps.Move(san="hold", uci="a1a2", win=55.0)])
    assert count == 900
    assert share == pytest.approx(90.0)


# -- storage ---------------------------------------------------------------


def _trap(**kwargs):
    base = {
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "line": ("e4", "d5"),
        "turn": "white",
        "escapes": [traps.Move(san="Nf3", uci="g1f3", win=52.0, games=5, share=1.0)],
        "losers": [traps.Move(san="d4", uci="d2d4", win=8.0, games=90, share=90.0)],
        "refutation": ["d4", "Qxd4"],
        "win_best": 52.0,
        "win_next": 8.0,
        "bait": 90.0,
        "bait_games": 90,
        "games": 100,
        "nodes": 2_000_000,
    }
    return traps.Trap(**{**base, **kwargs})


def test_a_trap_id_changes_when_the_engine_changes_its_mind():
    def with_escapes(*ucis):
        return _trap(escapes=[traps.Move(san="x", uci=u, win=50.0) for u in ucis])
    assert with_escapes("a1a2").id == with_escapes("a1a2").id
    assert with_escapes("a1a2").id != with_escapes("b1b2").id
    # Order is not part of the fact being identified.
    assert with_escapes("a1a2", "b1b2").id == with_escapes("b1b2", "a1a2").id


def test_traps_round_trip_through_the_store(tmp_path):
    found = _trap()
    with store.open_db(tmp_path / "t.sqlite") as db:
        assert traps.save(db, [found]) == 1
        traps.save(db, [found])          # a rescan refreshes rather than duplicates
        rows = traps.listing(db)
        assert len(rows) == 1
        assert rows[0].escapes[0].san == "Nf3"
        assert rows[0].refutation == ["d4", "Qxd4"]
        assert rows[0].severity == pytest.approx(44.0)
        assert traps.stats(db)["traps"] == 1
        assert traps.get(db, found.id).line == ("e4", "d5")


def test_the_trap_listing_filters_by_side_and_orders_by_what_was_asked(tmp_path):
    def one(uci, turn, bait, severity):
        return _trap(
            fen=f"F{uci}", turn=turn, bait=bait, win_next=52.0 - severity,
            escapes=[traps.Move(san="x", uci=uci, win=52.0)],
        )
    with store.open_db(tmp_path / "t.sqlite") as db:
        traps.save(db, [
            one("a1a2", "white", 10.0, 90.0),
            one("b1b2", "black", 60.0, 30.0),
            one("c1c2", "white", 40.0, 50.0),
        ])
        assert [t.bait for t in traps.listing(db, order="bait")] == [60.0, 40.0, 10.0]
        assert [t.severity for t in traps.listing(db, order="severity")] == [90.0, 50.0, 30.0]
        assert [t.turn for t in traps.listing(db, turn="white")] == ["white", "white"]
        assert len(traps.listing(db, min_bait=35.0)) == 2


def test_the_book_walk_visits_each_position_once(tmp_path):
    """Transpositions are one position in the book and must be one row here."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("e3",): (1000, 600, 0, 400),
            ("Nf3",): (1000, 600, 0, 400),
            ("e3", "e6"): (900, 500, 0, 400),
            ("e3", "e6", "Nf3"): (800, 450, 0, 350),
            ("Nf3", "e6"): (900, 500, 0, 400),
            ("Nf3", "e6", "e3"): (800, 450, 0, 350),
        })
        seen = [node.fen for node in book.walk(db, depth=4, min_games=100, min_share=1.0)]
    assert len(seen) == len(set(seen))
    assert seen[0] == book.book_fen(chess.variant.AtomicBoard())
