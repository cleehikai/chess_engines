"""Puzzle generation, theme detection, and the triviality oracle.

Split from `test_trainer.py` because these cover a different question. That
file pins the arithmetic — win% conversion, book counting, coverage. This one
pins *atomic rules*: what the detectors are allowed to notice, and what the
filters are allowed to throw away.

Three of the tests below are regressions for bugs that silently emptied the
corpus rather than crashing, which is the failure mode worth guarding here — a
generator that produces nothing looks exactly like a generator with nothing to
find. Each is named after the wrong thing it prevents.
"""

from __future__ import annotations

import itertools

import chess
import chess.variant
import pytest

from lab import tablebase
from trainer import difficulty, dump, endgames, generate, mine, puzzles, store, themes

#: Black has just played ...Ne4?? and Qxf7 explodes the king on e8. Used
#: wherever a test needs a real, legal, one-move atomic win.
BLAST = "r1bqkb1r/ppp1pppp/2n5/3p4/4n3/2P1PQ1N/PP1P1PPP/RNB1KB1R w KQkq - 2 5"


# -- theme detectors -------------------------------------------------------
#
# Written against positions rather than against the detectors' internals, so
# narrowing a definition (as `quiet` and `no-capture-fortress` both were, once
# their firing rates came out at 73% and 60%) breaks a test only if it changes
# what the theme means.


def test_king_blast_fires_when_the_solution_detonates_on_the_king():
    fired = themes.detect(BLAST, ["f3f7"])
    assert "king-blast" in fired
    # The king is not the only thing in the radius; a rook and a bishop go too.
    assert "blast-fork" in fired


def test_quiet_needs_a_capture_to_have_been_declined():
    """A quiet move in a position with nothing to take is not the `quiet` idea.

    The regression that mattered: without the clause this fired on 72.6% of the
    corpus, because most endgame positions have no capture in them at all and
    never did.
    """
    barren = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"
    assert "quiet" not in themes.detect(barren, ["e3e2"])

    # Same shape of move, but Rxd5 was on the board and was passed over.
    tempting = "4k3/8/8/3q3R/8/4K3/8/8 w - - 0 1"
    board = chess.variant.AtomicBoard(tempting)
    assert any(board.is_capture(move) for move in board.legal_moves)
    assert "quiet" in themes.detect(tempting, ["h5h6"])


def test_own_king_in_blast_sees_a_capture_the_defender_may_not_make():
    """The defensive idea with no standard-chess counterpart.

    White's rook stands next to Black's king. Black's queen attacks it and may
    not take it: the explosion would remove the king as well.
    """
    board = chess.variant.AtomicBoard("8/8/8/3Rk3/3q4/8/8/4K3 b - - 0 1")
    suicidal = themes._suicidal_captures(board)
    assert any(move.to_square == chess.D5 for move in suicidal)
    assert not any(
        move.to_square == chess.D5 and board.is_capture(move)
        for move in board.legal_moves
    )


def test_connected_kings_is_seen_and_is_legal():
    """Adjacent kings are legal in atomic and neither of them is in check."""
    fen = "8/8/8/3kK3/8/8/7R/8 w - - 0 1"
    board = chess.variant.AtomicBoard(fen)
    assert not board.is_check()
    assert "connected-kings" in themes.detect(fen, ["h2h3"])


def test_walk_refuses_an_illegal_line():
    with pytest.raises(ValueError):
        list(themes.walk(chess.variant.AtomicBoard().fen(), ["e2e5"]))


def test_phase_is_decided_by_material_before_ply():
    start = chess.variant.AtomicBoard()
    assert themes.phase_of(start, ply=2) == "opening"
    assert themes.phase_of(start, ply=40) == "middlegame"
    bare = chess.variant.AtomicBoard("4k3/8/8/8/8/8/8/QK6 w - - 0 1")
    assert themes.phase_of(bare, ply=2) == "endgame"


def test_provenance_themes_are_not_judged_on_breadth():
    """`tb-conversion` at 65% means 65% came from the tablebase, not a bug."""
    assert themes.Firing("tb-conversion", 94, 145).verdict.startswith("provenance")
    assert themes.Firing("quiet", 120, 145).verdict == "too broad — narrow it"
    assert themes.Firing("quiet", 0, 145).verdict == "never fires — delete it"
    assert themes.Firing("quiet", 0, 5).verdict == "too few puzzles to judge"


# -- the triviality oracle -------------------------------------------------


def test_greedy_takes_an_immediate_win():
    """`M0-greedy` must find a one-move king blast, or it filters nothing.

    It did not, until `lab.methods` was corrected. `is_variant_win()` is
    relative to the side to move, and after the push that is the *opponent*, so
    the test for "I just won" is `is_variant_loss()`. The inverted version
    scored every winning move at minus a million and played a2a3 instead.
    """
    board = chess.variant.AtomicBoard(BLAST)
    assert generate.greedy_moves(board) == {"f3f7"}
    assert generate.is_trivial(board, "f3f7")


def test_lab_greedy_agrees_with_the_oracle():
    """The filter claims to be M0-greedy; check that it still is."""
    from lab.methods import Budget, GreedyMaterialMethod

    board = chess.variant.AtomicBoard(BLAST)
    engine = GreedyMaterialMethod(seed=0)
    played = {engine.choose(board, Budget(nodes=1)).uci() for _ in range(20)}
    assert played == generate.greedy_moves(board)


def test_greedy_indifference_is_not_a_solution():
    """With no captures on the board every move ties, and a tie is a guess.

    Treating "the move is in the tie set" as "greedy solved it" rejected every
    quiet position, which is most endgames: 14 genuine only-move tablebase
    positions were discarded by this before it was fixed, and the endgame
    corpus came out empty rather than wrong.
    """
    board = chess.variant.AtomicBoard("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    picks = generate.greedy_moves(board)
    assert len(picks) == board.legal_moves.count() > generate.GREEDY_TIES
    assert not generate.is_trivial(board, next(iter(picks)))


def test_a_position_with_one_legal_move_is_never_a_puzzle():
    board = chess.variant.AtomicBoard("4k3/8/8/8/8/8/8/QK6 w - - 0 1")
    assert not generate.is_trivial(board, "a1a7")  # plenty of choice here
    forced = chess.variant.AtomicBoard("8/8/8/8/8/8/6q1/7K w - - 0 1")
    if forced.legal_moves.count() == 1:
        assert generate.is_trivial(forced, next(iter(forced.legal_moves)).uci())


# -- line trimming ---------------------------------------------------------


def _candidate(fen: str) -> generate.Candidate:
    return generate.Candidate(
        game_id="g", ply=0, fen=fen, solver="white", motive="win",
        setup_fen=None, setup_uci=None, setup_san=None,
        win_best=90.0, win_second=10.0, mine=False,
    )


def test_a_solution_never_ends_on_the_defenders_move():
    start = chess.variant.AtomicBoard().fen()
    walk = generate._Walk(
        candidate=_candidate(start),
        board=chess.variant.AtomicBoard(),
        line=["e2e4", "e7e5"],
    )
    assert generate._trim(walk) == ["e2e4"]


def test_trimming_an_empty_line_stays_empty():
    start = chess.variant.AtomicBoard().fen()
    walk = generate._Walk(candidate=_candidate(start), board=chess.variant.AtomicBoard())
    assert generate._trim(walk) == []


# -- puzzle storage --------------------------------------------------------


def _puzzle(**kwargs) -> puzzles.Puzzle:
    base = {
        "fen": "4k3/8/8/8/8/8/8/QK6 w - - 0 1",
        "solver": "white",
        "solution": ["a1a7"],
        "solution_san": ["Qa7"],
        "themes": ["tb-conversion"],
        "phase": "endgame",
        "source": "tablebase",
    }
    return puzzles.Puzzle(**{**base, **kwargs})


def test_the_id_is_a_content_hash_so_attempts_survive_regeneration():
    first = _puzzle()
    again = _puzzle(themes=["tb-conversion", "promotion"], nodes=4_000_000)
    assert first.id == again.id
    assert _puzzle(solution=["a1a8"], solution_san=["Qa8"]).id != first.id


def test_saving_twice_refreshes_the_row_and_keeps_the_attempts(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzle = _puzzle()
        puzzles.save(db, [puzzle])
        puzzles.record_attempt(db, puzzle.id, solved=True)
        puzzles.save(db, [_puzzle(themes=["tb-conversion", "long"])])
        stored = puzzles.get(db, puzzle.id)
        assert stored is not None
        assert stored.themes == ["tb-conversion", "long"]
        assert len(puzzles.history(db, puzzle.id)) == 1
        assert puzzles.theme_counts(db) == {"tb-conversion": 1, "long": 1}


def test_clearing_the_corpus_keeps_the_solve_history(tmp_path):
    """Puzzles are derived and droppable; what you did is neither."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzle = _puzzle()
        puzzles.save(db, [puzzle])
        puzzles.record_attempt(db, puzzle.id, solved=True)
        puzzles.clear(db)
        assert puzzles.get(db, puzzle.id) is None
        assert len(puzzles.history(db, puzzle.id)) == 1


def test_a_solved_puzzle_leaves_the_unsolved_queue(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        one = _puzzle()
        two = _puzzle(solution=["a1a8"], solution_san=["Qa8"])
        puzzles.save(db, [one, two])
        puzzles.record_attempt(db, one.id, solved=True)
        # A failed attempt is not a solve: it has to keep coming back.
        puzzles.record_attempt(db, two.id, solved=False, wrong=3)
        unsolved = puzzles.search(db, puzzles.Filter(unsolved=True))
    assert [p.id for p in unsolved] == [two.id]


def test_a_puzzle_you_have_touched_at_all_leaves_the_unseen_queue(tmp_path):
    """The no-repeats default: `unseen` is stricter than `unsolved`.

    A puzzle you failed is still unsolved, and the redo queue wants it back;
    the queue you sit down to in the morning does not, which is why the two
    are separate flags rather than one.
    """
    with store.open_db(tmp_path / "t.sqlite") as db:
        one = _puzzle()
        two = _puzzle(solution=["a1a8"], solution_san=["Qa8"])
        three = _puzzle(solution=["a1a6"], solution_san=["Qa6"])
        puzzles.save(db, [one, two, three])
        puzzles.record_attempt(db, one.id, solved=True)
        puzzles.record_attempt(db, two.id, solved=False, wrong=1)

        unseen = {p.id for p in puzzles.search(db, puzzles.Filter(unseen=True))}
        unsolved = {p.id for p in puzzles.search(db, puzzles.Filter(unsolved=True))}
    assert unseen == {three.id}
    assert unsolved == {two.id, three.id}


def test_a_hold_is_never_labelled_a_win_by_the_screening_pass(tmp_path):
    """The confirming pass has the last word on what you are playing for.

    Candidates are screened at the review pass's budget and re-judged at 4M
    nodes; 31 of this corpus's 215 `save` puzzles turned out to be winning
    under the deeper look, and were being announced as "only one move holds".
    """
    assert puzzles.motive_of(96.0, "save") == "win"
    assert puzzles.motive_of(46.0, "win") == "save"
    # Nothing measured, nothing to correct: the tablebase puzzles keep the
    # motive their exact category gave them.
    assert puzzles.motive_of(None, "save") == "save"

    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzles.save(db, [_puzzle(motive="save", win_before=96.0, tb_category=None)])
        stored = puzzles.search(db, puzzles.Filter())[0]
    assert stored.goal["kind"] == "win"
    assert stored.goal["exact"] is False
    assert stored.goal["win"] == 96.0


def test_an_exact_goal_comes_from_the_tablebase_not_from_the_engine():
    drawn = _puzzle(motive="save", tb_category="draw", win_before=None)
    assert drawn.goal == {
        "kind": "draw", "exact": True, "win": None, "category": "draw",
    }
    assert _puzzle(motive="win", tb_category="cursed-win").goal["kind"] == "win"


def test_filters_are_anded_together(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzles.save(db, [
            _puzzle(),
            _puzzle(solution=["a1a8"], solution_san=["Qa8"], source="game",
                    phase="opening", themes=["quiet"], motive="save"),
        ])
        assert puzzles.count(db, puzzles.Filter(source="game")) == 1
        assert puzzles.count(db, puzzles.Filter(theme="quiet", phase="opening")) == 1
        assert puzzles.count(db, puzzles.Filter(theme="quiet", phase="endgame")) == 0


def test_the_public_view_never_carries_the_solution():
    """The referee rule, applied to the answer rather than to legality."""
    puzzle = _puzzle(
        solution=["a1a7", "e8d8", "a7d7"], solution_san=["Qa7", "Kd8", "Qd7#"]
    )
    payload = puzzle.public()
    assert not any("a1a7" in str(value) for value in payload.values())
    assert payload["length"] == 2          # two moves to find, not three plies
    assert puzzle.reveal()["solution"][0] == "a1a7"


def test_solution_sans_are_read_off_the_running_board():
    sans = puzzles.solution_sans(BLAST, ["f3f7"])
    assert sans == ["Qxf7#"]


def test_annotate_fills_in_phase_and_themes():
    puzzle = puzzles.annotate(
        puzzles.Puzzle(
            fen=BLAST, solver="white", solution=["f3f7"], solution_san=["Qxf7#"],
            source="game", ply=8,
        )
    )
    assert puzzle.phase == "opening"
    assert "king-blast" in puzzle.themes


# -- tablebase endgames ----------------------------------------------------


def test_holding_moves_flips_the_opponents_verdict():
    """The tablebase reports each move from the *opponent's* point of view.

    A move keeps our win exactly when it leaves them lost, so reading the
    categories without flipping them would select the losing moves.
    """
    board = chess.variant.AtomicBoard("4k3/8/8/8/8/8/8/QK6 w - - 0 1")
    probe = tablebase.Probe(
        category="win",
        dtz=3,
        moves=[
            {"uci": "a1a7", "category": "loss", "dtz": -2},
            {"uci": "b1b2", "category": "draw", "dtz": 0},
            {"uci": "a1h1", "category": "win", "dtz": 5},
        ],
    )
    assert endgames.holding_moves(board, probe) == ["a1a7"]


def test_a_lost_position_is_never_an_endgame_puzzle():
    """"The only move that keeps me lost" is well defined and worth nothing."""
    board = chess.variant.AtomicBoard("4k3/8/8/8/8/8/8/QK6 w - - 0 1")
    probe = tablebase.Probe(
        category="loss", dtz=-4,
        moves=[{"uci": "a1a7", "category": "win", "dtz": 3}],
    )
    assert endgames.holding_moves(board, probe) == []


def test_holding_moves_ignores_moves_that_are_not_legal_here():
    board = chess.variant.AtomicBoard("4k3/8/8/8/8/8/8/QK6 w - - 0 1")
    probe = tablebase.Probe(
        category="win", dtz=3,
        moves=[{"uci": "h8h1", "category": "loss", "dtz": -2}],
    )
    assert endgames.holding_moves(board, probe) == []


def test_sampled_positions_are_legal_and_reachable():
    """Two constraints, and `is_valid()` only covers the first.

    `AtomicBoard` relaxes the into-check rules, and the relaxation reaches
    `status()`, so a position with the side *not* to move already in check
    passes validation. The Lichess tablebase answers 400 to exactly those, and
    they cost a 3,000-position run a third of its budget before the check was
    made directly.
    """
    fens = list(itertools.islice(endgames.sampled(120, seed=11), 120))
    assert len(fens) == 120
    for fen in fens:
        board = chess.variant.AtomicBoard(fen)
        assert board.is_valid()
        assert not board.is_game_over(claim_draw=True)
        assert endgames.piece_count(board) <= endgames.MAX_PIECES
        board.turn = not board.turn
        assert not board.is_check(), f"wrong side exposed to check: {fen}"


def test_sampling_is_reproducible():
    first = list(itertools.islice(endgames.sampled(20, seed=5), 20))
    again = list(itertools.islice(endgames.sampled(20, seed=5), 20))
    assert first == again


# -- deduplication by position ---------------------------------------------
#
# Atomic's opening tree is narrow and forcing, so the same critical position
# recurs across unrelated games constantly — 20 of one account's 86 games open
# 1.Nf3 f6. Without a position key, scraping a thousand rated games produces
# the same tactic a dozen times with a different game id on each copy.


def test_the_position_key_ignores_the_move_counters():
    """A transposition is the same position, whatever move number it arrives at."""
    early = "4k3/8/8/8/8/8/8/QK6 w - - 0 1"
    late = "4k3/8/8/8/8/8/8/QK6 w - - 14 37"
    assert puzzles.position_key(early) == puzzles.position_key(late)


def test_the_position_key_keeps_en_passant():
    """Usually irrelevant; when it is relevant in atomic it is the whole puzzle."""
    with_ep = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 3"
    without = "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 3"
    assert puzzles.position_key(with_ep) != puzzles.position_key(without)


def test_a_second_game_reaching_the_same_position_is_not_a_second_puzzle(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        first = generate.Candidate(
            game_id="aaa", ply=8, fen=BLAST, solver="white", motive="win",
            setup_fen=None, setup_uci=None, setup_san=None,
            win_best=95.0, win_second=10.0, mine=False,
        )
        # Same position, different game, and reached at a different move number
        # so the counters differ too.
        second = generate.Candidate(
            game_id="bbb", ply=12, fen=BLAST.replace(" 2 5", " 4 7"),
            solver="white", motive="win",
            setup_fen=None, setup_uci=None, setup_san=None,
            win_best=95.0, win_second=10.0, mine=False,
        )
        seen = puzzles.seen_positions(db)
        # No engine involved: everything is rejected at the confirming step for
        # want of cached analysis, and the duplicate never gets that far.
        _, report = generate.generate(
            db, [first, second], model=_flat_model(), nodes=1, seen=seen
        )
    assert report.candidates == 2
    assert report.duplicate == 1


def test_regenerating_the_same_moment_is_not_a_duplicate(tmp_path):
    """Re-running the generator must refresh a puzzle, not refuse it.

    `save` uses INSERT OR REPLACE precisely so that regenerating at a higher
    node budget updates the evaluations of a puzzle whose solution is
    unchanged. Deduplicating on the position alone would defeat that — the
    puzzle already in the corpus occupies its own position.
    """
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzles.save(db, [_puzzle(source="game", game_id="aaa", ply=8)])
        seen = puzzles.seen_positions(db)
        again = generate.Candidate(
            game_id="aaa", ply=8, fen="4k3/8/8/8/8/8/8/QK6 w - - 0 1",
            solver="white", motive="win",
            setup_fen=None, setup_uci=None, setup_san=None,
            win_best=95.0, win_second=10.0, mine=False,
        )
        _, report = generate.generate(
            db, [again], model=_flat_model(), nodes=1, seen=seen
        )
    assert report.duplicate == 0


def _flat_model():
    """A win% model, loaded once; these tests never reach the engine."""
    from trainer import winrate

    return winrate.load()


# -- the scraper's game filter ---------------------------------------------


def _raw(**headers) -> dump.RawGame:
    base = {
        "Termination": "Normal", "WhiteElo": "2000", "BlackElo": "1900",
        "Site": "https://lichess.org/abcd1234",
    }
    return dump.RawGame({**base, **headers}, "1. Nf3 f6 2. Nd4 Nh6 " * 6 + "1-0")


def test_only_rated_games_that_ended_over_the_board_are_scanned():
    """A flag fall says nothing about the position, so it is not evidence."""
    assert mine.eligible(_raw())
    assert not mine.eligible(_raw(Termination="Time forfeit"))
    assert not mine.eligible(_raw(Termination="Abandoned"))


def test_both_players_have_to_clear_the_rating_floor():
    assert not mine.eligible(_raw(BlackElo="1200"))
    assert not mine.eligible(_raw(WhiteElo="1200"))
    assert mine.eligible(_raw(BlackElo="1200"), min_elo=1000)
    # A missing rating is not a passing one.
    assert not mine.eligible(dump.RawGame({"Termination": "Normal"}, "1. e3 e6 1-0"))


def test_the_game_id_comes_off_the_site_header():
    assert mine.game_id(_raw()) == "abcd1234"
    assert mine.game_id(dump.RawGame({}, "")) is None


def test_a_scanned_game_is_remembered_even_when_it_yields_nothing(tmp_path):
    """Otherwise a resumed run rescans the games that were already rejected."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        mine._record(db, "abcd1234", "2026-07", _raw(), ply=24, nodes=20_000, kept=0)
        db.commit()
        assert mine.already_scanned(db) == {"abcd1234"}
        assert mine.stats(db)["games"] == 1


# -- the improvement report ------------------------------------------------


def test_a_puzzle_you_failed_is_in_the_redo_queue_and_an_untried_one_is_not(tmp_path):
    """`failed` is narrower than `unsolved`, and that is the whole point.

    `unsolved` includes every puzzle you have never seen, which is most of the
    corpus and is evidence of nothing. The redo queue is puzzles you tried.
    """
    with store.open_db(tmp_path / "t.sqlite") as db:
        tried = _puzzle()
        untried = _puzzle(solution=["a1a8"], solution_san=["Qa8"])
        puzzles.save(db, [tried, untried])
        puzzles.record_attempt(db, tried.id, solved=False, wrong=2)
        failed = puzzles.search(db, puzzles.Filter(failed=True))
        unsolved = puzzles.search(db, puzzles.Filter(unsolved=True))
    assert [p.id for p in failed] == [tried.id]
    assert {p.id for p in unsolved} == {tried.id, untried.id}


def test_only_the_first_attempt_at_a_puzzle_counts(tmp_path):
    """Drilling a position you have memorised is not evidence that you see it."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzle = _puzzle()
        puzzles.save(db, [puzzle])
        puzzles.record_attempt(db, puzzle.id, solved=False, wrong=1)
        for _ in range(5):
            puzzles.record_attempt(db, puzzle.id, solved=True)
        report = puzzles.improvement(db)
    assert report["overall"]["seen"] == 1
    assert report["overall"]["solved"] == 0     # the first go is the one that counts
    assert report["overall"]["attempts"] == 6


def test_a_thin_slice_gets_an_interval_and_no_verdict():
    def skill(solved, seen):
        low, high = puzzles.wilson(solved, seen)
        return puzzles.Skill(key="quiet", seen=seen, solved=solved, avg_ms=None,
                             lower=low, upper=high)

    thin = skill(2, 2)
    assert thin.rate == 100.0
    assert thin.verdict(overall=50.0) == "too few to judge"

    # Enough attempts, and the interval clears the overall rate in each
    # direction — those are the only two cases that earn a word.
    strong = skill(29, 30)
    assert strong.verdict(overall=50.0) == "strong"
    weak = skill(4, 30)
    assert weak.verdict(overall=50.0) == "weak"
    even = skill(15, 30)
    assert even.verdict(overall=50.0) == "as usual"


def test_the_interval_never_leaves_the_unit_range():
    """The reason it is Wilson: `p ± z·se` runs past 100% on 2 of 2."""
    low, high = puzzles.wilson(2, 2)
    assert 0.0 <= low <= high <= 100.0
    assert high == 100.0


# -- measured difficulty ---------------------------------------------------
#
# The engine is not run here. Everything below tests what a *profile* over the
# ladder is turned into, which is where the judgement lives: the searches
# themselves are `lab.methods`' business and are covered by the ladder
# calibration that produced the Elo numbers in the first place.

RUNGS = (
    difficulty.Rung(nodes=2_000, elo=1366.0, span=88.0, anchor=True),
    difficulty.Rung(nodes=16_000, elo=1691.0, span=131.0),
    difficulty.Rung(nodes=64_000, elo=1704.0, span=135.0),
    difficulty.Rung(nodes=1_024_000, elo=1951.0, span=258.0),
)


def _attempts(*solved: bool) -> list[difficulty.Attempt]:
    return [
        difficulty.Attempt(nodes=rung.nodes, elo=rung.elo, solved=hit, found=1 if hit else 0,
                           total=1, played=None if hit else "a1a2", ms=1)
        for rung, hit in zip(RUNGS, solved, strict=True)
    ]


def test_the_rating_is_the_weakest_level_that_solves_it():
    verdict = difficulty.verdict_from("x", _attempts(False, True, True, True), RUNGS)
    assert verdict.elo == 1691.0
    assert verdict.bound == ""
    assert verdict.solved_by == 16_000
    assert verdict.label == "1691"


def test_the_ladder_floor_is_reported_as_a_bound_not_a_number():
    """Solved by the weakest rung means "at most", and the ladder stops there.

    A one-move blast that a 1366-rated engine finds instantly might be worth
    1300 or 400; nothing here can tell the difference, and rounding it to 1366
    would invent a precision the measurement does not have.
    """
    verdict = difficulty.verdict_from("x", _attempts(True, True, True, True), RUNGS)
    assert verdict.bound == "below"
    assert verdict.label == "≤1366"
    assert "cannot see how much easier" in verdict.describe()


def test_a_puzzle_no_level_solves_is_bounded_above_the_top_rung():
    verdict = difficulty.verdict_from("x", _attempts(False, False, False, False), RUNGS)
    assert verdict.bound == "above"
    assert verdict.elo == 1951.0
    assert verdict.label == ">1951"
    assert verdict.solved_by is None


def test_a_stronger_level_missing_what_a_weaker_one_found_is_recorded():
    """16k and 64k measured 1691 ± 131 and 1704 ± 135 — one rung inside the
    other's error bar — so this happens, and hiding it would misdescribe the
    ladder rather than the puzzle."""
    verdict = difficulty.verdict_from("x", _attempts(False, True, False, True), RUNGS)
    assert verdict.elo == 1691.0
    assert not verdict.monotone


def test_a_rating_survives_the_corpus_being_regenerated(tmp_path):
    """The reason ratings are their own table: `save` is INSERT OR REPLACE."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzle = _puzzle()
        puzzles.save(db, [puzzle])
        difficulty.save(db, [difficulty.verdict_from(
            puzzle.id, _attempts(False, True, True, True), RUNGS)])
        puzzles.save(db, [_puzzle(nodes=4_000_000)])   # regenerated, same id
        stored = puzzles.get(db, puzzle.id)
        assert stored is not None and stored.rating == 1691.0
        assert stored.rating_text == "1691"
        assert difficulty.load(db, puzzle.id).solved_by == 16_000


def test_an_unrated_puzzle_never_matches_a_rating_band(tmp_path):
    """"Not measured" is not evidence of being easy, hard, or in between."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        rated = _puzzle()
        unrated = _puzzle(solution=["a1a8"], solution_san=["Qa8"])
        puzzles.save(db, [rated, unrated])
        difficulty.save(db, [difficulty.verdict_from(
            rated.id, _attempts(False, False, True, True), RUNGS)])
        assert puzzles.count(db, puzzles.Filter(min_rating=1500)) == 1
        assert puzzles.count(db, puzzles.Filter(max_rating=1500)) == 0
        assert puzzles.count(db, puzzles.Filter(rated=False)) == 1
        # Unrated puzzles sort last whichever way the corpus is ordered.
        for order in ("easiest", "hardest"):
            found = puzzles.search(db, puzzles.Filter(order=order))
            assert [p.id for p in found] == [rated.id, unrated.id]


def test_a_run_skips_what_is_already_rated_unless_forced(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        one, two = _puzzle(), _puzzle(solution=["a1a8"], solution_san=["Qa8"])
        puzzles.save(db, [one, two])
        difficulty.save(db, [difficulty.verdict_from(
            one.id, _attempts(True, True, True, True), RUNGS)])
        spec = puzzles.Filter(limit=100)
        assert [p.id for p in difficulty.pending(db, spec)] == [two.id]
        assert len(difficulty.pending(db, spec, force=True)) == 2


def test_a_recalibrated_ladder_leaves_old_ratings_visibly_stale(tmp_path):
    """Two Elo scales in one column would be invisible; this makes it a count."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        puzzle = _puzzle()
        puzzles.save(db, [puzzle])
        difficulty.save(db, [difficulty.verdict_from(
            puzzle.id, _attempts(False, True, True, True), RUNGS)])
        assert difficulty.stale(db, RUNGS) == 0
        moved = (*RUNGS[:1], difficulty.Rung(nodes=16_000, elo=1750.0, span=120.0),
                 *RUNGS[2:])
        assert difficulty.stale(db, moved) == 1


def test_the_solve_rate_is_reported_beside_the_engine_rating(tmp_path):
    """Plan §5's refinement: the two axes side by side, neither averaged away."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        hard = _puzzle()
        puzzles.save(db, [hard])
        difficulty.save(db, [difficulty.verdict_from(
            hard.id, _attempts(False, False, False, False), RUNGS)])
        puzzles.record_attempt(db, hard.id, solved=True)
        bands = difficulty.distribution(db)["bands"]
    assert [band["label"] for band in bands] == [">1951"]
    assert bands[0]["tried"] == 1 and bands[0]["rate"] == 100.0
    assert puzzles.wilson(0, 0) == (0.0, 100.0)
