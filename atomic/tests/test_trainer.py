"""Trainer unit tests — everything that does not need Fairy-Stockfish running.

The engine-dependent path (does FSF return sane MultiPV lines for an atomic
position) is covered by `lab`'s perft gate and by actually running `analyse`;
what is worth pinning down here is the arithmetic and the parsing, because
those are the parts that can be quietly wrong forever.
"""

from __future__ import annotations

import math

import chess
import chess.variant
import pytest

from trainer import (
    analysis,
    book,
    dump,
    fit_winrate,
    repertoire,
    review,
    store,
    winrate,
)

# -- win% model ------------------------------------------------------------


def test_fallback_model_is_labelled_unfitted():
    model = winrate.fallback_model()
    assert not model.fitted
    assert "UNFITTED" in model.describe()


def test_win_pct_is_monotonic_and_centred_on_the_intercept():
    model = winrate.WinModel(a=0.0, k=0.004, fitted=True, provenance={})
    assert model.win_pct(0) == pytest.approx(50.0)
    assert model.win_pct(-500) < model.win_pct(-100) < model.win_pct(100) < model.win_pct(500)
    assert 0.0 < model.win_pct(-100_000) < 1e-6
    assert model.even_cp == pytest.approx(0.0)


def test_a_nonzero_intercept_moves_the_even_point():
    """If the engine's zero is not an even game, `even_cp` has to say so."""
    model = winrate.WinModel(a=-0.8, k=0.004, fitted=True, provenance={})
    assert model.win_pct(0) < 50.0
    assert model.even_cp == pytest.approx(200.0)
    assert model.win_pct(model.even_cp) == pytest.approx(50.0)


def test_mate_scores_are_decisive_either_way():
    model = winrate.WinModel(a=0.0, k=0.004, fitted=True, provenance={})
    assert model.score_win_pct(None, 3) == 100.0
    assert model.score_win_pct(None, -3) == 0.0
    with pytest.raises(ValueError):
        model.score_win_pct(None, None)


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (0.0, "ok"),
        (9.9, "ok"),
        (10.0, "inaccuracy"),
        (19.9, "inaccuracy"),
        (20.0, "mistake"),
        (29.9, "mistake"),
        (30.0, "blunder"),
        (100.0, "blunder"),
    ],
)
def test_classification_thresholds(loss, expected):
    assert winrate.classify(loss) == expected


def test_model_round_trips_through_disk(tmp_path):
    model = winrate.WinModel(a=-0.5, k=0.0031, fitted=True, provenance={"positions": 7})
    path = tmp_path / "winrate.json"
    winrate.save(model, path)
    loaded = winrate.load(path)
    assert loaded.a == pytest.approx(model.a)
    assert loaded.k == pytest.approx(model.k)
    assert loaded.fitted
    assert loaded.provenance["positions"] == 7


def test_missing_model_file_falls_back_rather_than_raising(tmp_path):
    assert not winrate.load(tmp_path / "absent.json").fitted


# -- the fit itself --------------------------------------------------------


def test_irls_recovers_known_parameters():
    """Synthetic data with a known curve must come back with that curve.

    Guards the one piece of maths here that has no external reference to check
    it against: if IRLS is subtly wrong, every judgement downstream is wrong and
    nothing else in the system would notice.
    """
    import random

    rng = random.Random(7)
    true_a, true_k = -0.6, 0.0035
    pairs = []
    for _ in range(40_000):
        cp = rng.uniform(-1200, 1200)
        p = 1.0 / (1.0 + math.exp(-(true_a + true_k * cp)))
        pairs.append((cp, 1.0 if rng.random() < p else 0.0))
    a, k = fit_winrate.fit(pairs)
    assert a == pytest.approx(true_a, abs=0.06)
    assert k == pytest.approx(true_k, rel=0.08)


def test_log_loss_prefers_the_true_curve():
    pairs = [(cp, 1.0 if cp > 0 else 0.0) for cp in range(-900, 901, 10)]
    good = fit_winrate.log_loss(pairs, 0.0, 0.01)
    bad = fit_winrate.log_loss(pairs, 0.0, 0.0001)
    assert good < bad


def test_calibration_buckets_cover_the_sample():
    pairs = [(float(cp), 0.5) for cp in range(-500, 500)]
    rows = fit_winrate.calibration(pairs, 0.0, 0.004, buckets=5)
    assert len(rows) == 5
    assert sum(row["n"] for row in rows) == len(pairs)
    assert rows[0]["cp_low"] == -500
    assert rows[-1]["cp_high"] == 499


# -- dump parsing ----------------------------------------------------------


def test_movetext_strips_numbers_comments_nags_and_result():
    game = dump.RawGame(
        headers={"Result": "1-0"},
        movetext="1. Nf3 { [%eval 0.3] [%clk 0:03:00] } 1... f6?! $6 2. e3 { x } 1-0",
    )
    assert game.moves() == ["Nf3", "f6?!", "e3"]


def test_elo_headers_are_parsed_and_missing_ones_are_none():
    game = dump.RawGame(headers={"WhiteElo": "1834", "BlackElo": "?"}, movetext="")
    assert game.elo("White") == 1834
    assert game.elo("Black") is None
    assert game.elo("Green") is None


# -- replay ----------------------------------------------------------------


def test_positions_of_replays_san_and_reports_the_mover():
    plies = analysis.positions_of(["Nf3", "f6", "Nd4"])
    assert [p[3] for p in plies] == [chess.WHITE, chess.BLACK, chess.WHITE]
    assert [p[2] for p in plies] == ["Nf3", "f6", "Nd4"]
    assert plies[0][0].startswith("rnbqkbnr/pppppppp")
    # Each FEN is the position *before* the move, so ply 1's FEN already has Nf3.
    assert "N" in plies[1][0].split()[0]


def test_positions_of_rejects_illegal_san_loudly():
    with pytest.raises(ValueError, match="illegal SAN"):
        analysis.positions_of(["Nf3", "Qh8"])


def test_terminal_win_pct_only_fires_on_a_real_ending():
    import chess.variant

    assert analysis._terminal_win_pct(chess.variant.AtomicBoard(), chess.WHITE) is None

    # Qxd7 detonates next to the black king on e8: an atomic win for White,
    # even though the capturing queen dies too and there is no checkmate.
    board = chess.variant.AtomicBoard("4k3/3q4/8/8/8/8/8/3QK3 w - - 0 1")
    board.push_san("Qxd7")
    assert board.is_game_over(claim_draw=True)
    assert not board.is_checkmate()
    assert analysis._terminal_win_pct(board, chess.WHITE) == 100.0
    assert analysis._terminal_win_pct(board, chess.BLACK) == 0.0


# -- store -----------------------------------------------------------------


def test_analysis_cache_round_trip(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        lines = [{"uci": "g1f3", "cp": 250, "mate": None, "depth": 20}]
        store.store_analysis(db, "FEN", 1000, 2, "e", lines, 20, 1.0)
        db.commit()
        assert store.cached_analysis(db, "FEN", 1000, 2, "e") == lines
        assert store.cache_stats(db)["positions"] == 1


def test_cache_lookup_does_not_substitute_a_different_budget(tmp_path):
    """A judgement is only comparable with others made at the same budget."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        store.store_analysis(db, "FEN", 1_000_000, 2, "e", [{"uci": "a", "cp": 1}], 30, 1.0)
        db.commit()
        assert store.cached_analysis(db, "FEN", 50_000, 2, "e") is None
        assert store.cached_analysis(db, "FEN", 1_000_000, 1, "e") is None
        assert store.cached_analysis(db, "FEN", 1_000_000, 2, "other") is None


def test_meta_round_trip(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        assert store.get_meta(db, "nope") is None
        store.set_meta(db, "k", "v")
        store.set_meta(db, "k", "v2")
        assert store.get_meta(db, "k") == "v2"
        assert store.get_meta(db, "schema_version") == str(store.SCHEMA_VERSION)


def test_a_newer_schema_is_refused_rather_than_corrupted(tmp_path):
    path = tmp_path / "t.sqlite"
    with store.open_db(path) as db:
        store.set_meta(db, "schema_version", str(store.SCHEMA_VERSION + 1))
    with pytest.raises(RuntimeError, match="newer trainer"):
        store.connect(path)


# -- MultiPV ordering ------------------------------------------------------


def _line(cp=None, mate=None):
    return analysis.Line(uci="a1a2", cp=cp, mate=mate, depth=10)


def test_best_first_sorts_by_evaluation():
    lines = [_line(cp=10), _line(cp=300), _line(cp=-50)]
    assert [l.cp for l in analysis.best_first(lines)] == [300, 10, -50]


def test_best_first_puts_mates_where_they_belong():
    """A node limit can leave FSF's MultiPV list out of order; this fixes it."""
    ordered = analysis.best_first(
        [_line(cp=900), _line(mate=3), _line(mate=-2), _line(mate=1), _line(mate=-8)]
    )
    assert [(l.cp, l.mate) for l in ordered] == [
        (None, 1),    # mate in 1 beats mate in 3
        (None, 3),
        (900, None),  # any mate beats any evaluation
        (None, -8),   # being mated later is better than being mated sooner
        (None, -2),
    ]


def test_uniqueness_gap_is_never_negative_after_sorting():
    model = winrate.WinModel(a=0.0, k=0.004, fitted=True, provenance={})
    ordered = analysis.best_first([_line(cp=120), _line(cp=400)])
    assert ordered[0].win_pct(model) - ordered[1].win_pct(model) > 0


# -- recurring-mistake clustering ------------------------------------------


def _seed(db, rows):
    """rows: (game_id, my_color, ply, fen, san, best, loss, judgement, source)."""
    for gid, colour, ply, fen, san, best, loss, judgement, source in rows:
        db.execute(
            "INSERT OR IGNORE INTO games (id, site, account, my_color, rated, source, "
            "moves, ply, raw) VALUES (?,'lichess','me',?,1,?,'',1,'{}')",
            (gid, colour, source),
        )
        db.execute(
            "INSERT INTO plies (game_id, ply, fen, mover, uci, san, win_before, "
            "win_after, loss, judgement, best_san, nodes) "
            "VALUES (?,?,?,?,'a1a2',?,50,?,?,?,?,1000)",
            (gid, ply, fen, colour, san, 50 - loss, loss, judgement, best),
        )
    db.commit()


def test_recurring_positions_groups_by_fen_and_ignores_singletons(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            ("g1", "black", 3, "FEN_A", "Nh6", "d5", 40.0, "blunder", "lobby"),
            ("g2", "black", 3, "FEN_A", "Nh6", "d5", 42.0, "blunder", "lobby"),
            ("g3", "black", 5, "FEN_B", "Qb6", "Na6", 48.0, "blunder", "lobby"),
        ])
        clusters = review.recurring_positions(db)
        assert len(clusters) == 1
        assert clusters[0].fen == "FEN_A"
        assert clusters[0].count == 2
        assert clusters[0].played == {"Nh6": 2}
        assert clusters[0].avg_loss == pytest.approx(41.0)
        assert sorted(clusters[0].games) == ["g1", "g2"]


def test_repeated_moves_spans_positions_that_fen_grouping_would_split(tmp_path):
    """A move played by rote is wrong in several *different* positions."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            ("g1", "black", 3, "FEN_A", "Nh6", "d5", 40.0, "blunder", "lobby"),
            ("g2", "black", 3, "FEN_B", "Nh6", "e6", 36.0, "blunder", "lobby"),
        ])
        assert review.recurring_positions(db) == []  # nothing repeats by position
        moves = review.repeated_moves(db)
        assert len(moves) == 1
        assert moves[0].san == "Nh6"
        assert moves[0].count == 2
        assert moves[0].positions == 2
        assert moves[0].alternatives == {"d5": 1, "e6": 1}


def test_ai_games_are_excluded_unless_asked_for(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            ("g1", "black", 3, "FEN_A", "Nh6", "d5", 40.0, "blunder", "ai"),
            ("g2", "black", 3, "FEN_A", "Nh6", "d5", 40.0, "blunder", "ai"),
        ])
        assert review.recurring_positions(db) == []
        assert len(review.recurring_positions(db, include_ai=True)) == 1


def test_phase_table_rate_and_open_ended_last_bucket(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            ("g1", "white", 0, "F1", "e3", "Nf3", 40.0, "blunder", "lobby"),
            ("g2", "white", 4, "F2", "e4", "Nf3", 2.0, "ok", "lobby"),
            ("g3", "white", 99, "F3", "h3", "Bb2", 1.0, "ok", "lobby"),
        ])
        buckets = review.phase_table(db)
        assert buckets[0].moves == 2 and buckets[0].blunders == 1
        assert buckets[0].rate == pytest.approx(50.0)
        assert buckets[0].per_move == pytest.approx(21.0)
        assert buckets[-1].moves == 1  # ply 99 lands in the open-ended band
        assert buckets[-1].high > 10**5


def test_move_number_formats_both_colours():
    assert review.move_number(0) == "1."
    assert review.move_number(1) == "1..."
    assert review.move_number(4) == "3."


# -- opening book ----------------------------------------------------------


def test_recent_months_counts_back_over_a_year_boundary():
    assert book.recent_months(3, before="2026-02") == ["2026-01", "2025-12", "2025-11"]


def test_recent_months_skips_the_month_still_being_written():
    assert book.recent_months(1, before="2026-08") == ["2026-07"]


def _write_dump(path, games):
    """Write a synthetic `.pgn.zst`. games: (moves, result, white_elo, black_elo)."""
    import zstandard

    out = []
    for moves, result, white, black in games:
        out += [f'[Result "{result}"]', f'[WhiteElo "{white}"]', f'[BlackElo "{black}"]', ""]
        body = " ".join(
            f"{i // 2 + 1}. {san}" if i % 2 == 0 else san for i, san in enumerate(moves)
        )
        out += [f"{body} {result}", ""]
    path.write_bytes(zstandard.ZstdCompressor().compress("\n".join(out).encode()))
    return path


def _naive_counts(games, depth, min_games):
    """The obvious single-pass count, as the reference the staged one must match."""
    counts: dict[tuple[str, ...], int] = {}
    for moves, result, *_ in games:
        if result == "*":
            continue
        key: tuple[str, ...] = ()
        for san in moves[:depth]:
            key = (*key, san)
            counts[key] = counts.get(key, 0) + 1
    return {key: n for key, n in counts.items() if n >= min_games}


def test_staged_counting_matches_the_naive_single_pass(tmp_path):
    """The staged prune is an optimisation, not an approximation.

    Counting is monotone, so restricting deeper stages to survivors cannot lose
    a node that would have survived. This is the test that claim rests on.
    """
    import random

    rng = random.Random(20260810)
    alphabet = ["a3", "b3", "c3", "d3", "e3", "f3"]
    games = [
        (
            [rng.choice(alphabet) for _ in range(rng.randint(1, 10))],
            rng.choice(["1-0", "0-1", "1/2-1/2", "*"]),
            1500,
            1500,
        )
        for _ in range(4000)
    ]
    path = _write_dump(tmp_path / "d.pgn.zst", games)

    nodes, corpus = book.build_tree(
        [path], depth=8, min_games=10, stages=(2, 4, 6)
    )
    assert corpus == sum(1 for g in games if g[1] != "*")

    expected = _naive_counts(games, depth=8, min_games=10)
    assert {node.line: node.games for node in nodes} == expected
    assert expected  # the corpus is big enough for the comparison to mean something


def test_unfinished_games_are_not_counted(tmp_path):
    path = _write_dump(tmp_path / "d.pgn.zst", [
        (["e3", "e6"], "1-0", 1500, 1500),
        (["e3", "e6"], "*", 1500, 1500),
    ])
    nodes, corpus = book.build_tree([path], depth=4, min_games=1, stages=())
    assert corpus == 1
    assert {node.line: node.games for node in nodes}[("e3",)] == 1


def test_min_elo_filters_the_corpus(tmp_path):
    path = _write_dump(tmp_path / "d.pgn.zst", [
        (["e3"], "1-0", 2000, 2000),
        (["e3"], "1-0", 1000, 1000),
    ])
    nodes, corpus = book.build_tree([path], depth=2, min_games=1, min_elo=1800, stages=())
    assert corpus == 1
    assert nodes[0].games == 1 and nodes[0].elo == 2000


def test_min_games_prunes_rare_lines(tmp_path):
    path = _write_dump(tmp_path / "d.pgn.zst", [
        (["e3", "e6"], "1-0", 1500, 1500),
        (["e3", "e6"], "0-1", 1500, 1500),
        (["e3", "d6"], "1-0", 1500, 1500),
    ])
    nodes, _ = book.build_tree([path], depth=4, min_games=2, stages=())
    assert {node.line for node in nodes} == {("e3",), ("e3", "e6")}


def test_transposing_move_orders_pool_into_one_position(tmp_path):
    """1.Nf3 e6 2.e3 and 1.e3 e6 2.Nf3 reach one position; the book says so.

    Entries key on the position *before* the move, so the merge shows up on the
    fourth ply — the first move played from the shared position.
    """
    path = _write_dump(tmp_path / "d.pgn.zst", [
        (["Nf3", "e6", "e3", "d6"], "1-0", 1500, 1500),
        (["Nf3", "e6", "e3", "d6"], "1-0", 1500, 1500),
        (["e3", "e6", "Nf3", "d6"], "0-1", 1500, 1500),
        (["e3", "e6", "Nf3", "d6"], "0-1", 1500, 1500),
    ])
    nodes, _ = book.build_tree([path], depth=4, min_games=2, stages=())
    assert len({node.line for node in nodes}) == 8  # two distinct move orders

    entries = list(book.to_entries(nodes))
    board = chess.variant.AtomicBoard()
    for san in ("Nf3", "e6", "e3"):
        board.push_san(san)
    merged = [e for e in entries if e.fen == book.book_fen(board)]
    assert len(merged) == 1  # ...and one entry for the move out of the shared position
    assert merged[0].san == "d6" and merged[0].ply == 3
    assert merged[0].games == 4
    assert merged[0].white_wins == 2 and merged[0].black_wins == 2


def test_book_fen_drops_the_move_counters():
    board = chess.variant.AtomicBoard()
    board.push_san("Nf3")
    assert book.book_fen(board).count(" ") == 3
    assert board.fen().startswith(book.book_fen(board))


def test_score_is_from_the_point_of_view_of_whoever_plays_the_move():
    white = book.Entry(fen="f", san="e3", uci="e2e3", ply=0, games=10,
                       white_wins=7, draws=0, black_wins=3)
    black = book.Entry(fen="f", san="e6", uci="e7e6", ply=1, games=10,
                       white_wins=7, draws=0, black_wins=3)
    assert white.score == pytest.approx(70.0)
    assert black.score == pytest.approx(30.0)


def test_draws_count_a_half():
    entry = book.Entry(fen="f", san="e3", uci="e2e3", ply=0, games=10,
                       white_wins=4, draws=2, black_wins=4)
    assert entry.score == pytest.approx(50.0)


def _seed_book(db, entries):
    book.replace_book(db, entries, {"games": 0})


def test_lookup_accepts_a_full_fen_and_orders_by_popularity(tmp_path):
    board = chess.variant.AtomicBoard()
    fen = book.book_fen(board)
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed_book(db, [
            book.Entry(fen=fen, san="e3", uci="e2e3", ply=0, games=10,
                       white_wins=5, draws=0, black_wins=5),
            book.Entry(fen=fen, san="Nf3", uci="g1f3", ply=0, games=30,
                       white_wins=20, draws=0, black_wins=10),
        ])
        found = book.lookup(db, board.fen())  # full FEN, with the counters
        assert [entry.san for entry in found] == ["Nf3", "e3"]
        assert book.lookup(db, fen, min_games=20)[0].san == "Nf3"


def test_book_context_ranks_your_move_against_the_field(tmp_path):
    fen = book.book_fen(chess.variant.AtomicBoard())
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed_book(db, [
            book.Entry(fen=fen, san="Nf3", uci="g1f3", ply=0, games=75,
                       white_wins=45, draws=0, black_wins=30),
            book.Entry(fen=fen, san="e3", uci="e2e3", ply=0, games=25,
                       white_wins=10, draws=0, black_wins=15),
        ])
        context = review.book_context(db, fen, "e3", best_san="Nf3")
        assert context.rank == 2
        assert context.share == pytest.approx(25.0)
        assert context.played.score == pytest.approx(40.0)
        assert context.best.san == "Nf3"


def test_book_context_reports_a_move_nobody_else_plays(tmp_path):
    fen = book.book_fen(chess.variant.AtomicBoard())
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed_book(db, [
            book.Entry(fen=fen, san="Nf3", uci="g1f3", ply=0, games=75,
                       white_wins=45, draws=0, black_wins=30),
        ])
        context = review.book_context(db, fen, "a3")
        assert context.played is None
        assert context.rank is None
        assert context.share == 0.0
        assert context.moves  # the field's moves are still there to show


def test_rebuilding_the_book_replaces_it_rather_than_appending(tmp_path):
    fen = book.book_fen(chess.variant.AtomicBoard())
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed_book(db, [book.Entry(fen=fen, san="Nf3", uci="g1f3", ply=0, games=1)])
        _seed_book(db, [book.Entry(fen=fen, san="e3", uci="e2e3", ply=0, games=2)])
        assert [entry.san for entry in book.lookup(db, fen)] == ["e3"]


def test_the_analysis_cache_survives_the_book_schema_upgrade(tmp_path):
    """v1 stores must gain `book` without losing the expensive table."""
    path = tmp_path / "t.sqlite"
    with store.open_db(path) as db:
        store.store_analysis(db, "FEN", 1000, 2, "e", [{"uci": "a1a2"}], 12, 1.0)
        db.commit()
        store.set_meta(db, "schema_version", "1")
    with store.open_db(path) as db:
        assert store.cached_analysis(db, "FEN", 1000, 2, "e") == [{"uci": "a1a2"}]
        assert store.get_meta(db, "schema_version") == str(store.SCHEMA_VERSION)
        db.execute("SELECT * FROM book")  # the new table is there


def test_the_v7_migration_relabels_a_hold_the_deep_pass_says_is_won(tmp_path):
    """A stored motive is a claim about the position and has to be true.

    v6 recorded whatever the screening pass guessed. Where the confirming pass
    disagreed, the corpus told the solver "only one move holds" about a
    position it had itself scored at 96% — so the migration re-derives the
    label from `win_before`, and leaves the tablebase rows (which have none)
    alone.
    """
    from trainer import puzzles

    path = tmp_path / "t.sqlite"
    with store.open_db(path) as db:
        db.executemany(
            "INSERT INTO puzzles (id, source, fen, solver, solution, solution_san,"
            " themes, phase, motive, win_before, tb_category, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                ("won", "dump", "F", "white", "[]", "[]", "[]", "endgame",
                 "save", 96.0, None, 1.0),
                ("held", "dump", "F", "white", "[]", "[]", "[]", "endgame",
                 "save", 46.0, None, 1.0),
                ("exact", "tablebase", "F", "white", "[]", "[]", "[]", "endgame",
                 "save", None, "draw", 1.0),
            ],
        )
        db.commit()
        store.set_meta(db, "schema_version", "6")

    with store.open_db(path) as db:
        motives = {
            row["id"]: row["motive"] for row in db.execute("SELECT id, motive FROM puzzles")
        }
    assert motives == {"won": "win", "held": "save", "exact": "save"}
    assert store.win_floor() == puzzles.WIN_FLOOR


# -- repertoire ------------------------------------------------------------


def _book_from_lines(db, lines):
    """Build a book from {(san, san, ...): (games, white, draws, black)}.

    Each key is a full move sequence from the start; the last move of it is the
    book entry and everything before it locates the position.
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


def test_derive_takes_the_best_scoring_move_not_the_most_played(tmp_path):
    """The field's most popular move is what you are trying to beat."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("Nf3",): (1000, 400, 0, 600),   # popular, scores 40%
            ("e3",): (300, 210, 0, 90),      # rarer, scores 70%
        })
        nodes = repertoire.derive(db, "white", depth=1, min_games=200)
    assert [node.san for node in nodes] == ["e3"]
    assert nodes[0].score == pytest.approx(70.0)


def test_derive_will_not_recommend_a_move_on_a_thin_sample(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("Nf3",): (1000, 400, 0, 600),
            ("e3",): (10, 10, 0, 0),  # 100%, on ten games — noise, not a move
        })
        nodes = repertoire.derive(db, "white", depth=1, min_games=200)
    assert [node.san for node in nodes] == ["Nf3"]


def test_derive_prepares_common_replies_and_ignores_rare_ones(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("e3",): (1000, 600, 0, 400),
            ("e3", "f6"): (600, 400, 0, 200),
            ("e3", "Nc6"): (390, 200, 0, 190),
            ("e3", "d6"): (10, 5, 0, 5),        # 1% of the field: not prepared for
            ("e3", "f6", "Qh5+"): (500, 400, 0, 100),
            ("e3", "Nc6", "Nf3"): (300, 160, 0, 140),
            ("e3", "d6", "Nf3"): (10, 5, 0, 5),
        })
        nodes = repertoire.derive(db, "white", depth=4, min_games=200, min_share=3.0)
    assert sorted(node.san for node in nodes) == ["Nf3", "Qh5+", "e3"]


def test_derive_stops_at_the_depth_limit(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("e3",): (1000, 600, 0, 400),
            ("e3", "f6"): (900, 500, 0, 400),
            ("e3", "f6", "Nf3"): (800, 450, 0, 350),
            ("e3", "f6", "Nf3", "e6"): (700, 400, 0, 300),
            ("e3", "f6", "Nf3", "e6", "d4"): (600, 350, 0, 250),
        })
        shallow = repertoire.derive(db, "white", depth=2, min_games=200)
        deeper = repertoire.derive(db, "white", depth=5, min_games=200)
    assert [node.ply for node in shallow] == [0]
    assert [node.ply for node in deeper] == [0, 2, 4]


def test_a_transposition_becomes_one_node_not_two(tmp_path):
    """Two move orders reaching the same position must not get two answers."""
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("Nf3",): (1000, 600, 0, 400),
            ("Nf3", "f6"): (500, 300, 0, 200),
            ("Nf3", "Nc6"): (500, 300, 0, 200),
            # Both replies are answered by e3, and Black then transposes.
            ("Nf3", "f6", "e3"): (500, 300, 0, 200),
            ("Nf3", "Nc6", "e3"): (500, 300, 0, 200),
            ("Nf3", "f6", "e3", "Nc6"): (400, 240, 0, 160),
            ("Nf3", "Nc6", "e3", "f6"): (400, 240, 0, 160),
            ("Nf3", "f6", "e3", "Nc6", "d4"): (300, 180, 0, 120),
        })
        nodes = repertoire.derive(db, "white", depth=6, min_games=200)
    fens = [node.fen for node in nodes]
    assert len(fens) == len(set(fens))
    assert sum(1 for node in nodes if node.san == "d4") == 1


def test_max_nodes_truncates_the_deepest_lines_not_a_random_branch(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("e3",): (1000, 600, 0, 400),
            ("e3", "f6"): (500, 300, 0, 200),
            ("e3", "Nc6"): (500, 300, 0, 200),
            ("e3", "f6", "Nf3"): (500, 300, 0, 200),
            ("e3", "Nc6", "Nf3"): (500, 300, 0, 200),
        })
        nodes = repertoire.derive(db, "white", depth=6, min_games=200, max_nodes=2)
    assert len(nodes) == 2
    assert sorted(node.ply for node in nodes) == [0, 2]  # breadth, not depth


def test_a_black_repertoire_starts_after_whites_choices(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("e3",): (600, 300, 0, 300),
            ("Nf3",): (400, 200, 0, 200),
            ("e3", "f6"): (500, 100, 0, 400),
            ("Nf3", "f6"): (300, 60, 0, 240),
        })
        nodes = repertoire.derive(db, "black", depth=4, min_games=200)
    assert len(nodes) == 2
    assert {node.ply for node in nodes} == {1}
    assert {node.san for node in nodes} == {"f6"}


def test_validation_needs_the_position_before_and_after_each_move():
    start = book.book_fen(chess.variant.AtomicBoard())
    node = repertoire.Node(color="white", fen=start, san="e3", uci="e2e3", ply=0)
    fens = repertoire.validation_fens([node, node])
    assert len(fens) == 2  # de-duplicated across nodes
    assert fens[0].startswith(start)
    assert " b " in fens[1]


def _cache(db, fen, cp, uci="e2e3", nodes=1000, multipv=1):
    store.store_analysis(
        db, fen, nodes, multipv, analysis.ENGINE_ID,
        [{"uci": uci, "cp": cp, "mate": None, "depth": 20}], 20, 1.0,
    )
    db.commit()


_MODEL = winrate.WinModel(a=0.0, k=winrate.LICHESS_STANDARD_K, fitted=True, provenance={})


def test_validate_flags_a_book_move_the_engine_dislikes(tmp_path):
    """Popularity is not merit; this is the check that says so."""
    board = chess.variant.AtomicBoard()
    start = book.book_fen(board)
    board.push_san("e3")
    with store.open_db(tmp_path / "t.sqlite") as db:
        repertoire.save(db, [repertoire.Node(
            color="white", fen=start, san="e3", uci="e2e3", ply=0, games=500, score=70.0
        )])
        _cache(db, repertoire.full_fen(start), 300, uci="g1f3")  # engine prefers Nf3
        _cache(db, board.fen(), 0, uci="g8f6")                   # e3 leaves Black even
        flagged = repertoire.validate(
            db, "white", nodes_budget=1000, multipv=1, model=_MODEL
        )
        stored = repertoire.get(db, "white", start)
    assert [node.san for node in flagged] == ["e3"]
    assert stored.best_san == "Nf3"
    assert stored.agrees is False
    assert stored.val_loss == pytest.approx(25.1, abs=0.5)  # 75.1% -> 50%
    assert stored.san == "e3"  # flagged, not silently rewritten


def test_validate_leaves_a_move_the_engine_agrees_with_alone(tmp_path):
    board = chess.variant.AtomicBoard()
    start = book.book_fen(board)
    board.push_san("e3")
    with store.open_db(tmp_path / "t.sqlite") as db:
        repertoire.save(db, [repertoire.Node(
            color="white", fen=start, san="e3", uci="e2e3", ply=0
        )])
        _cache(db, repertoire.full_fen(start), 300, uci="e2e3")
        _cache(db, board.fen(), -300, uci="g8f6")
        flagged = repertoire.validate(
            db, "white", nodes_budget=1000, multipv=1, model=_MODEL
        )
        stored = repertoire.get(db, "white", start)
    assert flagged == []
    assert stored.agrees is True
    assert stored.val_loss == pytest.approx(0.0, abs=0.01)


def test_fix_promotes_the_engines_move_and_records_what_it_replaced(tmp_path):
    board = chess.variant.AtomicBoard()
    start = book.book_fen(board)
    board.push_san("e3")
    with store.open_db(tmp_path / "t.sqlite") as db:
        repertoire.save(db, [repertoire.Node(
            color="white", fen=start, san="e3", uci="e2e3", ply=0, games=500, score=70.0
        )])
        _cache(db, repertoire.full_fen(start), 300, uci="g1f3")
        _cache(db, board.fen(), 0, uci="g8f6")
        repertoire.validate(
            db, "white", nodes_budget=1000, multipv=1, model=_MODEL, fix=True
        )
        stored = repertoire.get(db, "white", start)
    assert stored.san == "Nf3"
    assert stored.source == "engine"
    assert stored.flagged is False
    assert "was e3" in stored.note


def test_saving_again_keeps_validation_unless_replace_is_asked_for(tmp_path):
    start = book.book_fen(chess.variant.AtomicBoard())
    with store.open_db(tmp_path / "t.sqlite") as db:
        repertoire.save(db, [repertoire.Node(
            color="white", fen=start, san="e3", uci="e2e3", ply=0, val_loss=4.0,
            val_nodes=1000,
        )])
        assert repertoire.get(db, "white", start).validated
        repertoire.save(db, [repertoire.Node(
            color="white", fen="OTHER", san="d4", uci="d2d4", ply=0
        )])
        assert repertoire.get(db, "white", start).validated  # untouched
        repertoire.save(
            db,
            [repertoire.Node(color="white", fen="OTHER", san="d4", uci="d2d4", ply=0)],
            replace=True,
        )
        assert repertoire.get(db, "white", start) is None


def test_coverage_ranks_gaps_by_what_they_cost_you(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            # reached three times, handled fine: not a gap worth an evening
            ("g1", "white", 2, "CHEAP", "Nf3", "Nf3", 1.0, "ok", "lobby"),
            ("g2", "white", 2, "CHEAP", "Nf3", "Nf3", 1.0, "ok", "lobby"),
            ("g3", "white", 2, "CHEAP", "Nf3", "Nf3", 1.0, "ok", "lobby"),
            # reached twice, lost both times
            ("g4", "white", 4, "DEAR", "Qh5", "d4", 40.0, "blunder", "lobby"),
            ("g5", "white", 4, "DEAR", "Qh5", "d4", 36.0, "blunder", "lobby"),
            # already covered by the repertoire
            ("g6", "white", 0, "KNOWN", "e3", "e3", 0.0, "ok", "lobby"),
        ])
        repertoire.save(db, [repertoire.Node(
            color="white", fen="KNOWN", san="e3", uci="e2e3", ply=0
        )])
        gaps, covered, total = repertoire.coverage(db, "white", max_ply=12)
    assert (covered, total) == (1, 6)
    assert [gap.fen for gap in gaps] == ["DEAR", "CHEAP"]
    assert gaps[0].cost == pytest.approx(76.0)
    assert gaps[0].played == {"Qh5": 2}


def test_coverage_only_counts_the_colour_asked_about(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            ("g1", "white", 0, "W", "e3", "e3", 0.0, "ok", "lobby"),
            ("g2", "black", 1, "B", "f6", "Nc6", 30.0, "blunder", "lobby"),
        ])
        _, _, white_total = repertoire.coverage(db, "white")
        black_gaps, _, black_total = repertoire.coverage(db, "black")
    assert white_total == 1
    assert black_total == 1
    assert [gap.fen for gap in black_gaps] == ["B"]


def test_deviations_are_the_lines_you_had_and_did_not_play(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _seed(db, [
            ("g1", "white", 0, "KNOWN", "Nf3", "e3", 12.0, "inaccuracy", "lobby"),
            ("g2", "white", 0, "KNOWN", "d4", "e3", 20.0, "mistake", "lobby"),
            ("g3", "white", 0, "KNOWN", "e3", "e3", 0.0, "ok", "lobby"),
        ])
        repertoire.save(db, [repertoire.Node(
            color="white", fen="KNOWN", san="e3", uci="e2e3", ply=0
        )])
        found = repertoire.deviations(db, "white")
    assert len(found) == 1
    assert found[0].count == 2  # the game where you did follow it is not a deviation
    assert found[0].played == {"Nf3": 1, "d4": 1}
    assert found[0].cost == pytest.approx(32.0)


def test_lines_of_names_each_node_by_the_moves_that_reach_it(tmp_path):
    with store.open_db(tmp_path / "t.sqlite") as db:
        _book_from_lines(db, {
            ("e3",): (1000, 600, 0, 400),
            ("e3", "f6"): (900, 500, 0, 400),
            ("e3", "f6", "Nf3"): (800, 450, 0, 350),
        })
        repertoire.save(db, repertoire.derive(db, "white", depth=4, min_games=200))
        rows = repertoire.lines_of(db, "white")
    assert [(path, node.san) for path, node in rows] == [
        ([], "e3"),
        (["e3", "f6"], "Nf3"),
    ]


def test_the_analysis_cache_survives_the_repertoire_schema_upgrade(tmp_path):
    path = tmp_path / "t.sqlite"
    with store.open_db(path) as db:
        store.store_analysis(db, "FEN", 1000, 2, "e", [{"uci": "a1a2"}], 12, 1.0)
        db.commit()
        store.set_meta(db, "schema_version", "2")
    with store.open_db(path) as db:
        assert store.cached_analysis(db, "FEN", 1000, 2, "e") == [{"uci": "a1a2"}]
        db.execute("SELECT * FROM repertoire")
