"""Command line for the atomic trainer.

    python -m trainer.cli import clee96326        # fetch atomic games from Lichess
    python -m trainer.cli analyse                 # deep pass over everything unanalysed
    python -m trainer.cli games                   # what has been imported
    python -m trainer.cli review <game-id>        # where the game was thrown
    python -m trainer.cli worst                   # your biggest mistakes, all games
    python -m trainer.cli patterns                # mistakes you make more than once
    python -m trainer.cli book build              # count an opening book from the dumps
    python -m trainer.cli book line Nf3 f6        # what the population plays here
    python -m trainer.cli openings                # the named opening index
    python -m trainer.cli openings name Nf3 f6 --as "..."   # name a line yourself
    python -m trainer.cli traps scan              # find the one-move-holds positions
    python -m trainer.cli traps list              # what the field falls for
    python -m trainer.cli repertoire build white  # derive a repertoire from the book
    python -m trainer.cli repertoire validate     # let the engine check every line
    python -m trainer.cli repertoire coverage     # the lines you keep needing and lack
    python -m trainer.cli puzzles generate        # mine puzzles from your games
    python -m trainer.cli puzzles endgames        # exact ones from the tablebase
    python -m trainer.cli puzzles mine            # scrape rated Lichess games
    python -m trainer.cli puzzles rate            # measure difficulty on the ladder
    python -m trainer.cli puzzles ratings         # the corpus by difficulty
    python -m trainer.cli puzzles list            # browse the corpus
    python -m trainer.cli puzzles show <id>       # one puzzle and its line
    python -m trainer.cli puzzles themes          # do the theme detectors work?
    python -m trainer.cli rejudge                 # re-score from cache (no engine)
    python -m trainer.cli status                  # store, cache, and win% model
    python -m trainer.cli serve                   # the browser UI, all of it
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import chess
import chess.variant

from . import (
    analysis,
    book,
    difficulty,
    dump,
    endgames,
    explain,
    generate,
    mine,
    openings,
    puzzles,
    repertoire,
    review,
    store,
    themes,
    traps,
    winrate,
)
from .ingest import lichess

JUDGEMENT_MARK = {
    "blunder": "??",
    "mistake": "?",
    "inaccuracy": "?!",
    "ok": "",
}


def _model(db: sqlite3.Connection) -> winrate.WinModel:
    model = winrate.load()
    if not model.fitted:
        print(
            "warning: the cp->win% curve has not been fitted for atomic yet, so\n"
            "         every judgement below uses Lichess's STANDARD-chess curve\n"
            "         and is wrong by an unknown amount. Fix with:\n"
            "             python -m trainer.fit_winrate\n"
            "         then `python -m trainer.cli rejudge` (free, no engine).\n",
            file=sys.stderr,
        )
    return model


# -- commands --------------------------------------------------------------


def cmd_import(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        print(f"importing atomic games for {args.user} from Lichess...")
        result = lichess.import_account(
            db,
            args.user,
            full=args.full,
            max_games=args.max,
            on_progress=lambda n: print(f"  {n} games...", end="\r"),
        )
        total = db.execute(
            "SELECT COUNT(*) AS n FROM games WHERE account = ?", (args.user.lower(),)
        ).fetchone()["n"]
    print(
        f"  fetched {result['seen']}, added {result['added']} new, "
        f"skipped {result['skipped']} non-games; {total} stored in total"
    )
    if result["added"]:
        print("Next: python -m trainer.cli analyse")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        model = _model(db)
        if args.rejudge:
            games = list(
                db.execute(
                    "SELECT * FROM games WHERE analysed_nodes = ? ORDER BY created_at DESC",
                    (args.nodes,),
                )
            )
            what = "re-scoring from cache"
        else:
            games = analysis.pending_games(db, args.nodes, args.account)
            what = "analysing"
        if args.limit:
            games = games[: args.limit]
        if not games:
            print("nothing to do")
            return 0

        workers = args.workers or analysis.default_workers()
        plies = sum(game["ply"] for game in games)
        print(
            f"{what} {len(games)} games ({plies} plies) at {args.nodes:,} nodes, "
            f"MultiPV {args.multipv}, {workers} workers"
        )
        started = time.time()

        def report(index: int, game: sqlite3.Row, judged: list[analysis.Judged]) -> None:
            blunders = sum(1 for j in judged if j.judgement == "blunder")
            elapsed = time.time() - started
            rate = (index + 1) / max(elapsed, 1e-9)
            left = (len(games) - index - 1) / max(rate, 1e-9)
            print(
                f"  [{index + 1}/{len(games)}] {game['id']} {game['ply']:>3} ply  "
                f"{blunders} blunders   {elapsed / 60:.1f}m elapsed, "
                f"~{left / 60:.0f}m left"
            )

        done = analysis.analyse_games(
            db,
            games,
            nodes=args.nodes,
            multipv=args.multipv,
            workers=workers,
            model=model,
            on_game=report,
        )
        print(f"done: {done} games in {(time.time() - started) / 60:.1f} minutes")
    return 0


def cmd_games(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        sql = "SELECT * FROM games"
        params: list[object] = []
        if args.account:
            sql += " WHERE account = ?"
            params.append(args.account.lower())
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(args.limit)
        rows = list(db.execute(sql, params))
        if not rows:
            print("no games imported yet — try `python -m trainer.cli import <user>`")
            return 0
        print(f"{'id':<10} {'date':<11} {'col':<6} {'res':<5} {'ply':>4} {'blun':>5}  opponent")
        for row in rows:
            counts = db.execute(
                "SELECT COUNT(*) AS n FROM plies WHERE game_id = ? AND mover = ? "
                "AND judgement = 'blunder'",
                (row["id"], row["my_color"] or "white"),
            ).fetchone()
            date = time.strftime("%Y-%m-%d", time.localtime((row["created_at"] or 0) / 1000))
            mine = row["my_color"]
            result = (
                "-" if mine is None
                else "win" if row["winner"] == mine
                else "draw" if row["winner"] is None
                else "loss"
            )
            opponent = row["black"] if mine == "white" else row["white"]
            blunders = counts["n"] if row["analysed_nodes"] else "-"
            print(
                f"{row['id']:<10} {date:<11} {mine or '?':<6} {result:<5} "
                f"{row['ply']:>4} {blunders!s:>5}  {opponent}"
            )
    return 0


def _game_or_die(db: sqlite3.Connection, game_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    if row is None:
        raise SystemExit(f"no game {game_id!r} in the store")
    if not row["analysed_nodes"]:
        raise SystemExit(f"game {game_id} has not been analysed yet")
    return row


def cmd_review(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        game = _game_or_die(db, args.game)
        mine = game["my_color"]
        print(
            f"https://lichess.org/{game['id']}  "
            f"{game['white']} ({game['white_rating']}) vs "
            f"{game['black']} ({game['black_rating']})  "
            f"{game['speed']}, {game['status']}"
        )
        rows = list(
            db.execute(
                "SELECT * FROM plies WHERE game_id = ? ORDER BY ply", (game["id"],)
            )
        )
        shown = [
            row for row in rows
            if row["loss"] >= winrate.INACCURACY
            and (args.both or mine is None or row["mover"] == mine)
        ]
        if not shown:
            print("  no inaccuracies or worse — clean game")
            return 0
        print(f"\n  {'move':<10} {'played':<8} {'win%':>13}  {'best':<8}  cost")
        for row in shown:
            number = row["ply"] // 2 + 1
            dots = "." if row["mover"] == "white" else "..."
            mark = JUDGEMENT_MARK[row["judgement"]]
            print(
                f"  {number:>3}{dots:<7}{row['san'] + mark:<8} "
                f"{row['win_before']:>5.1f} -> {row['win_after']:<5.1f}  "
                f"{row['best_san'] or '?':<8}  -{row['loss']:.0f}"
            )
        worst = max(shown, key=lambda r: r["loss"])
        print(
            f"\n  the game turned at move {worst['ply'] // 2 + 1}: "
            f"{worst['san']} instead of {worst['best_san']} "
            f"({worst['win_before']:.0f}% -> {worst['win_after']:.0f}%)"
        )
        print(f"  replay from:  {worst['fen']}")
    return 0


def _parse_move(board: chess.variant.AtomicBoard, text: str | None) -> str | None:
    """A move as UCI, accepting SAN too, which is what people type."""
    if not text:
        return None
    try:
        return board.parse_san(text).uci()
    except ValueError:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise SystemExit(f"illegal move: {text}") from None
        return move.uci()


def cmd_explain(args: argparse.Namespace) -> int:
    board = chess.variant.AtomicBoard(args.fen) if args.fen else chess.variant.AtomicBoard()
    for san in args.moves or []:
        board.push_san(san)
    fen = board.fen()
    move = _parse_move(board, args.move)
    versus = _parse_move(board, args.versus)

    with store.open_db() as db:
        def search(position: str, multipv: int) -> list[analysis.Line]:
            return analysis.evaluate_now(db, position, nodes=args.nodes, multipv=multipv)

        try:
            result = explain.explain(fen, search, move=move, versus=versus)
        finally:
            analysis.LIVE.close()

    if result["over"]:
        print(f"the game is over: {result['result']}")
        return 0
    print(fen)
    print(f"\n{result['summary']}\n")

    def show(heading: str, reasons: list[dict]) -> None:
        print(heading)
        if not reasons:
            print("  no threat, no forcing line, no special rule: the engine just "
                  "rates the resulting position highest")
        for reason in reasons:
            print(f"  [{reason['kind']}] {reason['text']}")
        print()

    best = result["best"]
    alt = result["alternative"]
    sections = [(f"What {best['san']} does ({best['win']:.0f}%):", result["reasons"])]
    if alt:
        sections.append((f"Why not {alt['san']} ({alt['win']:.0f}%, -{alt['loss']:.0f}):",
                         alt["reasons"]))
        if not result["subject_is_best"]:
            sections.reverse()
    for heading, reasons in sections:
        show(heading, reasons)
    return 0


def cmd_worst(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        rows = list(
            db.execute(
                "SELECT p.*, g.id AS gid, g.my_color, g.created_at, g.source "
                "FROM plies p JOIN games g ON g.id = p.game_id "
                "WHERE p.mover = g.my_color AND p.loss >= ? "
                + ("" if args.include_ai else "AND (g.source IS NULL OR g.source != 'ai') ")
                + "ORDER BY p.loss DESC LIMIT ?",
                (winrate.MISTAKE, args.limit),
            )
        )
        if not rows:
            print("nothing analysed yet")
            return 0
        print(f"{'game':<10} {'move':<9} {'played':<8} {'best':<8} {'cost':>5}  date")
        for row in rows:
            number = row["ply"] // 2 + 1
            dots = "." if row["mover"] == "white" else "..."
            date = time.strftime("%Y-%m-%d", time.localtime((row["created_at"] or 0) / 1000))
            print(
                f"{row['gid']:<10} {str(number) + dots:<9} {row['san']:<8} "
                f"{row['best_san'] or '?':<8} {row['loss']:>5.0f}  {date}"
            )
    return 0


def _book_note(context: review.BookContext, san: str, best_san: str | None) -> None:
    """One line on how the wider player base treats a move you got wrong."""
    if context.played is None:
        top = ", ".join(f"{e.san} {e.games:,}" for e in context.moves[:3])
        print(f"      book: {san} is off-book here; the field plays {top}")
        return
    line = (
        f"      book: {san} played by {context.share:.0f}% of the field "
        f"(#{context.rank} of {len(context.moves)}), scoring {context.played.score:.0f}%"
    )
    if context.best is not None and best_san != san:
        line += f"; {best_san} scores {context.best.score:.0f}% on {context.best.games:,}"
    print(line)


def cmd_patterns(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        if not db.execute("SELECT 1 FROM plies LIMIT 1").fetchone():
            print("nothing analysed yet — run `python -m trainer.cli analyse`")
            return 0

        print("where your games go wrong, by phase:\n")
        print(f"  {'plies':<12} {'moves':>6} {'blunders':>9} {'rate':>7} {'win% lost/move':>15}")
        for bucket in review.phase_table(db, include_ai=args.include_ai):
            span = f"{bucket.low}-{bucket.high}" if bucket.high < 10**6 else f"{bucket.low}+"
            print(
                f"  {span:<12} {bucket.moves:>6} {bucket.blunders:>9} "
                f"{bucket.rate:>6.1f}% {bucket.per_move:>14.1f}"
            )

        clusters = review.recurring_positions(
            db, max_ply=args.max_ply, min_count=args.min_count, include_ai=args.include_ai
        )
        print(
            f"\nsame position, mishandled {args.min_count}+ times "
            f"(first {args.max_ply} plies):\n"
        )
        if not clusters:
            print("  none — no position has cost you more than once yet")
        for cluster in clusters:
            played = ", ".join(f"{san} x{n}" for san, n in cluster.played.items())
            print(
                f"  {review.move_number(cluster.ply):<7} played {played:<22} "
                f"instead of {cluster.best_san or '?':<6} "
                f"- cost {cluster.avg_loss:.0f} win% each time"
            )
            for san in cluster.played:
                context = review.book_context(db, cluster.fen, san, cluster.best_san)
                if context.moves:
                    _book_note(context, san, cluster.best_san)
            print(f"      {cluster.fen}")
            print(f"      games: {', '.join(cluster.games)}")

        moves = review.repeated_moves(
            db, max_ply=args.max_ply, min_count=args.min_count, include_ai=args.include_ai
        )
        multi = [m for m in moves if m.positions > 1]
        if multi:
            print(
                "\nsame move, wrong in several different positions "
                "— a move played by rote:\n"
            )
            for cluster in multi:
                alts = ", ".join(f"{san} x{n}" for san, n in cluster.alternatives.items())
                print(
                    f"  {review.move_number(cluster.ply)}{cluster.san:<6} x{cluster.count} "
                    f"across {cluster.positions} positions - "
                    f"cost {cluster.avg_loss:.0f} each; best was {alts}"
                )
                seen: set[str] = set()
                for row in cluster.rows:
                    if row["fen"] in seen:
                        continue
                    seen.add(row["fen"])
                    context = review.book_context(
                        db, row["fen"], cluster.san, row["best_san"]
                    )
                    if context.moves:
                        _book_note(context, cluster.san, row["best_san"])
    return 0


def cmd_book_build(args: argparse.Namespace) -> int:
    months = args.months_list or book.recent_months(args.months)
    print(f"building the book from {len(months)} months: {months[-1]} .. {months[0]}")
    paths = [dump.fetch(month) for month in months]
    started = time.time()

    def on_month(path, nodes: int) -> None:
        month = path.name.removesuffix(".pgn.zst")[-7:]
        print(f"    {month}  {nodes:>9,} nodes so far", end="\r")

    def on_stage(depth: int, counted: int, kept: int) -> None:
        print(
            f"  depth {depth:>2}: counted {counted:>10,} lines, "
            f"{kept:>7,} at {args.min_games}+ games   ({time.time() - started:.0f}s)"
        )

    nodes, corpus = book.build_tree(
        paths,
        depth=args.depth,
        min_games=args.min_games,
        min_elo=args.min_elo,
        on_stage=on_stage,
        on_month=on_month,
    )
    entries = list(book.to_entries(nodes))
    with store.open_db() as db:
        written = book.replace_book(
            db,
            entries,
            {
                "months": ",".join(months),
                "games": corpus,
                "min_games": args.min_games,
                "min_elo": args.min_elo or 0,
                "depth": args.depth,
                "built_at": int(time.time()),
            },
        )
    positions = len({entry.fen for entry in entries})
    print(
        f"book: {written:,} moves over {positions:,} positions, "
        f"from {corpus:,} games in {(time.time() - started) / 60:.1f} minutes"
    )
    return 0


def _print_book(entries: list[book.Entry], indent: str = "  ") -> None:
    if not entries:
        print(f"{indent}not in the book — under the minimum game count")
        return
    total = sum(entry.games for entry in entries)
    print(f"{indent}{'move':<8} {'games':>9} {'share':>7} {'score':>7} {'elo':>6}")
    for entry in entries:
        print(
            f"{indent}{entry.san:<8} {entry.games:>9,} "
            f"{100.0 * entry.games / total:>6.1f}% {entry.score:>6.1f}% "
            f"{entry.elo or 0:>6}"
        )


def cmd_book_line(args: argparse.Namespace) -> int:
    import chess.variant

    board = chess.variant.AtomicBoard()
    for san in args.moves:
        try:
            board.push_san(san)
        except ValueError:
            raise SystemExit(f"illegal or unparseable move: {san}") from None
    with store.open_db() as db:
        if not book.stats(db)["moves"]:
            print("no book yet — run `python -m trainer.cli book build`")
            return 0
        entries = book.lookup(db, book.book_fen(board), min_games=args.min_games)
        line = " ".join(args.moves) or "the starting position"
        print(f"after {line}  ({'white' if board.turn else 'black'} to move):\n")
        _print_book(entries)
    return 0


def cmd_book_stats(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        info = book.stats(db)
    if not info["moves"]:
        print("no book yet — run `python -m trainer.cli book build`")
        return 0
    months = str(info["months"] or "").split(",")
    print(f"book:     {info['moves']:,} moves over {info['positions']:,} positions")
    print(f"corpus:   {int(info['games'] or 0):,} games, {len(months)} months")
    print(f"          {months[-1]} .. {months[0]}")
    print(
        f"cutoffs:  {info['min_games']}+ games per line, depth {info['depth']} ply"
        + (f", Elo {info['min_elo']}+" if int(info["min_elo"] or 0) else "")
    )
    return 0


# -- openings --------------------------------------------------------------


def _print_tree(nodes: list[openings.TreeNode], indent: str = "") -> None:
    for node in nodes:
        name = node.name
        mark = {"user": "*", "community": "c", "standard": "s", "descriptive": "d"}.get(
            name.source, " "
        )
        moves = " ".join(node.line)
        print(
            f"{indent}{mark} {name.short:<34} {moves:<34} "
            f"{node.games:>9,} {node.score:>5.1f}%"
        )
        _print_tree(node.children, indent + "  ")


def cmd_openings(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        if not book.stats(db)["moves"]:
            print("no book yet — run `python -m trainer.cli book build`")
            return 0
        tree = openings.tree(
            db, depth=args.depth, min_games=args.min_games, min_share=args.min_share
        )
        print(
            "  name                               line                          "
            "     games score"
        )
        _print_tree(tree)
        print("\n* yours   c community   s standard chess   d coined here")
    return 0


def cmd_openings_name(args: argparse.Namespace) -> int:
    line = " ".join(args.moves)
    with store.open_db() as db:
        if args.clear:
            openings.rename(db, line, None)
            print(f"cleared the name for {line or 'the start'}")
        elif args.name:
            openings.rename(db, line, args.name, args.note or "")
            print(f"{line or 'the start'} is now {args.name!r}")
        named = openings.name_line(db, args.moves)
    print(f"\n{named.full}")
    if named.aliases:
        print(f"  also called: {', '.join(named.aliases)}")
    print(f"  named by: {named.source}")
    if named.evidence:
        print(f"  because:  {', '.join(named.evidence)}")
    if named.note:
        print(f"  note:     {named.note}")
    elif named.family_note:
        print(f"  {named.family}:\n    {named.family_note}")
    return 0


# -- traps -----------------------------------------------------------------


def cmd_traps_scan(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        if not book.stats(db)["moves"]:
            print("no book yet — run `python -m trainer.cli book build`")
            return 1
        model = _model(db)
        todo = traps.positions(
            db,
            depth=args.depth,
            min_games=args.min_games,
            min_share=args.min_share,
            limit=args.limit,
        )
        print(
            f"scanning {len(todo)} book positions at {args.scan_nodes:,} nodes, "
            f"confirming at {args.nodes:,}, MultiPV {args.multipv}"
        )
        started = time.time()
        state = {"stage": ""}

        def progress(stage: str, done: int, total: int) -> None:
            if stage != state["stage"]:
                state["stage"] = stage
                print()
            print(f"  {stage} {done}/{total}...", end="\r")

        found, report = traps.scan(
            db, todo,
            model=model,
            scan_nodes=args.scan_nodes,
            confirm_nodes=args.nodes,
            multipv=args.multipv,
            workers=args.workers,
            on_progress=progress,
        )
        written = traps.save(db, found)
        print(f"\n{report.line()}")
        print(f"stored {written} in {(time.time() - started) / 60:.1f} minutes")
    return 0


def cmd_traps_list(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        rows = traps.listing(
            db,
            turn=args.turn,
            max_ply=args.max_ply,
            min_bait=args.min_bait,
            order=args.order,
            limit=args.limit,
        )
        if not rows:
            print("no traps stored — run `python -m trainer.cli traps scan`")
            return 0
        for trap in rows:
            name = openings.name_line(db, trap.line)
            print(f"\n{' '.join(trap.line)}   ({trap.turn} to move)")
            print(f"  {name.full}")
            holds = ", ".join(f"{m.san} ({m.win:.0f}%)" for m in trap.escapes)
            print(f"  holds:   {holds}")
            print(
                f"  loses:   everything else — best of them {trap.losers[0].san} "
                f"at {trap.win_next:.0f}%, a {trap.severity:.0f}-point drop"
            )
            print(
                f"  bait:    {trap.bait:.0f}% of {trap.games:,} games here "
                f"({trap.bait_games:,}) played a losing move"
            )
            if trap.refutation:
                print(f"  line:    {' '.join(trap.refutation)}")
        print(f"\n{len(rows)} traps.  {traps.stats(db)['traps']} stored in all.")
    return 0


# -- repertoire ------------------------------------------------------------


def _colors(args: argparse.Namespace) -> list[str]:
    return [args.color] if getattr(args, "color", None) else list(repertoire.COLORS)


def cmd_rep_build(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        if not book.stats(db)["moves"]:
            print("no book yet — run `python -m trainer.cli book build` first")
            return 1
        for color in _colors(args):
            nodes = repertoire.derive(
                db,
                color,
                depth=args.depth,
                min_games=args.min_games,
                min_share=args.min_share,
                max_replies=args.max_replies,
                max_nodes=args.max_nodes,
            )
            if not nodes:
                print(f"{color}: nothing derived — try a lower --min-games")
                continue
            repertoire.save(db, nodes, replace=True)
            deepest = max(node.ply for node in nodes)
            thin = sum(1 for node in nodes if (node.games or 0) < args.min_games * 5)
            print(
                f"{color}: {len(nodes)} nodes to ply {deepest + 1}, "
                f"{thin} of them on a thin sample"
            )
    print("\nNext: python -m trainer.cli repertoire validate")
    return 0


def _rep_row(path: list[str], node: repertoire.Node) -> str:
    """One repertoire line. The engine's verdict is spelled out, not marked.

    No `!?`/`??` suffixes here: those already mean something in chess notation
    and none of them means "the book and the engine disagree", which is the
    only thing this column is reporting.
    """
    line = " ".join(path) or "(start)"
    sample = f"[{node.games:,} games, {node.score:.0f}%]" if node.games else ""
    if not node.validated:
        verdict = "unchecked"
    elif node.agrees:
        verdict = f"engine agrees ({node.win_move:.0f}%)"
    elif node.flagged:
        verdict = (
            f"LEAKS {node.val_loss:.0f}% — engine plays {node.best_san} "
            f"({node.win_best:.0f}% vs {node.win_move:.0f}%)"
        )
    else:
        verdict = f"engine prefers {node.best_san} by {node.val_loss:.0f}%"
    return f"  {line:<34} -> {node.san:<7} {sample:<24} {verdict}"


def cmd_rep_show(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        for color in _colors(args):
            rows = repertoire.lines_of(db, color)
            info = repertoire.stats(db, color)
            if not rows:
                print(f"{color}: no repertoire — `repertoire build {color}`")
                continue
            print(
                f"\n{color}: {info['nodes']} nodes, {info['validated']} validated, "
                f"{info['flagged']} flagged, engine agrees on {info['agreed']}"
            )
            for path, node in rows:
                if args.flagged and not node.flagged:
                    continue
                print(_rep_row(path, node))
    return 0


def cmd_rep_validate(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        model = _model(db)
        started = time.time()
        for color in _colors(args):
            todo = repertoire.nodes(db, color)
            if not todo:
                continue
            need = repertoire.validation_fens(todo)
            print(
                f"{color}: {len(todo)} nodes, {len(need)} positions to check "
                f"at {args.nodes:,} nodes"
            )

            def progress(done: int, total: int) -> None:
                if done % 10 == 0 or done == total:
                    print(f"  analysed {done}/{total}...", end="\r")

            flagged = repertoire.validate(
                db,
                color,
                nodes_budget=args.nodes,
                multipv=args.multipv,
                workers=args.workers,
                model=model,
                fix=args.fix,
                on_progress=progress,
            )
            info = repertoire.stats(db, color)
            print(
                f"  done in {(time.time() - started) / 60:.1f}m — engine agrees on "
                f"{info['agreed']}/{info['nodes']}, {info['flagged']} still flagged"
            )
            for node in flagged:
                print(
                    f"    ply {node.ply + 1:<3} {node.san:<7} leaks "
                    f"{node.val_loss:.0f} win% vs {node.best_san} "
                    f"({node.win_move:.0f}% vs {node.win_best:.0f}%)"
                )
                print(f"      {node.fen}")
            if flagged and not args.fix:
                print(
                    "\n  These are the book disagreeing with the engine, not "
                    "necessarily errors.\n  Review them, or take the engine's "
                    "word with `repertoire validate --fix`."
                )
    return 0


def cmd_rep_coverage(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        for color in _colors(args):
            gaps, covered, total = repertoire.coverage(
                db, color, max_ply=args.max_ply, include_ai=args.include_ai
            )
            if not total:
                print(f"{color}: no analysed games as {color} yet")
                continue
            pct = 100.0 * covered / total
            print(
                f"\n{color}: the repertoire covered {covered}/{total} "
                f"({pct:.0f}%) of your first {args.max_ply} plies"
            )
            for gap in gaps[: args.limit]:
                played = ", ".join(f"{san} x{n}" for san, n in gap.played.items())
                field = book.lookup(db, gap.fen, min_games=1)[:3]
                print(
                    f"  {review.move_number(gap.ply):<7} x{gap.count:<3} "
                    f"cost {gap.cost:>4.0f} win% total — you played {played}"
                )
                if field:
                    print(
                        "      field: "
                        + ", ".join(f"{e.san} {e.score:.0f}% on {e.games:,}" for e in field)
                    )
                print(f"      {gap.fen}")

            drifted = repertoire.deviations(
                db, color, max_ply=args.max_ply, include_ai=args.include_ai
            )
            if drifted:
                print(f"\n{color}: positions you have a line for and did not play it:")
                for dev in drifted[: args.limit]:
                    played = ", ".join(f"{san} x{n}" for san, n in dev.played.items())
                    print(
                        f"  {review.move_number(dev.node.ply):<7} played {played:<20} "
                        f"instead of {dev.node.san:<7} — cost {dev.cost:.0f} win% total"
                    )
    return 0


# -- puzzles ---------------------------------------------------------------


def cmd_puz_generate(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        model = _model(db)
        started = time.time()
        state = {"stage": ""}

        def progress(stage: str, done: int, total: int) -> None:
            if stage != state["stage"]:
                state["stage"] = stage
                print()
            print(f"  {stage} {done}/{total}...", end="\r")

        written, report = generate.run(
            db,
            model=model,
            account=args.account,
            game_id=args.game,
            include_ai=args.include_ai,
            saves=not args.no_saves,
            nodes=args.nodes,
            multipv=args.multipv,
            unique_gap=args.min_gap,
            screen_gap=args.screen_gap,
            max_solver_moves=args.max_moves,
            limit=args.limit,
            workers=args.workers,
            on_progress=progress,
        )
        print(f"\n{report.line()}")
        print(f"stored {written} in {(time.time() - started) / 60:.1f} minutes")
        if written:
            print("Next: python -m trainer.cli serve")
    return 0


def cmd_puz_endgames(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        print(
            f"probing the Lichess atomic tablebase ({args.source}); "
            f"cached probes are free, new ones are rate limited"
        )

        def progress(seen: int, report: endgames.Report) -> None:
            print(f"  {seen} positions, {report.kept} puzzles...", end="\r")

        started = time.time()
        written, report = endgames.run(
            db,
            source=args.source,
            samples=args.samples,
            seed=args.seed,
            walk_length=args.walk_length,
            include_ai=args.include_ai,
            limit=args.limit,
            max_solver_moves=args.max_moves,
            on_progress=progress,
        )
        print(f"\n{report.line()}")
        print(f"stored {written} in {(time.time() - started) / 60:.1f} minutes")
    return 0


def cmd_puz_mine(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        model = _model(db)
        available = mine.months_available()
        months = args.months or available
        if not months:
            print(
                "no dumps in data/dumps. Pass --month 2026-07 --fetch to download one.",
                file=sys.stderr,
            )
            return 1
        already = len(mine.already_scanned(db))
        print(
            f"scraping up to {args.games} rated atomic games "
            f"(both players {args.min_elo}+) from {', '.join(months[:3])}"
            f"{'...' if len(months) > 3 else ''}"
        )
        print(f"  {already} games already scanned; those are skipped")

        started = time.time()
        state = {"stage": ""}

        def progress(stage: str, done: int, total: int) -> None:
            if stage != state["stage"]:
                state["stage"] = stage
                print()
            print(f"  {stage} {done}/{total}...", end="\r")

        written, report = mine.run(
            db,
            months=months,
            games=args.games,
            model=model,
            min_elo=args.min_elo,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            scan_nodes=args.scan_nodes,
            confirm_nodes=args.nodes,
            unique_gap=args.min_gap,
            screen_gap=args.screen_gap,
            max_solver_moves=args.max_moves,
            saves=not args.no_saves,
            batch=args.batch,
            workers=args.workers,
            fetch_missing=args.fetch,
            on_progress=progress,
        )
        print(f"\n{report.line()}")
        print(f"stored {written} in {(time.time() - started) / 60:.1f} minutes")
    return 0


def _puzzle_filter(args: argparse.Namespace) -> puzzles.Filter:
    return puzzles.Filter(
        theme=args.theme,
        phase=args.phase,
        source=args.source,
        motive=args.motive,
        mine=args.mine,
        unsolved=args.unsolved,
        unseen=getattr(args, "unseen", False),
        failed=getattr(args, "failed", False),
        min_rating=getattr(args, "min_rating", None),
        max_rating=getattr(args, "max_rating", None),
        limit=args.limit,
        order=args.order,
    )


def cmd_puz_rate(args: argparse.Namespace) -> int:
    """Give every puzzle to every calibrated level; keep the crossover point."""
    with store.open_db() as db:
        try:
            rungs = difficulty.ladder()
        except difficulty.NotCalibrated as exc:
            print(exc, file=sys.stderr)
            return 1
        spec = puzzles.Filter(
            theme=args.theme,
            phase=args.phase,
            source=args.source,
            motive=args.motive,
            mine=args.mine,
            limit=args.limit or 1_000_000,
            order="recent",
        )
        todo = difficulty.pending(db, spec, force=args.force)
        stale = difficulty.stale(db, rungs)
        print(
            f"ladder: {difficulty.LADDER_METHOD} at "
            + ", ".join(f"{rung.nodes:,}n = {rung.elo:.0f}" for rung in rungs)
        )
        if stale and not args.force:
            print(
                f"  {stale} stored rating(s) were measured against a different "
                f"calibration; `--force` re-rates them"
            )
        if not todo:
            print("nothing to rate — every matching puzzle already has a rating")
            return 0
        print(f"rating {len(todo)} puzzles against {len(rungs)} levels\n")

        def progress(stage: str, done: int, total: int) -> None:
            print(f"  {stage} {done}/{total}...", end="\r")

        _rated, report = difficulty.run(
            db, spec=spec, force=args.force, on_progress=progress
        )
        print(f"\n{report.line()}")
        if report.floor:
            print(
                f"\n  {report.floor} puzzles sit at the ladder's floor — the "
                f"weakest calibrated level solves them, so their true difficulty\n"
                f"  is below {rungs[0].elo:.0f} and this ladder cannot say how far."
            )
        if report.non_monotone:
            print(
                f"  {report.non_monotone} were solved by a level that a stronger "
                f"one missed. Two rungs are within each other's error bars, so\n"
                f"  that is expected at a low rate; a high one means the ladder "
                f"needs recalibrating."
            )
    return 0


def cmd_puz_ratings(args: argparse.Namespace) -> int:
    """The corpus by difficulty, with your own solve rate beside it."""
    with store.open_db() as db:
        info = difficulty.summary(db)
        if not info["rated"]:
            print("no puzzles rated yet — `python -m trainer.cli puzzles rate`")
            return 0
        print(
            f"rated {info['rated']} of {info['rated'] + info['unrated']} puzzles "
            f"against the {info['method']} ladder\n"
        )
        print(f"  {'difficulty':<12} {'puzzles':>8} {'tried':>7} {'you':>7}")
        for band in info["bands"]:
            rate = f"{band['rate']:.0f}%" if band["rate"] is not None else "-"
            print(
                f"  {band['label']:<12} {band['count']:>8} {band['tried']:>7} "
                f"{rate:>7}"
            )
        if info["unrated"]:
            print(f"  {'unrated':<12} {info['unrated']:>8}")
        print(
            "\n  Difficulty is the rating of the weakest calibrated level that "
            "finds the\n  whole line. `you` is your unaided solve rate on the "
            "puzzles you have tried:\n  engine-hard and human-hard are different "
            "axes, and this is where they show."
        )
        if info["non_monotone"]:
            print(
                f"\n  {info['non_monotone']} puzzles were solved by a level a "
                "stronger one missed — two rungs\n  of the ladder are inside each "
                "other's error bars."
            )
        if info.get("stale"):
            print(
                f"\n  {info['stale']} ratings predate the current level "
                "calibration; re-run with `rate --force`."
            )
    return 0


def cmd_puz_list(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        spec = _puzzle_filter(args)
        found = puzzles.search(db, spec)
        total = puzzles.count(db, spec)
        if not found:
            print("no puzzles match — try `python -m trainer.cli puzzles generate`")
            return 0
        print(f"{len(found)} of {total} matching\n")
        print(
            f"{'id':<18} {'src':<10} {'phase':<11} {'mv':>3} {'gap':>5} "
            f"{'elo':>6}  themes"
        )
        for puzzle in found:
            gap = f"{puzzle.unique_gap:.0f}" if puzzle.unique_gap else "-"
            print(
                f"{puzzle.id:<18} {puzzle.source:<10} {puzzle.phase:<11} "
                f"{puzzle.length:>3} {gap:>5} {puzzle.rating_text or '-':>6}  "
                f"{themes.describe(puzzle.themes)}"
            )
    return 0


def cmd_puz_show(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        puzzle = puzzles.get(db, args.id)
        if puzzle is None:
            raise SystemExit(f"no puzzle {args.id!r}")
        print(f"{puzzle.id}  {puzzle.source}, {puzzle.phase}, {puzzle.motive}")
        print(f"  {puzzle.solver} to move: {puzzle.fen}")
        if puzzle.setup_san:
            print(f"  they played {puzzle.setup_san} in game {puzzle.game_id}")
        print(f"  solution: {puzzles.line_san(puzzle)}")
        print(f"  themes:   {themes.describe(puzzle.themes)}")
        if puzzle.tb_category:
            print(f"  tablebase: {puzzle.tb_category} — exact, no engine involved")
        if puzzle.win_before is not None:
            second = (
                f", second best {puzzle.win_second:.0f}%"
                if puzzle.win_second is not None
                else ""
            )
            print(
                f"  win%:     {puzzle.win_before:.0f} after the right move{second} "
                f"(at {puzzle.nodes:,} nodes)"
            )
        verdict = difficulty.load(db, puzzle.id)
        if verdict is not None:
            print(f"  rating:   {verdict.describe()}")
            for attempt in verdict.attempts:
                mark = "found" if attempt.solved else f"played {attempt.played}"
                print(
                    f"    {attempt.nodes:>9,}n ({attempt.elo:.0f} Elo): {mark}"
                    + ("" if attempt.solved
                       else f" at move {attempt.found + 1} of {attempt.total}")
                )
        attempts = puzzles.history(db, puzzle.id)
        if attempts:
            solved = sum(1 for row in attempts if row["solved"])
            print(f"  attempts: {len(attempts)}, solved unaided {solved}")
    return 0


def cmd_puz_themes(args: argparse.Namespace) -> int:
    """Does each detector actually distinguish anything? Report, do not assume."""
    with store.open_db() as db:
        rates = themes.firing_rates(db)
        total = rates[0].total if rates else 0
        if not total:
            print("no puzzles yet — nothing to validate the detectors against")
            return 0
        print(f"theme firing rates over {total} puzzles:\n")
        print(f"  {'theme':<22} {'n':>5} {'rate':>7}  verdict")
        for firing in rates:
            print(
                f"  {firing.theme:<22} {firing.count:>5} {firing.rate:>6.1f}%  "
                f"{firing.verdict}"
            )
        untagged = db.execute(
            "SELECT COUNT(*) AS n FROM puzzles p WHERE NOT EXISTS "
            "(SELECT 1 FROM puzzle_themes t WHERE t.puzzle_id = p.id)"
        ).fetchone()["n"]
        print(f"\n  {untagged} puzzles ({100.0 * untagged / total:.0f}%) match no theme")
        print("\n  A theme that never fires describes nothing; one over 60% separates")
        print("  nothing. Both are detector bugs, not corpus facts.")
    return 0


def cmd_puz_stats(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        info = puzzles.stats(db)
        if not info["total"]:
            print("no puzzles yet — `python -m trainer.cli puzzles generate`")
            return 0
        print(
            f"corpus:   {info['total']} puzzles — {info['from_games']} from your "
            f"games, {info['from_dump']} scraped, "
            f"{info['from_tablebase']} from the tablebase"
        )
        print(
            f"          {info['mine']} from your own side of the board, "
            f"{info['saves']} defensive"
        )
        scraped = mine.stats(db)
        if scraped["games"]:
            print(
                f"scraped:  {scraped['games']} rated games scanned at "
                f"{scraped['scan_nodes']:,} nodes, "
                f"{scraped['months'][0]}..{scraped['months'][1]}"
            )
        phases = ", ".join(f"{n} {phase}" for phase, n in info["phases"].items())
        print(f"phases:   {phases}")
        if info["rated"]:
            print(
                f"rated:    {info['rated']} measured on the engine ladder "
                f"({info['rating_min']:.0f}..{info['rating_max']:.0f} Elo), "
                f"{info['unrated']} unrated"
            )
        else:
            print("rated:    none yet — `python -m trainer.cli puzzles rate`")
        print(
            f"solving:  {info['puzzles_solved']}/{info['puzzles_attempted']} solved "
            f"unaided over {info['attempts']} attempts"
        )
        top = list(info["themes"].items())[:6]
        print("themes:   " + ", ".join(f"{theme} {n}" for theme, n in top))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    return server.serve(host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_status(args: argparse.Namespace) -> int:
    with store.open_db() as db:
        games = db.execute(
            "SELECT COUNT(*) AS n, SUM(analysed_nodes IS NOT NULL) AS done, "
            "SUM(ply) AS plies FROM games"
        ).fetchone()
        judged = db.execute(
            "SELECT judgement, COUNT(*) AS n FROM plies p JOIN games g ON g.id = p.game_id "
            "WHERE p.mover = g.my_color GROUP BY judgement"
        ).fetchall()
        cache = store.cache_stats(db)
        print(f"store:    {store.DB_PATH}")
        print(
            f"games:    {games['n'] or 0} imported, {games['done'] or 0} analysed, "
            f"{games['plies'] or 0} plies"
        )
        print(
            f"cache:    {cache['positions']} positions "
            f"({cache['min_nodes'] or 0:,}-{cache['max_nodes'] or 0:,} nodes)"
        )
        info = book.stats(db)
        print(
            f"book:     {info['moves']:,} moves over {info['positions']:,} positions"
            if info["moves"]
            else "book:     not built — `python -m trainer.cli book build`"
        )
        rep = [
            f"{color} {info['nodes']} nodes"
            + (
                f" ({info['flagged']} flagged)"
                if info["validated"]
                else " (unvalidated)"
            )
            for color in repertoire.COLORS
            if (info := repertoire.stats(db, color))["nodes"]
        ]
        print(
            f"repertoire: {'; '.join(rep)}"
            if rep
            else "repertoire: none — `python -m trainer.cli repertoire build`"
        )
        puz = puzzles.stats(db)
        print(
            f"puzzles:  {puz['total']} — {puz['from_games']} mined, "
            f"{puz['from_tablebase']} exact; {puz['puzzles_solved']} solved unaided"
            if puz["total"]
            else "puzzles:  none — `python -m trainer.cli puzzles generate`"
        )
        print(f"{winrate.load().describe()}")
        if judged:
            counts = {row["judgement"]: row["n"] for row in judged}
            total = sum(counts.values())
            print(
                f"your moves: {total} judged — "
                + ", ".join(
                    f"{counts.get(k, 0)} {k}"
                    for k in ("blunder", "mistake", "inaccuracy", "ok")
                )
            )
    return 0


# -- wiring ----------------------------------------------------------------


def _speak_utf8() -> None:
    """Stop the Windows console mangling punctuation into `?`.

    Python encodes stdout with the legacy code page (437 here), so an em dash
    in any of the messages below comes out as a replacement character. Both
    streams are switched to UTF-8; `errors="replace"` keeps a terminal that
    genuinely cannot render it from taking the command down with it.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass  # not a real stream (captured in tests); nothing to fix


def main(argv: list[str] | None = None) -> int:
    _speak_utf8()
    parser = argparse.ArgumentParser(prog="trainer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="fetch a Lichess account's atomic games")
    p_import.add_argument("user")
    p_import.add_argument("--full", action="store_true", help="re-fetch everything")
    p_import.add_argument("--max", type=int, help="stop after this many games")
    p_import.set_defaults(func=cmd_import)

    p_an = sub.add_parser("analyse", help="deep Fairy-Stockfish pass over every ply")
    p_an.add_argument("--nodes", type=int, default=analysis.DEFAULT_NODES)
    p_an.add_argument("--multipv", type=int, default=analysis.DEFAULT_MULTIPV)
    p_an.add_argument("--workers", type=int, default=None)
    p_an.add_argument("--limit", type=int, help="only this many games")
    p_an.add_argument("--account", help="only this account's games")
    p_an.add_argument(
        "--rejudge",
        action="store_true",
        help="re-score already-analysed games from the cache (no engine)",
    )
    p_an.set_defaults(func=cmd_analyse)

    p_re = sub.add_parser("rejudge", help="alias for `analyse --rejudge`")
    p_re.add_argument("--nodes", type=int, default=analysis.DEFAULT_NODES)
    p_re.add_argument("--multipv", type=int, default=analysis.DEFAULT_MULTIPV)
    p_re.add_argument("--workers", type=int, default=None)
    p_re.add_argument("--limit", type=int)
    p_re.add_argument("--account")
    p_re.set_defaults(func=cmd_analyse, rejudge=True)

    p_games = sub.add_parser("games", help="list imported games")
    p_games.add_argument("--account")
    p_games.add_argument("--limit", type=int, default=30)
    p_games.set_defaults(func=cmd_games)

    p_review = sub.add_parser("review", help="where a game was thrown")
    p_review.add_argument("game", help="lichess game id")
    p_review.add_argument("--both", action="store_true", help="include the opponent")
    p_review.set_defaults(func=cmd_review)

    p_ex = sub.add_parser(
        "explain", help="why the engine plays its move, or why yours is worse"
    )
    p_ex.add_argument("fen", nargs="?", help="position (default: the start)")
    p_ex.add_argument("--moves", nargs="*", help="SAN moves to play from the position first")
    p_ex.add_argument("--move", help="the move to ask about (SAN or UCI); default: the engine's")
    p_ex.add_argument("--versus", help="compare against this move instead of the second best")
    p_ex.add_argument("--nodes", type=int, default=300_000, help="per search")
    p_ex.set_defaults(func=cmd_explain)

    p_worst = sub.add_parser("worst", help="your biggest mistakes across all games")
    p_worst.add_argument("--limit", type=int, default=20)
    p_worst.add_argument("--include-ai", action="store_true")
    p_worst.set_defaults(func=cmd_worst)

    p_pat = sub.add_parser("patterns", help="mistakes you make more than once")
    p_pat.add_argument("--max-ply", type=int, default=review.OPENING_PLY)
    p_pat.add_argument("--min-count", type=int, default=2)
    p_pat.add_argument("--include-ai", action="store_true")
    p_pat.set_defaults(func=cmd_patterns)

    p_book = sub.add_parser("book", help="the opening book counted from the dumps")
    book_sub = p_book.add_subparsers(dest="book_command", required=True)

    p_build = book_sub.add_parser("build", help="count a fresh book from the dumps")
    p_build.add_argument("--months", type=int, default=12, help="how many recent months")
    p_build.add_argument(
        "--month", dest="months_list", action="append",
        help="a specific YYYY-MM (repeatable); overrides --months",
    )
    p_build.add_argument("--min-games", type=int, default=book.DEFAULT_MIN_GAMES)
    p_build.add_argument("--min-elo", type=int, default=None)
    p_build.add_argument("--depth", type=int, default=book.DEFAULT_DEPTH)
    p_build.set_defaults(func=cmd_book_build)

    p_line = book_sub.add_parser("line", help="what the population plays in a position")
    p_line.add_argument("moves", nargs="*", help="SAN moves from the start")
    p_line.add_argument("--min-games", type=int, default=0)
    p_line.set_defaults(func=cmd_book_line)

    book_sub.add_parser("stats", help="what the book was built from").set_defaults(
        func=cmd_book_stats
    )

    p_open = sub.add_parser("openings", help="the named opening index")
    open_sub = p_open.add_subparsers(dest="openings_command")
    p_open.add_argument("--depth", type=int, default=6)
    p_open.add_argument("--min-games", type=int, default=5_000)
    p_open.add_argument("--min-share", type=float, default=8.0)
    p_open.set_defaults(func=cmd_openings)

    p_oname = open_sub.add_parser("name", help="name a line yourself, or read its name")
    p_oname.add_argument("moves", nargs="*", help="SAN moves from the start")
    p_oname.add_argument("--as", dest="name", help="the name to give this line")
    p_oname.add_argument("--note", help="why, or what to remember about it")
    p_oname.add_argument("--clear", action="store_true", help="drop your name for it")
    p_oname.set_defaults(func=cmd_openings_name)

    p_trap = sub.add_parser("traps", help="book positions where one move holds")
    trap_sub = p_trap.add_subparsers(dest="traps_command", required=True)

    p_tscan = trap_sub.add_parser("scan", help="walk the book and judge every position")
    p_tscan.add_argument("--depth", type=int, default=traps.DEFAULT_DEPTH)
    p_tscan.add_argument("--min-games", type=int, default=traps.DEFAULT_MIN_GAMES)
    p_tscan.add_argument("--min-share", type=float, default=traps.DEFAULT_MIN_SHARE)
    p_tscan.add_argument("--limit", type=int, default=None, help="positions to scan")
    p_tscan.add_argument("--scan-nodes", type=int, default=traps.SCAN_NODES)
    p_tscan.add_argument("--nodes", type=int, default=traps.CONFIRM_NODES)
    p_tscan.add_argument("--multipv", type=int, default=traps.MULTIPV)
    p_tscan.add_argument("--workers", type=int, default=None)
    p_tscan.set_defaults(func=cmd_traps_scan)

    p_tlist = trap_sub.add_parser("list", help="the traps, worst first")
    p_tlist.add_argument("--turn", choices=("white", "black"), default=None)
    p_tlist.add_argument("--max-ply", type=int, default=None)
    p_tlist.add_argument("--min-bait", type=float, default=0.0)
    p_tlist.add_argument("--order", choices=tuple(traps.ORDERS), default="bait")
    p_tlist.add_argument("--limit", type=int, default=20)
    p_tlist.set_defaults(func=cmd_traps_list)

    p_rep = sub.add_parser("repertoire", help="what you intend to play, checked")
    rep_sub = p_rep.add_subparsers(dest="repertoire_command", required=True)

    p_rbuild = rep_sub.add_parser("build", help="derive a repertoire from the book")
    p_rbuild.add_argument("color", nargs="?", choices=repertoire.COLORS)
    p_rbuild.add_argument("--depth", type=int, default=repertoire.DEFAULT_DEPTH)
    p_rbuild.add_argument("--min-games", type=int, default=repertoire.DEFAULT_MIN_GAMES)
    p_rbuild.add_argument("--min-share", type=float, default=repertoire.DEFAULT_MIN_SHARE)
    p_rbuild.add_argument("--max-replies", type=int, default=repertoire.DEFAULT_MAX_REPLIES)
    p_rbuild.add_argument("--max-nodes", type=int, default=repertoire.DEFAULT_MAX_NODES)
    p_rbuild.set_defaults(func=cmd_rep_build)

    p_rshow = rep_sub.add_parser("show", help="print the repertoire as lines")
    p_rshow.add_argument("color", nargs="?", choices=repertoire.COLORS)
    p_rshow.add_argument("--flagged", action="store_true", help="only what the engine dislikes")
    p_rshow.set_defaults(func=cmd_rep_show)

    p_rval = rep_sub.add_parser("validate", help="check every line with Fairy-Stockfish")
    p_rval.add_argument("color", nargs="?", choices=repertoire.COLORS)
    p_rval.add_argument("--nodes", type=int, default=analysis.DEFAULT_NODES)
    p_rval.add_argument("--multipv", type=int, default=analysis.DEFAULT_MULTIPV)
    p_rval.add_argument("--workers", type=int, default=None)
    p_rval.add_argument(
        "--fix", action="store_true",
        help="replace flagged moves with the engine's choice",
    )
    p_rval.set_defaults(func=cmd_rep_validate)

    p_rcov = rep_sub.add_parser("coverage", help="the lines you keep needing and lack")
    p_rcov.add_argument("color", nargs="?", choices=repertoire.COLORS)
    p_rcov.add_argument("--max-ply", type=int, default=repertoire.DEFAULT_DEPTH)
    p_rcov.add_argument("--limit", type=int, default=15)
    p_rcov.add_argument("--include-ai", action="store_true")
    p_rcov.set_defaults(func=cmd_rep_coverage)

    p_puz = sub.add_parser("puzzles", help="the generated puzzle corpus")
    puz_sub = p_puz.add_subparsers(dest="puzzles_command", required=True)

    p_pgen = puz_sub.add_parser("generate", help="mine puzzles from your games")
    p_pgen.add_argument("--nodes", type=int, default=generate.CONFIRM_NODES,
                        help="confirming budget per position")
    p_pgen.add_argument("--multipv", type=int, default=generate.CONFIRM_MULTIPV)
    p_pgen.add_argument("--min-gap", type=float, default=generate.UNIQUE_GAP,
                        help="win%% between best and second best to call it unique")
    p_pgen.add_argument("--screen-gap", type=float, default=generate.SCREEN_GAP,
                        help="the looser gap used to pick candidates from the cache")
    p_pgen.add_argument("--max-moves", type=int, default=generate.MAX_SOLVER_MOVES)
    p_pgen.add_argument("--limit", type=int, help="only the N widest-gap candidates")
    p_pgen.add_argument("--workers", type=int, default=None)
    p_pgen.add_argument("--account")
    p_pgen.add_argument("--game", help="only this game id")
    p_pgen.add_argument("--include-ai", action="store_true")
    p_pgen.add_argument("--no-saves", action="store_true",
                        help="only punish-the-blunder puzzles")
    p_pgen.set_defaults(func=cmd_puz_generate)

    p_pend = puz_sub.add_parser("endgames", help="exact puzzles from the tablebase")
    p_pend.add_argument(
        "--source", choices=("walk", "cache", "games", "random", "all"),
        default="walk",
        help="walk: scatter a position and play it out (finds new ones); "
             "cache: re-examine every probe already paid for (free)",
    )
    p_pend.add_argument("--samples", type=int, default=200,
                        help="how many positions to start from")
    p_pend.add_argument("--walk-length", type=int, default=endgames.WALK_LENGTH,
                        help="plies to follow each start")
    p_pend.add_argument("--seed", type=int, default=None)
    p_pend.add_argument("--limit", type=int, help="stop after this many puzzles")
    p_pend.add_argument("--max-moves", type=int, default=endgames.MAX_SOLVER_MOVES)
    p_pend.add_argument("--include-ai", action="store_true")
    p_pend.set_defaults(func=cmd_puz_endgames)

    p_pmine = puz_sub.add_parser(
        "mine", help="scrape puzzles from rated atomic games on Lichess"
    )
    p_pmine.add_argument("--games", type=int, default=200,
                         help="how many unseen rated games to scan")
    p_pmine.add_argument("--month", action="append", dest="months",
                         help="dump month, e.g. 2026-07; repeatable, newest first")
    p_pmine.add_argument("--min-elo", type=int, default=mine.MIN_ELO,
                         help="both players must be at least this strong")
    p_pmine.add_argument("--min-ply", type=int, default=mine.MIN_PLY)
    p_pmine.add_argument("--max-ply", type=int, default=mine.MAX_PLY)
    p_pmine.add_argument("--scan-nodes", type=int, default=mine.SCAN_NODES,
                         help="screening budget per ply (a calibrated rung)")
    p_pmine.add_argument("--nodes", type=int, default=generate.CONFIRM_NODES,
                         help="confirming budget per surviving position")
    p_pmine.add_argument("--min-gap", type=float, default=generate.UNIQUE_GAP)
    p_pmine.add_argument("--screen-gap", type=float, default=generate.SCREEN_GAP)
    p_pmine.add_argument("--max-moves", type=int, default=generate.MAX_SOLVER_MOVES)
    p_pmine.add_argument("--batch", type=int, default=40,
                         help="candidates to accumulate before a confirming pass")
    p_pmine.add_argument("--workers", type=int, default=None)
    p_pmine.add_argument("--fetch", action="store_true",
                         help="download a month that is not cached yet")
    p_pmine.add_argument("--no-saves", action="store_true",
                         help="only punish-the-blunder puzzles")
    p_pmine.set_defaults(func=cmd_puz_mine)

    p_plist = puz_sub.add_parser("list", help="browse the corpus")
    p_plist.add_argument("--theme", choices=tuple(themes.CATALOG))
    p_plist.add_argument("--phase", choices=puzzles.PHASES)
    p_plist.add_argument("--source", choices=puzzles.SOURCES)
    p_plist.add_argument("--motive", choices=puzzles.MOTIVES)
    p_plist.add_argument("--mine", action="store_true", help="from your own games")
    p_plist.add_argument("--unsolved", action="store_true",
                         help="never solved unaided, seen or not")
    p_plist.add_argument("--unseen", action="store_true",
                         help="never attempted at all")
    p_plist.add_argument("--failed", action="store_true",
                         help="tried and never solved — the redo queue")
    p_plist.add_argument("--min-rating", type=float, dest="min_rating",
                         help="only puzzles measured at this Elo or above")
    p_plist.add_argument("--max-rating", type=float, dest="max_rating")
    p_plist.add_argument("--limit", type=int, default=30)
    p_plist.add_argument("--order", default="recent",
                         choices=tuple(puzzles._ORDERS))
    p_plist.set_defaults(func=cmd_puz_list)

    p_prate = puz_sub.add_parser(
        "rate", help="measure difficulty against the calibrated engine ladder"
    )
    p_prate.add_argument("--theme", choices=tuple(themes.CATALOG))
    p_prate.add_argument("--phase", choices=puzzles.PHASES)
    p_prate.add_argument("--source", choices=puzzles.SOURCES)
    p_prate.add_argument("--motive", choices=puzzles.MOTIVES)
    p_prate.add_argument("--mine", action="store_true")
    p_prate.add_argument("--limit", type=int, default=0,
                         help="stop after this many (0 = every unrated puzzle)")
    p_prate.add_argument("--force", action="store_true",
                         help="re-rate puzzles that already have a rating")
    p_prate.set_defaults(func=cmd_puz_rate)

    puz_sub.add_parser(
        "ratings", help="the corpus by measured difficulty, and how you do on it"
    ).set_defaults(func=cmd_puz_ratings)

    p_pshow = puz_sub.add_parser("show", help="one puzzle, with its solution")
    p_pshow.add_argument("id")
    p_pshow.set_defaults(func=cmd_puz_show)

    puz_sub.add_parser(
        "themes", help="firing rate of every detector — are they any good?"
    ).set_defaults(func=cmd_puz_themes)
    puz_sub.add_parser("stats", help="what the corpus contains").set_defaults(
        func=cmd_puz_stats
    )

    p_serve = sub.add_parser("serve", help="the browser UI: puzzles, review, book")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8771)
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    sub.add_parser("status", help="store, cache, and win% model").set_defaults(
        func=cmd_status
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
