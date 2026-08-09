"""Command line for the atomic lab.

    python -m lab.cli setup                       # fetch Fairy-Stockfish + net
    python -m lab.cli calibrate --games 40        # measure the FSF ladder
    python -m lab.cli rate M0-greedy --games 20   # score a method on the ladder
    python -m lab.cli match M0-greedy M0-random   # head-to-head between methods
    python -m lab.cli ablate --games 40           # what each eval term is worth
    python -m lab.cli ratings                     # every Elo measured so far
    python -m lab.cli play                        # play the engine in a browser
    python -m lab.cli bench                       # FSF throughput on this box
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import chess.variant

from . import arena, config, fsf, methods, rating


def cmd_setup(args: argparse.Namespace) -> int:  # noqa: ARG001
    from . import setup_deps

    return setup_deps.main()


def cmd_bench(args: argparse.Namespace) -> int:
    """How fast is the gold standard on this machine? Sets expectations."""
    board = chess.variant.AtomicBoard()
    for nodes in (16_000, 256_000):
        cfg = fsf.FsfConfig(nodes=nodes)
        with fsf.open_engine(cfg) as engine:
            start = time.perf_counter()
            info = engine.analyse(board, cfg.limit)
            elapsed = time.perf_counter() - start
        print(
            f"{cfg.name:<20} depth {info.get('depth'):>2}  "
            f"{info.get('nodes', 0)/max(elapsed, 1e-9)/1e6:.2f} Mnps  "
            f"eval {info['score'].white()}"
        )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Play the FSF rungs against each other and fit their Elo."""
    rungs = list(fsf.LADDER)
    scores: dict[tuple[int, int], list[float]] = {}
    for lower, upper in itertools.pairwise(rungs):
        cfg_lo, cfg_hi = fsf.FsfConfig(nodes=lower), fsf.FsfConfig(nodes=upper)
        with fsf.open_engine(cfg_lo) as e_lo, fsf.open_engine(cfg_hi) as e_hi:
            print(f"\n{cfg_lo.name} vs {cfg_hi.name}")
            match = arena.play_match(
                arena.FsfPlayer(e_lo, cfg_lo),
                arena.FsfPlayer(e_hi, cfg_hi),
                args.games,
                seed=args.seed,
                progress=not args.quiet,
            )
        print(f"  => {match}")
        scores[(lower, upper)] = [match.score_a, len(match.games)]
        match.to_jsonl(config.RESULTS / "calibration.jsonl")

    # Each adjacent pair gives an Elo gap; chain them outward from the
    # reference rung. `scores[(lo, hi)]` holds the *lower* rung's score, so a
    # score below 50% means a positive gap in favour of the upper rung.
    gaps = {
        (lo, hi): -400.0 * _logit10(min(max(s / g, 1e-3), 1 - 1e-3))
        for (lo, hi), (s, g) in scores.items()
    }
    ratings = {rating.REFERENCE_RUNG: rating.REFERENCE_ELO}
    # Walk up from the reference rung, then down.
    for lo, hi in itertools.pairwise(rungs):
        if lo in ratings:
            ratings[hi] = ratings[lo] + gaps[(lo, hi)]
    for lo, hi in reversed(list(itertools.pairwise(rungs))):
        if hi in ratings and lo not in ratings:
            ratings[lo] = ratings[hi] - gaps[(lo, hi)]

    rating.save_ladder(
        ratings,
        note=f"{args.games} games/pair, calibrated {time.strftime('%Y-%m-%d')}",
    )
    print(f"\nladder written to {rating.LADDER_FILE}")
    for rung, elo in sorted(ratings.items()):
        print(f"  {rung:>9,} nodes  {elo:8.0f} Elo")
    return 0


def _logit10(p: float) -> float:
    """log10(p / (1-p)); the Elo gap is -400 * this."""
    import math

    return math.log10(p / (1.0 - p))


def cmd_rate(args: argparse.Namespace) -> int:
    method = methods.build(args.method)
    budget = _budget_from(args)
    if not rating.is_calibrated():
        print(
            "warning: ladder not calibrated, using the theoretical "
            "70-Elo-per-doubling prior. Run `calibrate` for a measured scale.\n"
        )
    matches = arena.gauntlet(
        method,
        budget,
        games_per_rung=args.games,
        seed=args.seed,
        log=config.RESULTS / f"{args.method}.jsonl",
    )
    result = rating.rate_from_matches(matches)
    print(f"\n=== {method.name} @ {budget} ===")
    for match in matches:
        print(f"  {match}")
    print(f"\n  {result}")

    row = {
        "method": method.name,
        "budget": str(budget),
        "elo": round(result.elo, 1),
        "low": round(result.low, 1),
        "high": round(result.high, 1),
        "games": result.games,
        "score": result.score,
        "bounded": result.bounded,
        "calibrated": rating.is_calibrated(),
        "date": time.strftime("%Y-%m-%d"),
    }
    with (config.RESULTS / "ratings.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    budget_a = _budget_from(args)
    # `--nodes-b` gives the second player a different budget. That is normally
    # unfair by design and so is opt-in, but it is the only way to ask how well
    # a method converts extra compute into strength — a question the ladder
    # cannot answer for a method far weaker than every rung on it.
    budget_b = methods.Budget(nodes=args.nodes_b) if args.nodes_b else budget_a
    a = arena.MethodPlayer(methods.build(args.method_a), budget_a)
    b = arena.MethodPlayer(methods.build(args.method_b), budget_b)
    match = arena.play_match(a, b, args.games, seed=args.seed, progress=not args.quiet)
    print(f"\n{match}")
    result = rating.rate([rating.Opponent(b.name, 0.0, match.score_a, len(match.games))])
    print(f"  {a.name} is {result} relative to {b.name}")
    match.to_jsonl(config.RESULTS / "matches.jsonl")
    return 0


def cmd_levels(args: argparse.Namespace) -> int:
    """Calibrate a method's node-budget ladder — the UI's difficulty levels.

    Why this exists rather than just rating each level on the FSF ladder: a
    gauntlet rating depends on how far up the ladder the method got before
    being shut out, so a *weaker* setting that survives to play more rungs can
    out-rate a stronger one. Measured directly, M1 at 2k nodes rated 1366 +/- 88
    while M1 at 64k rated 1254 +/- 167 — an ordering the head-to-head results
    flatly contradict.

    So this measures each level against one anchor level head-to-head and
    chains the gaps, which is the same construction `calibrate` uses for the
    Fairy-Stockfish rungs. The anchor's absolute rating still comes from the
    ladder; only the *spacing* comes from here.
    """
    from .server import LEVELS

    rungs = [nodes for _label, nodes in LEVELS]
    anchor = args.anchor if args.anchor in rungs else rungs[0]

    measured = rating.latest_ratings().get((args.method, f"{anchor}n"))
    if measured is None:
        print(
            f"no ladder rating for {args.method} @ {anchor}n — run\n"
            f"  lab.cli rate {args.method} --nodes {anchor} --games 40\n"
            "first, so the chain has something to anchor to."
        )
        return 1
    anchor_elo = measured["elo"]

    levels = {anchor: {"elo": anchor_elo, "span": (measured["high"] - measured["low"]) / 2,
                       "anchor": True, "games": measured["games"]}}
    for nodes in rungs:
        if nodes == anchor:
            continue
        a = arena.MethodPlayer(methods.build(args.method), methods.Budget(nodes=nodes))
        b = arena.MethodPlayer(methods.build(args.method), methods.Budget(nodes=anchor))
        print(f"\n{a.name} vs {b.name}")
        match = arena.play_match(a, b, args.games, seed=args.seed, progress=not args.quiet)
        print(f"  => {match}")
        match.to_jsonl(config.RESULTS / "levels.jsonl")
        delta = rating.rate([rating.Opponent(b.name, 0.0, match.score_a, len(match.games))])
        levels[nodes] = {
            "elo": anchor_elo + delta.elo,
            "span": (delta.high - delta.low) / 2,
            "anchor": False,
            "games": delta.games,
        }

    rating.save_levels(
        args.method, anchor,
        dict(sorted(levels.items())),
        f"{args.games} games/level vs {anchor}n, {time.strftime('%Y-%m-%d')}",
    )
    print(f"\nwritten to {rating.LEVELS_FILE}")
    for nodes, level in sorted(levels.items()):
        tag = "  <- anchor (absolute, from the ladder)" if level["anchor"] else ""
        print(f"  {nodes:>9,} nodes  {level['elo']:6.0f} +/- {level['span']:3.0f} Elo{tag}")
    return 0


def cmd_ratings(args: argparse.Namespace) -> int:
    """Everything measured so far, on one scale."""
    ladder = rating.load_ladder()
    print(f"=== Fairy-Stockfish ladder ({'measured' if rating.is_calibrated() else 'PRIOR'}) ===")
    for rung, elo in sorted(ladder.items()):
        anchor = "  <- anchor" if rung == rating.REFERENCE_RUNG else ""
        print(f"  {f'fsf-nnue-{rung // 1000}kn':<20} {elo:8.0f} Elo{anchor}")

    rows = rating.latest_ratings()
    if not rows:
        print("\nno methods rated yet; run `lab.cli rate <method> --nodes N`")
        return 0
    print("\n=== methods ===")
    for (method, budget), row in sorted(rows.items(), key=lambda kv: -kv[1]["elo"]):
        note = "" if row.get("calibrated", True) else "   (uncalibrated ladder)"
        print(
            f"  {method:<20} @{budget:<10} {rating.describe(row):<20}"
            f" {row['score']:>5}/{row['games']:<4}{note}"
        )
    levels = rating.load_levels()
    if levels:
        print(f"\n=== {levels['method']} difficulty levels ({levels['note']}) ===")
        for nodes, level in sorted(levels["levels"].items()):
            tag = "  <- anchor" if level["anchor"] else ""
            print(f"  {nodes:>9,} nodes  {level['elo']:6.0f} +/- {level['span']:3.0f} Elo{tag}")

    if not args.quiet:
        print(
            "\nThis scale is internal: the 64k-node rung is pinned at 2000 by"
            "\ndefinition. Only differences mean anything - do not compare these"
            "\nnumbers to Lichess or CCRL ratings."
        )
        budgets = {method for method, _ in rows}
        if any(sum(1 for m, _ in rows if m == method) > 1 for method in budgets):
            print(
                "\nNote: two gauntlet ratings of the *same* method at different"
                "\nbudgets are not comparable. `gauntlet` stops climbing once a"
                "\nmethod is shut out, so each budget faces a different rung set."
                "\nUse the difficulty-level table above, or `match --nodes-b`."
            )
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Serve the browser UI for playing against the engine."""
    from . import server

    return server.serve(host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_ablate(args: argparse.Namespace) -> int:
    """Measure what each evaluation term is worth, in Elo.

    Each ablation plays the *full* method head-to-head rather than climbing the
    ladder. That is a deliberate statistical choice: two near-equal opponents
    produce a mix of results, which is where the information is, whereas a
    gauntlet against a much stronger ladder produces a wall of losses that
    barely constrains anything. The ladder answers "how strong is this"; the
    ablation answers "which idea is doing the work", and only the second
    question needs an answer for the project to make progress.
    """
    budget = _budget_from(args)
    reference = args.reference
    variants = args.variants or [
        name
        for name in sorted(methods.REGISTRY)
        if name.startswith("M1-") and name != reference
    ]

    rows = []
    for name in variants:
        a = arena.MethodPlayer(methods.build(name), budget)
        b = arena.MethodPlayer(methods.build(reference), budget)
        print(f"\n{name} vs {reference}")
        match = arena.play_match(a, b, args.games, seed=args.seed, progress=not args.quiet)
        print(f"  => {match}")
        match.to_jsonl(config.RESULTS / "ablations.jsonl")
        # Rate the variant on a scale where the reference method sits at 0, so
        # the fitted number *is* the Elo delta.
        result = rating.rate(
            [rating.Opponent(reference, 0.0, match.score_a, len(match.games))]
        )
        rows.append((name, result))

    print(f"\n=== evaluation-term ablation @ {budget}, vs {reference} ===")
    for name, result in sorted(rows, key=lambda r: r[1].elo):
        term = name.removeprefix("M1-no-") if name.startswith("M1-no-") else name
        worth = "" if result.bounded else f"  ({term} is worth {-result.elo:+.0f} Elo)"
        print(f"  {name:<24} {result}{worth}")
    return 0


def _budget_from(args: argparse.Namespace) -> methods.Budget:
    if args.depth is not None:
        return methods.Budget(depth=args.depth)
    if args.seconds is not None:
        return methods.Budget(seconds=args.seconds)
    return methods.Budget(nodes=args.nodes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab", description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="download Fairy-Stockfish and the atomic net")
    sub.add_parser("bench", help="measure Fairy-Stockfish throughput here")

    p_cal = sub.add_parser("calibrate", help="measure the FSF ladder's Elo")
    p_cal.add_argument("--games", type=int, default=40)

    def add_budget(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nodes", type=int, default=10_000)
        p.add_argument("--depth", type=int)
        p.add_argument("--seconds", type=float)

    p_rate = sub.add_parser("rate", help="rate a method against the ladder")
    p_rate.add_argument("method", choices=sorted(methods.REGISTRY))
    p_rate.add_argument("--games", type=int, default=20)
    add_budget(p_rate)

    p_match = sub.add_parser("match", help="head-to-head between two methods")
    p_match.add_argument("method_a", choices=sorted(methods.REGISTRY))
    p_match.add_argument("method_b", choices=sorted(methods.REGISTRY))
    p_match.add_argument("--games", type=int, default=20)
    add_budget(p_match)
    p_match.add_argument(
        "--nodes-b", type=int, help="give method_b a different node budget"
    )

    sub.add_parser("ratings", help="show every rating measured so far")

    p_lvl = sub.add_parser("levels", help="calibrate the UI's difficulty levels")
    p_lvl.add_argument("method", nargs="?", default="M1-alphabeta",
                       choices=sorted(methods.REGISTRY))
    p_lvl.add_argument("--anchor", type=int, default=2_000,
                       help="the level whose absolute rating anchors the chain")
    p_lvl.add_argument("--games", type=int, default=60)

    p_play = sub.add_parser("play", help="play the engine in a browser")
    p_play.add_argument("--host", default="127.0.0.1")
    p_play.add_argument("--port", type=int, default=8770)
    p_play.add_argument("--no-browser", action="store_true")

    p_abl = sub.add_parser("ablate", help="what is each eval term worth, in Elo")
    p_abl.add_argument("--reference", default="M1-alphabeta", choices=sorted(methods.REGISTRY))
    p_abl.add_argument("--variants", nargs="*", choices=sorted(methods.REGISTRY))
    p_abl.add_argument("--games", type=int, default=40)
    add_budget(p_abl)

    args = parser.parse_args(argv)
    handlers = {
        "setup": cmd_setup,
        "bench": cmd_bench,
        "calibrate": cmd_calibrate,
        "rate": cmd_rate,
        "match": cmd_match,
        "ablate": cmd_ablate,
        "ratings": cmd_ratings,
        "levels": cmd_levels,
        "play": cmd_play,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
