"""Drop training positions that already appear in an earlier dataset.

Why this exists: `gen_data.py` dedups by zobrist hash inside a `seen` set that is
created per *worker* (and per *run*). So it has never deduplicated across shards,
and it cannot deduplicate against a previous dataset -- that file is never even
opened. Two consequences showed up in the v5/v6 data:

  * train4 is a byte-for-byte superset of train3. Both runs used the default
    --seed 1234, and worker w always seeds from `seed + w*7919`, so worker w
    replays the identical randomized games in the identical order. The 70k-game
    run regenerated all 40k old games and appended new ones.
  * 3.6% of train4's own lines are FENs that appear more than once, because
    independent workers transpose into the same (mostly opening) positions.

A fresh --seed fixes the first. Only a global pass fixes the second, and only
this tool can guarantee a new dataset shares no positions with an old one.

Usage:
    # See what a train5-vs-train4 dedup would remove, without writing:
    python -m tools.dedup --against data/train4 --clean data/train5

    # Actually rewrite train5's shards in place:
    python -m tools.dedup --against data/train4 --clean data/train5 --apply

    # Collapse a dataset's internal duplicates too (no --against):
    python -m tools.dedup --clean data/train4 --apply

`--against` may be repeated. Shards are rewritten in place; pass --backup to
keep a `.bak` of each shard first.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil


def shard_paths(prefix: str) -> list[str]:
    """`data/train4` -> its sorted .shard*.tsv files."""
    paths = sorted(glob.glob(f"{prefix}.shard*.tsv"))
    if not paths:
        raise SystemExit(f"no shards match {prefix}.shard*.tsv")
    return paths


def fens_in(prefix: str) -> set[str]:
    """Every FEN appearing in a dataset's shards."""
    seen: set[str] = set()
    for path in shard_paths(prefix):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                fen, _sep, rest = line.partition("\t")
                if rest:  # skip malformed lines with no tab
                    seen.add(fen)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True,
                    help="dataset prefix to filter, e.g. data/train5")
    ap.add_argument("--against", action="append", default=[],
                    help="dataset prefix whose positions must be removed from "
                         "--clean. Repeatable. Omit to only collapse --clean's "
                         "own internal duplicates")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite shards in place (default: dry run, report only)")
    ap.add_argument("--backup", action="store_true",
                    help="with --apply, save each shard to <shard>.bak first")
    args = ap.parse_args()

    # Seed the set with everything we must exclude. As we walk --clean we keep
    # adding to it, so a position surviving in one shard also suppresses its
    # copies in later shards -- internal dedup falls out of the same pass.
    seen: set[str] = set()
    for ref in args.against:
        before = len(seen)
        seen |= fens_in(ref)
        print(f"{ref}: {len(seen) - before:,} new reference FENs "
              f"({len(seen):,} total)")

    kept_total = dropped_ref = dropped_self = 0
    ref_fens = set(seen)  # snapshot: anything in here came from --against

    for path in shard_paths(args.clean):
        keep: list[str] = []
        n = 0
        for line in open(path, "r", encoding="utf-8"):
            fen, _sep, rest = line.partition("\t")
            if not rest:
                continue
            n += 1
            if fen in seen:
                if fen in ref_fens:
                    dropped_ref += 1
                else:
                    dropped_self += 1
                continue
            seen.add(fen)
            keep.append(line)

        kept_total += len(keep)
        pct = 100.0 * (n - len(keep)) / n if n else 0.0
        print(f"{os.path.basename(path)}: {n:,} -> {len(keep):,} "
              f"({pct:.1f}% dropped)")

        if args.apply:
            if args.backup:
                shutil.copy2(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(keep)

    total_dropped = dropped_ref + dropped_self
    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: "
          f"{kept_total:,} lines kept, {total_dropped:,} dropped "
          f"({dropped_ref:,} already in --against, "
          f"{dropped_self:,} internal duplicates)")
    if not args.apply and total_dropped:
        print("Re-run with --apply to rewrite the shards.")


if __name__ == "__main__":
    main()
