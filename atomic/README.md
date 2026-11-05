# Atomic Chess Lab

A measurement-first workbench for atomic chess engines.

The question is **not** "can I build an atomic engine" — it is **"which way of
thinking about atomic buys the most strength per unit of compute?"** Modern
atomic engines are weak relative to their standard-chess counterparts, the
variant's theory is thin, and its rules break enough standard-chess assumptions
that transplanted ideas fail in interesting, informative ways.

So: several methods, one measured Elo scale, anchored to
**Fairy-Stockfish 14 + the atomic NNUE net** — the gold standard.

## Why this is not a "beat Fairy-Stockfish" project

FSF with `atomic-2cf13ff256cc.nnue` gains **+719 Elo** over its own handcrafted
atomic eval and runs at **1.24 Mnps** on this machine. It is not going to be
beaten here, and pretending otherwise would produce a project that optimises for
a number it can never reach. Instead FSF is *throttled* to a ladder of node
limits, and every method is placed on that ladder with confidence intervals. The
result of each experiment is a statement like "blast-radius king safety is worth
280 ± 60 Elo", which is a real finding regardless of the absolute level.

## Quick start

```powershell
cd atomic
py -3.13 -m pip install -r requirements.txt

py -3.13 -m lab.cli setup            # fetch Fairy-Stockfish + the atomic net
py -3.13 -m lab.cli bench            # what the gold standard does on this box
py -3.13 -m lab.perft                # verify atomic rules against FSF
py -3.13 -m pytest tests -q          # rule-level edge cases

py -3.13 -m lab.cli calibrate --games 40      # measure the FSF Elo ladder
py -3.13 -m lab.cli rate M0-greedy --games 20 # place a method on it
py -3.13 -m lab.cli match M0-greedy M0-random # head-to-head
py -3.13 -m lab.cli ablate --games 40         # what each eval term is worth
py -3.13 -m lab.cli ratings                   # every Elo measured so far
py -3.13 -m lab.cli play                      # play the engine in a browser
```

And the trainer, which uses the same engine to work on *your* atomic:

```powershell
py -3.13 -m trainer.fit_winrate       # fit cp -> win% for atomic (once, ~40 min)
py -3.13 -m trainer.cli import clee96326   # your atomic games from Lichess
py -3.13 -m trainer.cli analyse       # deep Fairy-Stockfish pass over every ply
py -3.13 -m trainer.cli review <id>   # where that game was thrown
py -3.13 -m trainer.cli worst         # your biggest mistakes, across all games
py -3.13 -m trainer.cli explain --moves Nf3 --move h6   # why a move is bad, from the engine's own lines
py -3.13 -m trainer.cli concepts fit  # fit the readable concept model to the engine (for "positional" moves)
py -3.13 -m trainer.cli book build    # count an opening book from the dumps
py -3.13 -m trainer.cli book line Nf3 f6   # what the field plays here, and how it scores
py -3.13 -m trainer.cli openings      # the named opening index, in standard-chess grammar
py -3.13 -m trainer.cli openings name Nf3 f6 --as "..."   # call a line what you will remember
py -3.13 -m trainer.cli traps scan    # book positions where only one or two moves hold
py -3.13 -m trainer.cli traps list    # ...ordered by how much of the field walks in
py -3.13 -m trainer.cli patterns      # mistakes you repeat, against what the field does
py -3.13 -m trainer.cli repertoire build      # derive one from the book, per colour
py -3.13 -m trainer.cli repertoire validate   # let the engine check every line
py -3.13 -m trainer.cli repertoire coverage   # the lines you keep needing and lack

py -3.13 -m trainer.cli puzzles generate      # mine puzzles from your analysed games
py -3.13 -m trainer.cli puzzles mine          # scrape them from rated Lichess games
py -3.13 -m trainer.cli puzzles endgames      # exact ones, from the atomic tablebase
py -3.13 -m trainer.cli puzzles themes        # do the theme detectors distinguish anything?
py -3.13 -m trainer.cli puzzles rate          # measure difficulty on the calibrated ladder
py -3.13 -m trainer.cli puzzles ratings       # the corpus by difficulty, and how you do on it
py -3.13 -m trainer.cli serve                 # the whole thing in a browser
```

`serve` is the front door: puzzles on an analysis board you can rewind with the
arrow keys and explore with the engine once they are solved — it answers in
about a tenth of a second and keeps deepening until you move on — an improvement
report over your own solve history, game review with replay-from-the-blunder,
repeated-mistake clustering, the named opening index, the traps, the
repertoire, and a board to play any position out on — all at
http://127.0.0.1:8771. The generators, the trap scan, and the difficulty rater
can be started and watched from there too.

`openings` is what makes the book memorable. Atomic has no ECO and almost no
published names, so lines are named in the grammar standard chess already uses
— Defence, Gambit, Attack, Counterattack — and each suffix is a *test* over the
position rather than a flourish: a Gambit leaves material that survives being
taken under the explosion rules, and a Counterattack has to measurably gain
Black ground on the field's own numbers, which is why `1.Nf3 f6 2.e3 d5` earns
the word (+8.3 points) and `1.Nf3 d5` does not (−29.0). Every name carries its provenance — borrowed
from standard chess, used by the atomic community, coined here, or derived by
rule — and any of them can be overwritten with your own, which then wins
everywhere the line appears.

`traps scan` asks the question the puzzle miners cannot: not "where did a game
go wrong" but "where in the tree *can* it". A trap is a book position where at
most two moves keep the balance and the best alternative is at least 25 win%
worse and leaves the mover under a third — and then the interesting column,
which is not an engine opinion at all: the share of the field that actually
plays into it, counted from the dumps. Over the popular part of the book that
is 51 traps in 90 seconds, the worst of them walked into by 61% of 3,475 games.

`puzzles rate` is what makes the corpus browsable by difficulty rather than
only by theme. A puzzle's rating is the Elo of the weakest calibrated M1 level
that plays the whole solution line, held to exactly the standard you are — so
it is in the same units as every other number in `PROGRESS.md`, needs no
solvers but the engine, and is honest about its error bars from the first
puzzle. It is also honest about its floor: 74% of the current corpus is solved
by the ladder's weakest rung, and those are reported as `≤1366` rather than as
a reading.

`puzzles mine` is the corpus builder: it streams the published monthly Lichess
dumps, screens every ply of a rated game at a cheap calibrated budget, and puts
the survivors through the same confirming pass the personal miner uses. Puzzles
are deduplicated on the position itself, which matters more than it sounds —
atomic's opening tree is narrow enough that unrelated games reach the same
critical position constantly.

`ablate` is the command that answers the project's actual question. It plays
each ablation against the *full* method rather than up the ladder: two
near-equal opponents produce a mix of results, which is where the information
is, while a gauntlet against a much stronger ladder produces a wall of losses
that constrains almost nothing.

## Layout

```
atomic/
  docs/
    research-notes.md     what is actually known about atomic, with sources
    method-catalog.md     the methods under test, hypotheses, predictions
    trainer-plan.md       the trainer: puzzles, game review, repertoire
  lab/                    Python: harness, measurement, orchestration
    fsf.py                Fairy-Stockfish adapter + the throttle ladder
    arena.py              colour-balanced match runner
    rating.py             anchored Elo with likelihood-profile intervals
    methods.py            the Method interface, M0 baselines, M1 + ablations
    rules.py              atomic-specific board helpers
    perft.py              movegen correctness gate (reference from FSF)
    tablebase.py          Lichess atomic tablebase client (cached)
    server.py             the play-against-it UI, stdlib http.server
    cli.py                the command line
  trainer/                the product built on the lab (imports it, never edits it)
    winrate.py            cp -> win% *for atomic*, and move classification
    fit_winrate.py        the experiment that fits it, from real games
    analysis.py           deep FSF pass over a game; the cache everything reads
    dump.py               streaming reader for the Lichess atomic dumps
    book.py               the opening book, counted from 12 months of dumps
    openings.py           names for the lines: curated, derived, or yours
    traps.py              book positions where one move holds and the rest lose
    review.py             the mistakes you make more than once
    repertoire.py         derive lines from the book, then let the engine check them
    themes.py             atomic theme detectors, and their firing rates
    puzzles.py            what a puzzle is: model, storage, filters, attempts
    generate.py           mining puzzles from the analysis cache
    mine.py               scraping puzzles from the rated Lichess dumps
    endgames.py           exact puzzles from the atomic tablebase
    difficulty.py         puzzle difficulty, measured on the calibrated M1 ladder
    jobs.py               generators run in the background, watched from the page
    server.py             the trainer UI's routes (owns legality *and* the answer),
                          and the two opponents: M1 on the ladder, or FSF at full strength
    ingest/lichess.py     account import (no token needed)
    store.py              SQLite: games, cache, judgements, book, names, traps,
                          repertoire, puzzles
    cli.py                import / analyse / review / patterns / book / openings /
                          traps / repertoire / puzzles
  ui/                     no framework, no CDN, no build step
    index.html            the lab's play page
    trainer.html          the trainer, ten views behind a hash router
    trainer.js            those views
    board.js              the shared board: renders what the server sent, nothing more
    pieces/               cburnett SVGs, vendored (GPLv2+); see its README
    app.css               one dark theme
  core/                   Rust: the hot loop (shakmaty + PyO3)
    src/eval.rs           atomic-native evaluation, one flag per term
    src/search.rs         alpha-beta, blast-aware ordering, atomic quiescence
  tests/                  rule edge cases + perft reference data
  bin/, data/, results/   binaries, nets, and measured output (git-ignored)
```

## The measurement protocol

Four rules, none optional, because a sloppy protocol makes every number
downstream worthless:

1. **Colour-balanced pairs** over shared random openings. White's first-move
   advantage in atomic is roughly **three times** the standard-chess figure
   (Pav 2025) — an unbalanced sample measures the colour, not the method.
2. **One budget currency** across compared methods. `Budget` accepts exactly one
   of nodes/depth/seconds, and a method that cannot honour a currency must say
   so rather than silently overspending.
3. **Anchored to the calibrated ladder**, with likelihood-profile confidence
   intervals (small, lopsided samples make the normal approximation lie).
   Shutouts are reported as bounds, never as point estimates.
4. **Ablations, not just totals.** The finding is the delta.

The scale is internal by construction: `REFERENCE_RUNG` (64k nodes) is pinned at
2000 Elo. Only *differences* on this scale mean anything — do not compare these
numbers to Lichess or CCRL ratings.

## Why a Rust core

The style engine in this repo topped out at ~12k nodes/sec in pure Python, and
its own PROGRESS.md concluded that search speed, not eval quality, decided every
experiment it ran. Fairy-Stockfish is 100× faster. A harness whose methods are
100× slower than its yardstick cannot distinguish "this idea is worse" from
"this idea had less compute" — so the board and search live in `core/`, built on
[shakmaty](https://docs.rs/shakmaty), the same atomic implementation Lichess
runs in production.

Movegen speed is *not* the goal (the Chalmers thesis measured movegen at <1% of
play-time cost). Search and eval throughput is — and M1 now searches at 2–3.7
Mnps, faster than Fairy-Stockfish's own search, so a method that loses does so
on the merits of its ideas.

```powershell
winget install Rustlang.Rustup      # once
py -3.13 -m lab.build_core          # builds a wheel and installs it
py -3.13 -m lab.perft --backend rust --max-depth 5
```

Measured: **68 Mnps** versus python-chess's 0.11 — about 630×. All 38 perft
reference counts match Fairy-Stockfish exactly, including the adversarial cases
(pinned-but-regicidal moves, own-king-in-blast, en-passant blast radius).

`lab.build_core` uses `maturin develop` inside a virtualenv and falls back to
`maturin build` + `pip install` on a global Python, where `develop` refuses to
run.

## Playing it

`lab.cli play` serves a board at `http://127.0.0.1:8770`. The server is the
referee — it owns the position, validates every move against the perft-checked
`AtomicBoard`, and computes explosion squares — so the page never decides what
is legal. That keeps the genuinely hard part of atomic in the implementation
that has been checked against Fairy-Stockfish rather than reimplemented in
JavaScript, where it would drift.

Difficulty levels are node budgets, each shown with its **measured** rating. A
level that has not been rated yet says so instead of displaying a guess.

Those ratings come from `lab.cli levels`, which chains head-to-head gaps from
one anchor rather than rating each level on the ladder separately. Per-level
gauntlet ratings turn out not to be comparable to each other — `gauntlet` stops
climbing once a method is shut out, so each budget faces a different rung set,
and a weaker setting that survives to play more rungs can out-rate a stronger
one. Measured here, M1 at 2k nodes gauntleted *above* M1 at 64k. See
`PROGRESS.md`.

## Reading order

1. `docs/research-notes.md` — the rules and what each one does to engine design,
   real atomic piece values, the state of the art, and the closest prior attempt.
2. `docs/method-catalog.md` — what is being tested and what each experiment
   would falsify.
3. `PROGRESS.md` — what is measured so far.
