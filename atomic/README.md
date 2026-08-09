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
  ui/index.html           the board page (no framework, no CDN)
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

## Reading order

1. `docs/research-notes.md` — the rules and what each one does to engine design,
   real atomic piece values, the state of the art, and the closest prior attempt.
2. `docs/method-catalog.md` — what is being tested and what each experiment
   would falsify.
3. `PROGRESS.md` — what is measured so far.
