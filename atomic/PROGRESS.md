# Atomic Lab — Progress

Last updated: 2026-08-09

Goal: compare several approaches to building an atomic chess engine on one
measured Elo scale anchored to Fairy-Stockfish + atomic NNUE.

---

## Status

| Component | State |
|---|---|
| Research distilled from both papers + web | Done (`docs/research-notes.md`) |
| Method catalog with hypotheses/predictions | Done (`docs/method-catalog.md`) |
| Fairy-Stockfish gold standard, wired and verified | Done |
| Perft correctness gate vs FSF | Done — 8 positions, 38 reference counts |
| Arena (colour-balanced, atomic adjudication) | Done |
| Anchored Elo with profile CIs | Done |
| Lichess atomic tablebase client (cached) | Done, verified live |
| M0 baselines rated end-to-end | Done — **numbers stale, see the M0-greedy defect** |
| Rust core (shakmaty + PyO3) | Done — compiles, perft-verified, ~630× python-chess |
| FSF ladder calibrated | Done — 40 games/pair |
| M1 (atomic HCE + alpha-beta, in Rust) | Done — rated, ablated |
| M2–M5 methods | Not started |
| Trainer plan (puzzles, review, repertoire) | Done (`docs/trainer-plan.md`) |
| Atomic cp→win% curve, fitted | Done — 12k positions, 2.84× flatter than standard |
| Trainer store + analysis cache + Lichess ingest | Done |
| Game review (terminal) | Done |
| Opening book from the dumps | Done — 3.74M games, 90k moves, 48k positions |
| Repertoire, derived and engine-checked | Done — 181 nodes, 62% engine agreement |
| Puzzle generation | Done — 145 puzzles, 51 mined + 94 exact |
| Atomic theme detectors, with firing rates | Done — 15 themes, 12 of them earn their place |
| Trainer UI (puzzles, review, book, repertoire, play) | Done — `trainer.cli serve` |
| Puzzle scraper over the rated Lichess dumps | Done — two-tier funnel, deduplicated by position |
| Analysis board, arrow-key replay, improvement report | Done |

## Measured so far

**Fairy-Stockfish 14 + `atomic.nnue` on this box** (i7-class CPU, 1 thread):

| Config | Depth reached | Throughput | Startpos eval |
|---|---|---|---|
| 16k nodes | 10 | 0.23 Mnps | +308 cp |
| 256k nodes | 16 | 1.24 Mnps | +248 cp |

A **+250 to +310 cp evaluation of the starting position** is the headline
atomic fact: White's advantage is enormous compared with standard chess, which
matches Pav's regression finding that White's first-move edge is ~3× the
standard-chess value. This is why the arena enforces colour-balanced pairs.

**Move generation cross-check** — python-chess vs Fairy-Stockfish, 8 positions
chosen for the rules atomic implementations get wrong (adjacent kings, en
passant blast radius, pinned-but-regicidal moves, own-king-in-blast,
castling rights destroyed by explosion): 38 reference counts generated from
Fairy-Stockfish, and **all 38 matched by both python-chess and the Rust core**.
Three independent implementations agree on every atomic edge case. Reference
data in `tests/perft_positions.json`.

**Movegen throughput:**

| Backend | Rate |
|---|---|
| python-chess `AtomicBoard` | 0.11 Mnps |
| Rust core (shakmaty) | **68 Mnps** (~630×) |
| Fairy-Stockfish, full *search* | 1.24 Mnps |

The core is no longer the bottleneck by a wide margin — which is the point.
Per the Chalmers profiling, movegen should be <1% of play-time cost, and now it
is. Everything from here is search and eval throughput.

**M0 baselines:**

| Match | Result |
|---|---|
| `M0-greedy` vs `M0-random`, 20 games | +13 =0 -7 (65.0%) |
| `M0-greedy` vs `fsf-nnue-1kn`, 4 games | 0/4 — bounded `< 1242 Elo` |

> ### These two numbers are stale: M0-greedy could not see an immediate win
>
> Found on 2026-08-11 while building the puzzle generator's triviality filter,
> which uses M0-greedy as its "any beginner sees it" oracle. The method's mate
> detection was inverted and had never fired.
>
> `python-chess` defines `is_variant_win()` and `is_variant_loss()` relative to
> **the side to move**. After `board.push(move)` the side to move is the
> *opponent*, so "I just won" is `is_variant_loss()` — they have no king — and
> `is_variant_win()` after a push means our own king is gone, which
> `AtomicBoard` never generates as a legal move in the first place.
> `GreedyMaterialMethod` had them the other way round, so the early return was
> dead code and the `-1_000_000` penalty was applied to **winning** moves.
>
> Demonstrated on a position with Qxf7 available, exploding the king on e8:
> M0-greedy played `a2a3` and never found the win in 40 tries. With the
> predicates corrected it plays `f3f7` every time.
>
> This is one line in `lab/methods.py` and it is now fixed, with a regression
> test in `tests/test_puzzles.py`. What it invalidates:
>
> - the 65.0% against `M0-random` — a greedy engine that declines mate in one
>   is not the baseline the row claims to describe, and the number is very
>   probably an underestimate;
> - the 0/4 bound against `fsf-nnue-1kn`, for the same reason (the bound
>   direction is safe, its tightness is not);
> - **not** anything about M1, the FSF ladder, the ablations, or the 2×2 —
>   none of those route through this method.
>
> Both rows want re-measuring (`lab.cli match M0-greedy M0-random --games 200`).
> That has not been run, so they are left in place and labelled rather than
> quietly replaced. It is also worth reading as a small result in its own
> right: the claim recorded here that "one-ply explosion-aware material barely
> beats random" was partly a measurement of a bug.

The 65% is itself a small result: one-ply explosion-aware material barely beats
random. In standard chess the same comparison is far more lopsided. It is an
early, cheap confirmation of the paper's central claim that **material is a much
weaker signal in atomic than in standard chess** — the games ended in 8–20 ply
by king explosion, decided by tactics that material cannot see.

**The calibrated ladder** (40 games/pair, chained outward from the 64k anchor):

| FSF nodes | Elo | per doubling |
|---|---|---|
| 1,000 | 1521 | |
| 4,000 | 1775 | +127 |
| 16,000 | 1892 | +58 |
| 64,000 | **2000** (anchor) | +54 |
| 256,000 | 2070 | +35 |
| 1,024,000 | 2178 | +54 |

**Compute buys much less in atomic than the standard-chess prior assumed.**
Above 16k nodes it is roughly 45 Elo per doubling, against the 70 the
theoretical prior used. Each gap carries about ±55 Elo at 40 games and the
errors chain outward from the anchor, so the endpoints are softer than the
middle — but the 254-Elo gap between the 1k and 4k rungs is far outside that,
and is consistent with atomic being forcing enough that 1k nodes simply cannot
see the tactics.

**M1 — atomic-native handcrafted eval + alpha-beta** (Rust, `core/src/`):

| Measurement | Result |
|---|---|
| Search throughput | 2.1–3.7 Mnps (**faster than FSF's own search**) |
| vs `M0-greedy`, 20 games | +20 =0 -0 |
| Rated on the ladder @ 64k nodes | **1254 ± 167 Elo** (5.0/60) |

M1 is ~270 Elo below the *weakest* FSF rung while getting 64× its nodes. Games
last 22–54 ply, so it is being outplayed rather than blundering: this is an
evaluation-quality gap, which is what M2 onward exist to close.

### The M1 ablations — and the main negative result so far

Each evaluation term removed, then played head-to-head against full M1 (40
games at 32k nodes each). Head-to-head rather than up the ladder on purpose:
two near-equal opponents produce a mix of results, where the information is,
while a gauntlet against a much stronger ladder produces a wall of losses that
constrains almost nothing.

| Removed | Cost, Elo |
|---|---|
| material | **+269 ± 144** |
| *quiescence search* (search-side, for scale) | **+203 ± 128** |
| all atomic terms (material only) | +168 ± 122 |
| mobility | +137 ± 117 |
| piece-square tables | +61 ± 110 |
| **blast-radius king safety** | **+17 ± 109** |
| pawn shield | +0 ± 109 |

**The project's headline hypothesis — that atomic strength lives in
blast-radius king safety — does not survive contact with measurement in the
form it was posed.** The term is worth nothing, and material, the thing atomic
theory calls a weak signal, dominates.

The obvious rescue is "the search is deep enough to see the blast, so the
static term is redundant", which predicts the term pays at shallow depth and
decays as nodes rise. **It does not.** Measured at 2k / 8k / 32k / 128k nodes,
the term is worth +6 / +23 / +17 / +0 Elo — flat across a 64× range.

### Where the knowledge actually lives: a 2x2, at 400 games per arm

What does explain the flat sweep is *which part* of the search already covers
the term. A fuse is lit by a capture, and quiescence searches every capture, so
a capture-only quiescence spans the term's entire threat space at every depth.
Running the same ablation with and without quiescence, 400 games each:

| Blast-radius king safety is worth | Elo | 95% interval |
|---|---|---|
| with quiescence | **-24 ± 34** | [-58, +10] |
| without quiescence | **+55 ± 35** | [+20, +90] |

**The intervals do not overlap.** The term is genuinely valuable knowledge —
worth a solid +55 Elo when nothing else supplies it — and is then made entirely
redundant by a capture-only quiescence search, to the point of being mildly
counterproductive once quiescence is present.

So the hypothesis was right about *what* matters and wrong about *where to put
it*. In atomic, king safety is a **tactical** property, not a positional one,
and belongs in the search rather than the evaluation. That is a real departure
from standard chess, where king safety must be static because attacks assemble
over more plies than any search can reach — and it is the first finding here
that would change how someone builds an atomic engine.

For scale, quiescence itself is worth **+203 ± 128 Elo**, more than every
atomic-specific evaluation term put together.

### M1 converts compute into strength; the ladder just cannot see it

The ladder ratings for M1 at 4k / 16k / 64k nodes come out flat (1254 / 1296 /
1254) with ±170 intervals — but that is a limitation of rating a method against
opposition that beats it 90% of the time, not a property of M1. Measured
directly, against itself at 1k nodes, 60 games per point:

| M1 at | Elo above M1 @ 1k nodes |
|---|---|
| 4,000 | +260 ± 115 |
| 16,000 | +417 ± 162 |
| 256,000 | +483 ± 193 |
| 1,024,000 | +585 ± 258 |

M1 spans roughly 585 Elo from its weakest to its strongest setting, so the UI's
difficulty levels are real.

### Gauntlet ratings at different budgets are not comparable to each other

This is a harness defect worth recording, because it silently invalidates the
obvious way to build a difficulty menu. Rating M1 at each node budget against
the ladder gives:

| M1 nodes | gauntlet rating | games |
|---|---|---|
| 2,000 | **1366 ± 88** | 240 |
| 4,000 | 1254 ± 178 | 40 |
| 16,000 | 1296 ± 164 | 40 |
| 64,000 | 1254 ± 167 | 60 |
| 256,000 | 1520 ± 103 | 80 |
| 1,024,000 | 1523 ± 108 | 60 |

**M1 at 2k nodes out-rates M1 at 4k and at 64k**, which the head-to-head
results above contradict outright. The cause is structural rather than
statistical: `gauntlet` stops climbing once a method is shut out, so the rung
set differs per budget, and a *weaker* setting that survives to play more rungs
draws its estimate from different — and easier — opposition. The game counts
in that table are a symptom, not a coincidence.

Two consequences, both now implemented:

- `lab.cli levels` calibrates the UI's difficulty ladder by chaining measured
  head-to-head gaps from a single anchor, which is the same construction
  `calibrate` already uses for the Fairy-Stockfish rungs. Only the anchor's
  absolute value comes from a gauntlet; all the spacing is direct.
- **A gauntlet against a much stronger ladder is the wrong instrument for any
  question about a weak method.** `match --nodes-b` and `ablate` are the right
  ones. The gauntlet remains correct for what it was built for — placing one
  configuration on an absolute scale — and its numbers should be read as
  "roughly 1250–1550, poorly identified", not compared with each other.

## Playing the engine

```powershell
py -3.13 -m lab.cli ratings   # every Elo measured so far, on one scale
py -3.13 -m lab.cli levels    # calibrate the difficulty ladder (chained)
py -3.13 -m lab.cli play      # browser UI, http://127.0.0.1:8770
```

The calibrated difficulty levels, chained from the 2k anchor at 60 games each:

| Level | Nodes | Elo |
|---|---|---|
| Casual | 2,000 | 1366 ± 88 (anchor) |
| Club | 16,000 | 1691 ± 131 |
| Strong | 64,000 | 1704 ± 135 |
| Full strength | 1,024,000 | 1951 ± 258 |

Monotonic, unlike the per-budget gauntlet ratings. Club and Strong are within
each other's intervals, which is honest rather than tidy: 16k to 64k is only
two doublings and M1's returns to compute are shallow in that range.

The UI is served by `lab/server.py` (stdlib `http.server`, no new dependencies)
with `ui/index.html` as its only page. The server is the referee: it owns the
board, validates every move against the perft-checked
`chess.variant.AtomicBoard`, and computes explosion squares, so the page never
decides what is legal. Difficulty levels are node budgets, each labelled with
its *measured* rating — a level that has not been rated says so rather than
showing a guessed number.

## The trainer: what an atomic evaluation is actually worth

The trainer (`trainer/`, see `docs/trainer-plan.md`) needs to say "this move
threw the game away", which means converting an evaluation into a win
expectancy. Lichess does this with

    Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)

fitted to standard chess. Reusing it here was never going to work, and this
file already contained the reason: FSF+NNUE evaluates the atomic **starting
position** at +250 to +310 cp, which the standard curve reads as White scoring
~76%. So the curve was refitted for atomic.

**Method.** 12,000 positions, 3 per game, drawn by reservoir sampling from the
120,408 eligible games in the 2026-07 Lichess atomic dump (both players
Elo 1400–2200, `Termination: Normal` only — a flag fall says nothing about the
position). Each evaluated with FSF + `atomic.nnue` at 1M nodes, then regressed
against the game's actual result by IRLS on

    win% = 100 / (1 + exp(-(a + k * cp)))

Three positions per game rather than thirty because the statistical unit is the
*game result*: twenty positions from one game carry about one game's worth of
information, and counting them as twenty would shrink the interval by 4.5× for
free. The intercept is fitted rather than assumed, so that any systematic
offset in the net is reported instead of being smeared across every judgement.

**Result.**

| | atomic (fitted) | standard chess (Lichess) |
|---|---|---|
| slope `k` | **0.0012965** | 0.0036821 |
| intercept `a` | −0.1488 | 0 (assumed) |
| even at | **+115 cp** | 0 cp |
| log-loss on this sample | **0.5422** | 0.7490 |

**An atomic centipawn is worth about a third of a standard one** — the curve is
2.84× flatter. Converted at the evaluations this project already published:

| FSF eval | atomic win% | standard curve claimed |
|---|---|---|
| 0 cp | 46.3% | 50.0% |
| +250 cp | 54.4% | 71.5% |
| **+308 cp** (startpos) | **56.2%** | 75.7% |
| +500 cp | 62.2% | 86.3% |
| +1000 cp | 75.9% | 97.5% |

So the headline atomic fact recorded above needs restating. White's +250 to
+310 cp opening advantage is **not** a 76% expectancy; it is **56%**. The
evaluation is large and the advantage is real but modest, and the two facts were
only ever in tension because a standard-chess conversion was doing the talking.
That also explains, without any new measurement, why the FSF ladder buys only
~45 Elo per doubling above 16k nodes against a standard-chess prior of 70: if a
centipawn converts to a third as much win probability, the same evaluation gain
is worth correspondingly less on the board.

**Robustness.** Refitting on only the undecided positions (|cp| ≤ 800, which
drops every clamped mate score) gives k = 0.0012973 — a 0.06% difference, so
the slope is not an artefact of the ±1500 cp clamp. Calibration by evaluation
decile, observed versus predicted score:

| cp range | n | observed | predicted |
|---|---|---|---|
| −812..−280 | 1200 | 31.8% | 31.1% |
| −279..−34 | 1200 | 38.2% | 41.5% |
| −34..142 | 1200 | 50.7% | 47.7% |
| 142..381 | 1200 | 57.4% | 54.8% |
| 382..866 | 1200 | 62.6% | 65.1% |
| 866..1437 | 1200 | 71.2% | 79.3% |

The middle of the range — where "was that a blunder" is actually decided — is
calibrated to a few points. The extremes are not, and are deliberately not used:
`winrate.score_win_pct` maps a forced mate straight to 100%/0% rather than
through the curve.

**Caveat worth keeping.** This is a *human conversion* curve for the 1400–2200
band, not a perfect-play one. It answers "given this evaluation, how do games at
this level actually end", which is the right question for judging human moves
and the wrong one for judging engines. A band shift would move it, and the band
is recorded in `data/winrate_atomic.json` alongside the parameters.

### First run against a real account

86 rated+casual atomic games (`clee96326`, 1531 atomic), every ply analysed at
1M nodes with MultiPV 2 — 2,692 plies, ~20 minutes wall clock across 14
Fairy-Stockfish sessions. 1,346 of them were the account's own moves:
**56 blunders, 36 mistakes, 102 inaccuracies, 1,152 fine.**

More blunders than mistakes is the opposite of the standard-chess shape, and it
is what the variant predicts: atomic errors are not gradual positional drift,
they are a king that stops existing.

**Where the games are actually lost:**

| plies | moves | blunders | rate | win% lost per move |
|---|---|---|---|---|
| 0–9 | 393 | 30 | **7.6%** | 5.7 |
| 10–19 | 265 | 15 | 5.7% | 6.3 |
| 20–29 | 174 | 7 | 4.0% | 4.9 |
| 30–39 | 103 | 1 | 1.0% | 2.8 |
| 40+ | 393 | 3 | 0.8% | 1.9 |

Monotonic, and an order of magnitude across the game: **more than half of all
blunders happen in the first ten plies.** Whatever else the trainer does, the
opening is where the Elo is.

Two further cuts, both pointing the same way: blunders are 6.9% of captures but
3.8% of quiet moves, and when a blunder happened the move that should have been
played was a *quiet* move 50 times out of 56. The errors are not "missed a
capture" — they are king safety and structure, mishandled before move six.

**The first recurring pattern found, and the argument for repertoire work.**
`trainer patterns` clusters errors by position and by move. The top cluster:

| after | played | best (8M nodes, depth 18–20) |
|---|---|---|
| 1.Nf3 f6 **2.Nd4** | ...Nh6 ×3 | **Nh6** (−182cp; next best −594) |
| 1.Nf3 f6 **2.d4** | ...Nh6 ×2 | d5 (−175); Nh6 not in the top 5 |
| 1.Nf3 f6 **2.e4** | ...Nh6 ×2 | d5 (−190); Nh6 not in the top 5 |
| 1.Nf3 f6 **2.e3** | ...Nh6 ×1 | d5 (−325); Nh6 not in the top 5 |

`...Nh6` is played by rote after 1.Nf3 f6 whatever White does. Against 2.Nd4 it
is clearly best — a 12 win% margin over the alternative. Against 2.d4, 2.e4 and
2.e3 it does not appear in the top five and lost the game outright in all five
games (FSF finds a forced win afterwards, which is consistent with the
published human line that 2.e3 followed by 3.Ne5 is close to winning).

Average cost 41 win% each time it was played out of context. This is a
repertoire failure rather than a tactical one — the move is right, the branch
is not — and it is exactly the case that grouping by position alone would miss,
since the five bad instances sit in three different positions.

### An opening book for a variant that has none

There is no ECO for atomic, no published theory to speak of, and no "what does
everyone play here" table anywhere. `trainer book build` makes one from the
Lichess monthly dumps: **3,741,929 rated games over 12 months (2025-08 to
2026-07), 90,299 moves across 48,464 positions**, in 5.1 minutes.

**Why it is fast, and why it is exact.** A single month holds 2.46M distinct
move-prefixes to depth 20, so a naive year-long count is ~20M live Python
objects for a book that ends up keeping 124k of them. Counting is *monotone*,
though — a line played 20 times implies every prefix of it was played at least
20 times — so a node can only survive the cutoff if its ancestors did. The
count is therefore staged: depth 6 over everything, prune, depth 12 along the
survivors, prune, then depth 20.

| Stage | Lines counted | Surviving at 20+ games | Elapsed |
|---|---|---|---|
| depth 6 | 480,511 | 35,269 | 100s |
| depth 12 | 4,384,931 | 108,508 | 207s |
| depth 20 | 1,469,118 | 124,218 | 299s |

Peak memory is set by the surviving tree, not by the corpus, and no surviving
count is approximated. Verified two ways: the staged count reproduces a naive
single-pass count exactly on a synthetic corpus (a test), and on a real month
both methods independently returned 4,816 survivors at the 50-game cutoff.

Counting runs on move-prefixes because that needs no board; the book is then
*stored* by position, replaying each surviving line once at the end. So
transposing move orders pool their statistics — which matters in atomic, where
20 of these 86 games open 1.Nf3 f6.

**The score column is not the merit column.** A line played mostly by 1800s
outscores one played mostly by 1200s regardless of merit. Against this
account's own positions, the engine's best move and the field's best-scoring
move agree **56%** of the time (169 positions with 2+ moves over 200 games) —
far above the ~10% a random legal move would score, but well short of
interchangeable. The book stores average Elo per line precisely so that this
confound is visible rather than hidden; the engine remains the judge.

### What the book says about the mistakes

`trainer patterns` now annotates every repeated error with what the field does.
The `...Nh6` habit stops being just an engine opinion:

| after | Nh6 is | scores | best alternative |
|---|---|---|---|
| 1.Nf3 f6 2.d4 | #3 of 15, 13% of games | 37% | d5 — 45% on 29,465 games |
| 1.Nf3 f6 2.e4 | #3 of 17, 12% of games | 49% | d5 — 58% on 40,676 games |
| 1.Nf3 f6 2.e3 | #3 of 18, 9% of games | **18%** | e6 — 49% on 98,204 games |

The engine and 3.7M human games agree independently, which is a much stronger
claim than either alone. Two more that the book surfaced and the engine-only
report could not rank:

- **1.e3 f6 2.Qh5+ scores 94.9% over 4,431 games** — close to a forced win, and
  FSF's top move. This account plays 2.Qf3 instead (71%). That is a winning
  chance being declined, not a blunder being made.
- **1.e3 Nc6 scores 17.2% over 23,429 games**, against 51.9% for 1...e6. The
  worst-scoring reply in the top five, played twice here.

### A repertoire nobody could have written, checked by something that can

Atomic has no published repertoire and no ECO to organise one by, so the first
one is derived: walk the book from the start, take the best-*scoring* move at
your turn, branch on every reply the field actually plays (≥3% share and ≥200
games), breadth-first to 12 ply. 33 seconds of engine time later:

| | nodes | depth | engine agrees | flagged |
|---|---|---|---|---|
| White | 38 | 11 ply | 29 (76%) | 1 |
| Black | 143 | 12 ply | 84 (59%) | 4 |

**62% agreement overall**, consistent with the 56% measured on this account's
own positions and, again, nothing like interchangeable. That gap is the reason
`repertoire validate` exists: Fairy-Stockfish scores every node by the same
win% measure game review uses, and the five nodes where the derived move leaks
more than 10 win% are reported rather than silently overwritten. Three of the
four Black flags say the same thing — **…Qh4** is stronger than what the field
plays — which is one motif, not four findings, and exactly the kind of thing a
popularity statistic cannot tell you.

Validation cost almost nothing because the analysis cache is keyed by position:
181 nodes needed 352 positions, of which 345 had to be searched — the rest
were already paid for by the 86 reviewed games.

### Coverage: the repertoire and the games barely overlap

Crossing it against the 86 imported games is the blunt part:

| | covered | deviations |
|---|---|---|
| White | 46 / 250 plies (18%) | 1.e3 ×38, 1.Nf3 ×8 — the line says 1.Nh3 |
| Black | 48 / 206 plies (23%) | 1…f6 ×20 — the line says 1…d6 |

Low coverage here is the finding, not a defect: the tree diverges at ply 1
because this account does not play the move the book's best-scoring line
recommends, and everything downstream of a different first move is a different
tree. The ranked gaps are more useful than the percentage, and they converge: the top
three Black gaps are all *1.Nf3 f6* met by a different second move (2.Nd4 ×7,
2.e4 ×3, 2.d4 ×2), costing 61–85 win% each, and `…Nh6` is the reply in every
one of them. Same habit the pattern report found, now with a named hole to fill
rather than a diagnosis.

One caveat the numbers make concrete: `1…d6` is itself flagged (leaks 18 win%
against `…f6`), so on that node the *account* is right and the derived
repertoire is wrong. `repertoire validate --fix` promotes the engine's move
where that happens; it is opt-in because at ply 1 the same disagreement is the
unresolved 1.Nh3-vs-1.Nf3 question, and auto-resolving it would bury it.

### The first atomic puzzle corpus

There is no atomic puzzle set anywhere — Lichess's generator and Chess.com's
Puzzles are both standard-chess only — so the corpus had to be generated. 145
puzzles now exist, from two sources that fail in completely different ways and
therefore make a decent pair.

| Source | Puzzles | Ground truth |
|---|---|---|
| Mined from 95 analysed games | 51 (26 punish, 25 hold) | FSF + `atomic.nnue`, 4M nodes, MultiPV 3 |
| Scraped from 51 rated dump games | 35 (see the scraper section) | FSF + `atomic.nnue`, 4M nodes, MultiPV 3 |
| Lichess atomic tablebase | 94 (40 win, 54 hold) | **exact** — no engine involved |

**Two thirds of the "big swing" positions are not puzzles.** Stage 1 is a query
over the analysis cache rather than a search — every ply of every game already
has a MultiPV pass on it, so uniqueness is known everywhere for free — and it
found 98 candidates. Stage 2 confirms at 4M nodes and then walks the line
forward until it stops being forced:

| | count |
|---|---|
| candidates from the cache (free) | 98 |
| rejected: `M0-greedy` finds it | 25 |
| rejected: not unique at 4M nodes / MultiPV 3 | 19 |
| **kept** | **54** (51 after identical lines merge) |

The plan predicted the triviality problem and it is real: a quarter of the
candidates are positions any beginner solves. They pass the uniqueness test —
they *are* unique — which is exactly why uniqueness alone is not enough here
and standard-chess puzzle criteria do not transfer.

**Both halves of the corpus are majority defensive.** 72 of 145 puzzles ask you
to *hold* rather than to punish. That is not how a standard-chess puzzle set
looks, and it follows from the variant: this account's own review data already
said the refutation was a quiet move in 50 of 56 blunders, and a position where
exactly one move avoids losing is a more common critical moment in atomic than
a winning shot.

### Endgames: exact, and much harder to find than expected

The tablebase gives a puzzle definition with nothing to argue about — *exactly
one legal move preserves the result* — and `lab/tablebase.py` already spoke the
protocol. Getting a corpus out of it took three corrections, each of which
produced **zero puzzles** rather than wrong ones, which is the failure mode to
watch for in a generator.

1. **Scattering pieces at random does not work.** 3,000 isolated positions
   produced nothing: a random position is either hopeless or comfortably won,
   and a comfortably won position has a dozen moves that keep the win.
   *Fix:* walk. Scatter a start, then play it out following the tablebase's own
   best move 60% of the time and a random legal move otherwise. The blunders
   are the point — a blunder is what creates a position with one saving move.
   Every step needs a probe and every probe is also a candidate test, so the
   walk costs nothing over sampling.
2. **A third of the budget went to HTTP 400s.** `AtomicBoard` relaxes the
   into-check rules, and the relaxation reaches `status()`, so `is_valid()`
   happily accepts a position with the side *not* to move already in check. The
   tablebase refuses those, correctly. *Fix:* test it directly. A second class
   of failure — sustained 4 requests a second earning a 429 — now backs off and
   retries instead of discarding the position.
3. **The triviality filter rejected 100% of endgames.** In a position with no
   captures available every legal move leaves identical material, so the greedy
   tie set is *every legal move* and "is the solution in it?" is always yes. 14
   genuine only-move positions were thrown away by this before it was noticed.
   *Fix:* being indifferent between twelve moves is not solving anything.
   Greedy only counts as having found the move if its tie set is small enough
   that playing it is a preference (`GREEDY_TIES = 2`).

With those in place, re-examining the probe cache — 3,350 exact verdicts
already paid for — produced **94 puzzles in two minutes and zero requests**:

| | count |
|---|---|
| cached positions examined | 3,350 |
| rejected: more than one move holds | 2,846 |
| rejected: trivial | 95 |
| **kept** | **94** |

So only-move positions are about **5%** of reachable ≤6-piece atomic positions.
The probe cache behaves like the analysis cache and for the same reason: after
a filter changes, re-mining it is free.

### Do the theme detectors distinguish anything?

Standard tactical vocabulary does not survive here — a fork assumes the forking
piece collects one of the two targets, and in atomic the collector dies with
its victim — so the taxonomy was rebuilt from the rules as 15 detectors over
the solution line. They are hypotheses, and `trainer puzzles themes` is the
falsification step: a theme that never fires describes nothing, and one that
fires on everything separates nothing.

The first run said two of them fired on everything, and both were fixed:

| Theme | Before | After | What was wrong |
|---|---|---|---|
| `quiet` | 72.6% | **20.0%** | fired whenever the answer was not a capture — but most endgames have no capture to decline. Now requires that a capture *was* available and was passed over, which is the actual idea. |
| `no-capture-fortress` | 60.0% | **20.7%** | fired whenever the final position had no captures, which in a four-piece endgame is just what small endgames look like. Now requires that captures existed earlier in the line: the fuse has to go out, not merely be absent. |

Current firing rates over 145 puzzles: `tb-conversion` 65% and `opening-trap`
15% (provenance labels, exempt from the breadth test — 65% means 65% came from
the tablebase), `defensive-diffusion` 35%, `own-king-in-blast` 28%,
`no-capture-fortress` 21%, `quiet` 20%, `connected-kings` 12%, `pawn-immunity`
6%, `blast-fork` 6%, `king-blast` 5%, `promotion` 3%, `sacrifice` 3%. **Only 2%
of puzzles match no theme at all.**

Three detectors never fire, and each is informative rather than broken:

- **`king-blast` at 5%, `regicidal` at 0%** — not because the motif is rare but
  because it is *obvious*. A one-move king blast is precisely what `M0-greedy`
  finds, so the triviality filter removes almost all of them before they reach
  the corpus. The 7 that survive are king blasts a beginner would not see.
- **`ep-blast` at 0%** — en passant is rare, and rarer still as the point of a
  position. Kept; it costs nothing and would be a genuine find.
- **`long` at 0%** — no puzzle in the corpus needs four solver moves. Atomic
  lines are short: the variant's median game is 30 ply and forcing sequences
  resolve rather than extend. This is a fact about atomic, not a bug.

### The trainer has a front end

`trainer.cli serve` puts all of it in a browser at `127.0.0.1:8771`: puzzles
with hint/solution/streak and wrong-move feedback priced in win%, game review
with a win% graph and replay-from-the-blunder, repeated-mistake clustering
annotated with what the field plays, the opening book as a clickable tree, the
repertoire with its engine verdicts and coverage gaps, and a board to play any
position out against the calibrated ladder.

Stdlib `http.server` and SQLite, no framework and no build step, and lab's
referee rule extended one step: the server owns legality *and* the answer.
`/api/puzzle/<id>` returns the position, the themes and the number of moves to
find, never the moves — checking a move is a round trip, so a puzzle cannot be
solved with the devtools open.

### The puzzle page is an analysis board with a puzzle on it

Rebuilt around a **line of nodes with a cursor on it** — the game that led to
the position, the mistake, the moves you find, and afterwards anything you care
to try. `←` and `→` walk it, so "how did this arise" is answered by holding the
left arrow rather than by opening the game in another tab.

The thing worth recording is what *did not* need building. Stepping forward
during a puzzle would give away the solution, and the obvious fix is to disable
the key — but the page has no forward nodes to step into, because the moves you
have not found live on the server and were never sent. The referee rule already
covered it; a disabled key would only have been a second, weaker copy of a
guarantee that was already exact.

Once a puzzle is finished the same board becomes an analysis board: three
engine lines at 300k nodes with their variations in SAN, best-move arrows, a
win% eval bar, and free play from any node — clicking a move in a variation
walks the board into it, and every one of those moves still goes through
`/api/move` rather than being trusted. Evaluations land in the analysis cache
like every other one, so exploring a position may pay for a later generation
run rather than costing anything.

### An improvement report that refuses to over-read seven puzzles

`#/improve` breaks the solve history down by theme, phase, goal and source. Two
rules keep it from lying with small numbers, and both were the design:

- **The unit is your first attempt at each puzzle.** The queue deliberately
  brings failures back, so counting the fourth go at a position you have now
  memorised would make every drilled theme look like a strength.
- **Nothing is called strong or weak without an interval that says so.** Each
  rate carries a Wilson score interval, and a slice earns a verdict only when
  its interval clears your overall rate — the same standard the Elo tables
  above are held to, applied to a much smaller sample. On the current 7
  attempts nothing does, and the report says exactly that rather than ranking
  noise.

The tab is also where the redo queue lives: `failed` (tried and never solved)
rather than `unsolved` (which includes the 141 puzzles never seen, and is
evidence of nothing).

### Scraping the rated dumps, and what a cheap scan costs

`trainer puzzles mine` finally builds the corpus §3 of the plan deferred as
T5 — puzzles from *other people's* rated atomic games, which is the only supply
of atomic positions large enough not to be a portrait of one account's habits.

Games come from the published monthly `.pgn.zst` dumps, not the API: there is
no "rated atomic games in this band" endpoint and there was never going to be,
and crawling `/api/games/user` would be slow, rude, and against the terms.

The two-tier funnel the plan left in the code for exactly this is what makes it
affordable. A 1M-node pass on every ply is 30–60s a game, which is fine over
your own 86 games and is not fine over thousands of strangers'; the scan
therefore runs at **20k nodes**, which is a *calibrated rung* (1892 Elo) rather
than a guess, and only its survivors meet the same 4M-node confirming pass the
review path uses.

First run, 2026-07 dump, both players 1800+:

| | count |
|---|---|
| games read | 262 |
| eligible (rated 1800+, `Termination: Normal`, 12–120 ply) | 41 |
| scanned at 20k nodes | 40 (1,764 positions) |
| candidates from the scan | 43 |
| rejected: duplicate position | 2 |
| rejected: `M0-greedy` finds it | 7 |
| rejected: not unique at 4M nodes | 9 |
| **kept** | **25** (8 punish, 17 save) |

**66 seconds for 40 games**, so a thousand is about half an hour rather than a
night — and note the yield: 0.6 puzzles per game against 0.5 from the personal
corpus, which is the same rate from a hundredth of the compute per game.

Two things carry over unchanged from the personal miner and are worth seeing
survive a different corpus: **two thirds of candidates are still not puzzles**
(the triviality and uniqueness filters reject 16 of 43), and the corpus is
still **majority defensive** — 17 of 25 ask you to hold rather than to punish.
That was first read as a fact about this account's own games; it reproduces on
strangers', so it is a fact about atomic.

What a cheap scan costs is recall, not correctness — every position it proposes
is re-judged at 4M nodes before it becomes a puzzle, but positions it
*misjudges* never reach that pass and are lost silently. That is the honest
trade, and it is why the scan budget is a parameter and the node count is
recorded on every `mined_games` row.

**Deduplication by position turned out to be the load-bearing part.** Atomic's
opening tree is narrow and forcing — 20 of this account's 86 games open
1.Nf3 f6 — so the same critical position recurs across unrelated games
constantly. Puzzles are keyed on the first four FEN fields, so a transposition
counts as the same position and the counters do not split it, and the check
runs *before* the confirming pass rather than after: a duplicate rejected
afterwards has already cost 4M nodes. The provenance is carried alongside the
key so that re-running the generator over your own games still refreshes the
puzzles it made last time instead of refusing them as duplicates of themselves.

Scraped games go in their own `mined_games` table and deliberately **not** in
`games`. That table is your own history and every personal report — review,
patterns, repertoire coverage — is a query over it; a stranger's blunders have
no business in your habit clustering.

### Generation runs from the browser

All three generators (own games, dump scrape, tablebase) start from the puzzle
page and report progress while they run. One at a time, by construction: each
saturates Fairy-Stockfish across every core, so two would not go twice as fast,
they would go half as fast each and fight over the same SQLite writer. The
registry refuses a second rather than queueing it — a queue lets a mis-click
commit the machine to an hour of work invisibly.

### Difficulty, measured against the calibrated ladder

Puzzle difficulty is now a measurement rather than a missing column, built the
way `docs/trainer-plan.md` §5 specified:

> a puzzle's difficulty is the rating of the weakest calibrated level that
> finds the whole solution line.

Every puzzle is given to each rung of the chained M1 ladder (2k / 16k / 64k /
1024k nodes = 1366 / 1691 / 1704 / 1951 Elo), and the engine is held to exactly
the standard you are: it must play the stored solution move at every solver
ply, with the defence scripted from the line. That is what makes the number a
rating of *this puzzle* rather than of the position — accepting any move the
engine also likes would be measuring something else.

The whole corpus, 188 puzzles, at **0.46 s of engine time each** (all four
rungs, no early exit — 1.4 minutes for the lot):

| Difficulty | Puzzles | Tried | Solved unaided |
|---|---|---|---|
| ≤1366 (the ladder's floor) | 140 | 13 | 7 |
| 1691 | 19 | 4 | 2 |
| 1704 | 12 | 1 | 0 |
| 1951 | 7 | 0 | — |
| >1951 (nothing solved it) | 10 | 1 | 1 |

**The headline is the floor: 74% of the corpus sits under the weakest rung the
ladder has.** That is a fact about the ladder, not about the puzzles — M1 at
2,000 nodes is a 1366-rated engine, which is stronger than the audience a
training corpus is aimed at, and below it this construction cannot see. So the
floor is reported as a bound (`≤1366`) and never as a reading, the same way
`lab.rating` reports a shutout gauntlet. The next measurement that would pay
for itself is **lower rungs in `lab.cli levels`** — a 250n / 500n / 1000n
calibration would spread 140 puzzles that currently share one label.

Three results from the profile that were not obvious:

- **Line length is not difficulty.** 105 of the 140 one-move puzzles are at the
  floor — but 35 one-movers are *above* it, and 35 multi-move puzzles are at
  it. The corpus's old "shortest first" ordering was measuring something real
  and something different; both orderings are kept for that reason.
- **The tablebase corpus is the easy one.** 73 of 94 tablebase puzzles are at
  the floor against 44 of 59 mined from games and 23 of 35 scraped. Exact
  ground truth buys correctness, not depth.
- **6 puzzles were solved by a level a stronger one missed.** Expected, and the
  reason every rung is run rather than stopping at the first success: 16k and
  64k measured 1691 ± 131 and 1704 ± 135, one rung inside the other's error
  bar, so their disagreement is signal about the ladder and is recorded
  (`monotone = 0`) instead of hidden.

Ratings live in their own `puzzle_ratings` table, not in a column on `puzzles`,
because `puzzles.save` is `INSERT OR REPLACE` — regenerating the corpus at a
higher budget would erase them. A rating is a function of (position, solution),
which is exactly what the puzzle id hashes, so it survives a regeneration that
reproduces the id. The ladder's own fingerprint is stored with each verdict, so
recalibrating the levels leaves the old ratings *visibly* stale rather than
silently mixing two Elo scales in one column.

The improvement report grows a "by measured difficulty" table beside the theme
and phase ones. That is the plan's §5 refinement in its cheapest honest form:
no fitted engine→human correction, just your solve rate next to the engine's
rating so the places where the two axes come apart are visible rather than
averaged away.

### The board looks and behaves like a board

The trainer's pieces are the cburnett SVGs (the Lichess default set, vendored
under GPLv2+ in `ui/pieces/`) instead of the Unicode chess glyphs. The glyphs
failed for a specific reason worth recording: ♔ and ♚ are the outline and
filled versions of the *same* shape, so colour depended entirely on a
text-shadow and on how each font weighted them.

Input is pointer-based and supports both grammars, because people use both:
click-then-click, or pick a piece up and drop it. They share one gesture —
`pointerdown` selects and lifts, and what happens next decides which was meant
— so neither is a mode you can get stuck in. Clicking the selected piece again
deselects it; dropping onto a square that is not a legal target cancels;
dropping back where you started keeps the selection. The referee rule is
untouched: a drop still produces a UCI string drawn from `state.legal`, which
the server built, and the page still cannot invent a move.

### The analysis panel answers before it is finished

`/api/evaluate` used to be one 300k-node search per position, opened and closed
around its own Fairy-Stockfish process. A third of a second with nothing on
screen reads as a hang, and every abandoned look — arrowing back through a
line — paid for a process launch it then threw away, hash and all.

It is now a ladder: 40k, 150k, 600k, 2M nodes, each shown as it lands and each
its own cache key, served from a warm two-session pool. Measured on a middlegame
position: **0.16s to the first opinion** (0.06s once the pool is warm) and 2.0s
to the last, against 0.30s before to the only one. The searches themselves
cannot be interrupted, so abandonment happens between rungs — moving the cursor
costs at most the rung in flight rather than the whole budget — and a position
looked at twice comes back instantly at full depth.

Two findings fell out of it:

- **A pooled engine must be closed explicitly.** python-chess runs each
  engine's event loop on a *non-daemon* thread, and the interpreter joins those
  before it runs `atexit` handlers. A pool left open therefore hangs the process
  at shutdown — Ctrl-C on the server never returned — and registering the
  cleanup with `atexit` does not help, because by then it is already waiting.
- **Hint level has to be per move, not per puzzle.** The level was read off the
  count of hints taken, which is also the scoring number, so on a multi-move
  puzzle the *first* hint on the second move — a fresh question nobody had been
  helped with — handed over the whole move. Reproduced and fixed: the two
  counters are now separate.

The panel also shows the numbers, not just the bar: each line carries its
evaluation in pawns from White's point of view alongside the win%, and the
score sits at the winning end of the eval bar. Win% alone made a position a
piece up and a position that mates look identical, which in atomic — where the
two diverge much faster than in chess — is exactly the distinction being
studied. A chip under the board says whose move it is, which the instruction
line could not: it speaks about the puzzle, and stops being true the moment the
line is rewound or explored past.

### Four things the front end was quietly getting wrong

Found by using it, which is how this class of bug is always found. None of them
were crashes; each one made the tool say something untrue.

- **The eval bar went white when Black was winning.** The pale fill was pinned
  to the bottom of the bar and the *number* was inverted to compensate when the
  board was flipped — and the board is flipped whenever the solver is Black,
  which is half the corpus. So the commonest case drew a bar that read exactly
  backwards. The fill is White's share, in White's colour, at White's end of
  the board: it now moves to the top when the board flips, and "more black"
  always means Black is better. Placement and colour of the score are separate
  classes for the same reason — with Black at the bottom and Black ahead, the
  number belongs at the bottom *and* on the dark ground.
- **276 of 429 puzzles were unreachable.** The source filter offered "from
  games" and "tablebase" and had never been given the third source; the mined
  ones could only be seen by luck of the draw. Worse, the puzzle card *labelled*
  a puzzle from your own games "mined", which is what the dump ones are called
  everywhere else. Every source is now named after where it came from and all
  three are selectable.
- **The queue handed back puzzles you had already sat through.** `unsolved`
  (never solved cleanly) was the only exclusion available and it is the right
  rule for a redo queue, not for a session of new material. A stricter
  `unseen` — never attempted at all — is now the default, and the redo queue is
  the explicit `failed before` chip, which turns it off. 380 of the 429 are new.
- **The goal was implied rather than stated, and sometimes wrong.** Half this
  corpus is `save` material, where one move holds and there is no win to find;
  hunting a knockout in one of those is solving a puzzle nobody set. The puzzle
  now says what it is asking for, with its evidence — the tablebase's category
  where there is one, otherwise the engine's own win% for the solver, labelled
  as an opinion.

The last one turned up a real data defect behind the display one. `motive` was
recorded from the *screening* pass, at the review budget, and never revised by
the 4M-node confirming pass that decides everything else about a puzzle. **31 of
215 `save` puzzles were winning outright** under the deeper look — one of them
at 100% — and were being announced as "only one move holds". The label is now
derived from the confirmed evaluation (`puzzles.motive_of`), in the generator
and in a v6→v7 migration for the corpus already on disk.

### The play tab reaches the gold standard now

The trainer's play tab only ever offered M1, the lab's own alpha-beta engine, on
the four calibrated node budgets. Fairy-Stockfish with the atomic net has been
in this repo since the first day as the thing everything is *measured against*,
and there was no way to play it. There is now: three unthrottled levels at
0.2s, 1s and 3s a move, on every core, in their own warm session rather than
sharing the analysis pool — an opponent wants all the threads on one move, and
a 3-second search must not sit in front of the explore board's next request.

They carry no Elo, deliberately. The ladder is anchored by pinning throttled FSF
at 64k nodes to 2000, so full-strength FSF sits above the top of its own scale
by construction and any number here would be invented rather than measured.

### The openings have names now, and the names are testable

A book of 48,464 positions keyed by 4-field FEN is not something anybody
memorises. `trainer openings` gives every line a name in the grammar standard
chess already uses — Family: Variation — and the grammar is the point, because
each suffix is a *detector over the position* rather than a flourish:

| Suffix | The claim it makes | How it is tested |
|---|---|---|
| `Defence` | Black's answer to move one | ply 1, and assertive by none of the below |
| `Gambit` | material is offered | a capture wins material **and survives the blast** |
| `Attack` | it does something to them | sortie into their half, check, or a profitable hit |
| `Counterattack` | Black takes the initiative back | the above, **plus measurably gaining ground** |

That last row is the one worth arguing about, and it is the one that is
measured. Black's score in the field *after* the move, against Black's score
across the position it was played in:

| Line | Black scores here | ...after the move | Verdict |
|---|---|---|---|
| 1.e4 e5 2.Bc4 **d5** | 46.2% | 65.4% | **Counterattack** (+19.3) |
| 1.Nf3 f6 2.Nd4 e6 3.Nb5 **Bb4** | 32.5% | 46.9% | **Counter-Gambit** (+14.4) |
| 1.Nf3 f6 2.e3 **d5** | 38.6% | 47.0% | **Counterattack** (+8.3) |
| 1.Nf3 **d5** | 41.0% | 12.0% | Defence (−29.0) |
| 1.e4 **e5** | 57.6% | 42.0% | Defence (−15.6) |

The third row agrees with what atomic players say about 2...d5 — it drops two
pawns for an attack — without having been told so. The last two are the same
kind of move, a pawn into the centre, and are not counters at anything; the
field's own numbers separate them from the first three. Where the book is
empty, nothing is called a Counterattack at all: a missing measurement produces
a weaker claim, never a guessed one.

The first row is worth a second look, because the trap scanner below reached
the same position independently: 1.e4 e5 2.Bc4 is a trap in which **2...d5 is
the only move that holds**. Two constructions with nothing in common — one
counting human results, one asking Fairy-Stockfish how many moves survive —
picked out the same move.

Two atomic-specific corrections fell out of building it. **A gambit test has to
run through the explosion rules** — 1...e5 against 1.Nf3 is not a pawn offer,
because 2.Nxe5 loses the knight with it — and **a central thrust counts for
Black and not for White**, or ordinary space-gaining moves like 4.e4 get named
"the e4 Attack".

Every name carries where it came from: `standard` for one standard chess
already gives that move order (1.Nh3 is the Amar Opening in any variant),
`community` for one atomic players use (the Kapsarov Attack for 1.Nf3 f6
2.Nc3), `descriptive` for one coined here and admitting it, `derived` for one
this grammar produced. Above all of them is whatever you type in the UI, which
then wins everywhere the line appears — including in the repertoire list, which
is now grouped by opening rather than by derivation order.

### Traps: where it is possible to go wrong, not where somebody did

Every puzzle generator here mines mistakes that *were made*. Preparation wants
the other question, so `trainer traps scan` walks the book instead of a history
and asks, of each position: how many moves keep the balance?

A trap is at most **two** moves within 12 win% of the best, with the best
alternative at least **25 points** worse *and* under **32%** — losing, not
merely worse — from a position that was tenable to begin with, and with an
escape that `M0-greedy` does not find by counting material one ply deep.

**866 book positions → 51 traps, in 90 seconds** (screen at 400k nodes, confirm
survivors at 2M, MultiPV 5 throughout). Rejections are as informative as
acceptances: 91 positions where the alternatives do not actually lose, 40
already lost, 30 where three or more moves hold, 11 trivial.

The column that decides whether a critical position is a *trap* is the one that
is not an engine opinion at all — `bait`, the share of book games from the
position that played a non-escape:

| Line | Holds | Everything else | Walk in |
|---|---|---|---|
| 1.e3 e6 2.Qf3 Nf6 3.Bd3 | **3...Bb4** (41%) | 3...g6 at 6% | **61%** of 3,475 |
| 1.Nf3 f6 2.Nd4 c6 3.Nf5 e6 4.Ne7 Qa5 | **5.c3** (51%) | 5.b4 at 0% | 45% of 6,414 |
| 1.e4 e5 2.Bc4 | **2...d5** (56%) | 2...f6 at 0% | 42% of 17,501 |

One move holding and 4% of the field finding it is a trap; one move holding and
96% finding it is a position with a well-known answer. Both are stored; only
the ordering tells them apart, which is why bait is the default order.

MultiPV 5 bounds what can be *concluded*: a position where six moves hold comes
back as "more than two", the right answer for the wrong reason, so `trap_scans`
records the escape count and the node budget beside every verdict.

### Less on the screen while you are thinking

The puzzle view had four cards beside the board, a fourteen-control filter bar
and the generation forms, all permanently open. One of those cards is about the
move you are trying to find. The rest is extraneous cognitive load in Sweller's
sense — attention spent on nothing being learned — and progressive disclosure
(Nielsen, 1995) is the standard answer to that shape: fold what is used
occasionally, and label the fold with what is inside it.

Filters fold to a summary line naming every active one; the generation forms,
the keyboard map and the explanation fold too. The provenance card is
*deferred* rather than folded — it opens when the puzzle ends, which it should
always have done, since "endgame, tablebase, one move" narrows a search you are
meant to be doing yourself. What is on screen while you think is a board, a
goal, and one sentence.

## Next

1. **M2 — NNUE distilled from FSF.** The 2x2 above makes a falsifiable
   prediction about it: a net that has to re-encode tactics a cheap quiescence
   already provides should show worse strength-per-parameter than its
   standard-chess equivalent.
2. **Tune M1's weights** before using it as a baseline for M3/M4. They are
   round numbers chosen as hypotheses and never fitted. Now that
   `king_blast` is known to be redundant with quiescence, the interesting
   question is whether *any* static term survives contact with the search, or
   whether atomic evaluation should be material plus mobility and nothing else.
3. **Re-measure the M0 baselines.** `M0-greedy` was declining mate in one (see
   the defect box above); the two rows it produced are labelled stale and want
   `lab.cli match M0-greedy M0-random --games 200` behind them.
4. **Lower rungs on the M1 ladder.** Difficulty ratings are built (above) and
   immediately named their own limitation: 74% of the corpus is solved by the
   weakest calibrated level, so 140 puzzles share the label `≤1366`. A
   `lab.cli levels` run extended down to ~250–1000 nodes would give that
   majority a spread, and costs one calibration rather than any new
   construction.
5. **Ablate the search, not just the eval.** Quiescence is worth more than
   every evaluation term combined, which suggests the remaining strength is in
   search structure. Null-move pruning was left out of M1 on the argument that
   its central assumption is weak in atomic; that argument is now testable.

## Open questions worth an experiment

- FSF+NNUE prefers **1.Nh3** at 200k nodes, while published human atomic theory
  says **1.Nf3** (worth ~+80 Elo). Is the theory behind the engines?
- Does distilling from FSF (M2) inherit its blind spots in a way self-play (M3)
  avoids? Measuring *where the two disagree* is a result in itself.
- Atomic is extremely forcing and tactical. Does that favour deep narrow
  alpha-beta over broad MCTS at equal compute? The Chalmers thesis assumed MCTS
  without testing the alternative — this is the most interesting comparison
  available.
