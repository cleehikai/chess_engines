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
| M0 baselines rated end-to-end | Done |
| Rust core (shakmaty + PyO3) | Done — compiles, perft-verified, ~630× python-chess |
| FSF ladder calibrated | Done — 40 games/pair |
| M1 (atomic HCE + alpha-beta, in Rust) | Done — rated, ablated |
| M2–M5 methods | Not started |

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
difficulty levels are real. The lesson for the harness is that **a gauntlet
against a much stronger ladder is the wrong instrument for questions about a
weak method** — `match --nodes-b` and `ablate` are the right ones, and both
exist now.

## Playing the engine

```powershell
py -3.13 -m lab.cli ratings   # every Elo measured so far, on one scale
py -3.13 -m lab.cli play      # browser UI, http://127.0.0.1:8770
```

The UI is served by `lab/server.py` (stdlib `http.server`, no new dependencies)
with `ui/index.html` as its only page. The server is the referee: it owns the
board, validates every move against the perft-checked
`chess.variant.AtomicBoard`, and computes explosion squares, so the page never
decides what is legal. Difficulty levels are node budgets, each labelled with
its *measured* rating — a level that has not been rated says so rather than
showing a guessed number.

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
3. **Ablate the search, not just the eval.** Quiescence is worth more than
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
