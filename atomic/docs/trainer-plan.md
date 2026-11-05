# Atomic Trainer — plan

A proposal for two features on top of the existing lab:

1. **A puzzle generator** for atomic, tagged by phase (opening / middlegame /
   endgame) and by atomic-native tactical theme, with *measured* difficulty.
2. **Account linking + personal game review** — import your games, find where
   you threw the advantage, replay from that exact position, and detect the
   mistakes you make repeatedly (especially in openings).

Nothing like either exists for atomic today, which is the reason to build it.

## Decisions taken (2026-08-10)

- **Lichess only.** Chess.com atomic is not machine-reachable (§1) and is out of
  scope; no PGN-import fallback and no scraper for now.
- **Corpus v1 = your own games + tablebase-derived endgames.** The
  rating-filtered dump slice stays in the plan (§3) but is not built yet. The
  dumps are still needed for the opening book and the cp→win% fit.
- **Both halves in parallel**, on a shared T0 substrate (§7).
- **Stdlib `http.server` + SQLite.** No new dependencies; the referee rule
  (server owns the position, page never decides legality) holds in every mode.
- **Repertoire drilling is in scope** (§6a) — not just mistake detection.
- **Analysis budget: 30–60 s/game, strongest available engine.** That is
  Fairy-Stockfish + `atomic.nnue`, and it buys a deep pass on *every* ply rather
  than a cheap scan (§3).
- **Corpus = all atomic games, rated and casual, all time.** Games against
  Lichess AI are tagged (`source: ai`) and excluded from mistake clustering by
  default — a loss to a level-1 bot is not a habit — but remain importable and
  reviewable.

---

## 1. What the research settles

### There is no atomic puzzle corpus, anywhere

Lichess's puzzle database is standard chess only — the generator
([lichess-puzzler](https://github.com/ornicar/lichess-puzzler)) runs over
`database.lichess.org` standard games and standard Stockfish. Chess.com's
Puzzles are standard only too. So a corpus has to be *generated*, and this repo
already owns the two things needed to generate one: **Fairy-Stockfish + the
atomic NNUE net** (1.24 Mnps here, +719 Elo over its own HCE) and a
perft-verified atomic rules implementation.

### Lichess is fully usable; Chess.com is effectively closed for atomic

| | Lichess | Chess.com |
|---|---|---|
| Atomic games via API | **Yes** — `GET /api/games/user/{u}?perfType=atomic` | **No** |
| Auth needed for own public games | **No** | n/a |
| Bulk historical corpus | **Yes** — 28.9M rated atomic games, 3.69 GB of `.pgn.zst` | No |
| Tablebase | **Yes** — atomic, to 6 pieces, already wired in `lab/tablebase.py` | No |

Verified live during this research:

- `curl "https://lichess.org/api/games/user/<user>?perfType=atomic"` returns
  atomic games as NDJSON **with no token at all**. Filters: `since`, `until`,
  `max`, `vs`, `rated`, `color`, `sort`, plus `moves`/`clocks`/`evals`/
  `opening`/`tags`/`pgnInJson`. Rate limit ~20 games/s anonymous, 30 with
  OAuth, 60 for your own history. **OAuth/PKCE is therefore optional** — worth
  adding later for the higher rate and for private/ongoing games, but the v1
  can just take a username.
- Chess.com's [Published-Data API](https://www.chess.com/news/view/published-data-api)
  documents its `rules` enum as `chess, chess960, bughouse, kingofthehill,
  threecheck, crazyhouse` — **atomic is not in it**. Confirmed empirically:
  `api.chess.com/pub/player/hikaru/stats` returns only
  `chess_daily, chess960_daily, chess_rapid, chess_bullet, chess_blitz, fide,
  tactics, puzzle_rush` — no variant perfs at all. Chess.com's atomic lives on
  the separate Variants platform (`chess.com/variants/atomic`), whose archive
  is web-only. There is no supported machine path to those games.
- `https://database.lichess.org/` publishes **atomic** monthly dumps: 28,859,338
  rated games, 139 files, 3.69 GB compressed, Jan 2015 – Jul 2026. This is the
  puzzle mining corpus.
- `explorer.lichess.ovh?variant=atomic` returned **401** on every request during
  this research (the explorer has been degraded since the Feb 2026 OVH
  incident). Do not build on it — build the opening book from the dumps
  instead, which we want anyway since atomic openings have no ECO names.

### The move-quality math has to be re-derived for atomic

Lichess classifies mistakes by **win% drop**, not centipawn drop, using
`Win% = 50 + 50 * (2/(1 + exp(-0.00368208 * cp)) - 1)`
([source](https://lichess.org/page/accuracy)); thresholds are roughly >10 pts
inaccuracy, >20 mistake, >30 blunder.

That constant is fitted to **standard** chess and is wrong here. This repo has
already measured the smoking gun: FSF+NNUE evaluates the atomic *starting
position* at **+250 to +310 cp**, a position White certainly does not win 85%
of the time. Applying the standard curve would classify half of normal atomic
play as a blunder. So step one of everything below is to **refit the logistic
for atomic** — sample positions from the dump, evaluate at a fixed node count,
regress against the actual game result. That is a one-off experiment that fits
the existing measurement-first protocol, and its by-product (the true cp→win%
curve in atomic) is a publishable finding in its own right.

### Puzzle acceptance rules worth copying, and where atomic breaks them

From `lichess-puzzler/generator.py`, the constants that matter:

- candidate = a move where win-chance improves by **> 0.6** and the position was
  not already winning (prev score ≤ +300cp), with a minimum winning eval of
  +200cp;
- **uniqueness** = best move's win chance > second-best + **0.7** (MultiPV ≥ 2),
  or tablebase says only-move, or forced mate;
- solution lines have **odd length** (you move, they reply, …), minimum 2 ply,
  and the final move must not itself be an only-move;
- deep verification limits (`depth=50, nodes=25M`) applied only to survivors.

Two atomic-specific breakages to design around:

1. **Triviality.** Atomic is so forcing that a large share of "big swing"
   positions are one-move king-blasts that any beginner sees. The uniqueness
   test does not catch these — they are unique *and* obvious. Fix: reject any
   candidate that the lab's own **`M0-greedy`** (one-ply explosion-aware
   material, already implemented) or a shallow M1 solves. Cheap and it uses
   code that already exists.
2. **Difficulty ratings.** Lichess rates puzzles by Glicko from thousands of
   human solvers. We have one solver. Fix below (§5) — rate puzzles against the
   **already-Elo-calibrated engine ladder** instead of against people.

---

## 2. Architecture

Keep `lab/` as the measurement workbench it advertises itself as, and put the
product on top of it as a sibling package that *imports* `lab` and adds no new
concepts to it.

```
atomic/
  lab/                     unchanged: fsf, rules, methods, arena, rating, tablebase
  trainer/
    winrate.py             cp -> win% for atomic (fitted), move classification
    analysis.py            FSF analysis service + SQLite cache keyed by (fen, nodes, multipv)
    ingest/
      lichess.py           username -> games (NDJSON), incremental since last import
      pgn.py               shared parse -> Game rows (also the future PGN-import path)
    book.py                atomic opening book built from the dumps; line identity + naming
    repertoire.py          repertoire trie, engine validation, coverage vs your history
    themes.py              atomic theme detectors (§4)
    generate.py            the puzzle pipeline (§3)
    difficulty.py          puzzle rating from the engine ladder (§5)
    review.py              per-game mistake report, recurring-error clustering
    srs.py                 FSRS scheduling over your personal error positions
    store.py               SQLite schema + migrations
  ui/
    board.js               extracted from index.html, shared by all three pages
    index.html             play (existing)
    puzzles.html           puzzle mode
    review.html            game review + replay-from-the-blunder
    drill.html             repertoire drilling
  data/trainer.sqlite      games, positions, puzzles, attempts, schedule
```

`lab/server.py` grows a router (it is a single `do_GET`/`do_POST` today) or,
better, `trainer/server.py` mounts the new routes and reuses `EngineHolder`,
`_state()` and the "server is the referee" rule — which must not be relaxed:
the page never decides what is legal, in puzzle mode either.

Storage is SQLite for the same reason `lab/tablebase.py` already uses it: no
new dependency, and the analysis cache is the expensive asset.

---

## 3. The puzzle pipeline

A three-stage funnel, because deep analysis is the whole cost.

**Stage 0 — corpus.** Three independent sources, each producing different
puzzles. **v1 builds the second and third only**; the first is deferred (T5).

- *Human games* from the lichess dumps, filtered to a rating band (say both
  players ≥ 1800 atomic) so the losing side's mistake is a real mistake and not
  noise. This is the middlegame/opening supply. **Deferred.**
- *Your own games* (§6) — the personally relevant supply.
- *Tablebase*, which nobody else has used for this: enumerate ≤6-piece atomic
  positions, keep those where the tablebase says exactly one move preserves the
  win/draw, and you get **endgame puzzles with exact ground truth and no engine
  error at all**. This is the cleanest possible answer to "endgame puzzles",
  and `lab/tablebase.py` already speaks the protocol.

**Stage 1 — deep pass over every ply.** FSF + `atomic.nnue`, MultiPV=2, at a
node count calibrated to land in the 30–60 s/game budget. At 1.24 Mnps and a
30-ply median, that is roughly **1M nodes/ply** — and MultiPV=2 on every ply
means the uniqueness gap is already known everywhere, not just at the swings.
Keep plies where win% swings by more than the atomic blunder threshold; reject
anything `M0-greedy` finds.

Two consequences of spending this much up front, both good: the analysis cache
becomes the asset (re-review and puzzle generation over an imported game are
free afterwards), and game review and puzzle mining stop being separate passes.
The budget is a config knob, not a constant — the cache records the node count
each verdict was made at, so a corpus analysed cheaply can be re-deepened later
without being thrown away.

**Stage 2 — confirm** (~2–4M nodes, MultiPV=3) on survivors only: re-check the
swing at higher nodes, confirm uniqueness (best vs second-best win-chance gap),
and extend the solution line by alternating best-play until the line stops being
forced. Tablebase-resolve any leaf with ≤6 pieces.

*(If the dump-mining phase T5 ever happens, stage 1 there reverts to a cheap
~20k-node scan — 1M nodes/ply over 100k games is weeks of CPU. The two-tier
funnel stays in the code for exactly that reason.)*

**Stage 3 — annotate and store**: themes (§4), phase, opening line (§4),
difficulty (§5), source game link, and the *defender's* best tries so the UI can
punish a wrong second move properly rather than just saying "wrong".

Cost estimate on this box: FSF does 1.24 Mnps, atomic games median 30 ply. Stage
1 at 1M nodes × 30 ply ≈ **24 s/game** (MultiPV=2 costs some throughput, so call
it 30–40 s — the budget asked for); stage 2 at ~1.5 surviving plies ≈ **10
s/game**. A 1,000-game history is therefore a **~12-hour single-core import**,
or about 2 hours across 6 cores — a one-time overnight run, incremental
thereafter (a day's new games are a couple of minutes). Worth building the
importer to be resumable and parallel across games from the start; it is the one
place in this design where an interruption costs real time.

---

## 4. Concepts: phases, themes, openings

**Phase** is mechanical: opening = still inside the book (§ below) or ply ≤ 16;
endgame = ≤ 8 pieces or tablebase-reachable; middlegame = the rest. Atomic's
median game is 30 ply, so these boundaries are much earlier than in standard
chess and must not be inherited from it.

**Themes** need their own taxonomy — standard tactical themes (fork, pin, skewer,
discovered attack) largely do not survive contact with atomic, where a capture
destroys the capturer. A first list, each of which is a *detector function over
the solution line*, testable against the rules helpers already in `lab/rules.py`:

| Theme | Detector sketch |
|---|---|
| `king-blast` | solution capture's explosion square set contains the enemy king |
| `blast-fork` | one capture destroys ≥ 2 enemy non-pawn pieces |
| `regicidal-pin-break` | the winning move is illegal-looking: moves a pinned piece / ignores check, legal only because it explodes the king |
| `own-king-in-blast` | key defensive resource is that the opponent *cannot* capture without self-destructing |
| `connected-kings` | kings adjacent; the resulting no-capture zone is the point |
| `ep-blast` | en-passant, blast centred on the landing square |
| `pawn-immunity` | the point turns on pawns surviving explosions |
| `defensive-diffusion` | best move scatters pieces out of a future blast radius |
| `sacrifice-to-open-fuse` | a losing-material move that creates a forced blast next ply |
| `no-capture-fortress` | position is drawn/held because no capture is available that doesn't lose |
| `tb-conversion` | ≤6 pieces, exactly one move holds the tablebase category |
| `opening-trap` | occurs within book depth, from a known line |

These are hypotheses, exactly like the eval terms in `docs/method-catalog.md`,
and should be validated the same way — a theme that never fires, or that fires
on 40% of puzzles, is a bad theme and gets rewritten.

**Openings.** There is no ECO for atomic and the explorer is down, so build the
book from the dumps: a trie of positions to ply ~20, with game counts and score
by rating band, pruned to lines with ≥ N games. Note this means the dumps are
needed even though *mining* them is deferred — but only the first ~20 ply of
each game, which is cheap to extract and needs no engine. Assumption unless you
say otherwise: **the 12 most recent monthly files** (~300 MB compressed, streamed
and discarded, no full 3.69 GB download), which keeps the book representative of
how atomic is played now rather than in 2015. Line *identity* (so "I keep
losing this" can be computed) is the deepest book node the game reached. Line
*names* are a small hand-curated YAML seeded from community theory —
1.Nf3 f6 2.e3, 1.Nf3 f6 2.Nd4, 1.Nf3 f6 2.Nc3 Nh6 3.g4, 1.e3 — with an
auto-generated fallback label (`1.Nf3 f6 2.Nd4 — 14k games, 61% White`) so a
line without a name is still usable. Note the repo's own open question here:
FSF prefers **1.Nh3** while human theory says **1.Nf3**; the book will show
which the human data actually plays, and that disagreement is a puzzle theme in
itself.

---

## 5. Puzzle difficulty, measured rather than guessed

This is where the repo has an unfair advantage and should not imitate lichess.
`lab.cli levels` already produces an **Elo-calibrated ladder of engine
settings** (M1 at 2k/16k/64k/1024k nodes = 1366 / 1691 / 1704 / 1951, chained
from one anchor). So:

> A puzzle's difficulty is the rating of the weakest calibrated level that
> finds the whole solution line.

Give each puzzle to each rung; the crossover point is a difficulty in the same
Elo units as everything else in `PROGRESS.md`. It costs a few seconds per
puzzle, needs no users, and — unlike a Glicko bootstrap — is honest about its
error bars from day one. The known caveat from `PROGRESS.md` applies and is
already understood: use the *chained* level ratings, never per-budget gauntlet
ratings.

Refinement worth measuring later: engine-solvable and human-hard are not the
same axis (an engine finds a 6-ply forced blast instantly that a human never
sees). Once there is personal solve data, fit a correction from your own
attempts and report both numbers.

**Built 2026-08-11** as `trainer/difficulty.py`; see §10 for what measuring it
turned out to say about the ladder rather than about the puzzles.

---

## 6. Account linking and personal review

**Import.** `trainer ingest lichess <username>` pulls all `perfType=atomic`
games as NDJSON, incrementally (store the last `createdAt`, pass `since`), no
token required; an optional OAuth-PKCE flow can be added later for rate and for
private games. **Chess.com is out of scope** — there is no supported machine
path to its atomic games (§1), and a scraper against the Variants archive would
be fragile and against their ToS. If Chess.com games become important later, the
cheapest honest option is a PGN file/paste importer, which the ingest layer is
structured to accept without rework.

**Analyse.** Every imported game goes through the same stage-1/stage-2 machinery
with the atomic thresholds from §1, producing per-ply: eval, win%, delta, the
best move you missed, and a classification.

**The three outputs you asked for:**

1. **"Where I threw it."** Per game: the plies where win% fell the most, ranked;
   "you were +X and after this move you were −Y". Click one and the board opens
   at the position *before* the mistake.
2. **Replay from the critical position.** The existing play server already
   accepts an arbitrary FEN, so this is mostly UI: play out the position you
   ruined against the engine at a chosen level, from your side, and it tells you
   whether you converted this time. The same machinery gives "save the lost
   game" — start from a position where you were losing but the engine says it
   was holdable.
3. **Recurring mistakes.** This is the interesting one, and openings are where
   it will pay. Cluster your errors by (a) book line identity, (b) theme, (c)
   the blast pattern involved. Output: "in 1.Nf3 f6 2.e3, you have played
   3...Nh6? in 9 of 14 games and scored 21%", or "you miss `regicidal-pin-break`
   defences: 12 occurrences, 2 found". Each cluster becomes a personal puzzle
   set.

## 6a. Repertoire drilling

Mistake detection tells you *where* you leak; drilling is what stops it. In
scope, and atomic makes it easier than standard chess in one respect and harder
in another: games are short (median 30 ply) so a repertoire is shallow — 10–14
ply covers most of the useful theory — but there is no published repertoire to
import, and no ECO to organise it by.

**Repertoire model.** A repertoire is a set of lines per colour, stored as a
move trie over positions (not move lists — transpositions matter and atomic's
opening tree is narrow enough that they are common). Each node is: position,
your intended move, optional note, and a source. Sources, in the order they
should be supported:

1. **Derived from your own results** — the book (§4) plus your game history
   already knows which lines you reach and how you score. The first repertoire
   is generated, not authored: "you reach this position 40 times; here is the
   move that scores best in the dumps and here is what you actually play."
2. **Authored/edited** in the browser, on the existing board UI.
3. **Imported** from a Lichess study (`GET /api/study/{id}.pgn`, public, no
   auth) — this is how atomic theory is actually published today, so it is the
   real import path.

**Drilling.** You play the repertoire side; the trainer replies with a weighted
sample of what opponents actually play at your rating band (from the book), not
just the main line. Wrong move → shown the intended one, card scheduled by
FSRS. A repertoire node is *validated* against FSF: if your intended move loses
more than a threshold of win% versus the engine's best, the trainer flags it
rather than drilling you into a bad move. That check is the feature that a
generic repertoire trainer cannot offer here, because it requires an atomic
engine — which this repo has.

**Coverage report.** Cross the repertoire against your game history: which
positions you reach often but have no line for, ranked by frequency × how badly
you score there. That is the direct answer to "I make the same opening mistakes"
— it names the gap instead of waiting for you to lose again.

**Retention.** Personal error positions feed an FSRS scheduler
(`open-spaced-repetition`'s algorithm — the same one Anki 23.10+ and the chess
trainers Chessdriller/NeuralPGN use; ~20–30% fewer reviews than SM-2 at equal
retention). One card = one position you got wrong; grading comes from whether
you found the move unaided.

---

## 7. Suggested order of work

Both halves run in parallel off one shared substrate. T0 is the fork point —
nothing on either branch is trustworthy before it exists.

| Phase | Deliverable | Why |
|---|---|---|
| **T0** | atomic cp→win% fit, move classification, FSF analysis service + SQLite cache, store schema | shared by both branches; everything downstream is wrong without the refit |
| **T1a** | lichess ingest (username, incremental) + `trainer review <game>` in the terminal | fastest path to something used daily |
| **T1b** | tablebase endgame puzzle generator + theme detectors for `tb-conversion` | independent of T1a, exact ground truth, no engine error to debug |
| **T2a** | review UI + replay-from-the-blunder | turns T1a into practice instead of a report |
| **T2b** | game-derived puzzle generator (stage 1/2 funnel) over your own games | reuses T1a's analysis cache — every reviewed game is already scanned |
| **T3a** | opening book from the dumps + recurring-mistake clustering | the "I make the same opening mistakes" feature |
| **T3b** | full theme detectors + difficulty via the engine ladder | makes the corpus browsable by concept |
| **T4a** | repertoire: derived-from-history v1, engine validation, coverage report (§6a) | needs T3a's book and clustering; delivers the fix, not just the diagnosis |
| **T4b** | puzzle UI | needs a corpus to browse |
| **T5** | FSRS over one queue: error positions, puzzles, repertoire nodes | needs all three to exist |
| **T6** | repertoire editor + Lichess study import | authoring is only worth building once drilling is proven |
| **T7** | ~~*(deferred)*~~ rating-filtered dump mining for a general corpus | **built** — `trainer puzzles mine`, see below |

The two branches converge at T5, where personal errors, generated puzzles and
repertoire nodes become one scheduled review queue.

## 8. Risks

- **Analysis quality ≠ truth.** FSF+NNUE is the best available atomic judge but
  it is not a tablebase; puzzles mined from it inherit its blind spots (the
  1.Nh3 vs 1.Nf3 disagreement is a live example). Mitigation: tablebase-verify
  every endgame puzzle, and record the node count each verdict was made at so
  the corpus can be re-verified later rather than being frozen wrong.
- **Compute.** Mining the dumps is deferred (T5). v1's corpus is your own games
  plus tablebase endgames, which is minutes of CPU, not nights.
- **A corpus of your own games is small and self-selecting.** It will be thin on
  positions you never reach, and its themes will mirror your habits — which is
  the point for training but means the theme detectors get validated on a
  biased sample. Worth re-checking them against a dump slice before trusting the
  taxonomy (T5).
- **Chess.com.** Genuinely blocked, see §1; out of scope by decision.
- **Import is now a long job.** A deep pass on every ply at 30–60 s/game turns
  the first import into an overnight run (§3). Make it resumable, parallel, and
  incremental from day one, or it will be the thing that makes the tool annoying
  to use.
- **A derived repertoire can teach you the dumps' bad habits.** "What scores
  best at your rating band" is a popularity statistic, not truth — hence the FSF
  validation pass on every repertoire node (§6a). Where the two disagree is
  worth surfacing rather than silently resolving; the repo's existing 1.Nh3 vs
  1.Nf3 question is exactly this disagreement at ply 1.
- **Scope creep into `lab/`.** The lab's value is that its claims are measured.
  Keep the trainer out of it; import, never modify.

---

## 9. What T7 turned out to need (built 2026-08-11)

The deferred dump-mining phase is now `trainer/mine.py`. Three things the plan
above did not anticipate, all of which changed the design:

1. **Deduplication by position is not housekeeping, it is the feature.** §4
   already noted that atomic's opening tree is narrow and that 20 of 86 games
   open 1.Nf3 f6; the consequence for mining is that unrelated games reach the
   same critical position constantly, and a naive scrape produces one tactic a
   dozen times over. Puzzles are keyed on the first four FEN fields — so a
   transposition is the same position — and the check runs *before* stage 2,
   because a duplicate caught afterwards has already cost 4M nodes. The key
   carries the game and ply each copy came from, so regenerating your own
   corpus at a higher budget still refreshes rows rather than refusing them.
2. **Scraped games must not enter `games`.** Every personal feature in §6 is a
   query over that table. They go in `mined_games`, which also records games
   that produced nothing, because "scanned and rejected" is the fact a resumed
   run needs.
3. **The scan budget is a recall knob and should be labelled as one.** 20k
   nodes is a rung of the calibrated ladder (1892 Elo), not a guess, and
   everything it proposes is re-judged at 4M — but what it *misjudges* never
   reaches that pass and is lost without a trace. §8's "analysis quality ≠
   truth" risk applies twice over here, so the node count is stored on every
   row and the budget is a parameter.

§5's difficulty ratings were the obvious missing axis after this: a corpus
mined from strangers' games has no personal relevance to order it by. They are
built; §10 is what building them found.

---

## 10. What §5 turned out to measure (built 2026-08-11)

`trainer/difficulty.py` implements §5 unchanged in construction — every puzzle
against every calibrated rung of the M1 ladder, difficulty is the weakest one
that finds the whole line — and four things came out of it that the plan did
not anticipate.

1. **The ladder's floor, not the puzzles, is the binding constraint.** 140 of
   188 puzzles are solved by M1 at 2,000 nodes, and that rung is rated 1366
   Elo. Below it this construction is blind, so 74% of the corpus shares one
   label. The fix is a calibration, not a redesign: `lab.cli levels` extended
   down to ~250–1000 nodes. Until then a floor rating is written `≤1366` and is
   a bound, exactly as `lab.rating` writes a shutout gauntlet.
2. **Run every rung, not up to the first success.** §5 implies an early exit.
   The measured ladder does not support one: 16k and 64k nodes came out at
   1691 ± 131 and 1704 ± 135, one rung inside the other's error bar, and 6
   puzzles were duly solved by a level that a stronger one missed. That is
   information about the ladder and is stored (`monotone = 0`) rather than
   resolved silently. It costs ~0.46 s per puzzle to keep, which is nothing.
3. **The engine has to be held to the solver's exact standard.** Accepting any
   move that also wins would rate the *position*; requiring the stored solution
   move at every solver ply — the same test `server.puzzle_move` applies to you
   — rates the puzzle. This matters more in atomic than it would in standard
   chess, because the uniqueness filter already guarantees the alternatives are
   20 win% points worse and the temptation to accept them is correspondingly
   larger.
4. **Ratings cannot live on the `puzzles` row.** `puzzles.save` is `INSERT OR
   REPLACE`, so regenerating the corpus at a higher budget would erase them.
   They go in `puzzle_ratings`, keyed by the content-hash id, for the same
   reason `puzzle_attempts` is a separate table — and each verdict carries a
   fingerprint of the ladder it was measured against, so recalibrating the
   levels makes the old ratings countably stale instead of quietly
   incomparable.

The refinement §5 left for later is present in its cheapest form: the
improvement report tabulates your unaided solve rate *by* engine difficulty, so
the two axes sit beside each other. No correction is fitted — 20 attempts do
not support one — but the disagreement is now visible rather than assumed.

## 11. What using it found (2026-08-12)

Four defects, none of which produced an error message, and one of them was in
the data rather than the display.

1. **`motive` was a screening guess that nothing ever revised.** §3's funnel
   picks candidates at the review pass's budget and re-judges everything at 4M
   nodes — everything except the label saying what the solver is playing for.
   31 of 215 `save` puzzles were winning outright at the confirming budget, one
   of them at 100%, while the page told the solver "only one move holds". The
   label is now derived from the confirmed evaluation, and — since the corpus
   is expensive and must not be regenerated to fix a string — by migration for
   the rows already stored. The general rule this is an instance of: **any
   claim shown to the solver must come from the deepest pass that judged the
   position**, not from whichever pass first produced it.
2. **A `save` puzzle needs its goal stated, not implied.** Half this corpus is
   hold-the-draw material, which was the right call (§3) and is unusual enough
   that no solver arrives expecting it. Saying "only one move holds" in the
   verdict line is not enough: the puzzle now states the objective with its
   evidence, and the two kinds of evidence are visibly different — the
   tablebase's category is exact, the engine's win% is an opinion and is
   labelled as one. §8's "analysis quality ≠ truth" risk is a *user-facing*
   fact here, not just a note in a plan.
3. **A filter that cannot name a source hides it.** The scraper (§9) landed
   without its option in the source menu, so 276 of 429 puzzles could only be
   reached by luck of the ordering. `SOURCES` is three long and the UI knew two
   of them; worse, it displayed "mined" for the one source that was not mined.
4. **`unsolved` is the redo rule, not the no-repeats rule.** Puzzles came round
   again because the only exclusion available was "never solved cleanly", which
   is exactly what a redo queue wants and exactly what a new session does not.
   A stricter `unseen` is now the default and the redo queue is an explicit
   request.

§6's replay-from-the-blunder also gained the opponent it was missing: the play
tab can now be set to Fairy-Stockfish + `atomic.nnue` unthrottled, rather than
only to the throttled M1 ladder. Notably it carries no Elo — the ladder is
anchored *against* that engine (§5), so it sits above the top of its own scale
by construction.

## 12. Names, traps, and less on the screen (2026-08-15)

Three things, all of them consequences of the corpus finally being big enough
to be unusable by hand.

### 12a. §4's "small hand-curated YAML" was the wrong shape

The plan said line *names* would be a seed file with an auto-generated
fallback label (`1.Nf3 f6 2.Nd4 — 14k games, 61% White`). Building it showed
that a statistic is not a name: it changes when the book is rebuilt, it sorts
by nothing, and nobody remembers one. What a name has to do is be *stable* and
*structured*, which is what standard chess's grammar already provides — so
`trainer/openings.py` reproduces the grammar instead of the label, and every
suffix in it is a detector over the position, in exactly the sense §4's themes
are:

| Suffix | Test |
|---|---|
| `Defence` | Black's reply at ply 1, assertive by none of the tests below |
| `Gambit` | after the move a capture wins material *and survives the blast* |
| `Attack` | White sorties, checks, or hits a piece it can profitably take |
| `Counterattack` | the same by Black, **and** it gains ground on the field |
| `Counter-Gambit` | a Gambit that also gains ground |
| `Variation` | nothing fired |

Four things came out of writing it that the plan did not anticipate.

1. **"Counterattack" is a claim, so it has to be measured.** The obvious
   implementation — call any sharp Black move a counterattack — would name half
   the tree after an adjective. The test used instead is the book's own
   numbers: Black's score after the move against Black's score across the
   position it was played in. `1.Nf3 f6 2.e3 d5` lifts Black from 38.6% to
   47.0% and earns the word, which is also what community theory says about
   that move; `1.Nf3 d5` is the same kind of move and drops Black from 41.0% to
   12.0%, so it is a Defence and nothing more. Where there is no book, nothing
   is called a Counterattack at all — a missing measurement makes a weaker
   claim, never a guessed one.

   The check that this is measuring something real: `1.e4 e5 2.Bc4 d5` comes
   out a Counterattack at +19.3, and §12b's trap scan — which counts no human
   results at all — independently finds that 2...d5 is the *only* move holding
   that position.
2. **A gambit test has to go through the explosion rules.** In standard chess
   "left en prise" is a static question about defenders. Here the capturer dies
   with its victim, so half of what looks hanging is not: 1...e5 against 1.Nf3
   is not a pawn offer, because 2.Nxe5 loses a knight for it. The predicate
   plays every capture out.
3. **A central thrust counts for Black and not for White.** The first version
   let a pawn's double step into the centre make a move assertive for either
   side, and promptly named ordinary space-gaining moves like 4.e4 "the e4
   Attack". Hitting back in the centre is a *counter*; playing in the centre is
   just playing.
4. **The last move is not enough to name a line by.** Naming a line after its
   final branching move gave `1.e4 d5` and `1.e4 Nf6 2.Bc4 d5` the identical
   name. Names accumulate down the tree and elide at three parts, which is
   where this book stops colliding.

The provenance column earns its place. `standard` (the name standard chess
gives that exact move order — 1.Nh3 really is the Amar Opening), `community`
(names atomic players use: the Kapsarov Attack for 1.Nf3 f6 2.Nc3), and
`descriptive` (coined here, and saying so) are three different amounts of
authority, and a coined name that does not admit it is a small lie that
compounds. Above all three sits whatever you type in the UI, which is the
feature this was asked for: a repertoire of forty positions identified by
4-field FEN is a repertoire nobody can memorise.

### 12b. Traps are the question the miners cannot ask

`generate.py` and `mine.py` both find mistakes that *were made*. Preparation
needs the other question — where in the tree it is possible to go wrong — and
that is a walk over the book rather than over any history, so `trainer/traps.py`
is the first generator here whose cost is set by the size of the tree.

A trap: at most two moves within `HOLD_LOSS` of the best, the best alternative
at least `SEVERITY` worse *and* under `LOST_CEILING`, the position tenable to
begin with, and the escape not something `M0-greedy` finds — the same
triviality oracle the puzzle corpus uses (§3).

Two things worth recording.

1. **The number that makes it a trap rather than a critical position is not an
   engine number.** `bait` is the share of book games from the position that
   played a non-escape, counted from 3.7M rated games. One move holding and 4%
   of the field finding it is a trap; one move holding and 96% finding it is a
   position with a well-known answer. Both are stored; only the ordering
   distinguishes them, and it is the default ordering for that reason.
2. **MultiPV bounds what can be *concluded*, not just what can be seen.** A
   position where six moves hold comes back as "more than two hold" because
   five is all that was asked for. That is the right answer for the wrong
   reason, so `trap_scans` records the escape count and the budget beside every
   verdict — the same discipline §9.3 arrived at for the scraper's scan budget.

First run over the popular part of the book: 866 positions, 51 traps, 90
seconds. The worst is 1.e3 e6 2.Qf3 Nf6 3.Bd3, where only 3...Bb4 holds and 61%
of 3,475 games did something else; 1.e4 e5 2.Bc4 is the most-played one, where
only 2...d5 holds and 42% walk in.

### 12c. What the puzzle page was actually costing

The solving view had grown four cards beside the board — the verdict, the
engine, the puzzle's provenance and themes, the session's keyboard map — plus a
fourteen-control filter bar and the generation forms, all of it permanently
open. Only the verdict is about the move you are trying to find. The rest is
*extraneous* load in Sweller's sense: it competes for attention without
contributing anything to the thing being learned, and Nielsen's progressive
disclosure (1995) is the standard answer to exactly this shape — hide what is
needed occasionally behind one click, and label the click with what is inside.

So: the filter bar folds to a summary line naming every active filter, the
generation forms and the keyboard map fold, the explanation folds (it is worth
reading once, not on every visit), and the provenance card is *deferred* rather
than folded — it opens when the puzzle is over, which it should have done
anyway, since "endgame, tablebase, one move" narrows a search you are supposed
to be doing yourself. What is left while you think is a board, a goal, and a
sentence.

The same instinct fixed the repertoire list, which had been ordered by
`(ply, fen)` — the order the lines were derived in, and not an order anybody
thinks in. It is grouped under the opening each line belongs to, which is the
shape a repertoire is actually memorised in.
