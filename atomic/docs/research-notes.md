# Atomic chess: what is actually known

Distilled from the two papers in `atomic_docs/` plus current sources. Every
claim here is sourced, because the whole project is a measurement exercise and
an unsourced "fact" that quietly shapes an eval function is a bug.

---

## 1. The rules, and what each one does to engine design

Lichess ruleset (what Fairy-Stockfish and both papers implement):

| Rule | Consequence for an engine |
|---|---|
| A capture removes victim, **capturer**, and every **non-pawn** on the 8 adjacent squares | Captures are multi-piece swings. Static exchange evaluation from standard chess is meaningless; you need an explosion-aware SEE. |
| Pawns are immune to blast damage (but can be captured directly) | Pawns become structurally special — they survive melees that clear every piece. |
| Kings cannot capture (suicide) | The king is a pure liability in melee but can safely walk into "attacked" squares. |
| Kings may be adjacent; neither checks the other | Standard king-opposition and king-safety code is wrong. |
| A capture that explodes **your own** king is illegal | Legality depends on the *post-explosion* board, not the pre-move board. |
| You may ignore check, move a pinned piece, or move into check **if** the move explodes the enemy king | Pin detection cannot prune moves the way it does in chess. This is the single most commonly mis-implemented rule. |
| En passant explodes around the **landing** square | Easy to get wrong; covered by the `en-passant-blast` perft position. |
| Win = enemy king exploded, or ordinary checkmate | Two distinct terminal conditions to detect at every node. |

**The structural consequence: king safety is non-local.** A king dies to a
capture happening one square away from it, by a piece that never attacked it.
"Is my king safe" is not a function of the squares attacking the king, so every
king-safety term inherited from standard chess is measuring the wrong quantity.
This is the biggest single reason a standard-chess engine transplanted into
atomic plays badly.

## 2. Piece values are compressed, ply-dependent, and not chess's

Pav (2025), *Inferring Piece Value in Chess and Chess Variants*
(arXiv:2509.04691), logistic regression over **1.5M filtered Lichess atomic
games** (Jan 2015 – May 2025), controlling for Glicko-2 skill with a SIMEX
correction for rating noise.

Uniform-random snapshot, **normalised to pawn = 1**:

| Piece | Atomic | Standard chess (traditional) |
|---|---|---|
| Knight | **1.9** | 3 |
| Bishop | **2.5** | 3 |
| Rook | **4.9** | 5 |
| Queen | **10.2** | 9 |

Three things matter more than the numbers themselves:

1. **Bishops are worth clearly more than knights** (2.5 vs 1.9) — a much larger
   gap than the slight bishop edge Pav finds in standard chess.
2. **Values peak around ply 60 and then *decline*** (Table 10). In standard
   chess they keep rising through ply 90. An eval with static piece values is
   wrong at both ends of the game, in opposite directions.
3. **Material is a weaker signal overall.** Pav's summary: atomic and antichess
   have "very compressed valuations, and are thus more *positional* than
   standard chess." Because a single capture can remove several enemy pieces
   and the king cannot capture, small material edges convert to wins more
   readily *and* material imbalance predicts outcome less well.

Raw coefficients (Elo-equivalent points, uniform snapshot, Table 9): pawn 21.8,
knight 41.4, bishop 55.0, rook 106, queen 222. Passed pawns are large:
rank-7 passer is worth 87 (and **284** in the 10–19 ply bucket).

**White's first-move advantage is ~3× standard chess** (44.1 vs ~15 in the same
units; 45.4 in the 10–19 ply bucket). Any match protocol that is not
colour-balanced in pairs is dominated by this bias, not by the thing it is
trying to measure. `lab.arena` enforces the pairing.

## 3. The state of the art

**Fairy-Stockfish 14 + `atomic-2cf13ff256cc.nnue`** is the gold standard.
The net is worth **+719 Elo** over FSF's own handcrafted atomic evaluation
(fairy-stockfish.github.io/nnue, author belzedar_, 2022-08-24). On this machine
it runs **~1.24 Mnps** single-threaded and evaluates the start position at
**+250 to +310 cp for White** — a striking number, and consistent with Pav's
finding that White's advantage is far larger in atomic.

Everything else publicly available is far behind. The strongest Lichess atomic
bots historically sit around 2600 (AtomicChessBot), and the atomic bot
ecosystem is thin enough that there is no meaningful engine leaderboard.

**This is why the goal is not "beat Fairy-Stockfish".** It is: put competing
methods on one scale anchored to throttled FSF, and find which ideas buy the
most strength per unit of compute.

## 4. The Chalmers thesis — the closest prior attempt, and its gaps

Adolfsson et al. (2025), *A Self-Trained Chess Engine for Atomic Chess*,
Chalmers/Gothenburg bachelor's thesis. AlphaZero-style: C++ / LibTorch,
12 residual blocks × 128 channels, PUCT MCTS, self-play RL.

**Result: 1729 Lichess Elo** (top 10% of atomic players) from ~450,000
self-play games over ~150 hours on rented VMs, total cost ~1,840 SEK (~$175).

Their numbers worth stealing:

- **16.8M training positions** total, versus AlphaZero's ~1.2B — i.e. **1.4%**
  of the data AlphaZero needed to beat Stockfish. Their own conclusion is that
  data volume, not architecture, was the binding constraint.
- **Where the time goes at play time:** NN inference **65%**, MCTS **33%**,
  move generation **<1%**. Movegen speed is not the lever; eval throughput is.
- **Batching MCTS selections** for GPU inference was the single biggest speedup,
  with **virtual loss** to stop parallel selections colliding (collisions per
  move dropped from 11,000 → 300 at 1,000 iterations).
- **Replay buffer is essential, not optional.** With a buffer holding only the
  last step's samples, the model could not beat a *randomly initialised*
  network after 8 hours (47% win rate). With a 100k–200k buffer and 4,096
  mini-batches against 2,048 new positions per step, it improved steadily.
  Sampling the *whole* buffer each step overfit badly.
- **Progress stalled after ~2,000 training steps.** Model-4000 was slightly
  *stronger* than Model-8000.

**Their documented weaknesses — each one is a research opportunity:**

| Weakness | Why it happened | Direct fix available |
|---|---|---|
| Strong openings, weak middle/endgame | Median atomic game is 30 ply, so self-play data is dominated by openings | Lichess atomic **tablebase** for endgame ground truth; start some self-play from mid/endgame positions; weight samples by ply |
| No time management | Fixed MCTS iteration count regardless of clock | Trivial to add |
| Idle during opponent's turn | No pondering | Reuse the subtree after the opponent moves |
| Trades down while behind in material | No positional understanding of resourcefulness | May also be *correct* atomic play — worth testing rather than assuming |
| Bare 12-channel input | No move count, repetition, en passant rights, or position history | Cheap input additions |
| No symmetry augmentation | Never implemented | Horizontal mirror doubles data for free |

## 5. Free ground truth: the Lichess atomic tablebase

Lichess's `lila-tablebase` serves **atomic** alongside standard and antichess.
Verified working:

```
GET https://tablebase.lichess.ovh/atomic?fen=4k3/8/8/8/8/8/8/QK6_w_-_-_0_1
-> {"category":"win","dtz":3, "moves":[...per-move dtz and category...]}
```

DTZ and per-move win/draw/loss categories, with `variant_win` / `variant_loss`
flags for king-explosion terminals. This is perfect, free supervision for
exactly the phase of the game the Chalmers engine was weakest in — and no
published atomic engine appears to use it for training targets.

## 6. Atomic opening theory (thin, but real)

- **1.Nf3 is worth roughly +80 Elo** for White; the plan is 2.Ng5 or 2.Ne5
  threatening to detonate the black king's neighbourhood. **1...f6 is the only
  good reply.**
- Forced early mates exist and are short — this is why self-play games end fast
  and endgames stay under-trained.
- Interestingly, FSF+NNUE at 200k nodes on this machine prefers **1.Nh3**, not
  1.Nf3. Worth investigating; the human theory may simply be behind the engine.

There is no reliable atomic opening book, which is why `lab.arena` uses random
legal opening lines rather than a book: a book would bake in whichever theory
its author believed.

---

## Sources

- Adolfsson, Lewis, Rahmn, Rajamäe, Rungardt, Tafani (2025). *A Self-Trained
  Chess Engine for Atomic Chess.* Chalmers / University of Gothenburg.
  (`atomic_docs/AtomicChess2025.pdf`)
- Pav, S. E. (2025). *Inferring Piece Value in Chess and Chess Variants.*
  arXiv:2509.04691. (`atomic_docs/inferring_piece_value.pdf`)
- [Fairy-Stockfish NNUE downloads](https://fairy-stockfish.github.io/nnue/)
- [Fairy-Stockfish playing strength](https://github.com/fairy-stockfish/Fairy-Stockfish/wiki/Playing-strength)
- [lila-tablebase supported variants](https://deepwiki.com/lichess-org/lila-tablebase/1.2-supported-chess-variants)
- [Lichess atomic variant page](https://lichess.org/variant/atomic)
- [Andy Tockman's atomic guide](https://tck.mn/atomicguide/)
