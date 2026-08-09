# Method catalog

The point of this project is not to build *an* atomic engine. It is to find out
**which way of thinking about atomic buys the most strength per unit of
compute**, by putting several methods on one measured scale.

Each method below has a hypothesis, a reason atomic specifically makes it
interesting, and a falsifiable prediction. A method is "done" when it has an
Elo with error bars in `results/ratings.jsonl`.

---

## M0 — Floor `shipped`

`M0-random`, `M0-greedy` (1-ply explosion-aware material).

Not engines; they exist so the harness is exercised end to end and every later
number has a known floor. **Already informative:** greedy material beats random
only **65%** (13/20). In standard chess the same comparison is far more
lopsided — a first, cheap confirmation of Pav's "material is a weak signal in
atomic" finding.

## M1 — Atomic-native handcrafted eval + alpha-beta

Classical search with an eval built *from atomic facts* rather than ported from
chess:

- Pav's piece values, **interpolated by ply** (they peak ~ply 60 and decline).
- **Blast-radius king safety**: score the enemy pieces that could reach a
  square adjacent to your king, not the squares attacking your king.
- **Explosion-aware SEE** for move ordering and quiescence.
- **Atomic quiescence**: extend on *every* capture, since any capture is a
  potential multi-piece swing or instant win (see `rules.is_quiet`).

*Hypothesis:* most of the gap between a naive port and a real atomic engine is
king safety, not material.
*Prediction:* M1 beats a chess-eval-ported control by >300 Elo at equal nodes,
and the ablation that removes blast-radius king safety costs more than the
ablation that removes atomic piece values.

### Result: the prediction was wrong, and the way it was wrong is the finding

Built and measured (see `PROGRESS.md` for the numbers). The prediction fails on
its central claim. Removing **material** costs 269 Elo; removing **blast-radius
king safety** costs 17 ± 109 — indistinguishable from nothing. The ranking is
exactly backwards from the hypothesis.

The follow-up experiment is what makes this useful rather than merely negative.
The obvious escape hatch — "the search is deep enough to see the blast, so the
static term is redundant" — predicts that the term pays at shallow depth and
decays as nodes rise. It does not: measured at 2k, 8k, 32k and 128k nodes, the
term is worth +6, +23, +17 and +0 Elo, flat across a 64× range.

What does explain it is *which part of the search* covers the term. A fuse is
lit by a capture, and quiescence searches every capture, so a capture-only
quiescence already spans the whole threat space the term describes — at every
depth, which is why the depth sweep is flat. Turning quiescence off and
re-running the same ablation moves king safety from +17 ± 109 to +83 ± 91 Elo.
That is the predicted direction and roughly a 5× increase, though with a
confidence interval that still touches zero it is support, not proof.

Meanwhile quiescence itself is worth **+203 ± 128 Elo** — more than every
atomic-specific evaluation term put together.

*Revised claim, to be tested by M2–M4:* in atomic, king safety is a **tactical**
property rather than a positional one, and therefore belongs in the search, not
the evaluation. This is a real difference from standard chess, where king safety
must be static because attacks assemble over more plies than any search can
reach. If it holds, it predicts something specific and awkward for M2: an NNUE
distilled from FSF will spend a large fraction of its capacity re-encoding
tactics that a cheap quiescence search already gets for free, and should
therefore underperform its standard-chess strength-per-parameter.

## M2 — NNUE distilled from Fairy-Stockfish

Label positions with FSF+atomic-NNUE evals, train an efficiently-updatable net,
search with alpha-beta. This is the highest strength-per-GPU-hour route and the
one the previous engine in this repo was building toward before it hit the
Python speed wall.

*Hypothesis:* distillation dominates self-play at this compute budget — the
Chalmers thesis spent 150 hours to reach 1729 Elo from scratch, while a teacher
that is already superhuman is sitting right there.
*Prediction:* M2 exceeds the Chalmers result (~1729 Lichess Elo equivalent) in
under 10 GPU-hours on the 5070.

**Open question worth the experiment:** distillation inherits the teacher's
blind spots. If FSF's atomic net has systematic weaknesses, M2 cannot see past
them and M3 can. Measuring *where* M2 and M3 disagree is a result in itself.

## M3 — AlphaZero-style self-play (Chalmers replication + fixes)

Reproduce the thesis, then apply its own list of future work:

- **Tablebase-backed endgame targets** from `lab.tablebase` — attacks the
  thesis's single biggest documented weakness (strong openings, weak endgames)
  with free exact labels. No published atomic engine appears to do this.
- **Ply-weighted sampling** so 30-ply median games stop starving the endgame.
- **Mid/endgame self-play starts**, not just the initial position.
- **Horizontal-mirror augmentation** — doubles data for free; they never got to it.
- **Richer inputs**: move count, repetition, en passant rights, position history.

*Hypothesis:* the thesis was data-starved, not architecture-starved (their own
conclusion), so throughput fixes beat model-size changes.
*Prediction:* the tablebase-labelled variant beats a faithful replication by a
margin that shows up specifically in games lasting past ply 40.

## M4 — Hybrids

- **NNUE eval inside MCTS** (cheap eval, PUCT search).
- **Policy-guided alpha-beta**: use a policy net purely for move ordering.

*Hypothesis:* atomic's extremely forcing, tactical character favours deep
narrow search over broad statistical search, so alpha-beta with a learned move
order beats MCTS at equal compute.
*Prediction:* at 64k nodes, policy-ordered alpha-beta > MCTS with the same net.
This is the most interesting comparison in the project, because the Chalmers
thesis assumed MCTS without testing the alternative.

## M5 — Knowledge injection

Not a search method — an additive layer testable on top of any of the above:

- **Endgame tablebase probing** at the leaf (the Lichess API, cached locally).
- **Opening book** mined from high-rated Lichess atomic games.
- **Time management** and **pondering** (the thesis lost real Elo to both).

*Prediction:* these are the cheapest Elo in the project, and pondering alone is
worth more than any eval refinement of comparable effort.

---

## Measurement protocol

Non-negotiable, because a sloppy protocol makes every number above worthless:

1. **Colour-balanced pairs** from shared random openings. White's advantage in
   atomic is ~3× standard chess; an unbalanced sample measures the colour, not
   the method.
2. **One budget currency** across compared methods (`nodes` by default). A
   method that quietly overspends its budget produces a meaningless Elo.
3. **Anchored to the calibrated FSF ladder**, with likelihood-profile
   confidence intervals. Report the interval, always.
4. **Ablations, not just totals.** "M1 scored 1400" says little; "removing
   blast-radius king safety costs 280 ± 60 Elo" is the actual finding.
