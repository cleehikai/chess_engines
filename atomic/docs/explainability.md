# Making the engine explain itself — research and design

The problem, stated precisely: when the engine plays a move that does not make
sense to a human, the trainer shows an eval and a principal variation, and
neither one says **why**. This document surveys what the research can do about
that, works out which parts of it survive the move to atomic, and proposes a
staged plan that fits the rest of this lab: every stage has a hypothesis and a
measurement.

Sourcing caveat: arxiv.org was unreachable from the session that wrote this, so
paper details come from abstracts, search summaries, and, for Queen, its public
source code. Anything not checked against a paper body is marked *(unverified)*.

---

## 1. What the research has actually done

### 1.1 Queen: an LLM wired directly into the engine's activations (2026)

This is almost certainly "the paper where they trained an LLM tied to the
weights of the engine": *Language Models that Play Chess and Explain Their
Moves* (Princeton NLP, arXiv 2610.03695, Oct 2026). Code is at
`github.com/CDittmeier/queen`, and weights are in the
`princeton-nlp/queen-chess-models` HF collection.

What it is, read from `models/flamingo.py` and the repo layout:

| Part | Detail |
|---|---|
| Encoder | **Frozen** Lc0 BT5 transformer: 64 square tokens, width 1024 |
| Decoder | **Frozen** SmolLM3-3B (a Gemma-3-4B adapter also exists) |
| Bridge | 16 Flamingo-style gated cross-attention sublayers. Sublayer *i* reads encoder layer *i* and sits before decoder layer ≈ 2*i*. Each one is LN → MHA (Q from the text, K/V from the 64 squares) → `tanh(α)` gate → FFN → second gate. `W_O` starts at zero, so the frozen LM is unchanged at step 0. |
| Trained | Bridge plus new token embeddings only. There is an optional LoRA path. |
| Total | ~4B parameters |

The key idea is that the LM never sees a FEN it has to parse. It reads the
engine's **internal square representations at every depth**, so whatever Lc0
has computed about a square (attacked, hanging, part of a plan) is available
to the words.

The curriculum (`datagen/tasks/`, `configs/sample_instances/stage1..4`) is the
part to copy most directly:

1. **Board facts**: piece on square, square of piece, pieces on a
   file/rank/diagonal, piece count, material count.
2. **Move facts**: a piece's moves, a square's attackers, capture moves,
   checking moves, check parries, is-checkmate.
3–4. **The same questions *after* a given move sequence** ("forward" tasks).
   The bridge has to learn to update the board through the moves, with
   move-by-move chain-of-thought.
5. **Reasoning**: HCE motif annotation, HCE rationale from a search pipeline,
   verdict classification, plans from self-play games, and "sim rationale" (a
   plan-model search narrative).

Then comes **self-distillation** (`datagen/self_distill/`): the model writes
analyses, Stockfish at 100k nodes acts as the referee, and analyses with
mistakes or eval claims that disagree with the truth beyond a tolerance are
dropped (`--mistake-drop`, `--eval-truth-limit`). Accepted child analyses are
consolidated into root supervision, and the model retrains. Reported: **1782 →
2697 Elo over seven iterations**. Explanations are fluent and approach a
frontier LLM in coherence, but are **weaker in conceptual coherence on chess
motifs**. A `sol_distill` stage suggests the initial rationale data was
distilled from a frontier LLM *(unverified)*.

Two findings bear directly on our use:

- **Queen often disagrees with Leela's own move.** It is a new player that
  reads Leela's features, not a narrator of Leela's decisions. If the goal is
  "why did *the engine* play this", that is the wrong guarantee, and the design
  in §4 has to fix it.
- The encoder must be a **token-per-square transformer**. Nothing like that
  exists for atomic (see §3).

### 1.2 Grounding an LLM in engine-extracted concepts (CCC, 2024)

Kim et al., *Concept-guided Chess Commentary* (arXiv 2410.20811). They extract
concept vectors (king safety, pawn structure, …) from an expert engine's
activations, rank which concepts matter most for the move, and put the
top-ranked ones into the LLM prompt. Human judges rated the result above the
baselines, and on most metrics above the reference human comments. It also
introduced GCC-Eval, an LLM judge.

There is no training of the LM at all. It is cheap, and it is the closest
template for a first learned-concept explainer.

### 1.3 How badly ungrounded LLMs hallucinate (ACT-Eval, 2026)

Hebbar, Sheng, Oh, Viswanath, *Hallucinations on the Board* (arXiv
2608.04240). They split commentary into atomic claims and check each one with
15+ board and engine tools. Even in **standard** chess, GPT-5.4 without tools
gets **22%** of sub-claims wrong, and small open models get over 40%. LLM
judges hand out near-perfect scores too easily. Tools fix much of the factual
error but not the strategic coverage.

For atomic this is the most important number in this document, and it gets
worse. A general LLM's chess knowledge is standard-chess knowledge, and
`trainer/themes.py` already documents why that vocabulary breaks here: a fork's
collector dies with its victim, a pin binds nothing if moving the pinned piece
explodes the king, and kings may stand adjacent. Expect the hallucination rate
on atomic to be far above 22%, and the errors to be *confident rule errors*.
Verification is therefore a hard requirement, not an option.

### 1.4 Reading concepts out of engine networks

- **McGrath et al., *Acquisition of chess knowledge in AlphaZero*** (PNAS
  2022). Linear probes recover human concepts from AlphaZero's layers, and
  probing over training time shows when each concept appears.
- **Schut et al., *Concept discovery and transfer in AlphaZero*** (PNAS 2025).
  Concept vectors are found by convex optimisation and filtered for
  **teachability** (another agent can learn them) and **novelty** (absent from
  human games). Four world-champion-level GMs improved on held-out positions
  after studying the concept prototypes.
- **Jenner et al., *Learned look-ahead in Leela*** (NeurIPS 2024). Leela
  represents future optimal moves internally, and that representation is
  causally used: activation patching moves the output.
- **Sparse autoencoders and transcoders on Lc0.** Contrastive SAEs for planning
  (Poupart 2024, 2406.04028), sparse replacement layers with causal circuit
  tracing (*Tracing the Thought of a Grandmaster-level Chess Transformer*,
  2604.10158), and iterative inference via the logit lens (2508.21380).
- **Piece attribution by ablation.** A SHAP-style method removes pieces and
  measures the eval change (2510.25775).
- **NNUE.** No published work probes what Stockfish's NNUE learns. That is an
  open gap, and our atomic net is the same kind of network.

### 1.5 Lighter-weight explainers

ThinkFish, GrandMentor, chessxplain, and ez-chess all use the same pattern:
engine output plus a tactical-pattern detector go into an LLM. They report the
same failure: invented pins and missed back-rank checks, even with the PV in
the prompt. Feeding before *and* after positions helps. All of these are
standard-chess only, and none are measured carefully.

**Nothing in this survey touches atomic.** Every system above assumes chess
rules in its detectors, its training text, or its LLM's priors.

---

## 2. What "why" means for this engine

The three common answers to "why this move" are not the same thing:

| Kind of "why" | Faithful to the engine? | Readable? |
|---|---|---|
| **Search**: every alternative fails to a concrete line, and this one doesn't | **Yes, exactly.** For alpha-beta this *is* the reason. | No. It is a proof tree with thousands of nodes. |
| **Eval**: the leaf position scores better on features X, Y | Partly. It is the leaf's reason, not the root's. | Fairly, for an HCE. Hard for an NNUE. |
| **Concept**: "it defuses the f7 bomb" | Only if grounded in one of the above | Yes |

The lab has already measured which of these matters in atomic. Quiescence is
worth **+203 ± 128 Elo**, more than every atomic-specific eval term combined,
and blast-radius king safety is worth +55 without quiescence and nothing with
it (`method-catalog.md`, M1 result). In atomic, *king safety is a tactical
property that lives in the search*.

That decides the architecture. An explainer that only reads network weights
(a Queen-style bridge onto an eval net) would look in the place where atomic
keeps the *least* of its reasoning. The faithful explanation of an atomic move
is almost always a short forcing line, usually a capture sequence a few plies
long. **The core job is compressing the refutation tree into words, not
decoding the eval.** Weight-level methods are a second layer that adds
positional "why" on top.

### Why atomic moves look senseless, by category

These are the cases an explainer must cover. Each is a detector over a search
result, not an eval term.

1. **Defusal or prophylaxis.** The move steps a piece out of a blast radius, or
   off a square next to its own king, before a capture lands. It looks passive
   because the threat is a capture that has not happened yet. *Detect:* a
   null-move search before and after the move. The move removes a threat the
   null move exposes.
2. **Connected kings.** The king walks next to the enemy king, which makes it
   immune because no capture there is legal for either side. It looks suicidal
   to a standard-chess eye.
3. **Regicidal legality.** The move ignores a check or moves a pinned piece
   because it explodes the king. It looks illegal.
4. **Declined capture.** The "free" capture would explode one of the capturer's
   own pieces next to it, open a blast lane, or blow up its own king.
5. **Pawn immunity.** The move relies on pawns surviving a melee that clears
   everything else.
6. **Quiet killers.** The move is neither a capture nor a check, but leaves two
   unstoppable fuses. `themes.py` already detects `quiet` and `blast-fork` on
   puzzle lines.

---

## 3. What exists to build on

| Asset | Where | Use |
|---|---|---|
| FSF + atomic NNUE with MultiPV | `trainer/analysis.py`, `lab/fsf.py` | The engine being explained. Its refutation lines are the evidence. |
| Atomic rules and blast geometry | `lab/rules.py` (`explosion_squares`) | Ground truth for every claim |
| 15 atomic theme detectors with measured firing rates | `trainer/themes.py` | A ready-made concept vocabulary written against atomic rules, not chess |
| Rust core, 2–3.7 Mnps search, named eval terms | `core/src/` (`EvalFlags`) | Cheap re-searches for counterfactuals and ablations. Per-term eval breakdown. |
| cp → win% for atomic | `trainer/winrate.py` | Express "how much worse" in units people understand |
| Puzzle ratings in calibrated Elo, solve history | `trainer/difficulty.py`, store | **Measure whether an explanation teaches anything** |
| Analysis board UI | `trainer/server.py`, `ui/` | Where explanations appear |

What does **not** exist: any atomic network with per-square tokens. FSF's
atomic NNUE is HalfKA-style. Its first layer is a single accumulator vector
built from (king square, piece, square) features, with no square tokens for a
language model to attend to. Probes can read it (§4.3), but a Queen-style bridge
cannot use it without a new network. The methods catalog already plans such a
network as M3 (self-play policy/value) or M4 (policy-guided alpha-beta). The
explainer can reuse it.

---

## 4. The plan, in stages

Each stage works without the ones after it and is useful alone.

### Stage A: contrastive search explanations (no ML, days of work)

The most common real question is not "why this move" but **"why not the move I
wanted?"** That has an exact, faithful answer.

```
explain(fen, engine_move, my_move):
  1. MultiPV at the trainer's budget: values for both moves, as win%.
  2. Refutation of my_move: the opponent's best reply and its PV.
     Shrink it to the shortest prefix that already loses >= T win%.
     This is the "proof" line, usually 1-5 ply in atomic.
  3. Run themes.py detectors over the proof line, plus the §2 detectors
     (defusal via null-move diff, connected kings, regicidal, declined
     capture, pawn immunity).
  4. Threat diff: null-move search before and after engine_move.
     If the move removes a threat, that is the prophylaxis explanation.
  5. Blast map: for each capture in the proof line, the squares and pieces
     it destroys (lab.rules.explosion_squares) -> board overlay.
  6. Output structured facts only, each one a checked object:
     {kind: "refutation", line: [...], drop_win_pct: 38,
      themes: ["own-king-in-blast"], destroyed: ["Qd1","Bc1"], ...}
```

Rendering does not need an LLM. Atomic's concepts are few enough that
templates per detector work: *"After 14.Bxf7? Nxe5 the capture on e5 detonates
next to your king — d1 queen and c1 bishop go with it (−38%)."* The UI draws
the proof line as arrows and the blast as shaded squares.

Additions that cost a few lines each with the Rust core:

- **Piece-importance heatmap.** Remove each piece, re-search shallowly, and
  shade by |Δ win%|. This is the 2510.25775 method, and at 2–3 Mnps it runs in
  under a second per position.
- **Eval-term breakdown** (M1 only). Expose a `evaluate_terms(fen) -> dict` per
  `EvalFlags` term over PyO3. Report the term deltas from root to PV leaf.
  Useful for quiet moves. Do not show it for FSF moves, because M1's terms are
  not FSF's reasons.

*Hypothesis:* most "this move makes no sense" moments in atomic resolve to a
proof line of 4 ply or fewer, plus one of the existing theme detectors.
*Measurement:* run Stage A over the user's analysed blunders
(`trainer.cli worst`). Report (a) the share explained by a proof line of 4 ply
or fewer and (b) the share where **no** detector fires. Group (b) is the
concept vocabulary's blind spot, and it is the input to Stage C.

### Stage B: an LLM verbalizer with a claim checker (a week)

The LLM's only job is **phrasing**. It never decides facts.

1. Input: Stage A's fact objects, both FENs, a short **atomic rules primer**
   (the rules table from `research-notes.md` §1), and an explicit list of
   standard-chess intuitions that are false here.
2. Output: prose plus a **claim list**, each claim in a checkable schema: piece
   on square, X attacks Y, capture C destroys {…}, line L loses N%, and so on.
3. A checker built on `lab.rules` and the Rust core checks every claim. Any
   failure means regenerate, and after two failures, fall back to the Stage A
   template. This is ACT-Eval moved from evaluation time to generation time.
4. Log every rejected claim. The rejection rate per claim type measures how
   wrong LLM chess priors are in atomic, which is a small result in its own
   right.

*Prediction:* with no checker, an off-the-shelf frontier LLM makes rule errors
(captures by kings, "check" on adjacent kings, pins treated as binding) in
**more than 30%** of atomic explanations, against ACT-Eval's 22% for standard
chess. With the checker, shown text contains zero rule errors by construction,
and the cost is a fallback rate that should be reported.

### Stage C: what the networks know (research)

Applied to the FSF atomic NNUE accumulator now, and to the M2/M3 nets once they
exist:

- **Linear probes** for atomic concepts. Labels come for free from `lab.rules`:
  lit fuse (an enemy can capture next to my king), number of capturable bombs
  next to the king, kings connected, a capture that wins outright exists,
  pawn-shielded king, and each `themes.py` detector. Probe accuracy per layer
  says which atomic concepts the net computes statically. Combined with the
  quiescence finding, this predicts that **fuse concepts probe poorly in the
  NNUE**, because the search supplies them. That is a falsifiable test of the
  M1 "king safety belongs in search" claim from a completely different
  direction.
- **Unsupervised concept discovery** (an SAE on the accumulator, or the Schut
  method on a transformer). Atomic is where this pays most: theory is thin, so
  a large share of what an engine knows has never been written down. Novelty
  in Schut's sense is far more likely here than in standard chess.
- **Teachability test on the user, with the lab's own units.** Build concept
  prototype puzzles for a discovered concept, use `puzzles rate` for
  difficulty, and compare solve rates before and after studying the
  prototypes. Use held-out positions and a control concept. One subject is
  small, but the measurement is the same as the PNAS GM study.

Discovered concepts that pass become new detectors for Stages A and B. That is
how learned knowledge reaches explanations without a model reading weights at
run time.

### Stage D: a Queen-style atomic explainer (the long-range bet)

Only worth building once a token-per-square atomic transformer exists (the M3
or M4 policy/value net). There are two ways to get one:

- **Fine-tune an Lc0 BT net on atomic.** Lc0's input planes encode any atomic
  position, and its learned geometry (attacks, lines, adjacency) transfers. The
  explosion semantics have to be relearned. Distil from FSF: policy from
  MultiPV, value from the eval through `winrate`. This is cheap compared with
  self-play, and it is the M2 "distillation dominates" bet with a
  transformer.
- **Train a small native atomic transformer** (10–40M parameters) from
  scratch on FSF labels. It fits on the 5070, and doubles as M4's move-ordering
  policy.

Then the Queen recipe, adapted:

| Queen stage | Atomic version |
|---|---|
| 1. Board facts | The same tasks (piece on square, counts, …) |
| 2. Move facts | Replace *checking moves / check parries / is checkmate* with: **which pieces a capture destroys**, **is this capture legal (own king in blast)**, **which captures explode the enemy king**, **are kings connected**, **lit fuses on the board**. Labels from `lab.rules`. |
| 3–4. Forward | The same questions after a move sequence. This is critical in atomic, because explosions remove several pieces at once and board tracking is harder than in chess. |
| 5. Rationale | Stage A/B outputs, already checked, used as targets. Plus `themes.py` labels and tablebase verdicts (`lab.tablebase`) for exact endgame "why". |
| Self-distill | FSF is the referee: drop analyses whose claims fail the Stage B checker or whose eval claims miss FSF's by more than a tolerance. |

There is no human atomic commentary corpus, so the training text is
**entirely synthetic and checked**. That is a strength: Queen's main weakness
was conceptual coherence on motifs it learned from loosely grounded text, and
here every motif label comes from a rules-checked detector.

**The faithfulness fix** for Queen's "disagrees with its own engine" problem:
put the engine's chosen move and its proof line *into the prompt*, and train
the explainer to explain a given move, not to choose one. Keep the claim
checker at run time anyway.

Compute, honestly: the bridge trains with both sides frozen. SmolLM3-3B in
bf16 is about 6 GB of weights. A 3B decoder is tight but plausible on a 12 GB
5070 with gradient checkpointing and short sequences, and a 1–1.7B decoder is
comfortable. Queen's self-distillation used vLLM generation at scale plus
Stockfish refereeing, which is far beyond one GPU. Plan for a smaller decoder
and fewer iterations. *(Queen's own training compute is unverified.)*

*Hypothesis:* given an encoder of equal quality, the Stage D model's
explanations beat Stage B on **coverage** (positional "why" for quiet moves
that no detector describes), not on accuracy, because Stage B is already
checked.
*Measurement:* the Stage A blind-spot set from group (b), scored by the claim
checker for accuracy and by solve-rate gain for usefulness.

---

## 5. Recommendation

1. **Build Stage A now.** It is exact, faithful to FSF's real reasoning, needs
   nothing that does not already exist, and answers the question as actually
   asked: "why not my move?" Its blind-spot report says whether anything later
   is needed.
2. **Add Stage B** for readability once Stage A's facts are stable. The
   checker is not optional in atomic.
3. **Run Stage C's probes on the FSF NNUE.** It is cheap, publishable-grade,
   and tests the lab's central M1 claim from a new direction.
4. **Leave Stage D until M3 or M4 produces a transformer.** Then it is mostly
   data plumbing over Stages A–C, and its training data will already exist.

## 6. Stage A: what was built (2026-11-05)

`trainer/explain.py`, exposed three ways:

- **Play tab**: after any move, *Why Nf6?* explains the engine's last move,
  and *Was h6 OK?* explains yours.
- **Review tab**: every ply has *Why is h6 bad?* (or *Why Nf3?* when the move
  was fine).
- **Terminal**: `py -3.13 -m trainer.cli explain [FEN] --moves Nf3 --move h6`.

Each reason with a line has *Show on board*, which steps through it with ◂ ▸
(or the arrow keys) and lights the blast squares. Esc returns to the game.

What it computes, from four or five cached searches at 300k nodes each:

| Reason | How it is found |
|---|---|
| **refuted** | The move asked about, plus the engine's best reply line. Cut to the shortest prefix where the damage is done: the king explodes, or at least 1.5 pawns of atomic material has changed hands. Every explosion is spelled out: which pieces die on each side, the capturer included. |
| **parries** | A null move. Hand the opponent the turn, search, and compare. If passing would cost at least 15 win% more than the engine's move does, the move is a defence, and the threat line is shown. That threat line is marked as hypothetical, because the position after a pass never happened. |
| **threatens** | The same null-move test after the engine's move. A non-mate threat has to land within 5 plies and win material for the side making it, otherwise it is not reported. |
| **plan** | The engine's own principal variation, when it is forcing. It says so when even the best line loses material. |
| **check / wins** | Direct rules facts |
| **kings touch / out of the blast / own pieces / atomic legality** | Rules notes read off the move with `lab.rules`. "Out of the blast" is shown only when there was a real threat. |
| **positional** | No short forcing line explains the difference. **It says so instead of inventing a tactic.** These cases are the blind-spot set that Stage C exists for. |
| **close** | Within 5 win%. The engine's preference is noise at this budget. |

Checked so far:

- 17 tests in `tests/test_explain.py`. One runs the real engine.
- A stress run of ~25 random bad moves from engine self-play positions, read
  by hand. Every refutation and threat named was a real line the engine
  played.
- The Play and Review tabs driven end to end in a headless browser.

All of this used FSF's **handcrafted** atomic eval, because the NNUE file
could not be downloaded in the build container. On your machine the trainer
uses the net as usual. The explanations will be stronger, and they will not
always match the ones above.

Not yet measured: the Stage A blind-spot rate over your own blunders, from
§4. That needs your analysed games. The way to get it is to run *Why is X
bad?* over `trainer.cli worst` and count the **positional** answers.

### On rewriting the engine to be readable

An engine that is easier to read but weaker would explain the *wrong* moves:
you would learn its mistakes. The approach here does not need a readable
engine. The search-based reason is faithful to whatever engine produced it,
so the explanation stays tied to Fairy-Stockfish, the strongest atomic
engine available.

Readability does matter for the **positional** cases, where no short line
exists. The route there is a small evaluation built from named atomic
concepts and fitted to agree with FSF (Stage C). It would explain FSF's
quiet preferences in its own terms without having to play any games itself.

## Sources

- Queen: arXiv [2610.03695](https://arxiv.org/abs/2610.03695) · code
  [CDittmeier/queen](https://github.com/CDittmeier/queen)
- CCC: arXiv [2410.20811](https://arxiv.org/abs/2410.20811)
- ACT-Eval: arXiv [2608.04240](https://arxiv.org/abs/2608.04240)
- McGrath et al., PNAS 2022, *Acquisition of chess knowledge in AlphaZero*
- Schut et al., PNAS 2025, doi:10.1073/pnas.2406675122 · arXiv [2310.16410](https://arxiv.org/abs/2310.16410)
- Jenner et al., NeurIPS 2024 · arXiv [2406.00877](https://arxiv.org/abs/2406.00877)
- Contrastive SAEs for chess planning: arXiv [2406.04028](https://arxiv.org/abs/2406.04028)
- Tracing a GM-level chess transformer: arXiv [2604.10158](https://arxiv.org/abs/2604.10158)
- Iterative inference in Leela: arXiv [2508.21380](https://arxiv.org/abs/2508.21380)
- Piece-ablation attribution: arXiv [2510.25775](https://arxiv.org/abs/2510.25775)
- MATE (explanations to improve LLM move choice): arXiv [2411.06655](https://arxiv.org/abs/2411.06655)
- ThinkFish: [github.com/ronaldsuwandi/thinkfish](https://github.com/ronaldsuwandi/thinkfish)
