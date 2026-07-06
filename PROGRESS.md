# Project Progress

A style-configurable chess engine (strong as possible, distinct selectable
playing styles) — CPU now, RTX 5070 later. Python + python-chess + PyTorch.

Last updated: 2026-07-05

---

## Status by phase

| Phase | What | State |
|---|---|---|
| 0 | Environment, deps, Stockfish 18 labeler | ✅ Done |
| 1 | Alpha-beta engine + hand-crafted eval + pygame GUI | ✅ Done |
| 2 | UCI adapter + 7 style profiles + bounded shaping | ✅ Done |
| 3 | NNUE strength track (data-gen, train, infer, wiring) | ✅ Built & assessed |
| 4 | AlphaZero learning pipeline | ⏸️ Deferred (later) |
| 5 | Style fine-tuning + GUI polish | ✅ Done |
| — | Faster search (raise nodes/sec) | ✅ Done (6.9× hand, 2.3× NNUE) |
| — | Faster NNUE eval (Numba JIT) | ✅ Done (~8× eval; NNUE search now ≈ hand depth) |
| — | GUI features: premoves, setup, pass-and-play, analyzer, match viewer | ✅ Done |

---

## Phase 0–2 (done)

- **Engine core** (`engine/`): alpha-beta + iterative deepening + transposition
  table + quiescence + MVV-LVA/killer ordering (`search.py`); hand-crafted eval
  = material + piece-square tables + king safety + mobility + openness, which
  also emits positional *features* for the style layer (`evaluate.py`).
- **Style system** (`style/`): 7 distinct, selectable profiles — `balanced`,
  `sacrificer`, `attacker`, `defender`, `open`, `closed`, `grinder`
  (`profiles.py`). Applied as a **bounded** eval overlay that shrinks as the
  position becomes decided, so winning always dominates and no style ever plays
  a losing move (`shaping.py`). Verified: every style finds mate-in-1 and grabs
  a free queen.
- **Interfaces** (`ui/`): pygame GUI with click-to-move, eval bar, live style
  switch, depth control (`gui.py`, `run.py`); UCI adapter with `Style` option
  for Cutechess/Arena/Nibbler (`uci.py`).

## Phase 3 — NNUE (built, trained, honestly assessed)

Pipeline in `engine/nnue/`: `features.py` (HalfKP), `model.py` (10.5M-param
EmbeddingBag net), `gen_data.py` (parallel Stockfish-labeled data),
`dataset.py`, `train.py` (device-agnostic — auto-CUDA on the 5070),
`infer.py` (NumPy). Labeler: `tools/stockfish.exe` (SF18). Strength harness:
`tools/match.py`.

**Debugging arc (real):**
1. First net (97k positions) lost **0-6** — root cause: the data generator never
   labeled opening positions, so the net played nonsense openings and collapsed.
2. Fixed the generator to label every position (incl. openings), regenerated
   **257k** positions, retrained (lambda 0.8). Opening play became sound (real
   theory: Scandinavian `e4 d5 exd5 Qxd5 Nc3`); material eval correct
   (+580cp up a queen).
3. New result vs hand eval: **0-5-1** timed, **0W-2D-2L at equal depth-3**.

**Honest bottom line:** the NNUE eval is now *sound* but **not yet stronger**
than the tuned hand-crafted eval. Two understood bottlenecks:
- **Search speed** — pure-Python search ~1300 nps; NNUE's ~60µs/eval makes it
  search shallower than the fast hand eval at equal time (only depth 1-2/move).
- **Data scale** — 257k CPU positions isn't enough for a decisive eval edge.

**Levers to surpass it:** faster search (incremental accumulators / Cython) and
GPU-scale data on the 5070. Optimized `both_perspectives` 79µs→15µs and
`eval_cp` 104µs→63µs (parity-verified) as a first step.

---

## How to run

All commands are run from the project root (`C:\Users\caleb\chess`) in your
Anaconda Python. On Windows use PowerShell; the commands are the same.

### 0. One-time setup

```powershell
pip install -r requirements.txt
```

Everything else already exists in the repo: the Stockfish labeler
(`tools/stockfish.exe`) and a trained net (`data/nnue.pt`).

### 1. Play against it (GUI)

```powershell
python run.py                                    # you = White, balanced, hand eval
python run.py --style sacrificer                 # pick a style
python run.py --style defender --depth 5         # stronger + slower (higher depth)
python run.py --black --style attacker           # play Black (engine moves first)
python run.py --style open --nnue data/nnue.pt   # use the NNUE net instead of hand eval
```

In the window:
- **Click** a piece, then click its destination to move. Pawns auto-promote to
  a queen.
- **Style button** (right panel) cycles through the 7 styles live.
- **Depth −/+** changes strength (higher = stronger, slower). Depth 3–5 is a
  good range on your CPU.
- **New Game** restarts.
- The panel shows the **eval bar**, the engine's **plan** (its predicted line),
  its search **depth/speed**, and the **move list**.

### 2. Play it in a "real" chess GUI (UCI) — optional

The engine speaks UCI, so it loads in Arena, Cutechess, Nibbler, or Banksia.
Point the GUI at this command as the engine:

```
python -m ui.uci
```

Then set the `Style` option in that GUI. To use the NNUE net, set the `EvalFile`
option to `data/nnue.pt`, or set an environment variable before launching:

```powershell
$env:STYLE_ENGINE_NNUE = "data/nnue.pt"
```

### 3. Check how strong it is (engine vs engine)

```powershell
# NNUE net vs the hand-crafted eval, 6 games, 0.2s per move
python -m tools.match --nnue data/nnue.pt --games 6 --movetime 0.2

# Compare two nets against each other
python -m tools.match --a-nnue data/nnue.pt --b-nnue data/nnue_v2.pt --a-name v1 --b-name v2
```

### 3b. Get a realistic Elo rating (anchored to Stockfish)

`tools/match.py` only says "A beats B" — for an **absolute** number, rate a net
against Stockfish 18 capped to known Elo levels (`tools/rate.py`). It solves for
the net's Elo from the scores, prints a confidence interval, and appends a row
(version, #positions, label depth, lambda, epochs, time control, Elo ± CI) to
`data/ratings.csv` so you can track "more/better data → higher Elo" over v1..vN.

```powershell
# Rate a net: 24 games at each of 4 Stockfish levels, 0.2s/move
python -m tools.rate --nnue data/nnue.pt --version v1 --games 24 --movetime 0.2

# Wider ladder / longer thinking time for a tighter number
python -m tools.rate --nnue data/nnue_v2.pt --version v2 `
    --games 32 --movetime 0.3 --levels 1320,1500,1700,1900,2100
```

Realism caveats: the Elo is only meaningful **at the recorded time control**
(change `--movetime` and it moves); Stockfish's `UCI_Elo` is itself ±100ish but
**consistent**, which is what you need for progress tracking; and you need
**≥24 games/level** or the CI is too wide to trust a small gain (a 1–2 game run
is pure noise — the tool's CI will say so).

**Net metadata** — `train.py` now writes a `<net>.meta.json` sidecar (positions,
label depth, lambda, epochs) next to every net it saves; `rate.py` reads it so
the ratings log fills in automatically. Existing nets were backfilled from this
file. Keep versioning nets as `data/nnue.pt` (v1), `data/nnue_v2.pt` (v2), … and
pass `--version vN` when rating so each row ties to a concrete net.

---

## How to make it stronger (training more)

The engine's strength comes from the **NNUE net** (`data/nnue.pt`). You improve
it by training on **more, higher-quality data** — that is the single biggest
lever. The full loop is three steps: generate data → train → verify it's better.

### Step 1 — Generate more training data

Positions are labeled by Stockfish. More games = more data = a smarter net.

```powershell
# ~250k positions (what the current net used) — ~10 min on your CPU
python -m engine.nnue.gen_data --games 2400 --workers 8 --depth 9 --out data/train

# Bigger + higher quality: more games, deeper Stockfish labels (slower but better)
python -m engine.nnue.gen_data --games 8000 --workers 8 --depth 12 --out data/train
```

- `--games`   more = more data (the main knob). 2400 → ~250k positions.
- `--workers` parallel Stockfish processes. Keep at 8 (you have 12 threads).
- `--depth`   Stockfish label quality. 9 is fast; 12–14 gives sharper targets
  (each label takes longer). Higher depth ≈ smarter teacher ≈ stronger net.
- Data lands in `data/train.shard*.tsv`. To **add** to existing data instead of
  replacing, use a different `--out` prefix (e.g. `data/train2`) and pass a glob
  covering both to training (see below).

### Step 2 — Train the net

```powershell
# Train from scratch on all shards
python -m engine.nnue.train --shards "data/train*.shard*.tsv" `
    --epochs 60 --batch 8192 --lambda 0.8 --out data/nnue_v2.pt
```

- `--epochs`  more passes = better fit, up to a point (~50–80 is plenty). It
  saves the **best** net by validation loss automatically.
- `--lambda`  0.8 mixes mostly Stockfish eval with a little game outcome (a good
  default). 1.0 = pure eval, 0.0 = pure win/loss.
- `--batch`   8192 is fine; lower it only if you run out of memory.
- `--resume data/nnue.pt`  continue training an existing net instead of starting
  fresh (useful when you add new data).
- On CPU this is ~5s/epoch (~5 min total). **On your RTX 5070 it will be
  dramatically faster** — see the GPU note below. No code changes needed.

### Step 3 — Verify it actually got better

Always confirm the new net beats the old one before keeping it:

```powershell
python -m tools.match --a-nnue data/nnue.pt --b-nnue data/nnue_v2.pt `
    --a-name old --b-name new --games 10 --movetime 0.3
```

If `new` wins clearly, replace the net: copy `data/nnue_v2.pt` over
`data/nnue.pt` (or just point `--nnue` at the new file).

### Style-specific nets (optional)

Once you have a strong base net, bias a copy toward a style without losing
strength (trains on the same data, just weighting on-style positions more):

```powershell
python -m engine.nnue.finetune --base data/nnue.pt --style sacrificer `
    --shards "data/train*.shard*.tsv" --epochs 8 --out data/nnue_sacrificer.pt
python run.py --style sacrificer --nnue data/nnue_sacrificer.pt
```

### The realistic ceiling right now

On **CPU only**, two things cap strength: the pure-Python search speed
(~12k nps for the hand eval, ~3k with NNUE) and how much data you can afford to
label. The current net is *sound* but not yet beating the hand-crafted eval.

**The RTX 5070 is the real unlock.** When it arrives:
1. Reinstall the CUDA build of PyTorch (one command, in `requirements.txt`
   comments) — no code changes; the trainer auto-detects the GPU.
2. Generate **much** more data (e.g. `--games 20000 --depth 14`) and train for
   more epochs. This is where NNUE nets become genuinely strong (2400+).
3. Then run `finetune.py` per style.

For even more strength beyond that, the next engineering step is **incremental
NNUE accumulators** (update the eval as pieces move instead of recomputing) to
let the NNUE search deeper — noted below under "optional".

---

## Faster search (done)

Profiling found two hidden costs in the per-node hot path:
`can_claim_threefold_repetition()` (replays the whole game — called at *every*
node) and generating legal moves **three times** per node (checkmate check,
stalemate check, move ordering). Fixes in `engine/search.py` /
`engine/evaluate.py`:
- Replaced the expensive repetition check with `is_repetition(2)` +
  `halfmove_clock`.
- Generate legal moves **once** per node; derive mate/stalemate from the empty
  list; reuse the list for ordering.
- Replaced the mobility term's double legal-move-generation with a bitboard
  attack-span proxy.

Result: **hand eval 1,743 → 11,964 nps (6.9×)**, **NNUE 1,312 → 3,014 nps
(2.3×)**. All tactics/draw regression tests still pass. (Did not change the
strength verdict vs the hand eval — that's an eval-quality/data-scale issue, not
speed.)

## Faster NNUE eval — Numba JIT (done)

The NNUE eval was ~400µs/call in pure NumPy, making NNUE search shallower than
the hand eval at equal time. Rebuilt the inference hot path with **Numba-JIT
kernels** (`engine/nnue/kernels.py`) driving a new `FastNNUE`
(`engine/nnue/fast_infer.py`):
- **Fused bitboard feature transformer** — one compiled pass extracts HalfKP
  features straight from python-chess piece bitboards (no `piece_map`, no Python
  `both_perspectives`) and sums the transformer rows. This was the biggest win;
  pure-Python feature extraction had become the dominant cost.
- **Int16 quantized dense head** — the 2·256→32→32→1 head runs in fixed-point
  int matmuls. Quantization only pays off *under Numba* (NumPy has no fast int
  matmul — there it was a 2× pessimization). Verified within ≤14 cp of the float
  reference on non-draw positions.

Result: **eval 400µs → ~48µs (~8×)**; NNUE search nps went from 1820 → 2707,
now **≈ the hand eval's nps (2891)** and reaching equal or greater depth on
middlegame positions. Verified: FastNNUE picks the **same move as the reference
NNUE in 17/18 fixed-depth tests** (the one diff is two ~equal opening moves off
quantization rounding) — i.e. the speedup is "free," same decisions, deeper reach.

**Honest strength note:** with the depth handicap removed, NNUE *still* loses to
the hand eval at equal time (measured ~0-7/8 at 0.4s/move). This isolates the
remaining gap to **eval quality, not speed** — exactly the 257k-position data
ceiling PROGRESS predicted. The speedup was necessary (NNUE now searches as deep
as the hand eval) but not sufficient; the next lever is **GPU-scale training
data** on the 5070, not more CPU eval optimization. Incremental accumulators were
prototyped and verified exact, but proved *slower* than the fused from-scratch
JIT for this search's push/eval/pop leaf pattern (no accumulator reuse across
siblings), so the fused path is the shipped default. `numba>=0.62` added to
requirements; the engine falls back to the NumPy NNUE path if numba is absent.

## Phase 5 — style fine-tuning + GUI polish (done)

- **Style fine-tuning** (`engine/nnue/finetune.py` + `style_weight.py`): resume
  from the base net and continue training with a per-position *style weight*
  (high for on-style positions, e.g. sacrifices for `sacrificer`) applied to the
  loss. Eval targets are unchanged, so strength is preserved — the net just pays
  more attention to on-style positions. Verified end-to-end. Run:
  `python -m engine.nnue.finetune --base data/nnue.pt --style sacrificer --out data/nnue_sacrificer.pt`
  Most effective after GPU-scale base training (the "train strength first, then
  nudge" approach).
- **GUI polish** (`ui/gui.py`): added a live **move list** (SAN), the engine's
  **plan / principal variation** with eval + depth + nodes-per-second, and a
  wider panel. Verified headlessly.

## GUI features (done)

- **Premoves fixed + multi-premove** (`ui/gui.py`): highlight now shows only the
  piece's reachable squares (was every square); premoves are a queue played one
  per turn, validated against a virtual board so chained premoves are correct.
- **Position setup** (`Setup (FEN/PGN)` button): paste auto-detects FEN vs PGN;
  PGN auto-replays then hands control back.
- **Pass-and-play** (`--both` / live toggle): control both colors, engine idle.
- **Game analyzer** (`ui/analyze.py`, or the in-GUI `Analyze game` button):
  step through a game (← →, Home/End, F flip); background NNUE pass fills eval +
  best move + blunder/mistake/inaccuracy tags (eval-swing thresholds); press **S**
  for on-demand per-style move suggestions at the current position. Run:
  `python -m ui.analyze game.pgn` (defaults to `data/nnue.pt`, depth 6).
- **Match viewer** (`tools/watch.py`): the visual counterpart to `tools/match.py`
  — watch engine-vs-engine games animate live with running score; supports
  `--a-style`/`--b-style` so you can watch styles play each other. Space pauses,
  → steps. Shared board rendering lives in `ui/render.py`.

## Phase 4 — AlphaZero learning pipeline (LATER — not doing this yet)

**Status: intentionally deferred. Do not start until explicitly chosen.**

This phase is the "learn modern reinforcement learning" track from the original
plan — build a small AlphaZero-style self-play loop end to end. It is
*educational*, not the path to a strong engine on this hardware (self-play from
scratch on one GPU realistically caps out well below the NNUE track). It's here
so it's ready when you want to learn the RL pipeline — but it's parked for now.

When picked up, it would add `engine/az/`:
- `net.py` — a small residual network with **policy** (move probabilities) and
  **value** (who's winning) heads.
- `mcts.py` — Monte Carlo Tree Search that uses the net to guide the search.
- `selfplay.py` — generate games by self-play, parallelized across CPU cores
  with `multiprocessing`.
- `rl_train.py` — train the net on the self-play games, then repeat (the net
  plays itself, learns, and gets better in a loop).

Reasonable scope when the time comes: get the loop running end to end, confirm
the loss goes down and it beats a random/1-ply baseline. Strength is *not* the
goal — understanding every component is. Style would enter as a small, bounded
term in the self-play reward (same "winning dominates" rule as the NNUE track).
The self-play stays on CPU cores; only the net training moves to the 5070.

## Also next up

- **When the 5070 arrives**: reinstall CUDA torch, scale NNUE data/epochs (no
  code changes — trainer is device-agnostic), then run `finetune.py` per style.
  See "How to make it stronger" above.
- **Optional engineering**: incremental NNUE accumulators to push NNUE nps
  further so it can search deeper.
