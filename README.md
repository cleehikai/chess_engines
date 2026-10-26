# Style Chess Engine

A chess engine designed to play in **distinct, selectable styles** (sacrificer,
defender, open, closed, attacker, grinder, balanced) while staying **as strong
as possible** — winning always dominates; style is a bounded nudge.

Built to run on CPU now and scale to an RTX 5070 later (only training moves to
GPU; the engine itself is CPU-friendly).

## What works today (Phases 0–2)

- **Alpha-beta search** with iterative deepening, transposition table,
  quiescence, MVV-LVA + killer move ordering (`engine/search.py`).
- **Hand-crafted evaluation** (material + piece-square tables + king safety +
  mobility + openness) that also emits positional *features* (`engine/evaluate.py`).
- **7 configurable style profiles** (`style/profiles.py`) applied as a bounded,
  safety-capped eval overlay (`style/shaping.py`). Style bonuses shrink as the
  position becomes decided, so no style ever plays a losing move.
- **pygame GUI** — click-to-move board, eval bar, live style switcher, depth
  control (`ui/gui.py`, `run.py`).
- **UCI adapter** — plugs into Cutechess / Arena / Nibbler / Banksia for
  strength testing and engine-vs-engine matches, with `Style` as a UCI option
  (`ui/uci.py`).

## Install

```bash
pip install -r requirements.txt
```

When the RTX 5070 arrives, reinstall the CUDA torch build (see comment in
`requirements.txt`) — no code change needed.

## Play (GUI)

```bash
python run.py                          # you are White vs balanced engine
python run.py --style sacrificer --depth 4
python run.py --black --style defender
```

Cycle styles and change depth live from the side panel.

## Test strength / run style matches (UCI)

Point any UCI GUI at `python -m ui.uci`, or with `cutechess-cli`:

```bash
cutechess-cli \
  -engine name=Sacrificer cmd=python arg=-m arg=ui.uci option.Style=sacrificer \
  -engine name=Defender   cmd=python arg=-m arg=ui.uci option.Style=defender \
  -each proto=uci tc=10+0.1 -games 20
```

## NNUE (Phase 3 — strength track)

The NNUE pipeline lives in `engine/nnue/` and is fully working end-to-end:

```bash
# 1. Generate Stockfish-labeled training data (parallel across cores)
python -m engine.nnue.gen_data --games 800 --workers 8 --depth 9 --out data/train

# 2. Train the net (auto-uses CUDA on the 5070; CPU otherwise)
python -m engine.nnue.train --shards "data/train.shard*.tsv" \
    --epochs 60 --batch 8192 --lambda 1.0 --out data/nnue.pt

# 3. Play with the NNUE eval
python run.py --nnue data/nnue.pt --style sacrificer
# or in UCI:  set STYLE_ENGINE_NNUE=data/nnue.pt  (or the EvalFile UCI option)
```

`--lambda` mixes the target between Stockfish eval (1.0) and game result (0.0),
per the nnue-pytorch convention.

**Status / honest caveat (CPU proof stage):** the net is trained on ~257k
Stockfish-depth-9 positions (opening-inclusive). It evaluates material and
tactics correctly and plays sound openings (real theory lines like the
Scandinavian). BUT it is **not yet stronger than the tuned hand-crafted eval**
in timed games, for a concrete, understood reason: the pure-Python alpha-beta
search runs at only ~1300 nodes/sec, and each NNUE eval (~60µs) makes the search
shallower than the fast hand eval at equal time — so NNUE reaches only depth 1-2
per move and gets out-tactic'd. This is the well-known pure-Python NNUE speed
wall, not a training bug.

Paths to actually surpass the hand eval:
1. **Faster search** — the real lever. Incremental NNUE accumulator updates, or
   moving the search hot loop to Cython/C. (~1300 nps is the ceiling in pure
   Python.)
2. **GPU-scale data** on the 5070 — a much stronger net that wins even at low
   depth.
3. Compare/deploy at **fixed equal depth** rather than equal time, where the
   NNUE's positional knowledge is on even footing.

Run `python -m tools.match --nnue data/nnue.pt` to reproduce the strength check.

## Roadmap

- **Phase 4 — AlphaZero (learning track):** `engine/az/`. Small ResNet + MCTS +
  multiprocessing self-play. Educational; not the strength deliverable.
- **Phase 5 — style fine-tuning + GUI polish.**

See the full plan for hardware reality-check and strength expectations.

## Layout

```
engine/   board.py  search.py  evaluate.py   (+ nnue/, az/ later)
style/    profiles.py  shaping.py
ui/       gui.py  uci.py
run.py    launches the GUI
```

## Working in this repo (for AI assistants)

This repo has a prebuilt **graphify** knowledge graph in `graphify-out/`. When
answering questions about the codebase — architecture, how files relate, "where
does X live", "how does the NNUE pipeline flow" — query that existing graph
first (via the graphify skill) rather than blind grep. For mechanical chores
(git, editing files, running commands), skip graphify and just do the work.
