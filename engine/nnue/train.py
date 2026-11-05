"""Train the NNUE net. Device-agnostic: uses CUDA (the 5070) if available,
otherwise CPU.

The net is trained to predict a WDL-space value in [-1, 1] (tanh of the
Stockfish centipawn eval, optionally mixed with game result via --lambda). At
inference we invert the mapping to recover centipawns (see infer.py).

Usage:
    python -m engine.nnue.train --shards "data/train.shard*.tsv" \
        --epochs 30 --batch 8192 --lr 1e-3 --lambda 1.0 --out data/nnue.pt
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from engine.nnue import meta
from engine.nnue.dataset import NNUEDataset, collate
from engine.nnue.model import NNUE
from tools.estimate import announce


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/train.shard*.tsv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=1.0,
                    help="1.0=pure eval target, 0.0=pure game result")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--split-by-fen", dest="split_by_fen",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="partition train/val by FEN hash so every copy of a "
                         "position lands on one side. Splitting by line (the old "
                         "behaviour, --no-split-by-fen) leaks: duplicate FENs put "
                         "held-out positions back in train, so val loss measures "
                         "memorization. Nets before v7 used the leaky split")
    ap.add_argument("--out", default="data/nnue.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--warmup", type=int, default=3,
                    help="linear LR warmup epochs before cosine decay")
    ap.add_argument("--qat", action="store_true",
                    help="quantization-aware training: clip weights to the int "
                         "range used at inference so the deployed net matches "
                         "what was trained (removes the quantization Elo tax)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="L2 regularization. v4 overfit (train<<val) on 1.86M "
                         "positions; a small value (1e-4..1e-3) curbs that until "
                         "more data is available")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many epochs with no val "
                         "improvement (0 = disabled). Saves time once the cosine "
                         "tail stops improving")
    ap.add_argument("--min-delta", dest="min_delta", type=float, default=1e-4,
                    help="minimum val-loss improvement that counts as progress "
                         "for --patience. Epoch-to-epoch val noise is ~1e-4 here, "
                         "so without this a 1e-9 'improvement' resets patience "
                         "forever and early stopping never fires")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.get_num_threads()} CPU threads)")

    print("Loading dataset (encoding features)...")
    t0 = time.perf_counter()
    ds = NNUEDataset(args.shards, lambda_=args.lambda_,
                     max_samples=args.max_samples, as_wdl=True)
    print(f"  {len(ds):,} samples in {time.perf_counter()-t0:.0f}s")

    if args.split_by_fen:
        # Assign each position to val by a threshold on its FEN hash. Because the
        # key is a function of the FEN alone, every duplicate of a position goes
        # to the same side -- no leakage -- and the assignment is reproducible
        # across runs and across datasets (a position held out for v6 is still
        # held out for v7, so their val losses are comparable).
        cutoff = int(args.val_frac * (1 << 64))
        val_idx = [i for i, k in enumerate(ds.fen_keys) if k < cutoff]
        train_idx = [i for i, k in enumerate(ds.fen_keys) if k >= cutoff]
        if not val_idx:
            raise SystemExit("val split is empty; raise --val-frac")
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)
        leaked = len(set(ds.fen_keys[i] for i in val_idx)
                     & set(ds.fen_keys[i] for i in train_idx))
        print(f"  split by FEN: {len(train_idx):,} train / {len(val_idx):,} val "
              f"({len(val_idx)/len(ds):.1%}), {leaked} leaked positions")
    else:
        n_val = max(1, int(len(ds) * args.val_frac))
        n_train = len(ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
        print(f"  split by line (LEAKY): {n_train:,} train / {n_val:,} val")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model = NNUE().to(device)
    model.qat = args.qat
    if args.qat:
        print("QAT on: activations fake-quantized to the inference int grid.")
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # Cosine annealing over the FULL epoch budget with a short linear warmup.
    # The old StepLR halved the LR every 10 epochs, so by epoch ~25 it was tiny
    # and training stalled -- v2/v3 both saved their "best" net around epoch
    # 21-24 of 60 and never improved, i.e. the schedule starved the optimizer
    # rather than the model converging. Cosine keeps a useful LR across the whole
    # run and lands softly, which is the standard nnue-pytorch recipe.
    warmup = max(0, min(args.warmup, args.epochs - 1))

    def lr_at(epoch: int) -> float:
        # epoch is 0-indexed here.
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)
    loss_fn = torch.nn.MSELoss()

    # Calibration: ~2s/epoch per 250k samples on CPU; GPU is ~30x faster.
    per_epoch_s = 2.0 * (len(ds) / 250_000.0)
    if device.type == "cuda":
        per_epoch_s /= 30.0
    announce(
        f"Training ({args.epochs} epochs, {len(ds):,} samples on {device.type})",
        args.epochs * per_epoch_s,
        f"~{per_epoch_s:.1f}s/epoch x {args.epochs} epochs")

    best_val = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        te = time.perf_counter()
        for stm_i, stm_o, nstm_i, nstm_o, tgt, _w in train_dl:
            stm_i, stm_o = stm_i.to(device), stm_o.to(device)
            nstm_i, nstm_o = nstm_i.to(device), nstm_o.to(device)
            tgt = tgt.to(device)
            # Model outputs raw; squash to WDL space to match targets.
            pred = torch.tanh(model(stm_i, stm_o, nstm_i, nstm_o))
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * tgt.size(0)
            n += tgt.size(0)
        sched.step()

        val = _evaluate(model, val_dl, loss_fn, device)
        print(f"epoch {epoch:3d} | train {running/max(n,1):.4f} | "
              f"val {val:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
              f"{time.perf_counter()-te:.0f}s", flush=True)

        # Two separate thresholds, deliberately:
        #   * ANY improvement saves the checkpoint, so `args.out` is always the
        #     best net we ever saw.
        #   * Only an improvement larger than --min-delta counts as *progress*
        #     and resets patience. A long drizzle of 1e-6 gains would otherwise
        #     keep the run alive forever while saving nets that are all the same.
        if val < best_val:
            improved_enough = val < best_val - args.min_delta
            best_val = val
            best_epoch = epoch
            epochs_since_best = 0 if improved_enough else epochs_since_best + 1
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            # Sidecar metadata: how this net was trained. Read by tools/rate.py
            # so ratings can be logged against data quantity/quality.
            meta.write(args.out, {
                "shards": args.shards,
                "positions": meta.count_positions(args.shards),
                "label_depth": meta.label_depth_of(args.shards),
                "lambda": args.lambda_,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "resumed_from": args.resume,
                "best_val": round(best_val, 5),
                "trained_epoch": epoch,
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "val_split": "fen-hash" if args.split_by_fen else "random-line",
                "min_delta": args.min_delta,
                "patience": args.patience,
            })
            print(f"  saved best -> {args.out} (val {val:.4f})")
        else:
            epochs_since_best += 1

        if args.patience and epochs_since_best >= args.patience:
            print(f"early stop: no val improvement > {args.min_delta:g} in "
                  f"{args.patience} epochs (best {best_val:.4f} at epoch "
                  f"{best_epoch})")
            break


@torch.no_grad()
def _evaluate(model, dl, loss_fn, device) -> float:
    model.eval()
    tot = 0.0
    n = 0
    for stm_i, stm_o, nstm_i, nstm_o, tgt, _w in dl:
        stm_i, stm_o = stm_i.to(device), stm_o.to(device)
        nstm_i, nstm_o = nstm_i.to(device), nstm_o.to(device)
        tgt = tgt.to(device)
        pred = torch.tanh(model(stm_i, stm_o, nstm_i, nstm_o))
        tot += loss_fn(pred, tgt).item() * tgt.size(0)
        n += tgt.size(0)
    return tot / max(n, 1)


if __name__ == "__main__":
    train()
