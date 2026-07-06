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
import os
import time

import torch
from torch.utils.data import DataLoader

from engine.nnue import meta
from engine.nnue.dataset import NNUEDataset, collate
from engine.nnue.model import NNUE


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
    ap.add_argument("--out", default="data/nnue.pt")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.get_num_threads()} CPU threads)")

    print("Loading dataset (encoding features)...")
    t0 = time.perf_counter()
    ds = NNUEDataset(args.shards, lambda_=args.lambda_,
                     max_samples=args.max_samples, as_wdl=True)
    print(f"  {len(ds):,} samples in {time.perf_counter()-t0:.0f}s")

    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model = NNUE().to(device)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(
            torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from {args.resume}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
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

        if val < best_val:
            best_val = val
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
            })
            print(f"  saved best -> {args.out} (val {val:.4f})")


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
