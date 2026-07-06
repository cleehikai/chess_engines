"""Fine-tune a base NNUE net toward a specific style.

Starts from a trained base checkpoint and continues training on the SAME data,
but with a per-position style weight applied to the loss (see style_weight.py).
The eval targets are unchanged, so playing strength is preserved -- the net just
allocates more capacity to positions the style seeks, which nudges play toward
that style while keeping winning the objective.

This is the "train strength first, then nudge" approach. It is most effective
once the base net is strong (i.e. after GPU-scale training on the 5070), but the
machinery is complete and runnable now.

Usage:
    python -m engine.nnue.finetune --base data/nnue.pt --style sacrificer \
        --shards "data/train.shard*.tsv" --epochs 8 --lr 3e-4 \
        --out data/nnue_sacrificer.pt
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from engine.nnue.dataset import NNUEDataset, collate
from engine.nnue.model import NNUE
from style.profiles import PROFILES, STYLE_NAMES


def finetune() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base checkpoint to start from")
    ap.add_argument("--style", required=True, choices=STYLE_NAMES)
    ap.add_argument("--shards", default="data/train.shard*.tsv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)  # small: gentle nudge
    ap.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"data/nnue_{args.style}.pt"
    profile = PROFILES[args.style]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning base {args.base} -> style '{args.style}' on {device}")

    t0 = time.perf_counter()
    ds = NNUEDataset(args.shards, lambda_=args.lambda_,
                     max_samples=args.max_samples, as_wdl=True,
                     style_profile=profile)
    print(f"  {len(ds):,} samples in {time.perf_counter()-t0:.0f}s")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)

    model = NNUE().to(device)
    model.load_state_dict(torch.load(args.base, map_location=device,
                                     weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        wsum = 0.0
        te = time.perf_counter()
        for stm_i, stm_o, nstm_i, nstm_o, tgt, w in dl:
            stm_i, stm_o = stm_i.to(device), stm_o.to(device)
            nstm_i, nstm_o = nstm_i.to(device), nstm_o.to(device)
            tgt, w = tgt.to(device), w.to(device)
            pred = torch.tanh(model(stm_i, stm_o, nstm_i, nstm_o))
            # Weighted MSE: on-style positions count more.
            loss = (w * (pred - tgt) ** 2).sum() / w.sum().clamp_min(1e-6)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * w.sum().item()
            wsum += w.sum().item()
        print(f"epoch {epoch:2d} | weighted-loss {running/max(wsum,1):.4f} | "
              f"{time.perf_counter()-te:.0f}s", flush=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Saved style-tuned net -> {out}")


if __name__ == "__main__":
    finetune()
