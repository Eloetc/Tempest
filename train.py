#!/usr/bin/env python3
"""
Training des DRUNet-Restaurators.

Phase 1 (Bootstrap):   auf SYNTHETISCHEN Paaren -- braucht nur saubere Textbilder.
    python train.py --clean data/clean --out runs/synth --iters 40000

Phase 2 (Fine-Tuning): auf ECHTEN, ausgerichteten Aaronia-Captures.
    python train.py --real data/pairs --init runs/synth/best.pt --out runs/real --iters 8000
"""
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from tempest.model import DRUNet
from tempest.dataset import SyntheticPairs, RealPairs
from tempest.align import restoration_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", help="Verzeichnis mit sauberen Bildern (synthetisch)")
    ap.add_argument("--real", help="Verzeichnis mit .npy/.png-Paaren (Fine-Tuning)")
    ap.add_argument("--init", help="Checkpoint zum Weitertrainieren")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    if args.real:
        ds = RealPairs(args.real, patch=args.patch, length=args.batch * args.iters)
    elif args.clean:
        ds = SyntheticPairs(args.clean, patch=args.patch, length=args.batch * args.iters)
    else:
        raise SystemExit("Gib --clean (synthetisch) oder --real (Fine-Tuning) an.")

    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    pin_memory=(dev.type == "cuda"), drop_last=True)

    net = DRUNet().to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev))
        print(f"Initialisiert von {args.init}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    net.train()
    t0 = time.time()
    running = 0.0
    best = float("inf")
    for it, (x, y) in enumerate(dl, 1):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        pred = net(x)
        loss = restoration_loss(pred, y)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if it % 100 == 0:
            avg = running / 100
            running = 0.0
            ips = it / (time.time() - t0)
            print(f"iter {it:>7d}/{args.iters}  loss {avg:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {ips:.1f} it/s")
            if avg < best:
                best = avg
                torch.save(net.state_dict(), os.path.join(args.out, "best.pt"))
        if it % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.out, f"iter_{it}.pt"))
        if it >= args.iters:
            break

    torch.save(net.state_dict(), os.path.join(args.out, "final.pt"))
    print("Fertig. Bestes Modell:", os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
