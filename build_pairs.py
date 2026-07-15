#!/usr/bin/env python3
"""
Baut ausgerichtete Trainingspaare aus echten Aaronia-Captures (Phase 2).

Erwartete Eingabe: ein Verzeichnis, in dem zu jedem Motiv liegen:
    <name>.cf32   -> rohes IQ (complex64) vom SPECTRAN V6
    <name>.png    -> exakt das Bild, das dabei am Monitor stand (Ground Truth)

    python build_pairs.py --in captures --out data/pairs --fs 122e6 --timing 1080p60

Pro Motiv:
  1. IQ -> resample auf Pixeltakt -> ueber alle Frames mitteln (Rauschen runter)
  2. Phasenkorrelation gegen die Kanten des Originals -> pixelgenaue Ausrichtung
  3. Speichert <name>.npy (HxWx2, ausgerichtet) + kopiert <name>.png daneben
Diese Paare frisst dann RealPairs im Fine-Tuning.
"""
import argparse
import glob
import os
import shutil
import numpy as np
from PIL import Image

from tempest import timing as T
from tempest import reconstruct, align, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--timing", default="1080p60")
    ap.add_argument("--dtype", default="complex64")
    ap.add_argument("--preview", action="store_true", help="Magnitude-PNG zur Kontrolle")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tm = T.get(args.timing)

    iqs = sorted(glob.glob(os.path.join(args.inp, "*.cf32")) +
                 glob.glob(os.path.join(args.inp, "*.iq")))
    if not iqs:
        raise SystemExit(f"Keine IQ-Dateien (*.cf32/*.iq) in {args.inp}")

    for path in iqs:
        name = os.path.splitext(os.path.basename(path))[0]
        gt_path = os.path.join(args.inp, name + ".png")
        if not os.path.exists(gt_path):
            print(f"  [skip] kein Ground-Truth-PNG fuer {name}")
            continue

        iq = reconstruct.read_iq(path, dtype=args.dtype)
        deg = reconstruct.reconstruct(iq, args.fs, tm, average=True)

        ref = np.asarray(Image.open(gt_path).convert("L"), np.float32) / 255.0
        if ref.shape != deg.shape[:2]:
            ref = np.asarray(Image.open(gt_path).convert("L")
                             .resize((deg.shape[1], deg.shape[0])), np.float32) / 255.0

        aligned, (dy, dx) = align.align_to_reference(deg, ref)
        np.save(os.path.join(args.out, name + ".npy"), aligned)
        shutil.copy(gt_path, os.path.join(args.out, name + ".png"))
        if args.preview:
            Image.fromarray(synth.to_preview(synth.normalize_complex2(aligned))) \
                 .save(os.path.join(args.out, name + "_mag.png"))
        print(f"  {name}: Versatz (dy={dy}, dx={dx}) -> Paar gespeichert")

    print("Fertig ->", args.out)


if __name__ == "__main__":
    main()
