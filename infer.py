#!/usr/bin/env python3
"""
Inferenz: rohes Aaronia-IQ ODER ein degradiertes Frame  ->  lesbares Bild + Text.

Von echtem Capture:
    python infer.py --model runs/synth/best.pt --iq capture.cf32 --fs 122e6 \
                    --timing 1080p60 --harmonic 3 --out out.png

Von einem bereits rekonstruierten Frame (.npy HxWx2):
    python infer.py --model runs/synth/best.pt --frame frame.npy --out out.png

Optional --gt original.png fuer die Character Error Rate (CER).
"""
import argparse
import numpy as np
import torch
from PIL import Image

from tempest.model import DRUNet
from tempest import timing as T
from tempest import reconstruct, synth


def ocr(img01: np.ndarray) -> str:
    try:
        import pytesseract
    except ImportError:
        return "[pytesseract nicht installiert]"
    im = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
    return pytesseract.image_to_string(im).strip()


def cer(ref: str, hyp: str) -> float:
    # Levenshtein auf Zeichenebene
    r, h = ref.strip(), hyp.strip()
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + c)
    return d[len(r), len(h)] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iq", help="Rohes IQ (complex64) vom SDR")
    ap.add_argument("--frame", help="Vorrekonstruiertes Frame .npy (HxWx2)")
    ap.add_argument("--fs", type=float, help="Sample-Rate des IQ in Hz (z.B. 122e6)")
    ap.add_argument("--timing", default="1080p60")
    ap.add_argument("--harmonic", type=int, default=3,
                    help="nur informativ: welche Harmonische aufgenommen wurde")
    ap.add_argument("--dtype", default="complex64")
    ap.add_argument("--out", default="reconstructed.png")
    ap.add_argument("--gt", help="Original-PNG fuer CER")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tm = T.get(args.timing)

    if args.iq:
        if not args.fs:
            raise SystemExit("--fs (Sample-Rate) noetig bei --iq")
        iq = reconstruct.read_iq(args.iq, dtype=args.dtype)
        deg = reconstruct.reconstruct(iq, args.fs, tm, average=True)
        print(f"Rekonstruiert aus {len(iq)} IQ-Samples "
              f"(Harmonische {args.harmonic} @ {tm.harmonic(args.harmonic)/1e6:.1f} MHz)")
    elif args.frame:
        deg = np.load(args.frame)
    else:
        raise SystemExit("Gib --iq oder --frame an.")

    x = synth.normalize_complex2(deg).transpose(2, 0, 1)[None]
    x = torch.from_numpy(x.copy()).float().to(args.device)

    net = DRUNet().to(args.device)
    net.load_state_dict(torch.load(args.model, map_location=args.device))
    net.eval()
    with torch.no_grad():
        y = net(x).clamp(0, 1)[0, 0].cpu().numpy()

    Image.fromarray((y * 255).astype(np.uint8)).save(args.out)
    print("Gespeichert:", args.out)

    text = ocr(y)
    print("\n--- OCR ---\n" + text + "\n-----------")
    if args.gt:
        ref = ocr(np.asarray(Image.open(args.gt).convert("L"), np.float32) / 255.0)
        print(f"CER gegen Ground Truth: {cer(ref, text)*100:.1f}%")


if __name__ == "__main__":
    main()
