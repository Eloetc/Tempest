#!/usr/bin/env python3
"""
Erzeugt saubere Text-Screenshots als Ground Truth fuer das Bootstrap-Training
(Phase 1). Kein echtes Capture noetig.

    python gen_clean_text.py --out data/clean --n 400

Tipp: Mische spaeter ECHTE Screenshots deiner Ziel-Oberflaechen dazu
(Terminals, Editoren, Login-Masken) -- je naeher an der spaeteren Realitaet,
desto besser generalisiert das Netz.
"""
import argparse
import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORDS = ("System Login Passwort Benutzer Root Kernel Netzwerk Firewall Zugriff "
         "Verschluesselung Zertifikat Protokoll Server Datenbank Sitzung Token "
         "Konfiguration Dienst Prozess Speicher Adresse Schnittstelle").split()


def find_fonts():
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    return [c for c in cands if os.path.exists(c)] or [None]


def rand_line(rng):
    kind = rng.random()
    if kind < 0.4:
        return " ".join(rng.choice(WORDS) for _ in range(rng.integers(3, 8)))
    if kind < 0.7:
        u = "".join(rng.choice(list(string.ascii_lowercase)) for _ in range(rng.integers(4, 9)))
        p = "".join(rng.choice(list(string.ascii_letters + string.digits))
                    for _ in range(rng.integers(6, 12)))
        return f"{u}@host:~$ login  pass={p}"
    return "0x" + "".join(rng.choice(list("0123456789abcdef")) for _ in range(rng.integers(6, 16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fonts = find_fonts()

    for i in range(args.n):
        bg = int(rng.integers(230, 256))
        fg = int(rng.integers(0, 60))
        img = Image.new("L", (args.w, args.h), bg)
        d = ImageDraw.Draw(img)
        size = int(rng.integers(16, 30))
        fp = fonts[rng.integers(len(fonts))]
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        y = int(rng.integers(5, 25))
        while y < args.h - size:
            d.text((int(rng.integers(5, 40)), y), rand_line(rng), fill=fg, font=font)
            y += size + int(rng.integers(2, 12))
        img.save(os.path.join(args.out, f"clean_{i:05d}.png"))

    print(f"{args.n} Bilder -> {args.out}")


if __name__ == "__main__":
    main()
