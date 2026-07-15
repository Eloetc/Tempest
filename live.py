#!/usr/bin/env python3
"""
Echtzeit-Rekonstruktion aus dem HTTP-IQ-Stream der RTSA-Suite PRO.

    python live.py --url "http://192.168.1.50:54664/stream?format=float" \
                   --fs 122e6 --timing 1080p60 --harmonic 3 \
                   --model runs/real/best.pt --out live.png --interval 1.0

Ablauf je Zyklus:
  1. IQ aus dem Stream abholen (Hintergrund-Thread puffert fortlaufend)
  2. auf Pixeltakt resamplen, in Frames falten, ueber den Block mitteln
  3. gleitender Mittelwert (EMA) ueber Zyklen  -> starke Rauschminderung
     (Bild ist statisch, daher zulaessig und sehr wirksam)
  4. blinder Frame-Sync (kein Ground Truth im Live-Betrieb)
  5. DRUNet-Inferenz -> live.png (mit Bildbetrachter mit Auto-Refresh ansehen)
  6. optional OCR-Text in die Konsole

Beenden mit Strg-C.
"""
import argparse
import time
import numpy as np
import torch
from PIL import Image

from tempest import timing as T
from tempest import reconstruct as R
from tempest import synth
from tempest.httpsource import AaroniaHTTPSource
from tempest.model import DRUNet


def try_ocr(img01: np.ndarray) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    im = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
    return pytesseract.image_to_string(im).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Stream-URL des HTTP-Server-Blocks")
    ap.add_argument("--fs", type=float, required=True, help="Sample-Rate in Hz (= span*1.5)")
    ap.add_argument("--fmt", default="float", choices=["float", "int16"])
    ap.add_argument("--scale", type=float, default=1.0, help="Teiler bei fmt=int16")
    ap.add_argument("--timing", default="1080p60")
    ap.add_argument("--harmonic", type=int, default=3)
    ap.add_argument("--model", help="DRUNet-Checkpoint; ohne = nur Rohrekonstruktion")
    ap.add_argument("--out", default="live.png")
    ap.add_argument("--interval", type=float, default=1.0, help="Sekunden je Zyklus")
    ap.add_argument("--ema", type=float, default=0.7, help="Glaettung 0..1 (hoch=traeger)")
    ap.add_argument("--frames_per_cycle", type=int, default=8)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tm = T.get(args.timing)
    print(f"Timing {tm.name}: Pixeltakt {tm.pixel_clock/1e6:.1f} MHz, "
          f"Harmonische {args.harmonic} @ {tm.harmonic(args.harmonic)/1e6:.1f} MHz")
    print(f"fs = {args.fs/1e6:.3f} MHz  ->  {args.fs/tm.refresh:,.0f} Samples/Frame am Capture-Takt")

    net = None
    if args.model:
        net = DRUNet().to(args.device)
        net.load_state_dict(torch.load(args.model, map_location=args.device))
        net.eval()
        print(f"Modell geladen: {args.model} auf {args.device}")
    else:
        print("Kein Modell -> zeige nur die (gemittelte) Rohrekonstruktion.")

    src = AaroniaHTTPSource(args.url, fmt=args.fmt, scale=args.scale).start()

    # Warmup: genug fuer einen Frame am Capture-Takt
    need = int(args.fs / tm.refresh) + 1
    print(f"Warte auf Stream ({need:,} Samples fuer den ersten Frame) ...")
    if not src.wait_for_samples(need, timeout=20):
        print("Kein/zu wenig Datenfluss. Pruefe URL, Mission (laeuft sie?) und Firewall.")
        src.stop(); return
    print("Stream laeuft. Live-Rekonstruktion startet. Strg-C zum Beenden.\n")

    acc = None      # EMA-Akkumulator (HxWx2)
    offset = None
    cycle = 0
    try:
        while True:
            t0 = time.time()
            iq = src.drain()
            if len(iq) < need:
                time.sleep(max(0.0, args.interval - (time.time() - t0)))
                continue

            iq_pix = R.resample_to_pixelclock(iq, args.fs, tm)
            if offset is None or cycle % 10 == 0:      # Sync gelegentlich neu schaetzen
                offset = R.estimate_offset(iq_pix, tm)
            frames = R.fold_frames(iq_pix, tm, offset)
            k = min(args.frames_per_cycle, len(frames))
            block = frames[:k].mean(axis=0)
            active = R.crop_active(block, tm)
            cur = np.stack([active.real, active.imag], axis=-1).astype(np.float32)

            acc = cur if acc is None else args.ema * acc + (1 - args.ema) * cur

            if net is not None:
                x = synth.normalize_complex2(acc).transpose(2, 0, 1)[None]
                x = torch.from_numpy(x.copy()).float().to(args.device)
                with torch.no_grad():
                    img = net(x).clamp(0, 1)[0, 0].cpu().numpy()
            else:
                img = synth.to_preview(synth.normalize_complex2(acc)).astype(np.float32) / 255.0

            Image.fromarray((img * 255).astype(np.uint8)).save(args.out)
            cycle += 1
            msg = (f"[Zyklus {cycle:4d}] {len(iq):>9,} IQ  {k} Frames gemittelt  "
                   f"offset={offset}  {src.bytes_received/1e6:.1f} MB gesamt  -> {args.out}")
            print(msg)
            if args.ocr:
                txt = try_ocr(img).replace("\n", " ")[:80]
                if txt:
                    print("   OCR:", txt)

            dt = time.time() - t0
            if dt < args.interval:
                time.sleep(args.interval - dt)
    except KeyboardInterrupt:
        print("\nBeendet.")
    finally:
        src.stop()


if __name__ == "__main__":
    main()
