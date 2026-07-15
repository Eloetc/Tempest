"""
Synthetisches Vorwaertsmodell:  sauberes Bild  ->  degradiertes Emanationsbild.

Warum das der eleganteste Startpunkt ist
-----------------------------------------
Beim ueberwachten Training braucht DRUNet Paare (degradiert, sauber), die
PIXELGENAU ausgerichtet sind. Bei echten Captures ist genau diese Ausrichtung
das schwierigste Problem. deep-tempest loest das, indem sie zuerst auf
SYNTHETISCH erzeugten Paaren trainieren (perfekt ausgerichtet, unbegrenzt viele)
und danach auf echten Captures feinjustieren.

Dieses Modell ist eine PHYSIKALISCH PLAUSIBLE NAEHERUNG, kein exakter
TMDS-Simulator. Es bildet die drei Effekte ab, die das van-Eck-Bild praegen:

  1. Transitions-Betonung: TMDS strahlt Energie an Bit-Uebergaengen ab. Sichtbar
     wird v.a. die horizontale Ableitung des Bildes (vertikale Kanten von Text),
     flaechige Bereiche verschwinden. -> Geisterkanten.
  2. Bandbegrenzung: Der SDR erfasst nur einen Ausschnitt der Bandbreite rund um
     die Harmonische -> horizontale Unschaerfe.
  3. Komplexes Rauschen + Phasenrotation + Sub-Pixel-Jitter des Frame-Syncs.

Ausgabe: 2-Kanal-Array (Real, Imag) -- genau das, was DRUNet als Eingang bekommt
(kein AM-Demod, wir behalten die komplexen Werte, wie im deep-tempest-Paper).
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _to_gray01(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def degrade(
    clean: np.ndarray,
    *,
    edge_gain: float = 1.0,
    dc_gain: float = 0.15,
    h_blur_sigma: float = 0.8,
    v_blur_sigma: float = 0.3,
    snr_db: float = 8.0,
    phase: float | None = None,
    subpixel_shift: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    clean: HxW (oder HxWx3) Bild in [0,255] oder [0,1].
    Rueckgabe: HxWx2 float32 (Real, Imag), normiert.
    """
    rng = rng or np.random.default_rng()
    g = _to_gray01(clean)
    H, W = g.shape

    # 1) Transitions-Betonung: horizontale Ableitung + Rest-DC
    trans = np.zeros_like(g)
    trans[:, 1:] = g[:, 1:] - g[:, :-1]
    resp = dc_gain * g + edge_gain * trans

    # 2) Bandbegrenzung -> anisotrope Unschaerfe (horizontal staerker)
    if h_blur_sigma > 0:
        resp = gaussian_filter1d(resp, h_blur_sigma, axis=1, mode="nearest")
    if v_blur_sigma > 0:
        resp = gaussian_filter1d(resp, v_blur_sigma, axis=0, mode="nearest")

    # 3a) komplexe Traeger-Phase (unbekannte Ankopplung)
    phase = rng.uniform(0, 2 * np.pi) if phase is None else phase
    c = resp.astype(np.complex64) * np.exp(1j * phase)

    # 3b) Sub-Pixel-Shift (Frame-Sync-Jitter) via horizontale Roll-Interpolation
    shift = rng.uniform(-0.5, 0.5) if subpixel_shift is None else subpixel_shift
    if abs(shift) > 1e-3:
        f = np.fft.fft(c, axis=1)
        k = np.fft.fftfreq(W)
        c = np.fft.ifft(f * np.exp(-2j * np.pi * k * shift)[None, :], axis=1)

    # 3c) komplexes AWGN nach Ziel-SNR
    sig_p = np.mean(np.abs(c) ** 2) + 1e-12
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    noise = (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))
    noise *= np.sqrt(noise_p / 2.0)
    c = c + noise

    out = np.stack([c.real, c.imag], axis=-1).astype(np.float32)
    return out


def normalize_complex2(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zentriert/Skaliert ein HxWx2 Array pro Frame auf Einheitsvarianz."""
    m = x.mean(axis=(0, 1), keepdims=True)
    s = x.std(axis=(0, 1), keepdims=True) + eps
    return (x - m) / s


def to_preview(x2: np.ndarray) -> np.ndarray:
    """HxWx2 -> Magnitudenbild uint8 fuer Anzeige/Debug."""
    mag = np.sqrt(x2[..., 0] ** 2 + x2[..., 1] ** 2)
    lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
    mag = np.clip((mag - lo) / (hi - lo + 1e-6), 0, 1)
    return (mag * 255).astype(np.uint8)
