"""
IQ  ->  degradiertes komplexes Bild  (der Ersatz fuer die gr-tempest-Rohstufe).

Prinzip (identisch zu TempestSDR / gr-tempest, nur schlank in NumPy):

  * Der SDR nimmt komplexes IQ mit Rate fs auf, zentriert auf einer
    Pixeltakt-Harmonischen (z.B. 445.5 MHz fuer 1080p60).
  * Wir resamplen den Strom auf den PIXELTAKT f_p, sodass genau ein komplexes
    Sample pro Pixel entsteht: samples_per_frame = h_total * v_total.
  * Ein Frame-Block wird zu (v_total, h_total) gefaltet -> komplexes Bild.
  * Der aktive Bereich (h_active x v_active) wird nach dem Frame-Sync
    ausgeschnitten.

Frame-Sync
----------
Der Startoffset eines Frames ist unbekannt. Zwei Wege:
  (a) blind  -> estimate_offset() via Zeilen-/Frame-Autokorrelation
  (b) LABOR  -> tempest.align gegen das bekannte Originalbild (robuster).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

from .timing import VideoTiming


# ---------------------------------------------------------------------------
# IQ-Einlesen
# ---------------------------------------------------------------------------
def read_iq(path: str, dtype: str = "complex64", max_samples: int | None = None) -> np.ndarray:
    """
    Liest rohes, verschachteltes IQ.
      dtype='complex64'  -> float32 I, float32 Q (Standard bei SDR-Aufnahmen)
      dtype='complex128' -> float64 I, float64 Q
      dtype='int16'      -> int16 I, int16 Q  (auf [-1,1] skaliert)
    RTSA-Suite PRO exportiert IQ als float32-interleaved; das ist 'complex64'.
    """
    count = -1 if max_samples is None else max_samples * (2 if dtype == "int16" else 1)
    if dtype == "int16":
        raw = np.fromfile(path, dtype=np.int16, count=count).astype(np.float32) / 32768.0
        iq = raw[0::2] + 1j * raw[1::2]
    elif dtype == "complex128":
        iq = np.fromfile(path, dtype=np.complex128, count=count).astype(np.complex64)
    else:
        iq = np.fromfile(path, dtype=np.complex64, count=count)
    return iq


def write_iq(path: str, iq: np.ndarray) -> None:
    iq.astype(np.complex64).tofile(path)


# ---------------------------------------------------------------------------
# Resampling auf Pixeltakt und Faltung in Frames
# ---------------------------------------------------------------------------
def resample_to_pixelclock(iq: np.ndarray, fs: float, timing: VideoTiming,
                           max_denom: int = 4000) -> np.ndarray:
    """Resamplet IQ von Rate fs auf den Pixeltakt f_p (ein Sample pro Pixel)."""
    fp = timing.pixel_clock
    if abs(fs - fp) / fp < 1e-9:
        return iq.astype(np.complex64)
    frac = Fraction(fp / fs).limit_denominator(max_denom)
    up, down = frac.numerator, frac.denominator
    out = resample_poly(iq, up, down)          # arbeitet komplex
    return out.astype(np.complex64)


def fold_frames(iq_pix: np.ndarray, timing: VideoTiming, offset: int = 0):
    """
    Faltet einen auf Pixeltakt resampleten Strom in ganze Frames.
    Rueckgabe: Array (n_frames, v_total, h_total) complex64.
    """
    spf = timing.samples_per_frame
    x = iq_pix[offset:]
    n = len(x) // spf
    if n == 0:
        raise ValueError("Zu wenige Samples fuer einen vollen Frame.")
    x = x[: n * spf].reshape(n, timing.v_total, timing.h_total)
    return x


def crop_active(frame: np.ndarray, timing: VideoTiming) -> np.ndarray:
    """Schneidet den aktiven Bildbereich aus (Blanking entfernen)."""
    return frame[: timing.v_active, : timing.h_active]


def estimate_offset(iq_pix: np.ndarray, timing: VideoTiming, search: int | None = None) -> int:
    """
    Blinder Frame-Sync: findet den Startoffset per Autokorrelation der
    Betragsspur bei der Frame-Periode. Grob, aber ohne Ground Truth nutzbar.
    """
    spf = timing.samples_per_frame
    search = search or spf
    mag = np.abs(iq_pix[: 3 * spf])
    if len(mag) < 2 * spf:
        return 0
    a = mag[:spf]
    b = mag[spf:2 * spf]
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=spf)
    return int(np.argmax(corr[:search]))


def reconstruct(iq: np.ndarray, fs: float, timing: VideoTiming,
                offset: int | None = None, average: bool = True):
    """
    Volle Rohrekonstruktion: IQ -> komplexes (aktives) Frame.
      average=True mittelt ueber alle vollen Frames (starke Rauschminderung,
      da das Bild statisch ist -> genau dein Laborfall).
    Rueckgabe: HxWx2 float32 (Real, Imag) des aktiven Bereichs.
    """
    iq_pix = resample_to_pixelclock(iq, fs, timing)
    if offset is None:
        offset = estimate_offset(iq_pix, timing)
    frames = fold_frames(iq_pix, timing, offset)
    frame = frames.mean(axis=0) if average else frames[0]
    act = crop_active(frame, timing)
    return np.stack([act.real, act.imag], axis=-1).astype(np.float32)
