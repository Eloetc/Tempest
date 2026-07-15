"""
Alignment echter Captures gegen das bekannte Originalbild.

Der Labor-Trick
---------------
Beim ueberwachten Training muss die degradierte Emanation PIXELGENAU auf dem
Original liegen. Blind ist das schwer. Aber im Labor kennst du das Original --
also loest du den Frame-Sync als KREUZKORRELATION der Emanations-Magnitude
gegen die horizontale Ableitung des Originals (die Emanation zeigt v.a. Kanten).

Ablauf pro Motiv:
  1. Viele Frames aufnehmen und mitteln (Rauschen runter) -> reconstruct(average=True)
  2. align_to_reference() findet den ganzzahligen (dy, dx)-Versatz per Phasenkorrelation
  3. Das ausgerichtete HxWx2-Array + das Original-PNG bilden ein Trainingspaar.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = _SOBEL_X.t().contiguous()


def gradient_loss(pred, target):
    """Betont Kanten -> wichtig fuer Textschaerfe."""
    kx = _SOBEL_X.to(pred.device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(pred.device).view(1, 1, 3, 3)
    gx_p, gy_p = F.conv2d(pred, kx, padding=1), F.conv2d(pred, ky, padding=1)
    gx_t, gy_t = F.conv2d(target, kx, padding=1), F.conv2d(target, ky, padding=1)
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def restoration_loss(pred, target, w_grad=0.25):
    return charbonnier(pred, target) + w_grad * gradient_loss(pred, target)


# ---------------------------------------------------------------------------
# Alignment via Phasenkorrelation
# ---------------------------------------------------------------------------
def _h_gradient(img01: np.ndarray) -> np.ndarray:
    g = np.zeros_like(img01, dtype=np.float32)
    g[:, 1:] = np.abs(img01[:, 1:] - img01[:, :-1])
    return g


def phase_correlation(a: np.ndarray, b: np.ndarray):
    """Ganzzahliger (dy, dx)-Versatz, der b auf a schiebt."""
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-8
    r = np.fft.ifft2(R).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dy), int(dx)


def align_to_reference(deg2: np.ndarray, reference01: np.ndarray):
    """
    deg2        : HxWx2 rekonstruierte Emanation (Real, Imag)
    reference01 : HxW  Originalbild in [0,1]
    Rueckgabe   : (ausgerichtetes HxWx2, (dy, dx))
    """
    mag = np.sqrt(deg2[..., 0] ** 2 + deg2[..., 1] ** 2)
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    ref_edges = _h_gradient(reference01)
    ref_edges = (ref_edges - ref_edges.mean()) / (ref_edges.std() + 1e-6)
    dy, dx = phase_correlation(ref_edges, mag)
    aligned = np.roll(deg2, shift=(dy, dx), axis=(0, 1))
    return aligned, (dy, dx)
