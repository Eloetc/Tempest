"""
DRUNet (Denoising Residual U-Net), angepasst an den deep-tempest-Fall.

Architektur (nach Zhang et al., KAIR):
  * U-Net mit 4 Aufloesungsstufen (Kanaele 64/128/256/512)
  * Downsampling per strided conv (2x2, stride 2), Upsampling per transposed conv
  * je Stufe nb Residual-Bloecke (conv-relu-conv + Skip)
  * bias-frei

Anpassung an deep-tempest:
  * Eingang = 2 Kanaele (Real, Imag der komplexen Emanation) statt RGB+Noisemap.
    Wir demodulieren NICHT (kein AM), sondern lernen direkt aus den komplexen
    Samples auf das Bild -- das ist der Kernpunkt des Papers.
  * Ausgang = 1 Kanal (sauberes Graustufenbild; Text ist das Ziel).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def conv(in_c, out_c, k=3, s=1, p=1, bias=False):
    return nn.Conv2d(in_c, out_c, k, s, p, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv(c, c), nn.ReLU(inplace=True), conv(c, c))

    def forward(self, x):
        return x + self.body(x)


def downsample(in_c, out_c):
    return conv(in_c, out_c, k=2, s=2, p=0)


def upsample(in_c, out_c):
    return nn.ConvTranspose2d(in_c, out_c, 2, 2, 0, bias=False)


class DRUNet(nn.Module):
    def __init__(self, in_nc: int = 2, out_nc: int = 1,
                 nc=(64, 128, 256, 512), nb: int = 4):
        super().__init__()
        self.head = conv(in_nc, nc[0])

        self.enc1 = nn.Sequential(*[ResBlock(nc[0]) for _ in range(nb)])
        self.down1 = downsample(nc[0], nc[1])
        self.enc2 = nn.Sequential(*[ResBlock(nc[1]) for _ in range(nb)])
        self.down2 = downsample(nc[1], nc[2])
        self.enc3 = nn.Sequential(*[ResBlock(nc[2]) for _ in range(nb)])
        self.down3 = downsample(nc[2], nc[3])

        self.mid = nn.Sequential(*[ResBlock(nc[3]) for _ in range(nb)])

        self.up3 = upsample(nc[3], nc[2])
        self.dec3 = nn.Sequential(*[ResBlock(nc[2]) for _ in range(nb)])
        self.up2 = upsample(nc[2], nc[1])
        self.dec2 = nn.Sequential(*[ResBlock(nc[1]) for _ in range(nb)])
        self.up1 = upsample(nc[1], nc[0])
        self.dec1 = nn.Sequential(*[ResBlock(nc[0]) for _ in range(nb)])

        self.tail = conv(nc[0], out_nc)

    def forward(self, x):
        # Eingangsgroesse muss durch 8 teilbar sein (3x Downsampling) -> padden
        h, w = x.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")

        x0 = self.head(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        xm = self.mid(self.down3(x3))

        y3 = self.dec3(self.up3(xm) + x3)
        y2 = self.dec2(self.up2(y3) + x2)
        y1 = self.dec1(self.up1(y2) + x1)
        out = self.tail(y1 + x0)

        if ph or pw:
            out = out[..., :h, :w]
        return out


if __name__ == "__main__":
    net = DRUNet()
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(1, 2, 252, 316)   # ungerade Groesse -> testet Padding
    y = net(x)
    print(f"DRUNet: {n_params/1e6:.2f}M Parameter | in {tuple(x.shape)} -> out {tuple(y.shape)}")
