# tempest-ai — lokale KI zur Textrekonstruktion aus HDMI-Emanationen

Schlanke, lokale Neuimplementierung des **deep-tempest**-Ansatzes (Fernández et
al., LADC '24) für dein Laborsetup: abgeschirmtes Zelt mit Monitor + HDMI-Kabel,
zwei Aaronia SPECTRAN V6, Ubuntu-Laptop mit NVIDIA-GPU.

Statt den kompletten GNU-Radio-/gr-tempest-Stack aufzusetzen, ist die Signal­kette
hier in NumPy/PyTorch gebaut — reproduzierbar, offline, und der Laborvorteil
(du hast das Originalbild als Ground Truth) ist direkt eingebaut.

```
 Monitor ─HDMI─▶ Emanation ─▶ SPECTRAN V6 ─▶ IQ (.cf32)
                                                │
                              reconstruct.py: resample→Pixeltakt, Frames falten & mitteln
                                                │  (degradiertes komplexes Bild, 2 Kanäle)
                                                ▼
                                   DRUNet  (model.py)  ── lernt: komplex → sauberes Bild
                                                │
                                                ▼
                              Tesseract-OCR + Character Error Rate (infer.py)
```

## Warum das funktioniert (Kurzfassung)

- HDMI sendet Pixel per TMDS mit dem **Pixeltakt 148,5 MHz** (1080p60). Die
  Emanation liegt auf dessen Harmonischen — **exakt deine 148,5 / 297 / 445,5 MHz**
  (×1, ×2, ×3). Rechne selbst nach: `python -m tempest.timing`.
- deep-tempest' Kerntrick: **nicht** AM-demodulieren, sondern direkt aus den
  komplexen IQ-Samples auf das Bild lernen (inverses Problem). Das übernehmen wir.
- Dein Bild ist **statisch** → über viele Frames mitteln senkt das Rauschen massiv,
  bevor das Netz überhaupt anfängt (`reconstruct(average=True)`).
- Weil du im Labor das Original kennst, ist der Frame-Sync eine **Kreuzkorrelation
  gegen das Original** (`align.py`) statt blinder Autokorrelation — viel robuster.

---

## 1. Installation (Labor-Laptop)

```bash
sudo apt update && sudo apt install tesseract-ocr tesseract-ocr-deu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU-Torch passend zur Karte, z.B.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 2. Capture mit den Aaronia SPECTRAN V6

Der V6 streamt echtes IQ bis 245 MHz RTBW pro Gerät. Drei Wege, an die Daten zu
kommen — **empfohlen: (A) für saubere, reproduzierbare Trainingsdaten.**

**(A) RTSA-Suite PRO → Datei (empfohlen fürs Training).**
Center-Frequenz auf die **3. Harmonische = 445,5 MHz** (bewährt bei deep-tempest,
gute Kantenschärfe für Text). Bandbreite so hoch wie möglich (≥ 80 MHz, gern die
vollen ~245 MHz). Record starten, ein paar Sekunden pro Motiv aufnehmen
(= viele Frames → gutes Mitteln). Exportiere/wandle nach **rohem interleaved
float32-IQ** (`.cf32`, entspricht `complex64`). Notiere die **Sample-Rate `fs`**.

**(B) HTTP-Raw-IQ-Stream** aus RTSA-Suite PRO → live in die Pipeline
(Echtzeit-Modus, siehe Abschnitt 5).

**(C) SoapySDR / GNU-Radio-Plugin** von Aaronia, falls du lieber im GNU-Radio-Graph
bleibst und nur den DL-Teil hier nutzt.

> Zwei V6-Geräte? Zwei Optionen: (1) auf **verschiedene Harmonische** legen
> (z.B. 297 + 445,5 MHz) und die Rekonstruktionen später kombinieren (mehr SNR,
> MIMO-artig), oder (2) per RTSA koppeln für größere Echtzeit-Bandbreite auf einer
> Harmonischen. Fang mit **einem** Gerät auf 445,5 MHz an.

**Datei-Konvention fürs Training:** lege zu jedem Capture das Bild ab, das dabei
am Monitor stand:
```
captures/motiv001.cf32   captures/motiv001.png   (Ground Truth, 1920x1080)
captures/motiv002.cf32   captures/motiv002.png
```

## 3. Training

**Phase 1 — Bootstrap auf synthetischen Daten** (kein Capture nötig, sofort startbar):
```bash
python gen_clean_text.py --out data/clean --n 400        # + eigene Screenshots dazu!
python train.py --clean data/clean --out runs/synth --iters 40000 --batch 8
```
Das synthetische Vorwärtsmodell (`synth.py`) bildet den van-Eck-Look nach
(Geisterkanten, horizontale Unschärfe, komplexes Rauschen). So lernt DRUNet die
Grundaufgabe an unbegrenzten, perfekt ausgerichteten Paaren.

**Phase 2 — Fine-Tuning auf echten Aaronia-Captures:**
```bash
python build_pairs.py --in captures --out data/pairs --fs 122e6 --timing 1080p60 --preview
python train.py --real data/pairs --init runs/synth/best.pt --out runs/real --iters 8000
```
`build_pairs.py` rekonstruiert, mittelt über Frames und richtet jedes Capture
pixelgenau am Original aus. Fine-Tuning passt das Netz an deine reale Hardware,
Antenne und Zelt-Umgebung an — dafür brauchst du deutlich weniger Daten.

## 4. Inferenz / Auswertung

Direkt aus einem IQ-Capture:
```bash
python infer.py --model runs/real/best.pt --iq neu.cf32 --fs 122e6 \
                --timing 1080p60 --harmonic 3 --out ergebnis.png --gt original.png
```
Gibt das rekonstruierte Bild, den OCR-Text und die **CER** gegen das Original aus.
Ziel-Größenordnung laut Paper: von ~90 % CER (roh) auf **< 30 %** (mit Netz).

## 5. Echtzeit-Modus über HTTP (live.py)

Statt Dateien zu verarbeiten, zieht `live.py` den IQ-Strom direkt aus dem
**HTTP-Server-Block** der RTSA-Suite PRO und rekonstruiert fortlaufend.

**RTSA-Suite PRO einrichten (GUI):** Mission bauen
`SPECTRAN V6 (Rx, Center 445,5 MHz, Span groß) → HTTP Server Block`. Der Block
zeigt dir die Stream-URL, typisch `http://<laptop-ip>:54664/stream?format=float`
(oder `?format=int16&scale=1000000`). Optional streamt Rx2 über HTTP, während Rx1
parallel auf Platte aufnimmt.

**Sample-Rate:** Im IQ-Modus ist `fs = span × 1.5` (oder lies `stepFrequency` im
Paket). Diesen Wert gibst du `live.py` mit `--fs`.

```bash
python live.py --url "http://192.168.1.50:54664/stream?format=float" \
               --fs 15e6 --timing 1080p60 --harmonic 3 \
               --model runs/real/best.pt --out live.png --interval 1.0 --ocr
```

Pro Zyklus: IQ abholen → auf Pixeltakt resamplen → Frames falten & mitteln →
gleitender Mittelwert über Zyklen (EMA, weil das Bild statisch ist) → blinder
Frame-Sync → DRUNet → `live.png`. Öffne `live.png` in einem Viewer mit
Auto-Refresh (z.B. `feh -R 1 live.png`). Ohne `--model` siehst du die reine
gemittelte Rohrekonstruktion — praktisch zum Justieren von Frequenz und Antenne.

> Der HTTP-Client (`tempest/httpsource.py`) läuft stdlib-only in einem
> Hintergrund-Thread mit Ringpuffer und automatischem Reconnect. Getestet gegen
> einen Mock-Server: Struktur-Korrelation Rekonstruktion↔Original = 0.997.

---

## Dateien

| Datei | Zweck |
|---|---|
| `tempest/timing.py` | Video-Timings & Pixeltakt-Harmonische (1080p60 → 148,5/297/445,5 MHz) |
| `tempest/reconstruct.py` | IQ → resample auf Pixeltakt → Frames falten/mitteln → komplexes Bild |
| `tempest/synth.py` | Synthetisches Vorwärtsmodell (sauber → van-Eck-Emanation) für Phase 1 |
| `tempest/align.py` | Ground-Truth-Alignment (Phasenkorrelation) + Losses (Charbonnier + Gradient) |
| `tempest/model.py` | DRUNet (32 M Params), Eingang 2 Kanäle (Real/Imag), Ausgang 1 Kanal |
| `tempest/dataset.py` | `SyntheticPairs` (Phase 1) und `RealPairs` (Phase 2), zufällige Patches |
| `tempest/httpsource.py` | Streaming-IQ-Client für den RTSA-HTTP-Server-Block (Thread + Ringpuffer) |
| `gen_clean_text.py` | erzeugt Text-Screenshots als Ground Truth für den Bootstrap |
| `build_pairs.py` | echte Captures → ausgerichtete Trainingspaare |
| `train.py` / `infer.py` | Training bzw. Inferenz+OCR+CER (offline) |
| `live.py` | **Echtzeit-Rekonstruktion aus dem HTTP-IQ-Stream** |

## Ehrliche Grenzen

- Das **synthetische** Modell ist eine plausible Näherung, kein exakter
  TMDS-Simulator. Es bringt DRUNet gut in die Nähe; die letzten Prozentpunkte CER
  holt das **Fine-Tuning auf echten Captures** (Phase 2). Wer es exakter will:
  den GNU-Radio-Simulator aus dem originalen deep-tempest-Repo für die
  synthetische Stufe einhängen — `RealPairs`/`SyntheticPairs` bleiben kompatibel.
- Sauberer **Frame-Sync** ist das A und O. Wenn Ergebnisse schlecht sind, prüfe
  zuerst Ausrichtung (`--preview` in `build_pairs.py`) und `fs`, nicht das Netz.
- Ausschließlich für den **abgeschirmten Laboraufbau** mit eigenem Equipment.
```
