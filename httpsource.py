"""
Streaming-IQ-Client fuer den HTTP-Server-Block der Aaronia RTSA-Suite PRO.

Aufbau in RTSA-Suite PRO (GUI):
    SPECTRAN V6 (Rx, Center = 445.5 MHz, Span so gross wie moeglich)
        -> HTTP Server Block
Der Block zeigt dir die URL. Typisch:
    http://<laptop-ip>:54664/stream?format=float
    http://<laptop-ip>:54664/stream?format=int16&scale=1000000

Datenformat: fortlaufender, roher, interleavter Binaerstrom  I0 Q0 I1 Q1 ...
    format=float  -> float32 little-endian (8 Byte je komplexem Sample)
    format=int16  -> int16  little-endian, geteilt durch 'scale'

Sample-Rate fs bestimmst du NICHT aus dem rohen Stream, sondern aus der Mission:
    im IQ-Modus gilt  fs = span * 1.5  (bzw. packet.stepFrequency im SDK).
Gib fs den nachgelagerten Stufen explizit mit.

Der Client laeuft in einem Hintergrund-Thread (Producer) und puffert komplexe
Samples in einem Ringpuffer; die Live-Schleife holt sie per drain() ab.
"""
from __future__ import annotations
import threading
import time
import urllib.request
from collections import deque

import numpy as np


class AaroniaHTTPSource:
    def __init__(self, url: str, fmt: str = "float", scale: float = 1.0,
                 read_size: int = 1 << 16, max_blocks: int = 4000):
        """
        url       : vollstaendige Stream-URL des HTTP-Server-Blocks
        fmt       : 'float' (float32) oder 'int16'
        scale     : Teiler bei int16 (siehe ?scale=... in der URL)
        read_size : Bytes pro Socket-read
        max_blocks: Ringpuffer-Tiefe (alte Bloecke werden bei Ueberlauf verworfen
                    -> Echtzeit bevorzugt die frischesten Daten)
        """
        self.url = url
        self.fmt = fmt
        self.scale = scale
        self.read_size = read_size
        self._bpc = 8 if fmt == "float" else 4      # Bytes pro komplexem Sample
        self._buf: deque[np.ndarray] = deque(maxlen=max_blocks)
        self._leftover = b""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.bytes_received = 0
        self.connected = False

    # -- Parsing ------------------------------------------------------------
    def _parse(self, raw: bytes) -> np.ndarray:
        data = self._leftover + raw
        n = len(data) // self._bpc
        used = n * self._bpc
        chunk, self._leftover = data[:used], data[used:]
        if not chunk:
            return np.empty(0, np.complex64)
        if self.fmt == "int16":
            a = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / self.scale
        else:
            a = np.frombuffer(chunk, dtype="<f4")
        return (a[0::2] + 1j * a[1::2]).astype(np.complex64)

    # -- Producer-Thread ----------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(self.url, headers={"Accept": "*/*"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    self.connected = True
                    while not self._stop.is_set():
                        raw = r.read(self.read_size)
                        if not raw:
                            break
                        self.bytes_received += len(raw)
                        iq = self._parse(raw)
                        if len(iq):
                            with self._lock:
                                self._buf.append(iq)
            except Exception:
                self.connected = False
                if self._stop.is_set():
                    break
                time.sleep(1.0)   # Reconnect-Versuch

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def drain(self) -> np.ndarray:
        """Gibt alle aktuell gepufferten Samples zurueck und leert den Puffer."""
        with self._lock:
            if not self._buf:
                return np.empty(0, np.complex64)
            out = np.concatenate(list(self._buf))
            self._buf.clear()
        return out

    def wait_for_samples(self, n: int, timeout: float = 15.0) -> bool:
        """Blockiert, bis mindestens n Samples verfuegbar sind (fuer Warmup)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                have = sum(len(b) for b in self._buf)
            if have >= n:
                return True
            time.sleep(0.05)
        return False

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
