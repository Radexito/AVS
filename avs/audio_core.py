from __future__ import annotations

import logging
from collections import deque
from typing import Tuple
import numpy as np
import librosa
from scipy.signal import butter, lfilter

CAMELot_MAJOR = {
    "C": "8B", "C#": "3B", "Db": "3B", "D": "10B", "D#": "5B", "Eb": "5B",
    "E": "12B", "F": "7B", "F#": "2B", "Gb": "2B", "G": "9B", "G#": "4B",
    "Ab": "4B", "A": "11B", "A#": "6B", "Bb": "6B", "B": "1B",
}
CAMELot_MINOR = {
    "C": "5A", "C#": "12A", "Db": "12A", "D": "7A", "D#": "2A", "Eb": "2A",
    "E": "9A", "F": "4A", "F#": "11A", "Gb": "11A", "G": "6A", "G#": "1A",
    "Ab": "1A", "A": "8A", "A#": "3A", "Bb": "3A", "B": "10A",
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
PITCH_CLASS_NAMES = np.array(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])


class AudioAnalyzer:
    def __init__(self, rate: int = 48000, ring_seconds: float = 8.0, ema_alpha: float = 0.3, display_len: int = 100000):
        """Maintain audio buffers and compute BPM and key."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.RATE: int = rate
        self._ring_seconds: float = ring_seconds
        self._ema_alpha: float = ema_alpha
        self.data: np.ndarray = np.zeros(display_len, dtype=np.int16)
        self._ring: np.ndarray = np.zeros(int(self.RATE * self._ring_seconds), dtype=np.float32)
        self._ring_write_idx: int = 0
        self._ring_filled: int = 0
        self.bpm: float = 0.0
        self.key: str = "-"
        self._key_votes: deque[str] = deque(maxlen=5)
        self._chroma_ema: np.ndarray | None = None
        self._logger.debug("Initialized rate=%d ring_len=%d", self.RATE, self._ring.shape[0])

    def append_bytes_i16(self, vals: bytes, expected_chunk: int | None = None) -> None:
        """Append a chunk of int16 PCM bytes into buffers."""
        chunk_i16 = np.frombuffer(vals, dtype=np.int16)
        if expected_chunk is not None and chunk_i16.size != expected_chunk:
            self._logger.warning("Chunk size mismatch got=%d expected=%d", chunk_i16.size, expected_chunk)
        c = chunk_i16.size
        if c == 0:
            return
        if c >= self.data.size:
            self.data[:] = chunk_i16[-self.data.size:]
        else:
            self.data[:-c] = self.data[c:]
            self.data[-c:] = chunk_i16
        f = chunk_i16.astype(np.float32) / 32768.0
        n = f.size
        rlen = self._ring.shape[0]
        end = self._ring_write_idx + n
        if end <= rlen:
            self._ring[self._ring_write_idx:end] = f
        else:
            first = rlen - self._ring_write_idx
            self._ring[self._ring_write_idx:] = f[:first]
            self._ring[:end - rlen] = f[first:]
        self._ring_write_idx = (self._ring_write_idx + n) % rlen
        self._ring_filled = min(self._ring_filled + n, rlen)

    def butter_lowpass(self, cutoff: float, fs: int, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Design IIR lowpass filter coefficients."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def butter_lowpass_filter(self, data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
        """Apply IIR lowpass filter."""
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def calculate_bpm(self) -> float:
        """Estimate BPM from the display buffer."""
        peak = float(np.max(np.abs(self.data)))
        if peak == 0.0:
            return self.bpm
        signal = self.data.astype(np.float32) / peak
        signal_filtered = self.butter_lowpass_filter(signal, cutoff=10000.0, fs=self.RATE)
        tempo, _ = librosa.beat.beat_track(y=signal_filtered, sr=self.RATE)
        self.bpm = float(tempo)
        return self.bpm

    def _recent_from_ring(self, seconds: float) -> np.ndarray:
        """Return a recent mono slice from the analysis ring."""
        need = int(self.RATE * seconds)
        if self._ring_filled == 0:
            return np.array([], dtype=np.float32)
        got = min(need, self._ring_filled)
        rlen = self._ring.shape[0]
        start = (self._ring_write_idx - got) % rlen
        if start + got <= rlen:
            y = self._ring[start:start + got]
        else:
            first = rlen - start
            y = np.concatenate((self._ring[start:], self._ring[:got - first]))
        if y.size == 0:
            return y
        y = librosa.util.normalize(y.astype(np.float32), axis=None)
        return y

    def _chroma_short(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute median chroma from harmonic content."""
        if y.size == 0:
            return np.zeros(12, dtype=np.float32)
        y_harm, _ = librosa.effects.hpss(y)
        C = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=12)
        v = np.nan_to_num(np.median(C, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n = float(np.linalg.norm(v))
        out = v / n if n > 0.0 else v
        return out

    def _key_from_chroma(self, chroma: np.ndarray) -> Tuple[str, str, float]:
        """Infer key, camelot, and confidence from a normalized chroma vector."""
        major_t = MAJOR_PROFILE / np.linalg.norm(MAJOR_PROFILE)
        minor_t = MINOR_PROFILE / np.linalg.norm(MINOR_PROFILE)
        best = -1e9
        idx = 0
        mode = "major"
        for k in range(12):
            sMaj = float(np.dot(chroma, np.roll(major_t, k)))
            if sMaj > best:
                best = sMaj
                idx = k
                mode = "major"
            sMin = float(np.dot(chroma, np.roll(minor_t, k)))
            if sMin > best:
                best = sMin
                idx = k
                mode = "minor"
        tonic = PITCH_CLASS_NAMES[idx]
        camel = CAMELot_MAJOR.get(tonic, "-") if mode == "major" else CAMELot_MINOR.get(tonic, "-")
        return str(tonic), camel, float(best)

    def step_key_estimation(self, window_seconds: float = 8.0) -> tuple[str, str, float, str]:
        """Update smoothed key estimate and return tonic, camelot, confidence, current."""
        y = self._recent_from_ring(window_seconds)
        if y.size == 0:
            return "-", "-", 0.0, self.key
        chroma = self._chroma_short(y, self.RATE)
        if chroma.size != 12:
            return "-", "-", 0.0, self.key
        if self._chroma_ema is None:
            self._chroma_ema = chroma
        else:
            a = self._ema_alpha
            self._chroma_ema = a * chroma + (1.0 - a) * self._chroma_ema
        v = self._chroma_ema / (np.linalg.norm(self._chroma_ema) + 1e-12)
        tonic, camel, conf = self._key_from_chroma(v)
        if camel == "-":
            return tonic, camel, conf, self.key
        self._key_votes.append(camel)
        if len(self._key_votes) < self._key_votes.maxlen:
            return tonic, camel, conf, self.key
        vals, counts = np.unique(np.array(self._key_votes, dtype=object), return_counts=True)
        winner = vals[int(np.argmax(counts))]
        if winner != self.key:
            if self.key == "-":
                self.key = winner
            else:
                if int(np.max(counts)) >= 3:
                    self.key = winner
        return tonic, camel, conf, self.key
