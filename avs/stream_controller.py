from __future__ import annotations

import logging
from typing import Tuple
from collections import deque
import pyaudio
from qtpy import QtCore, QtWidgets
import numpy as np
import librosa
from scipy.signal import butter, lfilter

from avs.mic_thread import MicThread

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

class StreamController(QtWidgets.QWidget):
    def __init__(self):
        """Initialize audio stream controller, state buffers, and logger."""
        super(StreamController, self).__init__()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.data = np.zeros(100000, dtype=np.int16)
        self.CHUNK = 1024 * 2
        self.CHANNELS = 1
        self.RATE = 48000
        self.FORMAT = pyaudio.paInt16
        self.bpm = 0.0
        self.key: str = "-"
        self._key_votes: deque[str] = deque(maxlen=5)
        self._chroma_ema: np.ndarray | None = None
        self._ema_alpha: float = 0.3
        self._ring_seconds: float = 8.0
        self._ring: np.ndarray = np.zeros(int(self.RATE * self._ring_seconds), dtype=np.float32)
        self._ring_write_idx: int = 0
        self._ring_filled: int = 0
        self.audio = None
        self.stream = None
        self._logger.debug("Initialized with RATE=%d CHUNK=%d ring_len=%d", self.RATE, self.CHUNK, self._ring.shape[0])

    def setup_stream(self, selected_mic):
        """Create PyAudio stream and start worker timers."""
        self._logger.info("Opening input device index=%s", selected_mic.get("index"))
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=selected_mic["index"],
        )
        self.micthread = MicThread(self)
        self.micthread.start()
        self._logger.info("Mic thread started")

        self.bpm_timer = QtCore.QTimer()
        self.bpm_timer.timeout.connect(self.calculate_bpm)
        self.bpm_timer.start(5000)
        self._logger.debug("BPM timer started at 5s")

        self._key_timer = QtCore.QTimer()
        self._key_timer.timeout.connect(self.calculate_key_periodic)
        self._key_timer.start(15000)
        self._logger.debug("Key timer started at 15s")

    def butter_lowpass(self, cutoff: float, fs: int, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Design IIR lowpass filter coefficients."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        self._logger.debug("Designed lowpass cutoff=%.1f fs=%d order=%d norm=%.5f", cutoff, fs, order, normal_cutoff)
        return b, a

    def butter_lowpass_filter(self, data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
        """Apply IIR lowpass filter."""
        self._logger.debug("Filtering signal len=%d cutoff=%.1f", data.size, cutoff)
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def calculate_bpm(self) -> None:
        """Estimate BPM from recent audio."""
        peak = float(np.max(np.abs(self.data)))
        self._logger.debug("BPM calc peak=%f", peak)
        if peak == 0.0:
            self._logger.debug("BPM calc skipped due to zero peak")
            return
        signal = self.data.astype(np.float32) / peak
        signal_filtered = self.butter_lowpass_filter(signal, cutoff=10000.0, fs=self.RATE)
        tempo, _ = librosa.beat.beat_track(y=signal_filtered, sr=self.RATE)
        self.bpm = float(tempo)
        self._logger.info("BPM updated to %.2f", self.bpm)

    def append(self, vals: bytes) -> None:
        """Append mic chunk to both the display buffer and the analysis ring."""
        chunk_i16 = np.frombuffer(vals, dtype=np.int16)
        c = self.CHUNK
        if chunk_i16.size != c:
            self._logger.warning("Append chunk size mismatch got=%d expected=%d", chunk_i16.size, c)
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
        self._logger.debug("Ring wrote n=%d write_idx=%d filled=%d", n, self._ring_write_idx, self._ring_filled)

    def breakdown_stream(self) -> None:
        """Stop threads and close audio resources."""
        self._logger.info("Breaking down stream")
        try:
            self.micthread.terminate()
        except Exception as e:
            self._logger.exception("Mic thread terminate error: %s", e)
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception as e:
            self._logger.exception("Stream close error: %s", e)
        try:
            self.audio.terminate()
        except Exception as e:
            self._logger.exception("Audio terminate error: %s", e)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(400, loop.quit)
        loop.exec_()
        self._logger.info("Stream closed")

    def _recent_from_ring(self, seconds: float) -> np.ndarray:
        """Return a recent mono slice from the analysis ring."""
        need = int(self.RATE * seconds)
        if self._ring_filled == 0:
            self._logger.debug("Recent ring request seconds=%.2f got=0", seconds)
            return np.array([], dtype=np.float32)
        got = min(need, self._ring_filled)
        rlen = self._ring.shape[0]
        start = (self._ring_write_idx - got) % rlen
        if start + got <= rlen:
            y = self._ring[start:start + got]
        else:
            first = rlen - start
            y = np.concatenate((self._ring[start:], self._ring[:got - first]))
        self._logger.debug("Recent ring seconds=%.2f need=%d got=%d start=%d", seconds, need, got, start)
        if y.size == 0:
            return y
        y = librosa.util.normalize(y.astype(np.float32), axis=None)
        return y

    def _chroma_short(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Compute robust chroma from harmonic content for low-latency keying."""
        if y.size == 0:
            self._logger.debug("Chroma short skipped due to empty input")
            return np.zeros(12, dtype=np.float32)
        y_harm, _ = librosa.effects.hpss(y)
        C = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=12)
        v = np.nan_to_num(np.median(C, axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n = float(np.linalg.norm(v))
        out = v / n if n > 0.0 else v
        self._logger.debug("Chroma short median norm=%.6f", n)
        return out

    def _key_from_chroma(self, chroma: np.ndarray) -> Tuple[str, str, float]:
        """Infer key, camelot, and confidence from a normalized 12-d chroma vector."""
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
        self._logger.debug("Key from chroma tonic=%s mode=%s camel=%s score=%.6f", str(tonic), mode, camel, float(best))
        return str(tonic), camel, float(best)

    def calculate_key_periodic(self) -> None:
        """Periodically estimate musical key with smoothing and hysteresis."""
        y = self._recent_from_ring(8.0)
        if y.size == 0:
            self._logger.debug("Key calc skipped due to empty window")
            return
        chroma = self._chroma_short(y, self.RATE)
        if chroma.size != 12:
            self._logger.warning("Chroma vector wrong size=%d", chroma.size)
            return
        if self._chroma_ema is None:
            self._chroma_ema = chroma
        else:
            a = self._ema_alpha
            self._chroma_ema = a * chroma + (1.0 - a) * self._chroma_ema
        v = self._chroma_ema / (np.linalg.norm(self._chroma_ema) + 1e-12)
        tonic, camel, conf = self._key_from_chroma(v)
        if camel == "-":
            self._logger.debug("Key calc produced undefined camelot")
            return
        self._key_votes.append(camel)
        self._logger.debug("Key vote appended=%s votes=%s", camel, list(self._key_votes))
        if len(self._key_votes) < self._key_votes.maxlen:
            return
        vals, counts = np.unique(np.array(self._key_votes, dtype=object), return_counts=True)
        winner = vals[int(np.argmax(counts))]
        if winner != self.key:
            if self.key == "-":
                self.key = winner
                self._logger.info("Key initialized to %s", self.key)
            else:
                if int(np.max(counts)) >= 3:
                    old = self.key
                    self.key = winner
                    self._logger.info("Key changed %s -> %s", old, self.key)
        self._logger.debug("Key step tonic=%s conf=%.6f current=%s", tonic, conf, self.key)
