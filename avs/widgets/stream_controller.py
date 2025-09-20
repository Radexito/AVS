from __future__ import annotations

import logging

import numpy as np
import pyaudio
from qtpy import QtCore, QtWidgets
from avs.workers.mic_thread import MicThread
from avs.audio_core import AudioAnalyzer


class StreamController(QtWidgets.QWidget):
    def __init__(self):
        """Qt widget that manages audio devices, threading, and timers."""
        super(StreamController, self).__init__()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.CHUNK: int = 1024 * 2
        self.CHANNELS: int = 1
        self.RATE: int = 48000
        self.FORMAT = pyaudio.paInt16
        self.audio = None
        self.stream = None
        self.analyzer = AudioAnalyzer(rate=self.RATE)
        self.bpm: float = 0.0
        self.key: str = "-"
        self.micthread: MicThread | None = None
        self.bpm_timer: QtCore.QTimer | None = None
        self._key_timer: QtCore.QTimer | None = None
        self._logger.debug("Initialized with RATE=%d CHUNK=%d", self.RATE, self.CHUNK)

    def setup_stream(self, selected_mic):
        """Open the device, start mic thread, and start timers."""
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

    def append(self, vals: bytes) -> None:
        """Append one mic chunk into analysis."""
        self.analyzer.append_bytes_i16(vals, expected_chunk=self.CHUNK)

    def calculate_bpm(self) -> None:
        """Update BPM from analyzer."""
        self.bpm = float(self.analyzer.calculate_bpm())
        self._logger.info("BPM updated to %.2f", self.bpm)

    def calculate_key_periodic(self) -> None:
        """Step key estimation and propagate current key."""
        tonic, camel, conf, current = self.analyzer.step_key_estimation(window_seconds=8.0)
        if current != self.key and current != "-":
            old = self.key
            self.key = current
            if old == "-":
                self._logger.info("Key initialized to %s", self.key)
            else:
                self._logger.info("Key changed %s -> %s", old, self.key)
        self._logger.debug("Key step tonic=%s conf=%.6f current=%s", tonic, conf, self.key)

    def breakdown_stream(self) -> None:
        """Stop threads and close audio resources."""
        self._logger.info("Breaking down stream")
        try:
            if self.micthread is not None:
                self.micthread.terminate()
        except Exception as e:
            self._logger.exception("Mic thread terminate error: %s", e)
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
        except Exception as e:
            self._logger.exception("Stream close error: %s", e)
        try:
            if self.audio is not None:
                self.audio.terminate()
        except Exception as e:
            self._logger.exception("Audio terminate error: %s", e)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(400, loop.quit)
        loop.exec_()
        self._logger.info("Stream closed")

    @property
    def data(self) -> np.ndarray:
        """Display buffer of the latest int16 PCM samples."""
        return self.analyzer.data