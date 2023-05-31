import pyaudio
import numpy as np
from qtpy import QtCore, QtWidgets
from librosa.feature.rhythm import tempo

from mic_thread import MicThread


class StreamController(QtWidgets.QWidget):
    def __init__(self):
        super(StreamController, self).__init__()
        self.data = np.zeros(100000, dtype=np.int32)
        self.CHUNK = 1024 * 2
        self.CHANNELS = 1
        self.RATE = 48000
        self.FORMAT = pyaudio.paInt16
        self.bpm = 0
        self.audio = None
        self.stream = None

    def setup_stream(self, selected_mic):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=selected_mic['index'])
        self.micthread = MicThread(self)
        self.micthread.start()
        # Add a QTimer to periodically calculate the BPM
        self.bpm_timer = QtCore.QTimer()
        self.bpm_timer.timeout.connect(self.calculate_bpm)
        self.bpm_timer.start(5000)  # Adjust the interval as needed

    def calculate_bpm(self):
        signal = self.data / np.max(np.abs(self.data))
        bpm = tempo(y=signal, sr=self.RATE, hop_length=self.CHUNK)
        self.bpm = bpm.item()

    def append(self, vals):
        vals = np.frombuffer(vals, 'int16')
        c = self.CHUNK
        self.data[:-c] = self.data[c:]
        self.data[-c:] = vals

    def breakdown_stream(self):
        self.micthread.terminate()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(400, loop.quit)
        loop.exec_()
