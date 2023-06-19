import pyaudio
from qtpy import QtCore, QtWidgets
import numpy as np
import librosa
from scipy.signal import butter, lfilter

from mic_thread import MicThread


class StreamController(QtWidgets.QWidget):
    def __init__(self):
        super(StreamController, self).__init__()
        self.data = np.zeros(100000, dtype=np.int16)
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

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def calculate_bpm(self):
        signal = self.data / np.max(np.abs(self.data))
        signal_filtered = self.butter_lowpass_filter(signal, cutoff=10000, fs=self.RATE)
        bpm = librosa.beat.beat_track(y=signal_filtered, sr=self.RATE)
        self.bpm = bpm[0]  # bpm is a tuple, so we take the first element

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
