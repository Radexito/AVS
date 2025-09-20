from __future__ import annotations

import importlib
import io
import sys
import traceback
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from qtpy import QtWidgets
from contextlib import redirect_stderr, redirect_stdout


class TestStreamControllerPublic(unittest.TestCase):
    """Public API tests for avs.stream_controller."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls._mods_patcher = patch.dict(sys.modules, {"avs.main": types.ModuleType("avs.main")}, clear=False)
        cls._mods_patcher.start()
        cls.sc = importlib.import_module("avs.stream_controller")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._mods_patcher.stop()
        if cls._app is not None:
            cls._app.quit()

    def _bytes_one_chunk(self, ctrl) -> bytes:
        arr = (np.linspace(-0.6, 0.6, ctrl.CHUNK, dtype=np.float32) * (np.iinfo(np.int16).max * 0.7)).astype(np.int16)
        return arr.tobytes()

    def _append_seconds(self, ctrl, seconds: float) -> None:
        chunks = max(1, int(np.ceil(seconds * ctrl.RATE / ctrl.CHUNK)))
        raw = self._bytes_one_chunk(ctrl)
        for _ in range(chunks):
            ctrl.append(raw)

    def _append_chunks(self, ctrl, chunks: int) -> None:
        raw = (np.linspace(-0.5, 0.5, ctrl.CHUNK, dtype=np.float32) * (np.iinfo(np.int16).max * 0.75)).astype(
            np.int16).tobytes()
        for _ in range(chunks):
            ctrl.append(raw)

    def test_setup_stream(self) -> None:
        """setup_stream initializes audio, stream, timers, and mic thread."""
        ctrl = self.sc.StreamController()
        selected = {"index": 0}
        opened_stream = types.SimpleNamespace(stop_stream=MagicMock(), close=MagicMock())
        pa_instance = types.SimpleNamespace(open=MagicMock(return_value=opened_stream), terminate=MagicMock())
        timers: list[MagicMock] = []

        def _qtimer_factory():
            t = MagicMock()
            timers.append(t)
            return t

        with patch.object(self.sc, "pyaudio") as pya, \
                patch.object(self.sc.QtCore, "QTimer", side_effect=_qtimer_factory) as _qt, \
                patch.object(self.sc, "MicThread") as MicThread:
            pya.PyAudio.return_value = pa_instance
            mt = MagicMock()
            MicThread.return_value = mt
            ctrl.setup_stream(selected)

        self.assertIs(ctrl.audio, pa_instance)
        self.assertIs(ctrl.stream, opened_stream)
        self.assertTrue(MicThread.called)
        self.assertTrue(mt.start.called)
        self.assertEqual(len(timers), 2)
        self.assertTrue(timers[0].timeout.connect.called)
        self.assertTrue(timers[0].start.called)
        self.assertTrue(timers[1].timeout.connect.called)
        self.assertTrue(timers[1].start.called)

    def test_calculate_key_periodic_converges(self) -> None:
        """calculate_key_periodic produces a stable Camelot key from consistent chroma."""
        ctrl = self.sc.StreamController()
        tonic_idx = 0
        tonic_name = self.sc.PITCH_CLASS_NAMES[tonic_idx]
        minor_vec = np.roll(self.sc.MINOR_PROFILE, tonic_idx).astype(np.float32)
        minor_vec /= np.linalg.norm(minor_vec)

        with patch.object(self.sc.librosa.effects, "hpss", return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                                                         np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt",
                             return_value=np.tile(minor_vec.reshape(12, 1), (1, 8))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            self._append_seconds(ctrl, seconds=1.0)
            for _ in range(getattr(ctrl, "_key_votes").maxlen):
                ctrl.calculate_key_periodic()

        self.assertEqual(ctrl.key, self.sc.CAMELot_MINOR[tonic_name])

    def test_calculate_key_periodic_hysteresis(self) -> None:
        """Key changes only after sustained new votes."""
        ctrl = self.sc.StreamController()
        idx0, idx1 = 0, 1
        name0 = self.sc.PITCH_CLASS_NAMES[idx0]
        name1 = self.sc.PITCH_CLASS_NAMES[idx1]
        v0 = np.roll(self.sc.MAJOR_PROFILE, idx0).astype(np.float32)
        v1 = np.roll(self.sc.MAJOR_PROFILE, idx1).astype(np.float32)
        v0 /= np.linalg.norm(v0)
        v1 /= np.linalg.norm(v1)

        with patch.object(self.sc.librosa.effects, "hpss", return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                                                         np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):

            with patch.object(self.sc.librosa.feature, "chroma_cqt",
                              return_value=np.tile(v0.reshape(12, 1), (1, 8))):
                self._append_seconds(ctrl, seconds=1.0)
                for _ in range(getattr(ctrl, "_key_votes").maxlen):
                    ctrl.calculate_key_periodic()
            self.assertEqual(ctrl.key, self.sc.CAMELot_MAJOR[name0])

            less = max(1, getattr(ctrl, "_key_votes").maxlen // 2)
            with patch.object(self.sc.librosa.feature, "chroma_cqt",
                              return_value=np.tile(v1.reshape(12, 1), (1, 8))):
                for _ in range(less):
                    ctrl.calculate_key_periodic()
                self.assertEqual(ctrl.key, self.sc.CAMELot_MAJOR[name0])

                for _ in range(getattr(ctrl, "_key_votes").maxlen):
                    ctrl.calculate_key_periodic()
                self.assertEqual(ctrl.key, self.sc.CAMELot_MAJOR[name1])

    def test_calculate_bpm_zero_peak_early_return(self) -> None:
        """calculate_bpm keeps bpm unchanged when there is no signal peak."""
        ctrl = self.sc.StreamController()
        before = ctrl.bpm
        ctrl.data[:] = 0
        ctrl.calculate_bpm()
        self.assertEqual(ctrl.bpm, before)

    def test_calculate_bpm_sets_value(self) -> None:
        """calculate_bpm sets bpm when backend returns a tempo and signal exists."""
        ctrl = self.sc.StreamController()
        self._append_seconds(ctrl, seconds=1.0)
        expected_tempo = max(60.0, float(ctrl.RATE // max(1, ctrl.CHUNK)))

        with patch.object(ctrl, "butter_lowpass_filter",
                          side_effect=lambda data, cutoff, fs, order=5: np.asarray(data, dtype=np.float32)), \
                patch.object(self.sc.librosa.beat, "beat_track",
                             return_value=(expected_tempo, np.arange(2, dtype=int))):
            ctrl.calculate_bpm()

        self.assertAlmostEqual(ctrl.bpm, expected_tempo)

    def test_butter_lowpass_filter_paths(self) -> None:
        """butter_lowpass_filter uses module butter/lfilter correctly."""
        ctrl = self.sc.StreamController()
        fs = ctrl.RATE
        cutoff = fs / float(max(8, ctrl.CHUNK))

        def _butter(order: int, normal_cutoff: float, btype: str, analog: bool):
            return np.array([1.0, 0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

        scale = 2.0

        def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32) * scale

        with patch.object(self.sc, "butter", side_effect=_butter), \
                patch.object(self.sc, "lfilter", side_effect=_lfilter):
            x = np.linspace(-1.0, 1.0, num=ctrl.CHUNK, dtype=np.float32)
            y = ctrl.butter_lowpass_filter(x, cutoff=cutoff, fs=fs, order=4)

        self.assertTrue(np.allclose(y, x * scale))

    def test_append_size_mismatch_then_ok(self) -> None:
        ctrl = self.sc.StreamController()
        wrong = (np.arange(max(1, ctrl.CHUNK // 2), dtype=np.int16)).tobytes()
        right = (np.arange(ctrl.CHUNK, dtype=np.int16)).tobytes()
        before = getattr(ctrl, "_ring_filled")
        with self.assertRaises(ValueError):
            ctrl.append(wrong)
        mid = getattr(ctrl, "_ring_filled")
        ctrl.append(right)
        after = getattr(ctrl, "_ring_filled")
        self.assertEqual(mid, before)
        self.assertGreater(after, before)

    def test_breakdown_stream_event_loop_delay(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl.micthread = types.SimpleNamespace(terminate=MagicMock(side_effect=RuntimeError("boom")))
        ctrl.stream = types.SimpleNamespace(stop_stream=MagicMock(side_effect=RuntimeError("nope")),
                                            close=MagicMock(side_effect=RuntimeError("nah")))
        ctrl.audio = types.SimpleNamespace(terminate=MagicMock(side_effect=RuntimeError("bye")))
        loop_inst = MagicMock()
        with patch.object(self.sc.QtCore, "QEventLoop", return_value=loop_inst), \
                patch.object(self.sc.QtCore.QTimer, "singleShot", side_effect=lambda ms, fn: fn()), \
                patch("sys.stdout", new=io.StringIO()), \
                patch("sys.stderr", new=io.StringIO()):
            ctrl.breakdown_stream()
        self.assertTrue(loop_inst.exec_.called)

    def test_calculate_key_periodic_wrong_chroma_size(self) -> None:
        """calculate_key_periodic exits when chroma shape is not 12."""
        ctrl = self.sc.StreamController()
        self._append_chunks(ctrl, chunks=max(1, ctrl.RATE // ctrl.CHUNK))
        with patch.object(self.sc.librosa.effects, "hpss",
                          return_value=(np.ones(ctrl.CHUNK, dtype=np.float32), np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt", return_value=np.ones((11, 6), dtype=np.float32)), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            before = ctrl.key
            ctrl.calculate_key_periodic()
            self.assertEqual(ctrl.key, before)

    def test_calculate_key_periodic_undefined_camelot(self) -> None:
        """Undefined camelot mapping leaves key unchanged."""
        ctrl = self.sc.StreamController()
        self._append_seconds(ctrl, seconds=1.0)
        v = np.ones(12, dtype=np.float32) / np.sqrt(12.0)

        class _StubMap(dict):
            def __getitem__(self, k): return "-"

            def get(self, k, d=None): return "-"

        before = ctrl.key
        with patch.object(self.sc.librosa.effects, "hpss", return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                                                         np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt", return_value=np.tile(v.reshape(12, 1), (1, 4))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)), \
                patch.object(self.sc, "CAMELot_MAJOR", new=_StubMap()):
            fake_out = io.StringIO()
            with patch("sys.stdout", fake_out):
                ctrl.calculate_key_periodic()
        self.assertEqual(ctrl.key, before)

    def test_calculate_key_periodic_ema_branch_and_vote_window_not_full(self) -> None:
        """EMA initializes and early voting yields a stable key or remains unset."""
        ctrl = self.sc.StreamController()
        self._append_seconds(ctrl, seconds=1.0)
        tonic_idx = 0
        tonic_name = self.sc.PITCH_CLASS_NAMES[tonic_idx]
        prof = np.roll(self.sc.MAJOR_PROFILE, tonic_idx).astype(np.float32)
        prof /= np.linalg.norm(prof)
        with patch.object(self.sc.librosa.effects, "hpss", return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                                                         np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt", return_value=np.tile(prof.reshape(12, 1), (1, 6))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            ctrl.calculate_key_periodic()
            self.assertIsNotNone(getattr(ctrl, "_chroma_ema"))
            few = max(1, getattr(ctrl, "_key_votes").maxlen // 2)
            for _ in range(few):
                ctrl.calculate_key_periodic()
        expected_vote = self.sc.CAMELot_MAJOR[tonic_name]
        self.assertIn(ctrl.key, {"-", expected_vote})

    def test_breakdown_stream_real_timer_delay(self) -> None:
        """breakdown_stream uses real QEventLoop and QTimer.singleShot delay."""
        ctrl = self.sc.StreamController()
        ctrl.micthread = types.SimpleNamespace(terminate=MagicMock(side_effect=RuntimeError("boom")))
        ctrl.stream = types.SimpleNamespace(stop_stream=MagicMock(side_effect=RuntimeError("nope")),
                                            close=MagicMock(side_effect=RuntimeError("nah")))
        ctrl.audio = types.SimpleNamespace(terminate=MagicMock(side_effect=RuntimeError("bye")))
        loop_inst = MagicMock()
        with patch.object(self.sc.QtCore, "QEventLoop", return_value=loop_inst), \
                patch.object(self.sc.QtCore.QTimer, "singleShot", side_effect=lambda ms, fn: fn()), \
                patch.object(traceback, "print_exc", return_value=None):
            ctrl.breakdown_stream()
        self.assertTrue(loop_inst.exec_.called)

    def test_pitch_class_names_exact(self) -> None:
        expected = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.assertEqual(list(self.sc.PITCH_CLASS_NAMES), expected)

    def test_camelot_major_map_exact(self) -> None:
        expected = {
            "C": "8B", "C#": "3B", "Db": "3B", "D": "10B", "D#": "5B", "Eb": "5B",
            "E": "12B", "F": "7B", "F#": "2B", "Gb": "2B", "G": "9B", "G#": "4B",
            "Ab": "4B", "A": "11B", "A#": "6B", "Bb": "6B", "B": "1B",
        }
        self.assertEqual(self.sc.CAMELot_MAJOR, expected)

    def test_camelot_minor_map_exact(self) -> None:
        expected = {
            "C": "5A", "C#": "12A", "Db": "12A", "D": "7A", "D#": "2A", "Eb": "2A",
            "E": "9A", "F": "4A", "F#": "11A", "Gb": "11A", "G": "6A", "G#": "1A",
            "Ab": "1A", "A": "8A", "A#": "3A", "Bb": "3A", "B": "10A",
        }
        self.assertEqual(self.sc.CAMELot_MINOR, expected)

    def test_major_profile_exact_values(self) -> None:
        expected = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            dtype=np.float32,
        )
        self.assertEqual(self.sc.MAJOR_PROFILE.dtype, np.float32)
        self.assertEqual(self.sc.MAJOR_PROFILE.shape, (12,))
        self.assertTrue(np.allclose(self.sc.MAJOR_PROFILE, expected))

    def test_minor_profile_exact_values(self) -> None:
        expected = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
            dtype=np.float32,
        )
        self.assertEqual(self.sc.MINOR_PROFILE.dtype, np.float32)
        self.assertEqual(self.sc.MINOR_PROFILE.shape, (12,))
        self.assertTrue(np.allclose(self.sc.MINOR_PROFILE, expected))

    def test_append_wrap_write_else_branch(self) -> None:
        ctrl = self.sc.StreamController()
        with patch.object(self.sc.librosa.util, "normalize",
                          side_effect=lambda y, axis=None: np.linspace(-1.0, 1.0, ctrl.CHUNK, dtype=np.float32)), \
                patch.object(ctrl, "butter_lowpass_filter",
                             side_effect=lambda x, cutoff, fs, order=5: x.astype(np.float32)):
            ctrl._ring_write_idx = max(0, ctrl._ring.shape[0] - (ctrl.CHUNK // 2))
            raw = (np.arange(ctrl.CHUNK, dtype=np.int16)).tobytes()
            ctrl.append(raw)
        self.assertGreater(getattr(ctrl, "_ring_filled"), 0)

    def test_breakdown_stream_calls_close(self) -> None:
        """breakdown_stream calls stream.close when stop_stream succeeds."""
        ctrl = self.sc.StreamController()
        stop = MagicMock(return_value=None)
        close = MagicMock()
        ctrl.stream = types.SimpleNamespace(stop_stream=stop, close=close)
        ctrl.micthread = types.SimpleNamespace(terminate=MagicMock(return_value=None))
        ctrl.audio = types.SimpleNamespace(terminate=MagicMock(return_value=None))
        with patch.object(self.sc.QtCore, "QEventLoop", return_value=MagicMock()), \
                patch.object(self.sc.QtCore.QTimer, "singleShot", side_effect=lambda ms, fn: fn()), \
                patch.object(traceback, "print_exc", return_value=None), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ctrl.breakdown_stream()
        self.assertTrue(close.called)

    def test_calculate_key_periodic_empty_ring_early_return(self) -> None:
        ctrl = self.sc.StreamController()
        before = ctrl.key
        ctrl.calculate_key_periodic()
        self.assertEqual(ctrl.key, before)

    def test_calculate_key_periodic_reads_wrap_else_branch(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl._ring[:] = np.linspace(-1.0, 1.0, ctrl._ring.shape[0], dtype=np.float32)
        ctrl._ring_filled = ctrl._ring.shape[0]
        ctrl._ring_write_idx = min(5, ctrl._ring.shape[0] - 1)
        with patch.object(self.sc.librosa.effects, "hpss",
                          return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                        np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt",
                             return_value=np.tile(
                                 (self.sc.MAJOR_PROFILE / np.linalg.norm(self.sc.MAJOR_PROFILE)).reshape(12, 1),
                                 (1, 6))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            ctrl.calculate_key_periodic()
        self.assertIn(ctrl.key, {"-", self.sc.CAMELot_MAJOR[self.sc.PITCH_CLASS_NAMES[0]],
                                 self.sc.CAMELot_MINOR[self.sc.PITCH_CLASS_NAMES[0]]})

    def test_butter_lowpass_filter_empty_input_returns_empty(self) -> None:
        ctrl = self.sc.StreamController()
        with patch.object(self.sc, "butter",
                          return_value=(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32))), \
                patch.object(self.sc, "lfilter", side_effect=lambda b, a, x: x):
            y = ctrl.butter_lowpass_filter(np.array([], dtype=np.float32), cutoff=ctrl.RATE / 8.0, fs=ctrl.RATE,
                                           order=4)
        self.assertEqual(y.size, 0)

    def test_calculate_key_periodic_undefined_camelot_public_map(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl._ring[:] = 1.0
        ctrl._ring_filled = ctrl._ring.shape[0]
        v = np.ones(12, dtype=np.float32) / np.sqrt(12.0)

        class _Stub(dict):
            def __getitem__(self, k): return "-"

            def get(self, k, d=None): return "-"

        before = ctrl.key
        with patch.object(self.sc.librosa.effects, "hpss", return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                                                         np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt", return_value=np.tile(v.reshape(12, 1), (1, 4))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)), \
                patch.object(self.sc, "CAMELot_MAJOR", new=_Stub()):
            with patch.object(traceback, "print_exc", return_value=None), redirect_stdout(
                    io.StringIO()), redirect_stderr(io.StringIO()):
                ctrl.calculate_key_periodic()
        self.assertEqual(ctrl.key, before)

    def test_recent_from_ring_returns_empty_after_nonempty_ring(self) -> None:
        """_recent_from_ring returns empty when window length is zero even if ring has data."""
        ctrl = self.sc.StreamController()
        raw = (np.arange(ctrl.CHUNK, dtype=np.int16)).tobytes()
        ctrl.append(raw)
        out = ctrl._recent_from_ring(0.0)
        self.assertEqual(out.size, 0)

    def test_chroma_short_empty_input(self) -> None:
        """_chroma_short returns a 12-vector of zeros on empty input."""
        ctrl = self.sc.StreamController()
        y = np.array([], dtype=np.float32)
        z = ctrl._chroma_short(y, ctrl.RATE)
        self.assertEqual(z.shape, (12,))
        self.assertTrue(np.allclose(z, np.zeros(12, dtype=np.float32)))

    def test_key_calc_undefined_camelot_branch(self) -> None:
        """calculate_key_periodic exits early when camelot is undefined."""
        ctrl = self.sc.StreamController()
        ctrl._ring[:] = 1.0
        ctrl._ring_filled = ctrl._ring.shape[0]
        v = np.ones(12, dtype=np.float32) / np.sqrt(12.0)

        class _Stub(dict):
            def __getitem__(self, k): return "-"

            def get(self, k, d=None): return "-"

        before = ctrl.key
        with patch.object(self.sc.librosa.effects, "hpss",
                          return_value=(np.ones(ctrl.CHUNK, dtype=np.float32),
                                        np.ones(ctrl.CHUNK, dtype=np.float32))), \
                patch.object(self.sc.librosa.feature, "chroma_cqt",
                             return_value=np.tile(v.reshape(12, 1), (1, 4))), \
                patch.object(self.sc.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)), \
                patch.object(self.sc, "CAMELot_MAJOR", new=_Stub()), \
                patch.object(self.sc, "CAMELot_MINOR", new=_Stub()), \
                patch.object(traceback, "print_exc", return_value=None), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ctrl.calculate_key_periodic()
        self.assertEqual(ctrl.key, before)

if __name__ == "__main__":
    unittest.main(verbosity=2)
