from __future__ import annotations

import importlib
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np


class TestAudioCore(unittest.TestCase):
    """Unit tests for avs.audio_core.AudioAnalyzer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.core = importlib.import_module("avs.audio_core")

    def _bytes_of(self, n: int) -> bytes:
        x = (np.linspace(-0.6, 0.6, n, dtype=np.float32) * (np.iinfo(np.int16).max * 0.7)).astype(np.int16)
        return x.tobytes()

    def test_constants_exact(self) -> None:
        expected_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.assertEqual(list(self.core.PITCH_CLASS_NAMES), expected_names)
        major_expected = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                                  dtype=np.float32)
        minor_expected = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
                                  dtype=np.float32)
        self.assertTrue(np.allclose(self.core.MAJOR_PROFILE, major_expected))
        self.assertTrue(np.allclose(self.core.MINOR_PROFILE, minor_expected))

    def test_append_and_recent_nonempty(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        raw = self._bytes_of(2048)
        an.append_bytes_i16(raw, expected_chunk=2048)
        y = an._recent_from_ring(0.01)
        self.assertGreaterEqual(y.size, 1)

    def test_recent_zero_seconds_empty(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        an.append_bytes_i16(self._bytes_of(1024), expected_chunk=1024)
        out = an._recent_from_ring(0.0)
        self.assertEqual(out.size, 0)

    def test_chroma_short_empty(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        z = an._chroma_short(np.array([], dtype=np.float32), an.RATE)
        self.assertEqual(z.shape, (12,))
        self.assertTrue(np.allclose(z, np.zeros(12, dtype=np.float32)))

    def test_butter_lowpass_filter_paths(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        fs = an.RATE
        cutoff = fs / 16.0

        def _butter(order: int, normal_cutoff: float, btype: str, analog: bool):
            return np.array([1.0, 0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

        def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32) * 3.0

        with patch.object(self.core, "butter", side_effect=_butter), patch.object(self.core, "lfilter",
                                                                                  side_effect=_lfilter):
            x = np.linspace(-1.0, 1.0, num=1024, dtype=np.float32)
            y = an.butter_lowpass_filter(x, cutoff=cutoff, fs=fs, order=4)
        self.assertTrue(np.allclose(y, x * 3.0))

    def test_butter_lowpass_filter_empty(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        with patch.object(self.core, "butter",
                          return_value=(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32))), \
                patch.object(self.core, "lfilter", side_effect=lambda b, a, x: x):
            y = an.butter_lowpass_filter(np.array([], dtype=np.float32), cutoff=an.RATE / 8.0, fs=an.RATE, order=4)
        self.assertEqual(y.size, 0)

    def test_calculate_bpm_zero_peak(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        before = an.bpm
        an.data[:] = 0
        out = an.calculate_bpm()
        self.assertEqual(out, before)
        self.assertEqual(an.bpm, before)

    def test_calculate_bpm_sets_value(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        raw = self._bytes_of(4096)
        an.append_bytes_i16(raw, expected_chunk=4096)
        expected = 120.0
        with patch.object(an, "butter_lowpass_filter",
                          side_effect=lambda d, cutoff, fs, order=5: np.asarray(d, dtype=np.float32)), \
                patch.object(self.core.librosa.beat, "beat_track", return_value=(expected, np.arange(2, dtype=int))):
            out = an.calculate_bpm()
        self.assertAlmostEqual(out, expected)
        self.assertAlmostEqual(an.bpm, expected)

    def test_key_converges_minor(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        chunk = self._bytes_of(2048)
        for _ in range(100):
            an.append_bytes_i16(chunk, expected_chunk=2048)
        tonic_idx = 0
        tonic_name = self.core.PITCH_CLASS_NAMES[tonic_idx]
        minor_vec = np.roll(self.core.MINOR_PROFILE, tonic_idx).astype(np.float32)
        minor_vec /= np.linalg.norm(minor_vec)
        with patch.object(self.core.librosa.effects, "hpss",
                          return_value=(np.ones(2048, dtype=np.float32), np.ones(2048, dtype=np.float32))), \
                patch.object(self.core.librosa.feature, "chroma_cqt",
                             return_value=np.tile(minor_vec.reshape(12, 1), (1, 8))), \
                patch.object(self.core.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            for _ in range(an._key_votes.maxlen):
                an.step_key_estimation(window_seconds=1.0)
        self.assertEqual(an.key, self.core.CAMELot_MINOR[tonic_name])

    def test_key_hysteresis_major(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        chunk = self._bytes_of(2048)
        for _ in range(200):
            an.append_bytes_i16(chunk, expected_chunk=2048)
        idx0, idx1 = 0, 1
        name0 = self.core.PITCH_CLASS_NAMES[idx0]
        name1 = self.core.PITCH_CLASS_NAMES[idx1]
        v0 = np.roll(self.core.MAJOR_PROFILE, idx0).astype(np.float32)
        v1 = np.roll(self.core.MAJOR_PROFILE, idx1).astype(np.float32)
        v0 /= np.linalg.norm(v0)
        v1 /= np.linalg.norm(v1)
        with patch.object(self.core.librosa.effects, "hpss",
                          return_value=(np.ones(2048, dtype=np.float32), np.ones(2048, dtype=np.float32))), \
                patch.object(self.core.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            with patch.object(self.core.librosa.feature, "chroma_cqt", return_value=np.tile(v0.reshape(12, 1), (1, 8))):
                for _ in range(an._key_votes.maxlen):
                    an.step_key_estimation(window_seconds=1.0)
            self.assertEqual(an.key, self.core.CAMELot_MAJOR[name0])
            less = max(1, an._key_votes.maxlen // 2)
            with patch.object(self.core.librosa.feature, "chroma_cqt", return_value=np.tile(v1.reshape(12, 1), (1, 8))):
                for _ in range(less):
                    an.step_key_estimation(window_seconds=1.0)
                self.assertEqual(an.key, self.core.CAMELot_MAJOR[name0])
                for _ in range(an._key_votes.maxlen):
                    an.step_key_estimation(window_seconds=1.0)
                self.assertEqual(an.key, self.core.CAMELot_MAJOR[name1])

    def test_key_undefined_mapping_does_not_change(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        an._ring[:] = 1.0
        an._ring_filled = an._ring.shape[0]
        v = np.ones(12, dtype=np.float32) / np.sqrt(12.0)

        class _StubMap(dict):
            def __getitem__(self, k): return "-"

            def get(self, k, d=None): return "-"

        before = an.key
        with patch.object(self.core.librosa.effects, "hpss",
                          return_value=(np.ones(2048, dtype=np.float32), np.ones(2048, dtype=np.float32))), \
                patch.object(self.core.librosa.feature, "chroma_cqt", return_value=np.tile(v.reshape(12, 1), (1, 4))), \
                patch.object(self.core.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)), \
                patch.object(self.core, "CAMELot_MAJOR", new=_StubMap()), \
                patch.object(self.core, "CAMELot_MINOR", new=_StubMap()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            an.step_key_estimation(window_seconds=8.0)
        self.assertEqual(an.key, before)

    def test_append_bytes_i16_empty_chunk_returns(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        before_filled = an._ring_filled
        an.append_bytes_i16(b"", expected_chunk=0)
        self.assertEqual(an._ring_filled, before_filled)

    def test_recent_from_ring_empty_returns_empty(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        out = an._recent_from_ring(1.0)
        self.assertEqual(out.size, 0)

    def test_step_key_estimation_wrong_chroma_size(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        # fill ring so _recent_from_ring returns something
        an._ring[:] = 1.0
        an._ring_filled = an._ring.shape[0]
        with patch.object(self.core.librosa.effects, "hpss", return_value=(np.ones(128), np.ones(128))), \
                patch.object(self.core.librosa.feature, "chroma_cqt", return_value=np.ones((11, 6))), \
                patch.object(self.core.librosa.util, "normalize", side_effect=lambda y, axis=None: y):
            tonic, camel, conf, current = an.step_key_estimation(window_seconds=1.0)
        self.assertEqual(camel, "-")

    def test_step_key_estimation_camelot_dash_does_not_change(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        an._ring[:] = 1.0
        an._ring_filled = an._ring.shape[0]
        v = np.ones(12, dtype=np.float32) / np.sqrt(12.0)
        with patch.object(self.core.librosa.effects, "hpss", return_value=(np.ones(128), np.ones(128))), \
                patch.object(self.core.librosa.feature, "chroma_cqt", return_value=np.tile(v.reshape(12, 1), (1, 4))), \
                patch.object(self.core.librosa.util, "normalize", side_effect=lambda y, axis=None: y), \
                patch.object(an, "_key_from_chroma", return_value=("C", "-", 0.5)):
            tonic, camel, conf, current = an.step_key_estimation(window_seconds=1.0)
        self.assertEqual(camel, "-")

    def test_append_wraparound_else_branch(self) -> None:
        """Forces ring write to wrap at the end."""
        an = self.core.AudioAnalyzer(rate=48000)
        n = 2048
        an._ring_write_idx = max(0, an._ring.shape[0] - (n // 2))
        raw = (np.arange(n, dtype=np.int16)).tobytes()
        before = an._ring_filled
        an.append_bytes_i16(raw, expected_chunk=n)
        self.assertGreater(an._ring_filled, before)

    def test_append_chunk_bigger_than_display_overwrite(self) -> None:
        """When chunk >= display_len, display is replaced, not shifted."""
        an = self.core.AudioAnalyzer(rate=48000, display_len=64)
        big = (np.arange(128, dtype=np.int16)).tobytes()
        an.append_bytes_i16(big, expected_chunk=128)
        self.assertTrue(np.all(an.data == np.arange(64, 128, dtype=np.int16)))

    def test_step_key_estimation_early_vote_return(self) -> None:
        """Returns early while vote window not full yet."""
        an = self.core.AudioAnalyzer(rate=48000)
        an._ring[:] = 1.0
        an._ring_filled = an._ring.shape[0]
        v = (self.core.MAJOR_PROFILE / np.linalg.norm(self.core.MAJOR_PROFILE)).astype(np.float32)
        with patch.object(self.core.librosa.effects, "hpss",
                          return_value=(np.ones(256, dtype=np.float32), np.ones(256, dtype=np.float32))), \
                patch.object(self.core.librosa.feature, "chroma_cqt",
                             return_value=np.tile(v.reshape(12, 1), (1, 6))), \
                patch.object(self.core.librosa.util, "normalize",
                             side_effect=lambda y, axis=None: np.asarray(y, dtype=np.float32)):
            tonic, camel, conf, current = an.step_key_estimation(window_seconds=1.0)
        self.assertEqual(current, "-")
        self.assertEqual(camel, self.core.CAMELot_MAJOR[self.core.PITCH_CLASS_NAMES[0]])

    def test_append_bytes_size_mismatch_path(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        before = an._ring_filled
        small = (np.arange(8, dtype=np.int16)).tobytes()
        an.append_bytes_i16(small, expected_chunk=16)
        self.assertGreaterEqual(an._ring_filled, before)

    def test_step_key_estimation_empty_ring_early(self) -> None:
        an = self.core.AudioAnalyzer(rate=48000)
        tonic, camel, conf, current = an.step_key_estimation(window_seconds=1.0)
        self.assertEqual((tonic, camel, conf, current), ("-", "-", 0.0, "-"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
