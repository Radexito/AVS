from __future__ import annotations

import importlib
import io
import traceback
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

import numpy as np
from qtpy import QtWidgets


class TestStreamControllerPublic(unittest.TestCase):
    """Public API tests for avs.stream_controller.StreamController."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.sc = importlib.import_module("avs.widgets.stream_controller")
        cls.core = importlib.import_module("avs.audio_core")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._app is not None:
            cls._app.quit()

    def _bytes_one_chunk(self, ctrl) -> bytes:
        arr = (np.linspace(-0.6, 0.6, ctrl.CHUNK, dtype=np.float32) * (np.iinfo(np.int16).max * 0.7)).astype(np.int16)
        return arr.tobytes()

    def test_setup_stream(self) -> None:
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

    def test_append_delegates_to_analyzer(self) -> None:
        ctrl = self.sc.StreamController()
        called = {"n": 0, "size": 0}

        def _append(vals: bytes, expected_chunk: int | None = None) -> None:
            called["n"] += 1
            called["size"] = len(vals)
            self.assertEqual(expected_chunk, ctrl.CHUNK)

        ctrl.analyzer.append_bytes_i16 = _append
        raw = self._bytes_one_chunk(ctrl)
        ctrl.append(raw)
        self.assertEqual(called["n"], 1)
        self.assertEqual(called["size"], len(raw))

    def test_calculate_bpm_uses_analyzer(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl.analyzer.calculate_bpm = MagicMock(return_value=133.0)
        ctrl.calculate_bpm()
        self.assertAlmostEqual(ctrl.bpm, 133.0)

    def test_calculate_key_periodic_propagates(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl.key = "-"
        ctrl.analyzer.step_key_estimation = MagicMock(return_value=("C", "8B", 0.9, "8B"))
        ctrl.calculate_key_periodic()
        self.assertEqual(ctrl.key, "8B")
        ctrl.analyzer.step_key_estimation = MagicMock(return_value=("D", "10B", 0.9, "10B"))
        ctrl.calculate_key_periodic()
        self.assertEqual(ctrl.key, "10B")

    def test_breakdown_stream_event_loop_delay(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl.micthread = types.SimpleNamespace(terminate=MagicMock(side_effect=RuntimeError("boom")))
        ctrl.stream = types.SimpleNamespace(stop_stream=MagicMock(side_effect=RuntimeError("nope")),
                                            close=MagicMock(side_effect=RuntimeError("nah")))
        ctrl.audio = types.SimpleNamespace(terminate=MagicMock(side_effect=RuntimeError("bye")))
        loop_inst = MagicMock()
        with patch.object(self.sc.QtCore, "QEventLoop", return_value=loop_inst), \
             patch.object(self.sc.QtCore.QTimer, "singleShot", side_effect=lambda ms, fn: fn()), \
             patch.object(traceback, "print_exc", return_value=None), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            ctrl.breakdown_stream()
        self.assertTrue(loop_inst.exec_.called)

    def test_breakdown_stream_calls_close(self) -> None:
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

    def test_data_property_exposes_analyzer(self) -> None:
        ctrl = self.sc.StreamController()
        ctrl.analyzer.data[:] = 123
        self.assertTrue(np.all(ctrl.data == ctrl.analyzer.data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
