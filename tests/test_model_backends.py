import logging
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from model_backends import ParakeetBackend

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "test-temp"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class FakeParakeetModel:
    def __init__(self, *, fail_on_tensor: bool = False, emit_info: bool = False):
        self.fail_on_tensor = fail_on_tensor
        self.emit_info = emit_info
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.emit_info:
            logging.info("Initializing Lhotse CutSet from a single NeMo manifest")
        first = audio[0]
        if self.fail_on_tensor and isinstance(first, np.ndarray):
            raise RuntimeError("tensor path unsupported")
        return [SimpleNamespace(text="decoded text")]


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def make_backend(fake_model: FakeParakeetModel) -> ParakeetBackend:
    backend = ParakeetBackend.__new__(ParakeetBackend)
    backend._transcribe_lock = nullcontext()
    backend._prefer_tensor_transcribe = True
    backend.model = fake_model
    backend.device = "cpu"
    return backend


class ParakeetBackendTests(unittest.TestCase):
    def test_transcribe_prefers_tensor_path_and_suppresses_info_spam(self):
        backend = make_backend(FakeParakeetModel(emit_info=True))
        handler = _ListHandler()
        root = logging.getLogger()
        previous_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            with mock.patch("model_backends._silence_process_stdio", return_value=nullcontext()):
                with mock.patch("model_backends._temporary_directory_ignore_cleanup_errors", return_value=nullcontext()):
                    with mock.patch("model_backends._manual_temp_path", side_effect=AssertionError("fallback should not run")):
                        result = backend.transcribe(np.array([0.1, -0.2], dtype=np.float32), language="", beam_size=1)
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)

        self.assertEqual(result.text, "decoded text")
        self.assertTrue(backend._prefer_tensor_transcribe)
        self.assertIsInstance(backend.model.calls[0][0][0], np.ndarray)
        self.assertFalse(
            any("Initializing Lhotse CutSet from a single NeMo manifest" in message for message in handler.messages)
        )

    def test_transcribe_falls_back_to_temp_wav(self):
        backend = make_backend(FakeParakeetModel(fail_on_tensor=True))
        wav_path = TEST_TEMP_ROOT / "fallback.wav"
        wav_path.unlink(missing_ok=True)
        with mock.patch("model_backends._silence_process_stdio", return_value=nullcontext()):
            with mock.patch("model_backends._temporary_directory_ignore_cleanup_errors", return_value=nullcontext()):
                with mock.patch("model_backends._manual_temp_path", return_value=wav_path):
                    result = backend.transcribe(np.array([0.1, -0.2], dtype=np.float32), language="", beam_size=1)

        self.assertEqual(result.text, "decoded text")
        self.assertFalse(backend._prefer_tensor_transcribe)
        self.assertIsInstance(backend.model.calls[0][0][0], np.ndarray)
        self.assertIsInstance(backend.model.calls[1][0][0], str)
        self.assertFalse(wav_path.exists())

    def test_warmup_stays_nonfatal_when_tensor_path_fails(self):
        backend = make_backend(FakeParakeetModel(fail_on_tensor=True))
        with mock.patch("model_backends._silence_process_stdio", return_value=nullcontext()):
            with mock.patch("model_backends._temporary_directory_ignore_cleanup_errors", return_value=nullcontext()):
                with mock.patch(
                    "model_backends._manual_temp_path",
                    side_effect=RuntimeError("warmup should not use fallback temp wav"),
                ):
                    backend.warmup(np.zeros(16, dtype=np.float32))

        self.assertFalse(backend._prefer_tensor_transcribe)


if __name__ == "__main__":
    unittest.main()
