from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest import mock

from file_drop_worker import FileDropWorker, resolve_settings
from model_backends import TranscriptResult


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def transcribe(self, audio, language: str, beam_size: int):
        del audio, language, beam_size
        return TranscriptResult(
            "hello world",
            {
                "onnx_provider_active": "TensorrtExecutionProvider",
                "encoder_inference_ms": 12.5,
                "decoder_inference_ms": 7.5,
            },
        )


class FileDropWorkerTests(unittest.TestCase):
    def _dirs(self) -> dict[str, Path]:
        root = ROOT / ".tmp" / "unit-tests" / str(int(time.time() * 1000))
        return {
            "input": root / "input",
            "text": root / "text",
            "done": root / "done",
            "failed": root / "failed",
        }

    def test_resolve_settings_applies_onnx_provider_override(self) -> None:
        cfg = {
            "models": {
                "final_backend": "parakeet-onnx",
                "final": "demo-model",
                "final_device": "cuda",
                "final_compute": "float16",
                "language": "en",
                "parakeet_onnx": {"provider": "cuda"},
            },
            "file_drop": {},
        }
        settings = resolve_settings(cfg, onnx_provider="tensorrt")
        self.assertEqual(settings.final_backend, "parakeet-onnx")
        self.assertEqual(settings.backend_options["provider"], "tensorrt")

    def test_process_claimed_file_writes_text_json_and_moves_audio(self) -> None:
        dirs = self._dirs()
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        source = dirs["input"] / "sample.processing-999.wav"
        source.write_bytes(b"wav")
        cfg = {
            "audio": {"rate": 16000},
            "filters": {"ignored_exact_phrases": [], "disfluency_words": []},
            "models": {
                "final_backend": "parakeet",
                "final": "demo-model",
                "final_device": "cuda",
                "final_compute": "float16",
                "language": "en",
                "no_speech_threshold": 0.45,
                "log_prob_threshold": -0.8,
            },
            "file_drop": {
                "input_dir": str(dirs["input"]),
                "text_dir": str(dirs["text"]),
                "done_dir": str(dirs["done"]),
                "failed_dir": str(dirs["failed"]),
            },
        }
        settings = resolve_settings(
            cfg,
            input_dir=str(dirs["input"]),
            text_dir=str(dirs["text"]),
            done_dir=str(dirs["done"]),
            failed_dir=str(dirs["failed"]),
        )
        worker = FileDropWorker(cfg, settings)
        worker._backend = FakeBackend()

        with mock.patch("file_drop_worker._load_audio_for_transcription") as load_audio:
            load_audio.return_value = (
                [0.0, 0.1, 0.2],
                {
                    "source_sample_rate": 16000,
                    "target_sample_rate": 16000,
                    "channels": 1,
                    "audio_load_ms": 1.0,
                    "resample_ms": 0.0,
                    "audio_duration_s": 3.0,
                },
            )
            result = worker.process_claimed_file(source)

        self.assertTrue(Path(result["text_path"]).exists())
        self.assertTrue(Path(result["json_path"]).exists())
        self.assertTrue(Path(result["done_path"]).exists())
        txt = Path(result["text_path"]).read_text(encoding="utf-8")
        self.assertIn("hello world", txt)
        self.assertIn("onnx_provider_active: TensorrtExecutionProvider", txt)


if __name__ == "__main__":
    unittest.main()
