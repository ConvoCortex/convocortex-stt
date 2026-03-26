from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel


_RUNTIME_TEMP_DIR = Path(__file__).resolve().parent / ".tmp"


def _resolve_device(device: str) -> str:
    value = str(device).strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value or "cpu"


def _audio_to_pcm16(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.zeros(1, dtype=np.int16)
    clipped = np.clip(samples, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(value, (list, tuple)):
        parts = [_extract_text(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    return str(value).strip()


def _ensure_runtime_tempdir() -> str:
    path = _RUNTIME_TEMP_DIR
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path)
    os.environ["TMP"] = resolved
    os.environ["TEMP"] = resolved
    os.environ["TMPDIR"] = resolved
    tempfile.tempdir = resolved
    return resolved


@dataclass
class TranscriptResult:
    text: str


class TranscriptionBackend:
    name = "unknown"

    def warmup(self, audio: np.ndarray) -> None:
        self.transcribe(audio, language="", beam_size=1)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        raise NotImplementedError


class WhisperBackend(TranscriptionBackend):
    name = "whisper"

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        no_speech_threshold: float,
        log_prob_threshold: float,
    ) -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.no_speech_threshold = float(no_speech_threshold)
        self.log_prob_threshold = float(log_prob_threshold)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        kwargs: dict[str, Any] = {
            "beam_size": int(beam_size),
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
        }
        language = str(language or "").strip()
        if language:
            kwargs["language"] = language
        segs, _ = self.model.transcribe(np.asarray(audio, dtype=np.float32), **kwargs)
        return TranscriptResult(" ".join(seg.text for seg in segs).strip())


class ParakeetBackend(TranscriptionBackend):
    name = "parakeet"

    def __init__(self, model_name: str, device: str) -> None:
        _ensure_runtime_tempdir()
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as exc:
            raise RuntimeError(
                "Parakeet backend requires NeMo ASR. Install nemo_toolkit[asr] to use it."
            ) from exc

        self.device = _resolve_device(device)
        model_ref = str(model_name).strip()
        path = Path(model_ref)
        if path.exists() and path.is_file():
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=str(path))
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_ref)

        self.model.eval()
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Parakeet backend requested CUDA but no CUDA device is available.")
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")

    def warmup(self, audio: np.ndarray) -> None:
        del audio
        # NeMo's transcribe() creates temporary manifests internally.
        # On Windows, doing that during startup warmup is prone to file-lock
        # races, so we skip synthetic warmup and let the first real utterance
        # pay the one-time setup cost.
        return None

    def _transcribe_via_temp_wav(self, audio: np.ndarray) -> str:
        pcm16 = _audio_to_pcm16(audio)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm16.tobytes())
            result = self.model.transcribe([temp_path], batch_size=1)
            if isinstance(result, tuple):
                result = result[0]
            item = result[0] if isinstance(result, list) and result else result
            return _extract_text(item)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        del language, beam_size
        text = self._transcribe_via_temp_wav(np.asarray(audio, dtype=np.float32))
        return TranscriptResult(text)


def load_backend(
    backend_name: str,
    model_name: str,
    device: str,
    compute_type: str,
    no_speech_threshold: float,
    log_prob_threshold: float,
) -> TranscriptionBackend:
    backend = str(backend_name or "whisper").strip().lower()
    resolved_device = _resolve_device(device)
    if backend == "whisper":
        return WhisperBackend(
            model_name=model_name,
            device=resolved_device,
            compute_type=compute_type,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
        )
    if backend == "parakeet":
        return ParakeetBackend(model_name=model_name, device=resolved_device)
    raise ValueError(f"Unsupported transcription backend: {backend_name}")
