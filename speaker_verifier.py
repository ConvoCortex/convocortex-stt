from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("STT")


@dataclass(frozen=True)
class SpeakerVerificationResult:
    accepted: bool
    score: float | None
    threshold: float
    reason: str
    duration_s: float


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr)))
    if peak > 1.5:
        arr = arr / 32768.0
    return np.clip(arr, -1.0, 1.0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class SpeakerVerifier:
    def __init__(
        self,
        *,
        profile_path: str | Path,
        model_name: str = "ecapa_tdnn",
        requested_device: str = "cuda",
        sample_rate: int = 16000,
        threshold: float = 0.72,
        mode: str = "me-only",
    ) -> None:
        self.profile_path = Path(profile_path)
        self.model_name = str(model_name or "ecapa_tdnn").strip() or "ecapa_tdnn"
        self.requested_device = str(requested_device or "cuda").strip().lower() or "cuda"
        self.sample_rate = int(sample_rate)
        self.threshold = float(threshold)
        self.mode = str(mode or "me-only").strip() or "me-only"
        self._model = None
        self._device = None
        self._profile = None
        self._startup_logs: list[str] = []

    @property
    def resolved_device(self) -> str:
        self.ensure_ready(load_profile=False)
        return str(self._device)

    @property
    def profile_loaded(self) -> bool:
        return self._profile is not None

    def startup_logs(self) -> list[str]:
        return list(self._startup_logs)

    def ensure_ready(self, *, load_profile: bool = True) -> None:
        if self._model is None:
            import torch
            from nemo.collections.asr.models import EncDecSpeakerLabelModel

            preferred = self.requested_device
            if preferred == "cuda" and not torch.cuda.is_available():
                preferred = "cpu"
            if preferred not in {"cuda", "cpu"}:
                preferred = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = preferred
            self._model = EncDecSpeakerLabelModel.from_pretrained(self.model_name)
            self._model = self._model.to(self._device)
            self._model.eval()
            self._startup_logs.append(
                f"profile model={self.model_name} device={self._device} profile={self.profile_path}"
            )
        if load_profile and self._profile is None:
            self._profile = self.load_profile()

    def has_profile(self) -> bool:
        return self.profile_path.exists()

    def load_profile(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            raise FileNotFoundError(f"speaker profile not found: {self.profile_path}")
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        centroid = np.asarray(payload.get("centroid") or [], dtype=np.float32)
        if centroid.size == 0:
            raise ValueError(f"speaker profile centroid missing: {self.profile_path}")
        payload["centroid"] = centroid
        return payload

    def save_profile(self, profile: dict[str, Any]) -> None:
        serializable = dict(profile)
        centroid = np.asarray(serializable.get("centroid") or [], dtype=np.float32).reshape(-1)
        if centroid.size == 0:
            raise ValueError("speaker profile centroid is empty")
        serializable["centroid"] = centroid.tolist()
        self.profile_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self._profile = {
            **serializable,
            "centroid": centroid,
        }

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.ensure_ready(load_profile=False)
        normalized = _normalize_audio(audio)
        if normalized.size == 0:
            raise ValueError("speaker embedding audio is empty")
        if int(sample_rate) != self.sample_rate:
            import torchaudio
            import torch

            tensor = torch.from_numpy(normalized).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, int(sample_rate), self.sample_rate)
            normalized = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        emb, _ = self._model.infer_segment(normalized)
        vector = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("speaker embedding vector is empty")
        return vector

    def create_profile(
        self,
        samples: list[np.ndarray],
        *,
        sample_rate: int,
        sample_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("speaker enrollment requires at least one sample")
        embeddings = [self.embed_audio(sample, sample_rate) for sample in samples]
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        return {
            "version": 1,
            "mode": self.mode,
            "model": self.model_name,
            "device": self._device,
            "sample_rate": self.sample_rate,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "embedding_dim": int(centroid.size),
            "centroid": centroid,
            "enrollment_samples": sample_metadata or [],
        }

    def verify_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        threshold: float | None = None,
        min_duration_s: float = 0.0,
    ) -> SpeakerVerificationResult:
        self.ensure_ready(load_profile=True)
        normalized = _normalize_audio(audio)
        duration_s = float(normalized.size) / float(sample_rate) if sample_rate else 0.0
        target_threshold = float(self.threshold if threshold is None else threshold)
        if normalized.size == 0:
            return SpeakerVerificationResult(False, None, target_threshold, "empty_audio", duration_s)
        if duration_s < float(min_duration_s):
            return SpeakerVerificationResult(False, None, target_threshold, "insufficient_audio", duration_s)
        embedding = self.embed_audio(normalized, sample_rate)
        centroid = np.asarray(self._profile["centroid"], dtype=np.float32).reshape(-1)
        cosine = _cosine_similarity(embedding, centroid)
        score = (cosine + 1.0) / 2.0
        accepted = bool(score >= target_threshold)
        return SpeakerVerificationResult(
            accepted=accepted,
            score=float(score),
            threshold=target_threshold,
            reason="accepted" if accepted else "below_threshold",
            duration_s=duration_s,
        )

