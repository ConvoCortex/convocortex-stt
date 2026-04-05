from __future__ import annotations

import json
import logging
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("STT")
PROFILE_FORMAT_VERSION = 3


@dataclass(frozen=True)
class SpeakerVerificationResult:
    accepted: bool
    score: float | None
    threshold: float
    reason: str
    duration_s: float
    window_scores: tuple[float, ...] = ()


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
        self.window_seconds = 3.0
        self.window_hop_seconds = 1.5
        self.minimum_window_seconds = 1.5

    def _embedding_to_score(self, embedding: np.ndarray, references: np.ndarray) -> float:
        if references.size == 0:
            return 0.0
        similarities = np.dot(references, embedding) / (
            np.linalg.norm(references, axis=1) * np.linalg.norm(embedding) + 1e-12
        )
        cosine = float(np.max(similarities))
        return (cosine + 1.0) / 2.0

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

    def corpus_fingerprint(self, sample_paths: list[Path]) -> str:
        hasher = hashlib.sha256()
        for path in sorted(sample_paths, key=lambda item: str(item).lower()):
            stat = path.stat()
            hasher.update(str(path.resolve()).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime_ns)).encode("ascii"))
            hasher.update(str(int(stat.st_size)).encode("ascii"))
        return hasher.hexdigest()

    def profile_matches_corpus(self, sample_paths: list[Path]) -> bool:
        if not self.profile_path.exists():
            return False
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if int(payload.get("version") or 0) < PROFILE_FORMAT_VERSION:
            return False
        if "profile_self_score_min" not in payload or "profile_self_score_avg" not in payload or "profile_self_score_max" not in payload:
            return False
        if not payload.get("reference_embeddings"):
            return False
        sample_entries = list(payload.get("samples") or [])
        if not sample_entries or any(entry.get("self_score") is None for entry in sample_entries):
            return False
        return str(payload.get("samples_fingerprint") or "") == self.corpus_fingerprint(sample_paths)

    def load_profile(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            raise FileNotFoundError(f"speaker profile not found: {self.profile_path}")
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        raw_centroid = payload.get("centroid")
        if raw_centroid is None:
            raw_centroid = []
        centroid = np.asarray(raw_centroid, dtype=np.float32)
        if centroid.size == 0:
            raise ValueError(f"speaker profile centroid missing: {self.profile_path}")
        reference_embeddings = np.asarray(payload.get("reference_embeddings") or [], dtype=np.float32)
        if reference_embeddings.ndim == 1 and reference_embeddings.size:
            reference_embeddings = reference_embeddings.reshape(1, -1)
        payload["centroid"] = centroid
        payload["reference_embeddings"] = reference_embeddings
        return payload

    def save_profile(self, profile: dict[str, Any]) -> None:
        serializable = dict(profile)
        raw_centroid = serializable.get("centroid")
        if raw_centroid is None:
            raw_centroid = []
        centroid = np.asarray(raw_centroid, dtype=np.float32).reshape(-1)
        if centroid.size == 0:
            raise ValueError("speaker profile centroid is empty")
        raw_references = serializable.get("reference_embeddings")
        if raw_references is None:
            raw_references = []
        reference_embeddings = np.asarray(raw_references, dtype=np.float32)
        if reference_embeddings.ndim == 1 and reference_embeddings.size:
            reference_embeddings = reference_embeddings.reshape(1, -1)
        serializable["centroid"] = centroid.tolist()
        serializable["reference_embeddings"] = reference_embeddings.tolist()
        self.profile_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self._profile = {
            **serializable,
            "centroid": centroid,
            "reference_embeddings": reference_embeddings,
        }

    def load_audio_file(self, path: Path) -> tuple[np.ndarray, int]:
        import torchaudio

        waveform, source_rate = torchaudio.load(str(path))
        if waveform.numel() == 0:
            raise ValueError(f"speaker sample is empty: {path}")
        if waveform.ndim == 1:
            mono = waveform
        else:
            mono = waveform.mean(dim=0)
        return mono.detach().cpu().numpy().astype(np.float32).reshape(-1), int(source_rate)

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        normalized = _normalize_audio(audio)
        if normalized.size == 0:
            raise ValueError("speaker embedding audio is empty")
        if int(sample_rate) != self.sample_rate:
            import torchaudio
            import torch

            tensor = torch.from_numpy(normalized).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, int(sample_rate), self.sample_rate)
            normalized = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return normalized

    def embed_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.ensure_ready(load_profile=False)
        normalized = self._prepare_audio(audio, sample_rate)
        return self._embed_prepared_audio(normalized)

    def _embed_prepared_audio(self, normalized: np.ndarray) -> np.ndarray:
        emb, _ = self._model.infer_segment(normalized)
        vector = emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("speaker embedding vector is empty")
        return vector

    def _iter_prepared_windows(self, normalized: np.ndarray) -> list[np.ndarray]:
        total = int(normalized.size)
        if total == 0:
            return []
        window = max(1, int(self.window_seconds * self.sample_rate))
        hop = max(1, int(self.window_hop_seconds * self.sample_rate))
        minimum_window = max(1, int(self.minimum_window_seconds * self.sample_rate))
        slices: list[tuple[int, int]] = []
        if total <= window:
            slices.append((0, total))
        else:
            start = 0
            while start + window <= total:
                slices.append((start, start + window))
                start += hop
            tail_start = max(0, total - window)
            if not slices or tail_start > slices[-1][0]:
                slices.append((tail_start, total))

        windows: list[np.ndarray] = []
        seen: set[tuple[int, int]] = set()
        for start, end in slices:
            if (start, end) in seen:
                continue
            seen.add((start, end))
            if end - start < minimum_window:
                continue
            windows.append(normalized[start:end])
        return windows

    def _windowed_scores(self, audio: np.ndarray) -> list[float]:
        self.ensure_ready(load_profile=True)
        normalized = self._prepare_audio(audio, self.sample_rate)
        reference_embeddings = np.asarray(self._profile.get("reference_embeddings") or [], dtype=np.float32)
        if reference_embeddings.ndim == 1 and reference_embeddings.size:
            reference_embeddings = reference_embeddings.reshape(1, -1)
        if reference_embeddings.size == 0:
            centroid = np.asarray(self._profile["centroid"], dtype=np.float32).reshape(-1)
            reference_embeddings = centroid.reshape(1, -1)
        scores: list[float] = []
        for segment in self._iter_prepared_windows(normalized):
            vector = self._embed_prepared_audio(segment)
            scores.append(self._embedding_to_score(vector, reference_embeddings))
        return scores

    def create_profile(
        self,
        samples: list[np.ndarray],
        *,
        sample_rate: int,
        sample_metadata: list[dict[str, Any]] | None = None,
        samples_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("speaker profile build requires at least one sample")
        embeddings = [self.embed_audio(sample, sample_rate) for sample in samples]
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        return {
            "version": PROFILE_FORMAT_VERSION,
            "mode": self.mode,
            "model": self.model_name,
            "device": self._device,
            "sample_rate": self.sample_rate,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "embedding_dim": int(centroid.size),
            "centroid": centroid,
            "samples_fingerprint": str(samples_fingerprint or ""),
            "samples": sample_metadata or [],
        }

    def build_profile_from_paths(self, sample_paths: list[Path]) -> dict[str, Any]:
        normalized_paths = [Path(path) for path in sample_paths]
        if not normalized_paths:
            raise ValueError("speaker sample directory is empty")
        all_embeddings: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        sample_embeddings: list[list[np.ndarray]] = []
        for path in sorted(normalized_paths, key=lambda item: str(item).lower()):
            audio, source_rate = self.load_audio_file(path)
            prepared = self._prepare_audio(audio, source_rate)
            windows = self._iter_prepared_windows(prepared)
            window_embeddings = [self._embed_prepared_audio(window) for window in windows]
            if not window_embeddings:
                continue
            sample_embeddings.append(window_embeddings)
            all_embeddings.extend(window_embeddings)
            metadata.append(
                {
                    "path": str(path),
                    "source_sample_rate": int(source_rate),
                    "duration_s": round(float(audio.size) / float(source_rate), 6) if source_rate else 0.0,
                    "window_count": len(window_embeddings),
                }
            )
        if not all_embeddings:
            raise ValueError("speaker sample directory did not yield any usable windows")
        reference_embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
        centroid = np.mean(reference_embeddings, axis=0).astype(np.float32)
        self_scores: list[float] = []
        for idx, embeddings in enumerate(sample_embeddings):
            other_embeddings = [
                emb
                for other_idx, group in enumerate(sample_embeddings)
                for emb in group
                if other_idx != idx
            ]
            if other_embeddings:
                references = np.stack(other_embeddings, axis=0).astype(np.float32)
            else:
                references = reference_embeddings
            window_scores = [self._embedding_to_score(embedding, references) for embedding in embeddings]
            score = float(max(window_scores))
            self_scores.append(score)
            metadata[idx]["self_score"] = round(score, 6)
            metadata[idx]["window_scores"] = [round(float(item), 6) for item in window_scores]
        return {
            "version": PROFILE_FORMAT_VERSION,
            "mode": self.mode,
            "model": self.model_name,
            "device": self._device,
            "sample_rate": self.sample_rate,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "embedding_dim": int(centroid.size),
            "centroid": centroid,
            "reference_embeddings": reference_embeddings,
            "samples_fingerprint": self.corpus_fingerprint(normalized_paths),
            "samples": metadata,
            "profile_self_score_min": round(float(min(self_scores)), 6),
            "profile_self_score_avg": round(float(sum(self_scores) / len(self_scores)), 6),
            "profile_self_score_max": round(float(max(self_scores)), 6),
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
        prepared = self._prepare_audio(normalized, sample_rate)
        window_scores = self._windowed_scores(prepared)
        if not window_scores:
            return SpeakerVerificationResult(False, None, target_threshold, "insufficient_audio", duration_s)
        score = float(max(window_scores))
        accepted = bool(score >= target_threshold)
        return SpeakerVerificationResult(
            accepted=accepted,
            score=float(score),
            threshold=target_threshold,
            reason="accepted" if accepted else "below_threshold",
            duration_s=duration_s,
            window_scores=tuple(round(float(item), 6) for item in window_scores),
        )
