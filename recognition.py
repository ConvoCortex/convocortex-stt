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
PROFILE_FORMAT_VERSION = 6
DEFAULT_THRESHOLD_MARGIN = 0.03
DEFAULT_THRESHOLD_FLOOR = 0.65
DEFAULT_THRESHOLD_CEILING = 0.92
DEFAULT_TOP_K_MATCHES = 3
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


@dataclass(frozen=True)
class RecognitionResult:
    accepted: bool
    score: float | None
    threshold: float
    delta: float | None
    reason: str
    duration_s: float
    speaker: str | None = None
    match_scores: tuple[float, ...] = ()


def discover_speaker_samples(root_dir: Path) -> dict[str, list[Path]]:
    root = Path(root_dir)
    if not root.exists():
        return {}
    speakers: dict[str, list[Path]] = {}
    for speaker_dir in sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda item: item.name.lower(),
    ):
        samples = sorted(
            [
                path
                for path in speaker_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
            ],
            key=lambda item: item.name.lower(),
        )
        if samples:
            speakers[speaker_dir.name] = samples
    return speakers


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(arr)))
    if peak > 1.5:
        arr = arr / 32768.0
    return np.clip(arr, -1.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def derive_threshold_from_self_scores(
    scores: list[float] | tuple[float, ...],
    *,
    margin: float = DEFAULT_THRESHOLD_MARGIN,
    floor: float = DEFAULT_THRESHOLD_FLOOR,
    ceiling: float = DEFAULT_THRESHOLD_CEILING,
) -> float:
    if not scores:
        return _clamp(floor, floor, ceiling)
    return _clamp(float(min(scores)) - float(margin), floor, ceiling)


def format_recognition_annotation(score: float | None, threshold: float, delta: float | None) -> str:
    score_text = "n/a" if score is None else f"{float(score):.3f}"
    delta_text = "n/a" if delta is None else f"{float(delta):+0.3f}"
    return f"recognition {score_text}/{float(threshold):.3f} ({delta_text})"


class RecognitionEngine:
    def __init__(
        self,
        *,
        profile_path: str | Path,
        model_name: str = "ecapa_tdnn",
        requested_device: str = "cuda",
        sample_rate: int = 16000,
        threshold_override: float | None = None,
    ) -> None:
        self.profile_path = Path(profile_path)
        self.model_name = str(model_name or "ecapa_tdnn").strip() or "ecapa_tdnn"
        self.requested_device = str(requested_device or "cuda").strip().lower() or "cuda"
        self.sample_rate = int(sample_rate)
        self.threshold_override = None if threshold_override is None else float(threshold_override)
        self._model = None
        self._device = None
        self._profile = None
        self._startup_logs: list[str] = []
        self.top_k_matches = DEFAULT_TOP_K_MATCHES

    def _embedding_to_similarity_scores(self, embedding: np.ndarray, references: np.ndarray) -> np.ndarray:
        if references.size == 0:
            return np.asarray([], dtype=np.float32)
        similarities = np.dot(references, embedding) / (
            np.linalg.norm(references, axis=1) * np.linalg.norm(embedding) + 1e-12
        )
        return ((similarities + 1.0) / 2.0).astype(np.float32)

    def _aggregate_similarity_scores(self, similarities: np.ndarray) -> tuple[float, list[float]]:
        if similarities.size == 0:
            return 0.0, []
        ordered = np.sort(similarities.astype(np.float32))[::-1]
        top_k = ordered[: max(1, int(self.top_k_matches))]
        rounded = [round(float(item), 6) for item in top_k.tolist()]
        return float(np.mean(top_k)), rounded

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
                f"cache model={self.model_name} device={self._device} cache={self.profile_path}"
            )
        if load_profile and self._profile is None:
            self._profile = self.load_profile()

    def has_profile(self) -> bool:
        return self.profile_path.exists()

    def corpus_fingerprint(self, speaker_paths: dict[str, list[Path]] | list[Path]) -> str:
        hasher = hashlib.sha256()
        normalized = self._normalize_speaker_paths(speaker_paths)
        for speaker_name in sorted(normalized.keys(), key=str.lower):
            hasher.update(speaker_name.encode("utf-8"))
            for path in sorted(normalized[speaker_name], key=lambda item: str(item).lower()):
                stat = path.stat()
                hasher.update(str(path.resolve()).encode("utf-8"))
                hasher.update(str(int(stat.st_mtime_ns)).encode("ascii"))
                hasher.update(str(int(stat.st_size)).encode("ascii"))
        return hasher.hexdigest()

    def _normalize_speaker_paths(self, speaker_paths: dict[str, list[Path]] | list[Path]) -> dict[str, list[Path]]:
        if isinstance(speaker_paths, dict):
            normalized: dict[str, list[Path]] = {}
            for speaker_name, paths in speaker_paths.items():
                resolved = [Path(path) for path in paths]
                if resolved:
                    normalized[str(speaker_name)] = resolved
            return normalized
        paths = [Path(path) for path in speaker_paths]
        return {"speaker-1": paths} if paths else {}

    def profile_matches_corpus(self, speaker_paths: dict[str, list[Path]] | list[Path]) -> bool:
        if not self.profile_path.exists():
            return False
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if int(payload.get("version") or 0) < PROFILE_FORMAT_VERSION:
            return False
        speakers = dict(payload.get("speakers") or {})
        if not speakers:
            return False
        if not str(payload.get("speakers_fingerprint") or ""):
            return False
        for speaker_data in speakers.values():
            required = {
                "corpus_self_score_min",
                "corpus_self_score_p10",
                "corpus_self_score_avg",
                "corpus_self_score_max",
                "active_threshold",
                "threshold_source",
            }
            if not required.issubset(speaker_data.keys()):
                return False
            if not speaker_data.get("reference_embeddings"):
                return False
            sample_entries = list(speaker_data.get("samples") or [])
            if not sample_entries or any(entry.get("self_score") is None for entry in sample_entries):
                return False
        if str(payload.get("speakers_fingerprint") or "") != self.corpus_fingerprint(speaker_paths):
            return False
        return True

    def load_profile(self) -> dict[str, Any]:
        if not self.profile_path.exists():
            raise FileNotFoundError(f"recognition cache not found: {self.profile_path}")
        payload = json.loads(self.profile_path.read_text(encoding="utf-8"))
        speakers = dict(payload.get("speakers") or {})
        if not speakers:
            raise ValueError(f"recognition cache speakers missing: {self.profile_path}")
        normalized_speakers: dict[str, dict[str, Any]] = {}
        for speaker_name, raw_speaker in speakers.items():
            speaker_payload = dict(raw_speaker or {})
            raw_centroid = speaker_payload.get("centroid")
            if raw_centroid is None:
                raw_centroid = []
            centroid = np.asarray(raw_centroid, dtype=np.float32)
            if centroid.size == 0:
                raise ValueError(f"recognition cache centroid missing for speaker={speaker_name}: {self.profile_path}")
            raw_reference_embeddings = speaker_payload.get("reference_embeddings")
            if raw_reference_embeddings is None:
                raw_reference_embeddings = []
            reference_embeddings = np.asarray(raw_reference_embeddings, dtype=np.float32)
            if reference_embeddings.ndim == 1 and reference_embeddings.size:
                reference_embeddings = reference_embeddings.reshape(1, -1)
            speaker_payload["centroid"] = centroid
            speaker_payload["reference_embeddings"] = reference_embeddings
            normalized_speakers[str(speaker_name)] = speaker_payload
        payload["speakers"] = normalized_speakers
        return payload

    def save_profile(self, profile: dict[str, Any]) -> None:
        serializable = dict(profile)
        speakers = dict(serializable.get("speakers") or {})
        normalized_speakers: dict[str, dict[str, Any]] = {}
        for speaker_name, raw_speaker in speakers.items():
            speaker_payload = dict(raw_speaker or {})
            raw_centroid = speaker_payload.get("centroid")
            if raw_centroid is None:
                raw_centroid = []
            centroid = np.asarray(raw_centroid, dtype=np.float32).reshape(-1)
            if centroid.size == 0:
                raise ValueError(f"recognition cache centroid is empty for speaker={speaker_name}")
            raw_references = speaker_payload.get("reference_embeddings")
            if raw_references is None:
                raw_references = []
            reference_embeddings = np.asarray(raw_references, dtype=np.float32)
            if reference_embeddings.ndim == 1 and reference_embeddings.size:
                reference_embeddings = reference_embeddings.reshape(1, -1)
            speaker_payload["centroid"] = centroid.tolist()
            speaker_payload["reference_embeddings"] = reference_embeddings.tolist()
            normalized_speakers[str(speaker_name)] = speaker_payload
        serializable["speakers"] = normalized_speakers
        self.profile_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        self._profile = self.load_profile()

    def load_audio_file(self, path: Path) -> tuple[np.ndarray, int]:
        import torchaudio

        waveform, source_rate = torchaudio.load(str(path))
        if waveform.numel() == 0:
            raise ValueError(f"recognition sample is empty: {path}")
        if waveform.ndim == 1:
            mono = waveform
        else:
            mono = waveform.mean(dim=0)
        return mono.detach().cpu().numpy().astype(np.float32).reshape(-1), int(source_rate)

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        normalized = _normalize_audio(audio)
        if normalized.size == 0:
            raise ValueError("recognition embedding audio is empty")
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
            raise ValueError("recognition embedding vector is empty")
        return vector

    def active_threshold(self, override: float | None = None, *, speaker: str | None = None) -> float:
        self.ensure_ready(load_profile=True)
        if override is not None:
            return float(override)
        if self.threshold_override is not None:
            return float(self.threshold_override)
        speaker_name = speaker or next(iter(self._profile.get("speakers", {})), None)
        if not speaker_name:
            return derive_threshold_from_self_scores([])
        speaker_payload = dict(self._profile.get("speakers", {}).get(speaker_name) or {})
        active = speaker_payload.get("active_threshold")
        if active is None:
            scores = [float(entry.get("self_score")) for entry in (speaker_payload.get("samples") or []) if entry.get("self_score") is not None]
            return derive_threshold_from_self_scores(scores)
        return float(active)

    def threshold_source(self, override: float | None = None) -> str:
        if override is not None or self.threshold_override is not None:
            return "config"
        self.ensure_ready(load_profile=True)
        return str(self._profile.get("threshold_source") or "derived")

    def _reference_match_scores(self, audio: np.ndarray, *, speaker: str) -> list[float]:
        self.ensure_ready(load_profile=True)
        normalized = self._prepare_audio(audio, self.sample_rate)
        speaker_payload = dict(self._profile.get("speakers", {}).get(speaker) or {})
        raw_reference_embeddings = speaker_payload.get("reference_embeddings")
        if raw_reference_embeddings is None:
            raw_reference_embeddings = []
        reference_embeddings = np.asarray(raw_reference_embeddings, dtype=np.float32)
        if reference_embeddings.ndim == 1 and reference_embeddings.size:
            reference_embeddings = reference_embeddings.reshape(1, -1)
        if reference_embeddings.size == 0:
            centroid = np.asarray(speaker_payload["centroid"], dtype=np.float32).reshape(-1)
            reference_embeddings = centroid.reshape(1, -1)
        vector = self._embed_prepared_audio(normalized)
        similarities = self._embedding_to_similarity_scores(vector, reference_embeddings)
        _, top_matches = self._aggregate_similarity_scores(similarities)
        return top_matches

    def build_profile_from_speakers(self, speaker_paths: dict[str, list[Path]]) -> dict[str, Any]:
        self.ensure_ready(load_profile=False)
        normalized_speakers = self._normalize_speaker_paths(speaker_paths)
        if not normalized_speakers:
            raise ValueError("recognition root directory does not contain any speaker sample folders")
        speakers_payload: dict[str, dict[str, Any]] = {}
        embedding_dim = 0
        for speaker_name in sorted(normalized_speakers.keys(), key=str.lower):
            normalized_paths = normalized_speakers[speaker_name]
            if not normalized_paths:
                continue
            all_embeddings: list[np.ndarray] = []
            metadata: list[dict[str, Any]] = []
            sample_embeddings: list[list[np.ndarray]] = []
            for path in sorted(normalized_paths, key=lambda item: str(item).lower()):
                audio, source_rate = self.load_audio_file(path)
                prepared = self._prepare_audio(audio, source_rate)
                embedding = self._embed_prepared_audio(prepared)
                sample_embeddings.append([embedding])
                all_embeddings.append(embedding)
                metadata.append(
                    {
                        "path": str(path),
                        "speaker": speaker_name,
                        "source_sample_rate": int(source_rate),
                        "duration_s": round(float(audio.size) / float(source_rate), 6) if source_rate else 0.0,
                    }
                )
            if not all_embeddings:
                continue
            reference_embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
            centroid = np.mean(reference_embeddings, axis=0).astype(np.float32)
            embedding_dim = max(embedding_dim, int(centroid.size))
            self_scores: list[float] = []
            for idx, embeddings in enumerate(sample_embeddings):
                other_embeddings = [
                    emb
                    for other_idx, group in enumerate(sample_embeddings)
                    for emb in group
                    if other_idx != idx
                ]
                references = np.stack(other_embeddings, axis=0).astype(np.float32) if other_embeddings else reference_embeddings
                similarities = self._embedding_to_similarity_scores(embeddings[0], references)
                score, top_matches = self._aggregate_similarity_scores(similarities)
                self_scores.append(score)
                metadata[idx]["self_score"] = round(score, 6)
                metadata[idx]["match_scores"] = top_matches
            self_score_p10 = float(np.percentile(np.asarray(self_scores, dtype=np.float32), 10))
            active_threshold = derive_threshold_from_self_scores(self_scores)
            threshold_source = "derived"
            if self.threshold_override is not None:
                active_threshold = float(self.threshold_override)
                threshold_source = "config"
            speakers_payload[speaker_name] = {
                "speaker": speaker_name,
                "embedding_dim": int(centroid.size),
                "centroid": centroid,
                "reference_embeddings": reference_embeddings,
                "samples_fingerprint": self.corpus_fingerprint({speaker_name: normalized_paths}),
                "samples": metadata,
                "sample_count": len(metadata),
                "corpus_self_score_min": round(float(min(self_scores)), 6),
                "corpus_self_score_p10": round(self_score_p10, 6),
                "corpus_self_score_avg": round(float(sum(self_scores) / len(self_scores)), 6),
                "corpus_self_score_max": round(float(max(self_scores)), 6),
                "active_threshold": round(float(active_threshold), 6),
                "threshold_source": threshold_source,
            }
        if not speakers_payload:
            raise ValueError("recognition root directory did not yield any usable speaker samples")
        return {
            "version": PROFILE_FORMAT_VERSION,
            "model": self.model_name,
            "device": self._device,
            "sample_rate": self.sample_rate,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "embedding_dim": embedding_dim,
            "speakers_fingerprint": self.corpus_fingerprint(normalized_speakers),
            "speaker_names": sorted(speakers_payload.keys(), key=str.lower),
            "speaker_count": len(speakers_payload),
            "speakers": speakers_payload,
        }

    def build_profile_from_paths(self, sample_paths: list[Path]) -> dict[str, Any]:
        return self.build_profile_from_speakers({"speaker-1": [Path(path) for path in sample_paths]})

    def verify_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        threshold: float | None = None,
        min_duration_s: float = 0.0,
    ) -> RecognitionResult:
        self.ensure_ready(load_profile=True)
        normalized = _normalize_audio(audio)
        duration_s = float(normalized.size) / float(sample_rate) if sample_rate else 0.0
        if normalized.size == 0:
            return RecognitionResult(False, None, self.active_threshold(threshold), None, "empty_audio", duration_s)
        if duration_s < float(min_duration_s):
            return RecognitionResult(False, None, self.active_threshold(threshold), None, "insufficient_audio", duration_s)
        prepared = self._prepare_audio(normalized, sample_rate)
        best_speaker = None
        best_score = None
        best_threshold = self.active_threshold(threshold)
        best_delta = None
        best_match_scores: list[float] = []
        for speaker_name in sorted(self._profile.get("speakers", {}).keys(), key=str.lower):
            match_scores = self._reference_match_scores(prepared, speaker=speaker_name)
            if not match_scores:
                continue
            score = float(sum(match_scores) / len(match_scores))
            if best_score is None or score > best_score:
                current_threshold = self.active_threshold(threshold, speaker=speaker_name)
                best_speaker = speaker_name
                best_score = score
                best_threshold = current_threshold
                best_delta = float(score - current_threshold)
                best_match_scores = match_scores
        if best_score is None:
            return RecognitionResult(False, None, best_threshold, None, "insufficient_audio", duration_s)
        accepted = bool(best_score >= best_threshold)
        return RecognitionResult(
            accepted=accepted,
            score=float(best_score),
            threshold=best_threshold,
            delta=round(float(best_delta), 6) if best_delta is not None else None,
            reason="accepted" if accepted else "below_threshold",
            duration_s=duration_s,
            speaker=best_speaker,
            match_scores=tuple(round(float(item), 6) for item in best_match_scores),
        )

