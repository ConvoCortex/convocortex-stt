from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import config
from model_backends import load_backend


logger = logging.getLogger("STT")

SUPPORTED_AUDIO_EXTENSIONS = (
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
)


def _resolve_path(value: str | None, default: str) -> Path:
    raw = str(value or "").strip() or default
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _normalize_ignored_transcript(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,!?;:'\"()[]{}")


def _normalize_disfluency_phrase(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,!?;:'\"()[]{}")


def _cleanup_transcript_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\{\s*\}", "", cleaned)
    cleaned = re.sub(r"^[,;:!?.\-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _phrase_pattern(phrase: str) -> str:
    parts = [re.escape(part) for part in phrase.split() if part]
    return r"(?:[\s-]+)".join(parts)


def _should_ignore_transcript(text: str, ignored_phrases: Iterable[str]) -> bool:
    normalized = _normalize_ignored_transcript(text)
    if not normalized:
        return True
    return normalized in set(_normalize_ignored_transcript(item) for item in ignored_phrases if str(item).strip())


def _strip_disfluencies(text: str, disfluency_words: Iterable[str]) -> str:
    cleaned = str(text)
    for phrase in disfluency_words:
        normalized_phrase = _normalize_disfluency_phrase(str(phrase))
        if not normalized_phrase:
            continue
        cleaned = re.sub(
            rf"(?i)(?<!\w){_phrase_pattern(normalized_phrase)}(?!\w)",
            " ",
            cleaned,
        )
    return _cleanup_transcript_text(cleaned)


def _format_duration_ms(value_ms: float) -> float:
    return round(float(value_ms), 3)


def _format_seconds(value_s: float) -> float:
    return round(float(value_s), 6)


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _claim_name(path: Path, pid: int) -> Path:
    return path.with_name(f"{path.stem}.processing-{pid}{path.suffix}")


def _load_audio_for_transcription(path: Path, target_rate: int) -> tuple[np.ndarray, dict]:
    import torch
    import torchaudio

    load_started = time.perf_counter()
    waveform, source_rate = torchaudio.load(str(path))
    load_elapsed_ms = (time.perf_counter() - load_started) * 1000.0
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if waveform.ndim == 1:
        mono = waveform
        channel_count = 1
    else:
        channel_count = int(waveform.shape[0])
        mono = waveform.mean(dim=0)

    resample_ms = 0.0
    if int(source_rate) != int(target_rate):
        resample_started = time.perf_counter()
        mono = torchaudio.functional.resample(mono.unsqueeze(0), int(source_rate), int(target_rate)).squeeze(0)
        resample_ms = (time.perf_counter() - resample_started) * 1000.0

    audio = mono.detach().cpu().numpy().astype(np.float32).reshape(-1)
    duration_seconds = float(audio.shape[0]) / float(target_rate)
    return audio, {
        "source_sample_rate": int(source_rate),
        "target_sample_rate": int(target_rate),
        "channels": channel_count,
        "audio_load_ms": _format_duration_ms(load_elapsed_ms),
        "resample_ms": _format_duration_ms(resample_ms),
        "audio_duration_s": _format_seconds(duration_seconds),
    }


@dataclass
class FileDropSettings:
    enabled: bool
    auto_start_worker: bool
    poll_interval_seconds: float
    settle_seconds: float
    input_dir: Path
    text_dir: Path
    done_dir: Path
    failed_dir: Path
    worker_log: Path
    supported_extensions: tuple[str, ...]
    final_backend: str
    final_model: str
    final_device: str
    final_compute: str
    language: str
    no_speech_threshold: float
    log_prob_threshold: float
    write_json_sidecar: bool
    backend_options: dict[str, object]


def resolve_settings(
    cfg: dict,
    *,
    input_dir: str | None = None,
    text_dir: str | None = None,
    done_dir: str | None = None,
    failed_dir: str | None = None,
    final_backend: str | None = None,
    final_model: str | None = None,
    final_device: str | None = None,
    final_compute: str | None = None,
    language: str | None = None,
    onnx_provider: str | None = None,
) -> FileDropSettings:
    file_cfg = cfg.get("file_drop", {}) or {}
    models_cfg = cfg.get("models", {}) or {}
    supported = tuple(
        sorted(
            {
                str(item).strip().lower()
                for item in (file_cfg.get("supported_extensions") or SUPPORTED_AUDIO_EXTENSIONS)
                if str(item).strip()
            }
        )
    ) or SUPPORTED_AUDIO_EXTENSIONS
    resolved_backend = str(final_backend or models_cfg.get("final_backend", "whisper")).strip()
    backend_key = resolved_backend.lower().replace("-", "_")
    raw_backend_options = models_cfg.get(backend_key, {})
    backend_options = dict(raw_backend_options or {}) if isinstance(raw_backend_options, dict) else {}
    if onnx_provider is not None:
        backend_options["provider"] = str(onnx_provider).strip()
    return FileDropSettings(
        enabled=bool(file_cfg.get("enabled", False)),
        auto_start_worker=bool(file_cfg.get("auto_start_worker", True)),
        poll_interval_seconds=max(0.2, float(file_cfg.get("poll_interval_seconds", 2.0))),
        settle_seconds=max(0.0, float(file_cfg.get("settle_seconds", 1.0))),
        input_dir=_resolve_path(input_dir or file_cfg.get("input_dir"), "input"),
        text_dir=_resolve_path(text_dir or file_cfg.get("text_dir"), "text"),
        done_dir=_resolve_path(done_dir or file_cfg.get("done_dir"), "done"),
        failed_dir=_resolve_path(failed_dir or file_cfg.get("failed_dir"), "failed"),
        worker_log=_resolve_path(file_cfg.get("worker_log"), "stt-file-worker.log"),
        supported_extensions=supported,
        final_backend=resolved_backend,
        final_model=str(final_model or models_cfg.get("final", "")).strip(),
        final_device=str(final_device or models_cfg.get("final_device", "auto")).strip(),
        final_compute=str(final_compute or models_cfg.get("final_compute", "default")).strip(),
        language=str(language if language is not None else models_cfg.get("language", "")).strip(),
        no_speech_threshold=float(models_cfg.get("no_speech_threshold", 0.6)),
        log_prob_threshold=float(models_cfg.get("log_prob_threshold", -1.0)),
        write_json_sidecar=bool(file_cfg.get("write_json_sidecar", True)),
        backend_options=backend_options,
    )


class FileDropWorker:
    def __init__(self, cfg: dict, settings: FileDropSettings) -> None:
        self.cfg = cfg
        self.settings = settings
        self.rate = int(cfg.get("audio", {}).get("rate", 16000))
        self.ignored_phrases = tuple(cfg.get("filters", {}).get("ignored_exact_phrases", []) or [])
        self.disfluency_words = tuple(cfg.get("filters", {}).get("disfluency_words", []) or [])
        self._backend = None

    def ensure_ready(self) -> None:
        for directory in (
            self.settings.input_dir,
            self.settings.text_dir,
            self.settings.done_dir,
            self.settings.failed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.settings.worker_log.parent.mkdir(parents=True, exist_ok=True)
        if self._backend is None:
            logger.info(
                "[file-drop] loading backend=%s model=%s device=%s",
                self.settings.final_backend,
                self.settings.final_model,
                self.settings.final_device,
            )
            self._backend = load_backend(
                backend_name=self.settings.final_backend,
                model_name=self.settings.final_model,
                device=self.settings.final_device,
                compute_type=self.settings.final_compute,
                no_speech_threshold=self.settings.no_speech_threshold,
                log_prob_threshold=self.settings.log_prob_threshold,
                backend_options=self.settings.backend_options,
            )

    def _discover_candidates(self) -> list[Path]:
        input_dir = self.settings.input_dir
        if not input_dir.exists():
            return []
        files: list[Path] = []
        for path in sorted(input_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                continue
            if ".processing-" in path.name:
                continue
            if path.suffix.lower() not in self.settings.supported_extensions:
                continue
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds < self.settings.settle_seconds:
                continue
            files.append(path)
        return files

    def _claim(self, path: Path) -> Path | None:
        claimed = _claim_name(path, os.getpid())
        try:
            return path.replace(claimed)
        except FileNotFoundError:
            return None
        except PermissionError:
            return None

    def _write_outputs(self, source_name: str, transcript: str, metadata: dict) -> tuple[Path, Path | None]:
        stem = Path(source_name).stem
        txt_path = _unique_path(self.settings.text_dir, stem, ".txt")
        header_lines = [
            f"# source_file: {metadata['source_file']}",
            f"# backend: {metadata['backend']}",
            f"# model: {metadata['model']}",
            f"# device: {metadata['device']}",
            f"# audio_duration_s: {metadata['audio_duration_s']}",
            f"# audio_load_ms: {metadata['audio_load_ms']}",
            f"# resample_ms: {metadata['resample_ms']}",
            f"# inference_ms: {metadata['inference_ms']}",
            f"# total_job_ms: {metadata['total_job_ms']}",
            f"# realtime_factor: {metadata['realtime_factor']}",
            f"# throughput_x: {metadata['throughput_x']}",
            f"# ignored_exact_match: {str(metadata['ignored_exact_match']).lower()}",
            f"# generated_at: {metadata['generated_at']}",
            "",
        ]
        if metadata.get("onnx_provider_active"):
            header_lines.insert(4, f"# onnx_provider_active: {metadata['onnx_provider_active']}")
        if metadata.get("encoder_inference_ms") is not None:
            header_lines.insert(9, f"# encoder_inference_ms: {metadata['encoder_inference_ms']}")
        if metadata.get("decoder_inference_ms") is not None:
            header_lines.insert(10, f"# decoder_inference_ms: {metadata['decoder_inference_ms']}")
        txt_path.write_text("\n".join(header_lines) + transcript.strip() + "\n", encoding="utf-8")

        json_path = None
        if self.settings.write_json_sidecar:
            json_path = _unique_path(self.settings.text_dir, stem, ".json")
            json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return txt_path, json_path

    def _move_claimed(self, claimed_path: Path, target_dir: Path) -> Path:
        target = _unique_path(target_dir, Path(claimed_path.name).stem.replace(f".processing-{os.getpid()}", ""), claimed_path.suffix)
        return claimed_path.replace(target)

    def process_claimed_file(self, claimed_path: Path) -> dict:
        self.ensure_ready()
        started = time.perf_counter()
        source_name = claimed_path.name.replace(f".processing-{os.getpid()}", "")
        audio, audio_meta = _load_audio_for_transcription(claimed_path, self.rate)

        infer_started = time.perf_counter()
        result = self._backend.transcribe(
            audio,
            language=self.settings.language,
            beam_size=5,
        )
        raw_text = result.text
        inference_ms = (time.perf_counter() - infer_started) * 1000.0

        ignored_exact_match = _should_ignore_transcript(raw_text, self.ignored_phrases)
        cleaned_text = "" if ignored_exact_match else _strip_disfluencies(raw_text, self.disfluency_words)
        total_ms = (time.perf_counter() - started) * 1000.0
        duration_s = float(audio_meta["audio_duration_s"])
        realtime_factor = (inference_ms / 1000.0) / duration_s if duration_s > 0 else 0.0
        throughput_x = duration_s / (inference_ms / 1000.0) if inference_ms > 0 else 0.0
        metadata = {
            "source_file": source_name,
            "backend": self.settings.final_backend,
            "model": self.settings.final_model,
            "device": self.settings.final_device,
            "language": self.settings.language,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "raw_text": raw_text,
            "text": cleaned_text,
            "ignored_exact_match": ignored_exact_match,
            "inference_ms": _format_duration_ms(inference_ms),
            "total_job_ms": _format_duration_ms(total_ms),
            "realtime_factor": _format_seconds(realtime_factor),
            "throughput_x": _format_seconds(throughput_x),
            **audio_meta,
            **dict(getattr(result, "metrics", {}) or {}),
        }
        text_path, json_path = self._write_outputs(source_name, cleaned_text, metadata)
        done_path = self._move_claimed(claimed_path, self.settings.done_dir)
        metadata["text_path"] = str(text_path)
        metadata["json_path"] = str(json_path) if json_path else ""
        metadata["done_path"] = str(done_path)
        if json_path is not None:
            json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        logger.info(
            "[file-drop] %s -> %s (inference=%sms total=%sms throughput=%sx)",
            source_name,
            text_path.name,
            metadata["inference_ms"],
            metadata["total_job_ms"],
            metadata["throughput_x"],
        )
        return metadata

    def process_one_available(self) -> dict | None:
        for candidate in self._discover_candidates():
            claimed = self._claim(candidate)
            if claimed is None:
                continue
            try:
                return self.process_claimed_file(claimed)
            except Exception as exc:
                logger.exception("[file-drop] failed processing %s: %s", candidate.name, exc)
                failed_target = _unique_path(
                    self.settings.failed_dir,
                    Path(candidate.name).stem,
                    candidate.suffix,
                )
                claimed.replace(failed_target)
                error_report = _unique_path(self.settings.text_dir, Path(candidate.name).stem, ".error.txt")
                error_report.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                return {
                    "source_file": candidate.name,
                    "failed_path": str(failed_target),
                    "error_report": str(error_report),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return None

    def run(self, *, once: bool = False, idle_exit: bool = False) -> int:
        self.ensure_ready()
        logger.info(
            "[file-drop] watching input=%s text=%s done=%s failed=%s",
            self.settings.input_dir,
            self.settings.text_dir,
            self.settings.done_dir,
            self.settings.failed_dir,
        )
        processed_any = False
        while True:
            result = self.process_one_available()
            if result is not None:
                processed_any = True
                if once:
                    return 0
                continue
            if once or idle_exit:
                return 0 if processed_any or idle_exit else 1
            time.sleep(self.settings.poll_interval_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convocortex STT file-drop transcription worker")
    parser.add_argument("--once", action="store_true", help="Process at most one discovered file and exit")
    parser.add_argument("--idle-exit", action="store_true", help="Exit immediately if the input folder is empty")
    parser.add_argument("--input-dir", help="Override file-drop input directory")
    parser.add_argument("--text-dir", help="Override file-drop transcript output directory")
    parser.add_argument("--done-dir", help="Override processed-audio output directory")
    parser.add_argument("--failed-dir", help="Override failed-audio directory")
    parser.add_argument("--final-backend", help="Override models.final_backend for this worker")
    parser.add_argument("--final-model", help="Override models.final for this worker")
    parser.add_argument("--final-device", help="Override models.final_device for this worker")
    parser.add_argument("--final-compute", help="Override models.final_compute for this worker")
    parser.add_argument("--language", help="Override models.language for this worker")
    parser.add_argument("--onnx-provider", help="Override models.parakeet_onnx.provider for this worker")
    return parser


def _ensure_worker_log_handler(path: Path) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        root = logging.getLogger()
    target = str(path.resolve())
    for handler in root.handlers:
        base = getattr(handler, "baseFilename", None)
        if base and str(Path(base).resolve()) == target:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(file_handler)


def run_from_args(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config.load()
    settings = resolve_settings(
        cfg,
        input_dir=args.input_dir,
        text_dir=args.text_dir,
        done_dir=args.done_dir,
        failed_dir=args.failed_dir,
        final_backend=args.final_backend,
        final_model=args.final_model,
        final_device=args.final_device,
        final_compute=args.final_compute,
        language=args.language,
        onnx_provider=args.onnx_provider,
    )
    _ensure_worker_log_handler(settings.worker_log)
    worker = FileDropWorker(cfg, settings)
    return worker.run(once=bool(args.once), idle_exit=bool(args.idle_exit))


if __name__ == "__main__":
    raise SystemExit(run_from_args(sys.argv[1:]))
