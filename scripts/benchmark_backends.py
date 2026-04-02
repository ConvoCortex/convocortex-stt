from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_backends import TranscriptResult, load_backend


DEFAULT_AUDIO = REPO_ROOT / "voice-sample" / "Mmm, and here's the kicker. I found a secret recip.mp3"
DEFAULT_WARMUP_SECONDS = 1.0
DEFAULT_ITERATIONS = 3


@dataclass
class BenchmarkSpec:
    label: str
    backend: str
    model_name: str
    device: str
    compute_type: str
    language: str


@dataclass
class BenchmarkResult:
    label: str
    backend: str
    model_name: str
    device: str
    compute_type: str
    audio_path: str
    audio_seconds: float
    startup_ready_seconds: float
    prime_seconds: float
    hot_run_seconds: list[float]
    hot_run_rtf: list[float]
    hot_run_text: list[str]
    mean_hot_seconds: float
    mean_hot_rtf: float
    median_hot_rtf: float


def get_specs() -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            label="parakeet-v2",
            backend="parakeet",
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            device="cuda",
            compute_type="float16",
            language="en",
        ),
        BenchmarkSpec(
            label="whisper-large-v3-turbo",
            backend="whisper",
            model_name="large-v3-turbo",
            device="cuda",
            compute_type="float16",
            language="en",
        ),
    ]


def _load_audio(path: Path) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform.squeeze(0).to(torch.float32).cpu().numpy()


def _clip_audio(audio: np.ndarray, seconds: float) -> np.ndarray:
    frames = max(1, int(round(seconds * 16000)))
    return np.asarray(audio[:frames], dtype=np.float32)


def _transcribe_once(backend: Any, audio: np.ndarray, language: str, beam_size: int) -> tuple[TranscriptResult, float]:
    started = time.perf_counter()
    result = backend.transcribe(audio, language=language, beam_size=beam_size)
    elapsed = time.perf_counter() - started
    return result, elapsed


def run_benchmark(spec: BenchmarkSpec, audio: np.ndarray, audio_path: Path, iterations: int, warmup_seconds: float) -> BenchmarkResult:
    warmup_audio = _clip_audio(audio, warmup_seconds)

    startup_started = time.perf_counter()
    backend = load_backend(
        backend_name=spec.backend,
        model_name=spec.model_name,
        device=spec.device,
        compute_type=spec.compute_type,
        no_speech_threshold=0.45,
        log_prob_threshold=-0.8,
    )
    backend.warmup(warmup_audio)
    prime_result, prime_seconds = _transcribe_once(backend, warmup_audio, language=spec.language, beam_size=1)
    startup_ready_seconds = time.perf_counter() - startup_started

    del prime_result

    audio_seconds = float(len(audio) / 16000.0)
    hot_run_seconds: list[float] = []
    hot_run_rtf: list[float] = []
    hot_run_text: list[str] = []

    for _ in range(iterations):
        result, elapsed = _transcribe_once(backend, audio, language=spec.language, beam_size=1)
        hot_run_seconds.append(elapsed)
        hot_run_rtf.append(elapsed / audio_seconds if audio_seconds > 0 else float("inf"))
        hot_run_text.append(result.text)

    return BenchmarkResult(
        label=spec.label,
        backend=spec.backend,
        model_name=spec.model_name,
        device=spec.device,
        compute_type=spec.compute_type,
        audio_path=str(audio_path),
        audio_seconds=audio_seconds,
        startup_ready_seconds=startup_ready_seconds,
        prime_seconds=prime_seconds,
        hot_run_seconds=hot_run_seconds,
        hot_run_rtf=hot_run_rtf,
        hot_run_text=hot_run_text,
        mean_hot_seconds=statistics.fmean(hot_run_seconds),
        mean_hot_rtf=statistics.fmean(hot_run_rtf),
        median_hot_rtf=statistics.median(hot_run_rtf),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark transcription backend startup and hot inference throughput.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--backend-label", choices=[spec.label for spec in get_specs()], default=None)
    return parser.parse_args()


def run_single(spec: BenchmarkSpec, audio_path: Path, iterations: int, warmup_seconds: float) -> dict[str, Any]:
    audio = _load_audio(audio_path)
    return asdict(run_benchmark(spec, audio, audio_path, iterations, warmup_seconds))


def run_all(audio_path: Path, iterations: int, warmup_seconds: float) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for spec in get_specs():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--audio",
            str(audio_path),
            "--iterations",
            str(iterations),
            "--warmup-seconds",
            str(warmup_seconds),
            "--backend-label",
            spec.label,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark subprocess failed for {spec.label} with code {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        results.append(json.loads(proc.stdout))
    return {
        "audio_path": str(audio_path),
        "iterations": int(iterations),
        "warmup_seconds": float(warmup_seconds),
        "results": results,
    }


def main() -> int:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio sample not found: {audio_path}")

    if args.backend_label:
        spec = next(spec for spec in get_specs() if spec.label == args.backend_label)
        payload: dict[str, Any] = run_single(spec, audio_path, args.iterations, args.warmup_seconds)
    else:
        payload = run_all(audio_path, args.iterations, args.warmup_seconds)

    text = json.dumps(payload, indent=2)
    print(text)

    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
