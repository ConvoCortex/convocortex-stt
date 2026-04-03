from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from file_drop_worker import _load_audio_for_transcription
from model_backends import load_backend


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    backend: str
    backend_options: dict[str, Any]


DEFAULT_SPECS = (
    BenchmarkSpec("parakeet", "parakeet", {}),
    BenchmarkSpec("parakeet-tensorrt", "parakeet-tensorrt", {}),
)


def _default_input() -> Path:
    for search_root in (ROOT / "files" / "done", ROOT / "transcribe-input"):
        candidates = sorted(search_root.glob("*"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("No input files found under transcribe-input/")


def _results_dir() -> Path:
    path = ROOT / "benchmark-results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark fixed-file transcription backends")
    parser.add_argument("--input", default="", help="Audio file to benchmark")
    parser.add_argument("--repeat", type=int, default=3, help="Measured repetitions per backend")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions per backend")
    parser.add_argument("--backends", default="parakeet,parakeet-tensorrt", help="Comma-separated backend list to benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load()
    input_value = str(args.input).strip()
    input_path = Path(input_value) if input_value else _default_input()
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark input not found: {input_path}")

    audio, audio_meta = _load_audio_for_transcription(input_path, int(cfg.get("audio", {}).get("rate", 16000)))
    microphone_cfg = cfg.get("microphone", {}) or {}
    final_model = str(microphone_cfg.get("final", "")).strip()
    final_device = str(microphone_cfg.get("final_device", "auto")).strip()
    final_compute = str(microphone_cfg.get("final_compute", "default")).strip()
    language = str(microphone_cfg.get("language", "")).strip()
    no_speech_threshold = float(microphone_cfg.get("no_speech_threshold", 0.6))
    log_prob_threshold = float(microphone_cfg.get("log_prob_threshold", -1.0))
    requested_backends = [item.strip() for item in str(args.backends or "").split(",") if item.strip()]
    spec_map = {spec.backend: spec for spec in DEFAULT_SPECS}
    specs = [spec_map[name] for name in requested_backends if name in spec_map]
    if not specs:
        raise ValueError("No supported benchmark backends were selected")

    def _backend_options_for(backend_name: str) -> dict[str, Any]:
        key = str(backend_name or "").strip().lower().replace("-", "_")
        value = microphone_cfg.get(key, {})
        options = dict(value or {}) if isinstance(value, dict) else {}
        options["runtime_role"] = "benchmark"
        return options

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = _results_dir() / f"{timestamp}.json"
    results: list[dict[str, Any]] = []

    for spec in specs:
        try:
            backend = load_backend(
                backend_name=spec.backend,
                model_name=final_model,
                device=final_device,
                compute_type=final_compute,
                no_speech_threshold=no_speech_threshold,
                log_prob_threshold=log_prob_threshold,
                backend_options=_backend_options_for(spec.backend) | spec.backend_options,
            )
            for _ in range(max(0, int(args.warmup))):
                backend.transcribe(audio, language=language, beam_size=5)
            runs: list[dict[str, Any]] = []
            text = ""
            for rep in range(max(1, int(args.repeat))):
                started = time.perf_counter()
                result = backend.transcribe(audio, language=language, beam_size=5)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                text = result.text
                runs.append(
                    {
                        "rep": rep + 1,
                        "elapsed_ms": round(elapsed_ms, 3),
                        "metrics": dict(result.metrics or {}),
                    }
                )
            elapsed_values = [row["elapsed_ms"] for row in runs]
            best_ms = min(elapsed_values)
            avg_ms = statistics.mean(elapsed_values)
            duration_s = float(audio_meta["audio_duration_s"])
            results.append(
                {
                    "name": spec.name,
                    "backend": spec.backend,
                    "backend_options": spec.backend_options,
                    "input_file": str(input_path),
                    "audio_duration_s": duration_s,
                    "best_ms": round(best_ms, 3),
                    "avg_ms": round(avg_ms, 3),
                    "best_realtime_factor": round((best_ms / 1000.0) / duration_s, 6),
                    "best_throughput_x": round(duration_s / (best_ms / 1000.0), 6),
                    "runs": runs,
                    "text": text,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": spec.name,
                    "backend": spec.backend,
                    "backend_options": spec.backend_options,
                    "input_file": str(input_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(input_path),
        "audio_meta": audio_meta,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    for row in results:
        if "error" in row:
            print(f"{row['name']}: ERROR {row['error']}")
        else:
            print(
                f"{row['name']}: best={row['best_ms']}ms avg={row['avg_ms']}ms "
                f"rtf={row['best_realtime_factor']} throughput={row['best_throughput_x']}x"
            )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
