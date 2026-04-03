from __future__ import annotations

import os
import sys
import tempfile
import threading
import wave
import hashlib
import itertools
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
import logging as pylogging

import numpy as np
import torch
from faster_whisper import WhisperModel


_RUNTIME_TEMP_DIR = Path(__file__).resolve().parent / ".tmp"
_PARAKEET_STDIO_LOCK = threading.Lock()
_MANUAL_TEMP_COUNTER = itertools.count()


@contextmanager
def _silence_process_stdio():
    # Some NeMo/Lhotse paths write directly to process stdout/stderr, bypassing
    # Python logging and high-level redirect helpers.
    stdout_fd = stderr_fd = None
    saved_stdout_fd = saved_stderr_fd = None
    devnull_fd = None
    try:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        if saved_stdout_fd is not None and stdout_fd is not None:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)
        if saved_stderr_fd is not None and stderr_fd is not None:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


@contextmanager
def _temporary_directory_ignore_cleanup_errors(*modules):
    original = tempfile.TemporaryDirectory
    module_originals = []
    manual_root = _RUNTIME_TEMP_DIR / "manual-tempdirs"
    manual_root.mkdir(parents=True, exist_ok=True)

    class ManualTemporaryDirectory:
        def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
            del ignore_cleanup_errors
            parent = Path(dir) if dir else manual_root
            parent.mkdir(parents=True, exist_ok=True)
            prefix_value = prefix or "tmp"
            suffix_value = suffix or ""
            token = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}-{next(_MANUAL_TEMP_COUNTER)}"
            self.name = str(parent / f"{prefix_value}{token}{suffix_value}")
            Path(self.name).mkdir(parents=True, exist_ok=True)

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, tb):
            return False

    tempfile.TemporaryDirectory = ManualTemporaryDirectory
    for module in modules:
        module_tempfile = getattr(module, "tempfile", None)
        if module_tempfile is not None:
            module_originals.append((module_tempfile, module_tempfile.TemporaryDirectory))
            module_tempfile.TemporaryDirectory = ManualTemporaryDirectory
    try:
        yield
    finally:
        for module_tempfile, module_original in module_originals:
            module_tempfile.TemporaryDirectory = module_original
        tempfile.TemporaryDirectory = original


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


def _hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def _prepare_persistent_nemo_extract(restore_path: Path, unpack_fn) -> Path:
    extract_root = _RUNTIME_TEMP_DIR / "nemo-extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    stat = restore_path.stat()
    key = _hash_key(str(restore_path.resolve()), str(stat.st_size), str(stat.st_mtime_ns))
    candidate = extract_root / key
    marker = candidate / "model_weights.ckpt"
    if marker.exists():
        return candidate
    if candidate.exists():
        fallback = extract_root / f"{key}-{int(time.time() * 1000)}"
        unpack_fn(str(restore_path), str(fallback))
        return fallback
    candidate.mkdir(parents=True, exist_ok=True)
    unpack_fn(str(restore_path), str(candidate))
    return candidate


def _manual_temp_path(suffix: str) -> Path:
    root = _RUNTIME_TEMP_DIR / "manual-files"
    root.mkdir(parents=True, exist_ok=True)
    token = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}-{next(_MANUAL_TEMP_COUNTER)}"
    return root / f"{token}{suffix}"


@dataclass
class TranscriptResult:
    text: str
    metrics: dict[str, Any] | None = None


class TranscriptionBackend:
    name = "unknown"

    def warmup(self, audio: np.ndarray) -> None:
        self.transcribe(audio, language="", beam_size=1)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        raise NotImplementedError


class FasterWhisperBackend(TranscriptionBackend):
    name = "faster-whisper"

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        no_speech_threshold: float,
        log_prob_threshold: float,
    ) -> None:
        self._transcribe_lock = threading.Lock()
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.no_speech_threshold = float(no_speech_threshold)
        self.log_prob_threshold = float(log_prob_threshold)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        with self._transcribe_lock:
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
        self._transcribe_lock = threading.Lock()
        _ensure_runtime_tempdir()
        try:
            import nemo.collections.asr as nemo_asr
            import nemo.core.connectors.save_restore_connector as nemo_save_restore_module
            from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
            from nemo.utils import logging as nemo_logging
        except Exception as exc:
            raise RuntimeError(
                "Parakeet backend requires NeMo ASR. Install nemo_toolkit[asr] to use it."
            ) from exc

        try:
            nemo_logging.set_verbosity(pylogging.ERROR)
        except Exception:
            pass
        try:
            nemo_logging.remove_stream_handlers()
        except Exception:
            pass
        for logger_name in ("nemo", "nemo_logger", "lhotse"):
            logger = pylogging.getLogger(logger_name)
            logger.setLevel(pylogging.ERROR)
            logger.propagate = False
            for handler in list(logger.handlers):
                try:
                    logger.removeHandler(handler)
                except Exception:
                    pass

        self.device = _resolve_device(device)
        model_ref = str(model_name).strip()
        path = Path(model_ref)
        save_restore_connector = SaveRestoreConnector()
        # NeMo/Lhotse can print directly to stdout/stderr during model load and
        # device moves as well, so suppress that startup chatter too.
        with _PARAKEET_STDIO_LOCK:
            with _silence_process_stdio():
                with _temporary_directory_ignore_cleanup_errors(nemo_save_restore_module):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        if path.exists() and path.is_dir():
                            save_restore_connector.model_extracted_dir = str(path)
                            self.model = nemo_asr.models.ASRModel.restore_from(
                                restore_path=str(path),
                                save_restore_connector=save_restore_connector,
                            )
                        elif path.exists() and path.is_file():
                            extracted_dir = _prepare_persistent_nemo_extract(
                                path,
                                SaveRestoreConnector._unpack_nemo_file,
                            )
                            save_restore_connector.model_extracted_dir = str(extracted_dir)
                            self.model = nemo_asr.models.ASRModel.restore_from(
                                restore_path=str(path),
                                save_restore_connector=save_restore_connector,
                            )
                        else:
                            cached_model = nemo_asr.models.ASRModel.from_pretrained(
                                model_ref,
                                return_model_file=True,
                            )
                            cached_path = Path(str(cached_model))
                            if cached_path.is_file():
                                extracted_dir = _prepare_persistent_nemo_extract(
                                    cached_path,
                                    SaveRestoreConnector._unpack_nemo_file,
                                )
                                save_restore_connector.model_extracted_dir = str(extracted_dir)
                                self.model = nemo_asr.models.ASRModel.restore_from(
                                    restore_path=str(cached_path),
                                    save_restore_connector=save_restore_connector,
                                )
                            elif cached_path.is_dir():
                                save_restore_connector.model_extracted_dir = str(cached_path)
                                self.model = nemo_asr.models.ASRModel.restore_from(
                                    restore_path=str(cached_path),
                                    save_restore_connector=save_restore_connector,
                                )
                            else:
                                self.model = nemo_asr.models.ASRModel.from_pretrained(
                                    model_ref,
                                    save_restore_connector=save_restore_connector,
                                )

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
            temp_path = str(_manual_temp_path(".wav"))
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm16.tobytes())
            # NeMo/Lhotse emits noisy warnings and progress text directly to
            # stdout/stderr during transcribe(). Keep the app console focused on
            # STT runtime output instead of internal dataloader chatter.
            with _PARAKEET_STDIO_LOCK:
                with _temporary_directory_ignore_cleanup_errors():
                    with _silence_process_stdio():
                        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                            result = self.model.transcribe(
                                [temp_path],
                                batch_size=1,
                                num_workers=0,
                                verbose=False,
                            )
            if isinstance(result, tuple):
                result = result[0]
            item = result[0] if isinstance(result, list) and result else result
            return _extract_text(item)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        del language, beam_size
        with self._transcribe_lock:
            text = self._transcribe_via_temp_wav(np.asarray(audio, dtype=np.float32))
            return TranscriptResult(text)


def _import_onnxruntime_with_cuda_preload():
    try:
        import tensorrt_libs  # type: ignore

        dll_dir = os.path.dirname(tensorrt_libs.__file__)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    import onnxruntime
    try:
        onnxruntime.preload_dlls()
    except Exception:
        pass
    return onnxruntime


def _normalize_backend_options(options: dict[str, Any] | None) -> dict[str, Any]:
    return dict(options or {})


def _resolve_onnx_provider(options: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]] | None, str]:
    onnxruntime = _import_onnxruntime_with_cuda_preload()
    requested = str(options.get("provider", "auto")).strip().lower()
    device_id = int(options.get("device_id", 0))
    available = set(onnxruntime.get_available_providers())
    extra = dict(options.get("provider_options", {}) or {})
    if requested in {"tensorrt", "tensorrtexecutionprovider", "trt"} and "TensorrtExecutionProvider" in available:
        cache_dir = _resolve_path_like(options.get("trt_cache_dir"), _RUNTIME_TEMP_DIR / "trt-cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "device_id": device_id,
            "trt_fp16_enable": bool(options.get("trt_fp16_enable", True)),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
        } | extra
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider"], [trt_options, {"device_id": device_id}], "tensorrt"
    if requested in {"cuda", "cudaexecutionprovider"} and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider"], [{"device_id": device_id} | extra], "cuda"
    if requested in {"cpu", "cpuexecutionprovider"}:
        return ["CPUExecutionProvider"], None, "cpu"
    if "TensorrtExecutionProvider" in available:
        cache_dir = _resolve_path_like(options.get("trt_cache_dir"), _RUNTIME_TEMP_DIR / "trt-cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        trt_options = {
            "device_id": device_id,
            "trt_fp16_enable": bool(options.get("trt_fp16_enable", True)),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
        } | extra
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider"], [trt_options, {"device_id": device_id}], "tensorrt"
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider"], [{"device_id": device_id} | extra], "cuda"
    return ["CPUExecutionProvider"], None, "cpu"


def _resolve_path_like(value: Any, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


class ParakeetEncoderRuntimeBackend(TranscriptionBackend):
    name = "parakeet-cuda"

    def __init__(
        self,
        model_name: str,
        device: str,
        *,
        provider_label: str,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._transcribe_lock = threading.Lock()
        self.provider_label_requested = str(provider_label).strip().lower()
        self.options = _normalize_backend_options(options)
        self.base = ParakeetBackend(model_name=model_name, device=device)
        self.model = self.base.model
        self.device = self.base.device
        self.onnxruntime = _import_onnxruntime_with_cuda_preload()
        self.model_key = _hash_key(model_name, self.device, self.provider_label_requested)
        self.encoder_path = self._ensure_encoder_export()
        self.providers, self.provider_options, self.provider_label = _resolve_onnx_provider(
            {"provider": self.provider_label_requested, **self.options}
        )
        session_options = self.onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        max_threads = int(self.options.get("max_threads", 0))
        if max_threads > 0:
            session_options.inter_op_num_threads = max_threads
            session_options.intra_op_num_threads = max_threads
        self.encoder_session = self.onnxruntime.InferenceSession(
            str(self.encoder_path),
            sess_options=session_options,
            providers=self.providers,
            provider_options=self.provider_options,
        )
        self.encoder_input_names = [node.name for node in self.encoder_session.get_inputs()]
        self.encoder_output_names = [node.name for node in self.encoder_session.get_outputs()]

    def warmup(self, audio: np.ndarray) -> None:
        del audio
        # TensorRT engine build on synthetic startup audio makes app startup feel
        # hung. Let the first real decode pay that one-time cost instead.
        return None

    def _ensure_encoder_export(self) -> Path:
        export_dir = _resolve_path_like(self.options.get("export_dir"), _RUNTIME_TEMP_DIR / "onnx-cache")
        export_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = export_dir / f"parakeet-encoder-{self.model_key}.onnx"
        if encoder_path.exists():
            return encoder_path
        with _PARAKEET_STDIO_LOCK:
            with _silence_process_stdio():
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.model.encoder.export(str(encoder_path), check_trace=False)
        return encoder_path

    def transcribe(self, audio: np.ndarray, language: str, beam_size: int) -> TranscriptResult:
        del language, beam_size
        with self._transcribe_lock:
            model_device = next(self.model.parameters()).device
            audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0).to(model_device)
            audio_length = torch.tensor([audio_tensor.shape[-1]], dtype=torch.int64, device=model_device)
            with torch.inference_mode():
                processed_signal, processed_signal_len = self.model.preprocessor(input_signal=audio_tensor, length=audio_length)
            encoder_inputs = {
                self.encoder_input_names[0]: processed_signal.detach().cpu().numpy(),
                self.encoder_input_names[1]: processed_signal_len.detach().cpu().numpy().astype(np.int64, copy=False),
            }
            encoder_started = time.perf_counter()
            encoder_output, encoder_lengths = self.encoder_session.run(self.encoder_output_names[:2], encoder_inputs)
            encoder_ms = (time.perf_counter() - encoder_started) * 1000.0
            encoder_output_t = torch.from_numpy(np.asarray(encoder_output)).to(model_device)
            encoder_lengths_t = torch.from_numpy(np.asarray(encoder_lengths)).to(model_device)
            decoder_started = time.perf_counter()
            hypotheses = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output_t,
                encoder_lengths_t,
                return_hypotheses=False,
            )
            decoder_ms = (time.perf_counter() - decoder_started) * 1000.0
            text = hypotheses[0].text if hypotheses else ""
            return TranscriptResult(
                text=text,
                metrics={
                    "onnx_provider_requested": self.provider_label_requested,
                    "onnx_provider_resolved": self.provider_label,
                    "onnx_provider_active": self.encoder_session.get_providers()[0] if self.encoder_session.get_providers() else "",
                    "encoder_inference_ms": round(encoder_ms, 3),
                    "decoder_inference_ms": round(decoder_ms, 3),
                    "onnx_encoder_path": str(self.encoder_path),
                },
            )


class ParakeetCudaBackend(ParakeetEncoderRuntimeBackend):
    name = "parakeet-cuda"

    def __init__(self, model_name: str, device: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, device=device, provider_label="cuda", options=options)


class ParakeetTensorRTBackend(ParakeetEncoderRuntimeBackend):
    name = "parakeet-tensorrt"

    def __init__(self, model_name: str, device: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, device=device, provider_label="tensorrt", options=options)


def load_backend(
    backend_name: str,
    model_name: str,
    device: str,
    compute_type: str,
    no_speech_threshold: float,
    log_prob_threshold: float,
    backend_options: dict[str, Any] | None = None,
) -> TranscriptionBackend:
    backend = str(backend_name or "faster-whisper").strip().lower()
    resolved_device = _resolve_device(device)
    if backend == "faster-whisper":
        return FasterWhisperBackend(
            model_name=model_name,
            device=resolved_device,
            compute_type=compute_type,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
        )
    if backend == "parakeet":
        return ParakeetBackend(model_name=model_name, device=resolved_device)
    if backend == "parakeet-cuda":
        return ParakeetCudaBackend(model_name=model_name, device=resolved_device, options=backend_options)
    if backend == "parakeet-tensorrt":
        return ParakeetTensorRTBackend(model_name=model_name, device=resolved_device, options=backend_options)
    raise ValueError(f"Unsupported transcription backend: {backend_name}")
