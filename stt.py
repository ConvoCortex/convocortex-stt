"""
convocortex-stt — headless speech-to-text engine

Dual-model pipeline: fast CPU model for realtime partials,
accurate GPU model for final results after silence.
Output via configurable local handlers. NATS optional.
"""

import sys
import time
import queue
import re
import threading
import logging
from collections import deque

import numpy as np
import pyaudio
import pvporcupine
from faster_whisper import WhisperModel

import config
import state as state_store

cfg = config.load()
state_store.init(cfg)
_persisted = state_store.load()

# ── Models ────────────────────────────────────────────────────────────────────
FINAL_MODEL        = cfg["models"]["final"]
FINAL_DEVICE       = cfg["models"]["final_device"]
FINAL_COMPUTE      = cfg["models"]["final_compute"]
REALTIME_MODEL     = cfg["models"]["realtime"]
REALTIME_DEVICE    = cfg["models"]["realtime_device"]
LANGUAGE           = cfg["models"]["language"]

# ── Audio ─────────────────────────────────────────────────────────────────────
RATE            = cfg["audio"]["rate"]
CHUNK           = cfg["audio"]["chunk"]
VAD_THRESHOLD   = cfg["audio"]["vad_threshold"]
BUFFER_SECONDS  = cfg["audio"]["buffer_seconds"]
SILENCE_TIMEOUT = cfg["audio"]["silence_timeout"]
PREFERRED_INPUT_DEVICE = str(cfg["audio"].get("input_device", "")).strip()

# ── Realtime ──────────────────────────────────────────────────────────────────
RT_CHECK_INTERVAL = cfg["realtime"]["check_interval"]
MIN_CHUNKS        = cfg["realtime"]["min_chunks"]
MAX_CHUNKS        = cfg["realtime"]["max_chunks"]
RT_QUEUE_SIZE     = cfg["realtime"]["queue_size"]

# ── Sleep / Wake ───────────────────────────────────────────────────────────────
_sw_cfg = cfg.get("sleep_wake", {})
SLEEP_START_SLEEPING = bool(_sw_cfg.get("start_sleeping", True))
SLEEP_HOTKEY_TOGGLE = str(cfg["hotkeys"].get("sleep_toggle", "")).strip()
TYPE_TOGGLE_HOTKEY = str(cfg["hotkeys"].get("typing_toggle", "")).strip()
SLEEP_STOP_WORDS = [
    str(w).strip().lower()
    for w in _sw_cfg.get("stop_words", ["stop", "pause"])
    if str(w).strip()
]
WAKEWORD_ENABLED = bool(_sw_cfg.get("enabled", True))
WAKEWORD_BACKEND = str(_sw_cfg.get("backend", "pvporcupine")).strip().lower()
WAKEWORD = str(_sw_cfg.get("wake_word", "bumblebee")).strip().lower()
WAKEWORD_SENSITIVITY = float(_sw_cfg.get("sensitivity", 0.8))
WAKEWORD_DROP_SECONDS = float(_sw_cfg.get("drop_audio_after_wake_seconds", 0.4))

# ── Feedback sounds ───────────────────────────────────────────────────────────
_fb_cfg = cfg.get("feedback", {})
FEEDBACK_ENABLED = bool(_fb_cfg.get("enabled", True))
FEEDBACK_ON_SOUND = str(_fb_cfg.get("on_sound", "sounds/on.ogg")).strip()
FEEDBACK_OFF_SOUND = str(_fb_cfg.get("off_sound", "sounds/off.ogg")).strip()
FEEDBACK_SILENCE_SOUND = str(_fb_cfg.get("silence_sound", "sounds/silence.ogg")).strip()
FEEDBACK_OUTPUT_DEVICE = str(_fb_cfg.get("output_device", "")).strip()

# ── Voice commands (minimal) ──────────────────────────────────────────────────
VOICE_COMMANDS_ENABLED = bool(cfg.get("voice_commands", {}).get("enabled", False))
ENTER_COMMAND_ENABLED = (
    VOICE_COMMANDS_ENABLED
    and bool(cfg.get("voice_commands", {}).get("enter", {}).get("enabled", False))
)
ENTER_COMMAND_WORDS = [
    str(w).strip()
    for w in cfg.get("voice_commands", {}).get("enter", {}).get("words", [])
    if str(w).strip()
]
TYPE_TOGGLE_COMMAND_ENABLED = (
    VOICE_COMMANDS_ENABLED
    and bool(cfg.get("voice_commands", {}).get("type_at_cursor_toggle", {}).get("enabled", False))
)
TYPE_TOGGLE_COMMAND_WORDS = [
    str(w).strip().lower()
    for w in cfg.get("voice_commands", {}).get("type_at_cursor_toggle", {}).get("words", [])
    if str(w).strip()
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("STT")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("ctranslate2").setLevel(logging.WARNING)

# ── Event dispatch ────────────────────────────────────────────────────────────
# Handlers are registered at startup. Each receives every event dict.
# Schema:
#   {"type": "partial", "text": str, "epoch": int, "t": float, "inference_ms": int}
#   {"type": "final",   "text": str, "epoch": int, "t": float, "inference_ms": int}
#   {"type": "status",  "value": str}   # "recording" | "idle" | "sleeping" | "working"
#   {"type": "system",  "event": str, ...extra}

_handlers: list = []

def register_handler(fn):
    _handlers.append(fn)

def dispatch(event: dict):
    for fn in _handlers:
        try:
            fn(event)
        except Exception as e:
            logger.error(f"[handler:{fn.__name__}] {e}")

# ── VAD wrapper ───────────────────────────────────────────────────────────────
class SileroVAD:
    def __init__(self, model):
        self.model = model

    def is_speech(self, audio_chunk):
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(audio_chunk).unsqueeze(0)
            return self.model(tensor, RATE).item()


class FeedbackAudio:
    def __init__(self, enabled: bool, on_path: str, off_path: str, silence_path: str, output_device: str):
        self.enabled = enabled
        self.on_path = on_path
        self.off_path = off_path
        self.silence_path = silence_path
        self.output_device = output_device
        self._p_sfx = None
        self._p_silence = None
        self._output_device_index = None
        self._sound_queue = queue.Queue()
        self._running = False
        self._clips = {}
        self._silence_thread = None
        self._sound_thread = None
        self._lock = threading.Lock()

        if not self.enabled:
            return
        try:
            import torchaudio

            self._clips["on"] = self._load_clip(torchaudio, self.on_path)
            self._clips["off"] = self._load_clip(torchaudio, self.off_path)
            self._clips["silence"] = self._load_clip(torchaudio, self.silence_path)
            self._p_sfx = pyaudio.PyAudio()
            self._p_silence = pyaudio.PyAudio()
            self._output_device_index = self._resolve_output_device_index(self._p_sfx, self.output_device)
            self._running = True
            self._sound_thread = threading.Thread(target=self._sound_worker, daemon=True, name="feedback-sound")
            self._silence_thread = threading.Thread(target=self._silence_worker, daemon=True, name="feedback-silence")
            self._sound_thread.start()
            self._silence_thread.start()
            if self._output_device_index is None:
                logger.info("[feedback] Enabled (default output device)")
            else:
                logger.info(f"[feedback] Enabled (output device index={self._output_device_index})")
        except Exception as e:
            logger.warning(f"[feedback] Disabled: {e}")
            self.enabled = False

    def _resolve_output_device_index(self, p_instance, device_name: str):
        if not device_name:
            return None
        target = device_name.strip().lower()
        for i in range(p_instance.get_device_count()):
            info = p_instance.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) <= 0:
                continue
            if str(info.get("name", "")).strip().lower() == target:
                return int(info.get("index", i))
        logger.warning(f"[feedback] Configured output_device not found: {device_name}. Using default.")
        return None

    def _load_clip(self, torchaudio, path: str):
        wav, sr = torchaudio.load(path)
        wav = wav.detach().cpu().numpy().T.astype(np.float32)
        if wav.ndim == 1:
            wav = wav[:, np.newaxis]
        return {"data": np.ascontiguousarray(wav), "sr": int(sr), "channels": int(wav.shape[1])}

    def _play_clip_blocking(self, clip_name: str, pa_instance):
        if not self.enabled:
            return
        if pa_instance is None:
            return
        clip = self._clips.get(clip_name)
        if not clip:
            return
        stream = None
        try:
            open_kwargs = dict(
                format=pyaudio.paFloat32,
                channels=clip["channels"],
                rate=clip["sr"],
                output=True,
            )
            if self._output_device_index is not None:
                open_kwargs["output_device_index"] = self._output_device_index
            stream = pa_instance.open(**open_kwargs)
            stream.write(clip["data"].tobytes())
        except Exception as e:
            logger.warning(f"[feedback] Playback error ({clip_name}): {e}")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass

    def _sound_worker(self):
        while self._running:
            try:
                name = self._sound_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._play_clip_blocking(name, self._p_sfx)
            finally:
                self._sound_queue.task_done()

    def _silence_worker(self):
        while self._running:
            self._play_clip_blocking("silence", self._p_silence)

    def play_on(self):
        if self.enabled:
            self._sound_queue.put("on")

    def play_off(self):
        if self.enabled:
            self._sound_queue.put("off")

    def shutdown(self):
        if not self.enabled:
            return
        with self._lock:
            self._running = False
        try:
            if self._silence_thread:
                self._silence_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._sound_thread:
                self._sound_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._p_sfx is not None:
                self._p_sfx.terminate()
        except Exception:
            pass
        try:
            if self._p_silence is not None:
                self._p_silence.terminate()
        except Exception:
            pass


def main():
    import signal
    def _sigterm(_s, _f):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    print(f"Initializing STT (Realtime: {REALTIME_DEVICE}, Final: {FINAL_DEVICE})...")
    sys.stdout.flush()

    resources = {}
    load_errors = []
    load_errors_lock = threading.Lock()
    warmup_audio = np.zeros(RATE, dtype=np.float32)

    def record_load_error(message: str):
        with load_errors_lock:
            load_errors.append(message)

    # ── Async model loading ───────────────────────────────────────────────────

    def load_final():
        try:
            logger.info(f"Loading {FINAL_MODEL} ({FINAL_DEVICE})...")
            m = WhisperModel(FINAL_MODEL, device=FINAL_DEVICE, compute_type=FINAL_COMPUTE)
            m.transcribe(warmup_audio, beam_size=1)
            resources['final_model'] = m
            logger.info("Final model ready.")
        except Exception as e:
            record_load_error(f"Final load error: {e}")

    def load_realtime():
        if not REALTIME_MODEL:
            logger.info("Realtime model disabled (no model configured). Partials will not fire.")
            resources['realtime_model'] = None
            return
        try:
            logger.info(f"Loading {REALTIME_MODEL} ({REALTIME_DEVICE})...")
            compute = "default" if REALTIME_DEVICE == "cpu" else FINAL_COMPUTE
            m = WhisperModel(REALTIME_MODEL, device=REALTIME_DEVICE, compute_type=compute)
            m.transcribe(warmup_audio, beam_size=1)
            resources['realtime_model'] = m
            logger.info("Realtime model ready.")
        except Exception as e:
            record_load_error(f"Realtime load error: {e}")

    def load_audio_stack():
        import torch
        try:
            logger.info("Loading Silero VAD...")
            try:
                model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad', model='silero_vad',
                    source='local', onnx=True
                )
            except Exception:
                model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad', model='silero_vad',
                    force_reload=False, onnx=True
                )
            resources['vad'] = SileroVAD(model)

            logger.info("Initializing PyAudio...")
            p = pyaudio.PyAudio()
            resources['p_instance'] = p
            def _iter_input_devices():
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        yield info

            def _find_device_by_name(name: str, host_api: int | None = None):
                target = name.strip().lower()
                if not target:
                    return None
                for info in _iter_input_devices():
                    if host_api is not None and info.get("hostApi") != host_api:
                        continue
                    if str(info.get("name", "")).strip().lower() == target:
                        return info
                return None

            def _supports_format(info: dict) -> bool:
                try:
                    p.is_format_supported(
                        RATE,
                        input_device=int(info["index"]),
                        input_channels=1,
                        input_format=pyaudio.paInt16
                    )
                    return True
                except Exception:
                    return False

            default_info = p.get_default_input_device_info()
            info = None

            if PREFERRED_INPUT_DEVICE:
                configured_info = _find_device_by_name(PREFERRED_INPUT_DEVICE)
                if configured_info and _supports_format(configured_info):
                    info = configured_info
                    logger.info(f"[device] Startup from config: {configured_info['name']}")
                else:
                    logger.warning(f"[device] Configured startup device unavailable: {PREFERRED_INPUT_DEVICE}")
            else:
                persisted_name = str(_persisted.get("last_input_device_name") or "").strip()
                persisted_host_api = _persisted.get("last_input_device_host_api")
                if persisted_name:
                    persisted_info = _find_device_by_name(persisted_name, persisted_host_api)
                    if not persisted_info:
                        persisted_info = _find_device_by_name(persisted_name)
                    if persisted_info and _supports_format(persisted_info):
                        info = persisted_info
                        logger.info(f"[device] Startup from persisted state: {persisted_info['name']}")
                    else:
                        logger.warning(f"[device] Persisted startup device unavailable: {persisted_name}")

            if info is None:
                info = default_info

            open_kwargs = dict(
                format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, frames_per_buffer=CHUNK
            )
            try:
                stream = p.open(
                    input_device_index=int(info['index']),
                    **open_kwargs
                )
            except Exception as open_error:
                if int(info['index']) != int(default_info['index']):
                    logger.warning(f"[device] Startup device open failed ({info['name']}): {open_error}")
                    logger.warning(f"[device] Falling back to OS default: {default_info['name']}")
                    info = default_info
                    stream = p.open(
                        input_device_index=int(info['index']),
                        **open_kwargs
                    )
                else:
                    raise
            resources['stream'] = stream
            resources['dev_name'] = info['name']
            resources['dev_idx'] = int(info['index'])
            resources['host_api'] = info['hostApi']
            logger.info(f"Audio: {info['name']}")
        except Exception as e:
            record_load_error(f"Audio stack error: {e}")

    threads = [
        threading.Thread(target=load_final),
        threading.Thread(target=load_realtime),
        threading.Thread(target=load_audio_stack),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    if load_errors:
        for msg in load_errors:
            logger.error(msg)
        sys.exit(1)

    final_model = resources['final_model']
    rt_model    = resources['realtime_model']
    vad         = resources['vad']
    stream      = resources['stream']
    p_instance  = resources['p_instance']
    dev_name    = resources['dev_name']

    # ── Register handlers ─────────────────────────────────────────────────────
    import handlers as h
    handler_extras = h.register_all(cfg, register_handler)
    feedback = FeedbackAudio(
        enabled=FEEDBACK_ENABLED,
        on_path=FEEDBACK_ON_SOUND,
        off_path=FEEDBACK_OFF_SOUND,
        silence_path=FEEDBACK_SILENCE_SOUND,
        output_device=FEEDBACK_OUTPUT_DEVICE,
    )

    # ── State ─────────────────────────────────────────────────────────────────
    final_queue   = queue.Queue()
    rt_queue      = queue.Queue(maxsize=RT_QUEUE_SIZE)
    state_lock    = threading.Lock()
    state         = {'speech_epoch': 0, 'finalized_epoch': -1}
    rt_cancel     = threading.Event()
    mode_lock     = threading.Lock()
    mode_state    = {"sleeping": bool(_persisted.get("sleeping", SLEEP_START_SLEEPING))}
    typing_lock   = threading.Lock()
    default_typing_enabled = bool(cfg["output"]["type_at_cursor"]["enabled"])
    typing_state = {
        "enabled": bool(_persisted.get("type_at_cursor_enabled", default_typing_enabled))
    }

    type_set_enabled = handler_extras.get("type_at_cursor_set_enabled")
    if type_set_enabled:
        type_set_enabled(typing_state["enabled"])
    else:
        typing_state["enabled"] = False

    # ── State persistence helper ──────────────────────────────────────────────
    current_device_state = [{"name": dev_name, "host_api": resources['host_api']}]

    def persist():
        with mode_lock:
            sleeping = mode_state["sleeping"]
        with typing_lock:
            typing_enabled = typing_state["enabled"]
        state_store.save({
            "sleeping": sleeping,
            "type_at_cursor_enabled": typing_enabled,
            "last_input_device_name": current_device_state[0]["name"],
            "last_input_device_host_api": current_device_state[0]["host_api"],
        })

    def set_sleeping(sleeping: bool, reason: str = ""):
        with mode_lock:
            was_sleeping = mode_state["sleeping"]
            if was_sleeping == sleeping:
                return
            mode_state["sleeping"] = sleeping
        rt_cancel.set()
        if sleeping:
            with state_lock:
                state["speech_epoch"] += 1
                state["finalized_epoch"] = state["speech_epoch"]
            feedback.play_off()
        else:
            feedback.play_on()
        logger.info(f"[mode] {'Sleeping' if sleeping else 'Working'}" + (f" ({reason})" if reason else ""))
        dispatch({"type": "status", "value": "sleeping" if sleeping else "working"})
        persist()

    def set_typing_enabled(enabled: bool, reason: str = ""):
        if not type_set_enabled:
            logger.info("[typing] type_at_cursor handler is not active.")
            return
        with typing_lock:
            if typing_state["enabled"] == bool(enabled):
                return
            typing_state["enabled"] = bool(enabled)
            current = typing_state["enabled"]
        if type_set_enabled:
            type_set_enabled(current)
        if current:
            feedback.play_on()
        else:
            feedback.play_off()
        logger.info(f"[typing] {'Enabled' if current else 'Disabled'}" + (f" ({reason})" if reason else ""))
        dispatch({"type": "status", "value": "typing_enabled" if current else "typing_disabled"})
        persist()

    def toggle_typing_enabled(reason: str = "") -> bool:
        with typing_lock:
            next_value = not typing_state["enabled"]
        set_typing_enabled(next_value, reason=reason)
        with typing_lock:
            return typing_state["enabled"]

    # ── Workers ───────────────────────────────────────────────────────────────

    def apply_voice_commands(final_text: str) -> tuple[str, dict]:
        if not ENTER_COMMAND_ENABLED or not ENTER_COMMAND_WORDS:
            return final_text, {}

        text = final_text.strip()
        press_enter_after = False
        words = sorted(ENTER_COMMAND_WORDS, key=len, reverse=True)

        for trigger in words:
            pat = re.compile(rf"^\s*{re.escape(trigger)}(?:[\s\.,!?;:]+|$)", re.IGNORECASE)
            m = pat.match(text)
            if m:
                press_enter_after = True
                text = text[m.end():].strip()
                break

        for trigger in words:
            pat = re.compile(
                rf"(?:^|[\s\.,!?;:]+){re.escape(trigger)}(?:[\s\.,!?;:]+)?$",
                re.IGNORECASE
            )
            m = pat.search(text)
            if m:
                press_enter_after = True
                text = text[:m.start()].strip()
                break

        actions = {}
        if press_enter_after:
            actions["press_enter_after"] = True
        return text, actions

    def final_worker():
        while True:
            try:
                epoch, t0, audio_data = final_queue.get()
                with mode_lock:
                    if mode_state["sleeping"]:
                        continue
                audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
                logger.info(f"[FINAL] Processing {len(audio)/RATE:.2f}s audio...")

                t_start = time.time()
                segs, _ = final_model.transcribe(audio, language=LANGUAGE, beam_size=5)
                raw_text = " ".join(s.text for s in segs).strip()
                inf_ms = int((time.time() - t_start) * 1000)
                t = time.time() - t0

                normalized = raw_text.lower().strip().strip(".,!?;:")
                if normalized in SLEEP_STOP_WORDS:
                    set_sleeping(True, reason=f"voice:{normalized}")
                    continue
                if TYPE_TOGGLE_COMMAND_ENABLED and normalized in TYPE_TOGGLE_COMMAND_WORDS:
                    toggle_typing_enabled(reason=f"voice:{normalized}")
                    continue

                text, actions = apply_voice_commands(raw_text)
                if text or actions:
                    ev = {
                        "type": "final",
                        "text": text,
                        "epoch": epoch,
                        "t": round(t, 3),
                        "inference_ms": inf_ms,
                    }
                    if actions:
                        ev["actions"] = actions
                    with mode_lock:
                        currently_sleeping = mode_state["sleeping"]
                    if currently_sleeping:
                        logger.info(f"[FINAL +{t:.2f}s] → SLEEPING (dropped) ({inf_ms}ms)")
                    else:
                        if text:
                            logger.info(f"[FINAL +{t:.2f}s] → OUT  '{text}' ({inf_ms}ms)")
                        else:
                            logger.info(f"[FINAL +{t:.2f}s] → OUT  <action-only> ({inf_ms}ms)")
                        dispatch(ev)
                        if actions.get("press_enter_after"):
                            feedback.play_on()
                else:
                    logger.info(f"[FINAL +{t:.2f}s] <empty> ({inf_ms}ms)")

                with state_lock:
                    if state['finalized_epoch'] < epoch:
                        state['finalized_epoch'] = epoch
                        rt_cancel.set()
                try:
                    while True:
                        rt_queue.get_nowait(); rt_queue.task_done()
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"[final_worker] {e}")
            finally:
                try: final_queue.task_done()
                except: pass

    def realtime_worker():
        while True:
            try:
                job = rt_queue.get()
                if job is None:
                    continue
                epoch, t0, audio_data = job

                with state_lock:
                    if (epoch != state['speech_epoch']
                            or state['finalized_epoch'] >= epoch
                            or rt_cancel.is_set()):
                        continue
                with mode_lock:
                    if mode_state["sleeping"]:
                        continue

                if not (MIN_CHUNKS <= len(audio_data) <= MAX_CHUNKS):
                    continue

                audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
                t_start = time.time()
                segs, _ = rt_model.transcribe(audio, language=LANGUAGE, beam_size=1)
                text = " ".join(s.text for s in segs).strip()
                inf_ms = int((time.time() - t_start) * 1000)
                t = time.time() - t0

                with state_lock:
                    if (epoch != state['speech_epoch']
                            or state['finalized_epoch'] >= epoch
                            or rt_cancel.is_set()):
                        continue
                with mode_lock:
                    if mode_state["sleeping"]:
                        continue

                if text:
                    ev = {"type": "partial", "text": text, "epoch": epoch,
                          "t": round(t, 3), "inference_ms": inf_ms}
                    logger.info(f"[PARTIAL +{t:.2f}s] → OUT  '{text}' ({inf_ms}ms)")
                    dispatch(ev)
            except Exception as e:
                logger.error(f"[realtime_worker] {e}")
            finally:
                try: rt_queue.task_done()
                except: pass

    threading.Thread(target=final_worker, daemon=True).start()
    if rt_model:
        threading.Thread(target=realtime_worker, daemon=True).start()

    # ── Hotkeys ───────────────────────────────────────────────────────────────

    def setup_hotkeys():
        try:
            import keyboard as kb
        except ImportError:
            logger.warning("[hotkeys] keyboard not installed, hotkeys disabled.")
            return

        hk = cfg["hotkeys"]

        if hk.get("clipboard_accumulate_cycle") and "clipboard_accumulate_reset" in handler_extras:
            kb.add_hotkey(hk["clipboard_accumulate_cycle"], handler_extras["clipboard_accumulate_reset"])
            logger.info(f"[hotkey] clipboard_accumulate_cycle = {hk['clipboard_accumulate_cycle']}")

        if hk.get("device_cycle"):
            kb.add_hotkey(hk["device_cycle"], lambda: cycle_input_device())
            logger.info(f"[hotkey] device_cycle = {hk['device_cycle']}")

        if SLEEP_HOTKEY_TOGGLE:
            def _sleep_toggle():
                with mode_lock:
                    sleeping = mode_state["sleeping"]
                set_sleeping(not sleeping, reason="hotkey")
            kb.add_hotkey(SLEEP_HOTKEY_TOGGLE, _sleep_toggle)
            logger.info(f"[hotkey] sleep_toggle = {SLEEP_HOTKEY_TOGGLE}")

        if TYPE_TOGGLE_HOTKEY:
            kb.add_hotkey(TYPE_TOGGLE_HOTKEY, lambda: toggle_typing_enabled(reason="hotkey"))
            logger.info(f"[hotkey] typing_toggle = {TYPE_TOGGLE_HOTKEY}")


    setup_hotkeys()
    persist()

    # ── NATS control surface ──────────────────────────────────────────────────

    def setup_nats_control():
        if not cfg["nats"]["enabled"]:
            return
        try:
            import nats as natslib
        except ImportError:
            return

        import asyncio
        import json

        url     = cfg["nats"]["url"]
        subject = cfg["nats"]["subject_control"]

        async def _run():
            try:
                nc = await natslib.connect(url)
            except Exception as e:
                logger.warning(f"[nats:control] Could not connect: {e}")
                return

            async def _handler(msg):
                try:
                    payload = json.loads(msg.data.decode())
                    cmd = payload.get("cmd", "").lower()
                except Exception:
                    return

                if cmd == "sleep":
                    set_sleeping(True, reason=f"nats:{cmd}")
                elif cmd == "wake":
                    set_sleeping(False, reason=f"nats:{cmd}")
                elif cmd == "sleep_toggle":
                    with mode_lock:
                        sleeping = mode_state["sleeping"]
                    set_sleeping(not sleeping, reason="nats:sleep_toggle")
                elif cmd == "typing_toggle":
                    toggle_typing_enabled(reason="nats:typing_toggle")
                elif cmd == "typing_enable":
                    set_typing_enabled(True, reason="nats:typing_enable")
                elif cmd == "typing_disable":
                    set_typing_enabled(False, reason="nats:typing_disable")
                elif cmd == "status_query":
                    with mode_lock:
                        sleeping = mode_state["sleeping"]
                    with typing_lock:
                        typing_enabled = typing_state["enabled"]
                    reply = {
                        "sleeping": sleeping,
                        "mode": "sleeping" if sleeping else "working",
                        "typing_at_cursor_enabled": typing_enabled,
                    }
                    if msg.reply:
                        await nc.publish(msg.reply, json.dumps(reply).encode())
                elif cmd == "device_cycle":
                    cycle_input_device()
                elif cmd == "shutdown":
                    logger.info("[nats:control] Shutdown command received.")
                    import signal
                    signal.raise_signal(signal.SIGTERM)

            await nc.subscribe(subject, cb=_handler)
            logger.info(f"[nats:control] Subscribed to {subject}")
            await asyncio.Event().wait()

        def _run_loop():
            asyncio.run(_run())

        threading.Thread(target=_run_loop, daemon=True, name="nats-control").start()

    setup_nats_control()

    # ── Wake word detector ────────────────────────────────────────────────────
    wakeword_engine = None
    wakeword_frame_length = 0
    wakeword_buffer = np.array([], dtype=np.int16)
    if WAKEWORD_ENABLED:
        if WAKEWORD_BACKEND not in {"pvporcupine", "pvp"}:
            logger.warning(f"[wake] Unsupported backend '{WAKEWORD_BACKEND}', wake word disabled.")
        elif not WAKEWORD:
            logger.warning("[wake] Empty wake_word, wake word disabled.")
        else:
            try:
                wakeword_engine = pvporcupine.create(
                    keywords=[WAKEWORD],
                    sensitivities=[WAKEWORD_SENSITIVITY]
                )
                wakeword_frame_length = int(wakeword_engine.frame_length)
                if int(wakeword_engine.sample_rate) != RATE:
                    logger.warning(
                        f"[wake] Sample-rate mismatch: wake detector {wakeword_engine.sample_rate}Hz vs STT {RATE}Hz"
                    )
                logger.info(f"[wake] Enabled backend={WAKEWORD_BACKEND} word={WAKEWORD}")
            except Exception as e:
                logger.warning(f"[wake] Could not initialize wake detector: {e}")
    else:
        logger.info("[wake] Disabled by config.")

    def detect_wakeword(chunk: np.ndarray) -> bool:
        nonlocal wakeword_buffer
        if wakeword_engine is None:
            return False
        wakeword_buffer = np.concatenate((wakeword_buffer, chunk.astype(np.int16)))
        while wakeword_buffer.size >= wakeword_frame_length > 0:
            frame = wakeword_buffer[:wakeword_frame_length]
            wakeword_buffer = wakeword_buffer[wakeword_frame_length:]
            try:
                idx = wakeword_engine.process(frame.tolist())
                if idx >= 0:
                    return True
            except Exception as e:
                logger.warning(f"[wake] Detector error: {e}")
                return False
        return False

    # ── Device cycling ────────────────────────────────────────────────────────
    # IMPORTANT: stream.close()/open() must only happen in the main loop thread.
    # The hotkey thread only writes to _pending_device; the main loop consumes it.

    current_device_idx = [resources['dev_idx']]
    _pending_device    = [None]   # (idx, name) set by hotkey, consumed by main loop
    _host_api          = resources['host_api']  # only cycle within this host API
    _bad_devices       = set()    # device indices that failed to open; skipped in cycle
    _ALIAS_NAMES       = {'Primary Sound Capture Driver', 'Microsoft Sound Mapper',
                          'Microsoft Sound Mapper - Input', 'Microsoft Sound Mapper - Output'}

    def _enumerate_input_devices():
        """Read-only enumeration — safe to call from any thread."""
        devices = []
        seen_names = set()
        for i in range(p_instance.get_device_count()):
            info = p_instance.get_device_info_by_index(i)
            if info['maxInputChannels'] <= 0:
                continue
            if info['hostApi'] != _host_api:
                continue
            name = info['name']
            if name in _ALIAS_NAMES or name in seen_names or i in _bad_devices:
                continue
            try:
                p_instance.is_format_supported(
                    RATE, input_device=i,
                    input_channels=1, input_format=pyaudio.paInt16
                )
            except Exception:
                continue
            seen_names.add(name)
            devices.append((i, name))
        return devices

    def cycle_input_device():
        """Hotkey handler — only signals the main loop, never touches the stream."""
        if _pending_device[0] is not None:
            return  # previous cycle not yet consumed
        devices = _enumerate_input_devices()
        if len(devices) <= 1:
            logger.info("[device] Only one input device available.")
            return
        idxs = [d[0] for d in devices]
        try:
            pos = idxs.index(current_device_idx[0])
        except ValueError:
            pos = -1
        _pending_device[0] = devices[(pos + 1) % len(devices)]

    # ── Main loop ─────────────────────────────────────────────────────────────
    ring_buffer      = deque(maxlen=int(BUFFER_SECONDS * RATE / CHUNK))
    recording_buffer = []
    is_recording     = False
    silence_counter  = 0
    silence_limit    = int(SILENCE_TIMEOUT * (RATE / CHUNK))
    last_rt_update   = 0.0
    current_t0       = None
    last_error       = ""
    drop_chunks_after_wake = 0

    dispatch({"type": "system", "event": "startup",
              "device": dev_name,
              "models": {"final": FINAL_MODEL, "realtime": REALTIME_MODEL}})
    dispatch({"type": "status", "value": "sleeping" if mode_state["sleeping"] else "working"})
    print("STT Ready!")
    sys.stdout.flush()

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if last_error:
                logger.info("Recovered.")
                last_error = ""

            # ── Device switch (signalled by hotkey thread) ────────────────────
            pending = _pending_device[0]
            if pending is not None:
                _pending_device[0] = None
                _open_kwargs = dict(
                    format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                )
                # Try candidates in order; skip bad ones until one works or we exhaust all
                candidates = [pending] + []  # may grow if we auto-advance past failures
                _tried = {current_device_idx[0]}
                _all_devices = _enumerate_input_devices()
                _idxs = [d[0] for d in _all_devices]
                _start_pos = next(
                    (i for i, d in enumerate(_all_devices) if d[0] == pending[0]), 0
                )
                for _attempt in range(len(_all_devices)):
                    target_pos = (_start_pos + _attempt) % len(_all_devices)
                    next_idx, next_name = _all_devices[target_pos]
                    if next_idx in _tried:
                        continue
                    _tried.add(next_idx)
                    prev_idx = current_device_idx[0]
                    try:
                        stream.stop_stream()
                        stream.close()
                        stream = p_instance.open(input_device_index=next_idx, **_open_kwargs)
                        vdata = stream.read(CHUNK, exception_on_overflow=False)
                        vchunk = np.frombuffer(vdata, dtype=np.int16).astype(np.float32)
                        rms = int(np.sqrt(np.mean(vchunk ** 2)))
                        current_device_idx[0] = next_idx
                        current_device_state[0] = {"name": next_name, "host_api": _host_api}
                        wakeword_buffer = np.array([], dtype=np.int16)
                        ring_buffer.clear(); recording_buffer.clear(); is_recording = False
                        logger.info(f"[device] Cycled to: {next_name} (rms={rms})")
                        dispatch({"type": "system", "event": "device_changed", "device": next_name})
                        if target_pos == 0:
                            feedback.play_off()
                        else:
                            feedback.play_on()
                        persist()
                        break
                    except Exception as e:
                        _bad_devices.add(next_idx)
                        logger.warning(f"[device] Skipping {next_name}: {e}")
                        try: stream.stop_stream(); stream.close()
                        except: pass
                        try:
                            stream = p_instance.open(input_device_index=prev_idx, **_open_kwargs)
                            current_device_idx[0] = prev_idx
                        except: pass
                continue

            chunk = np.frombuffer(data, dtype=np.int16)

            with mode_lock:
                sleeping = mode_state["sleeping"]
            if sleeping:
                if is_recording or recording_buffer:
                    ring_buffer.clear()
                    recording_buffer = []
                    is_recording = False
                    silence_counter = 0
                    current_t0 = None
                if detect_wakeword(chunk):
                    set_sleeping(False, reason=f"wake:{WAKEWORD}")
                    drop_chunks_after_wake = int(max(0.0, WAKEWORD_DROP_SECONDS) * RATE / CHUNK)
                    ring_buffer.clear()
                    recording_buffer = []
                    is_recording = False
                    silence_counter = 0
                    current_t0 = None
                continue

            if drop_chunks_after_wake > 0:
                drop_chunks_after_wake -= 1
                continue

            prob  = vad.is_speech(chunk.astype(np.float32) / 32768.0)

            if prob > VAD_THRESHOLD:
                silence_counter = 0
                if not is_recording:
                    with state_lock:
                        state['speech_epoch'] += 1
                        state['finalized_epoch'] = -1
                        rt_cancel.clear()
                        epoch = state['speech_epoch']
                    current_t0 = time.time()
                    recording_buffer = list(ring_buffer)
                    ring_buffer.clear()
                    is_recording = True
                    dispatch({"type": "status", "value": "recording"})
                recording_buffer.append(chunk)
            else:
                if is_recording:
                    recording_buffer.append(chunk)
                    silence_counter += 1
                    if silence_counter > silence_limit:
                        with state_lock: epoch = state['speech_epoch']
                        is_recording = False
                        final_queue.put((epoch, current_t0, list(recording_buffer)))
                        recording_buffer = []
                        current_t0 = None
                        dispatch({"type": "status", "value": "idle"})
                else:
                    ring_buffer.append(chunk)

            if rt_model and is_recording and (time.time() - last_rt_update > RT_CHECK_INTERVAL):
                with state_lock:
                    ep  = state['speech_epoch']
                    fin = state['finalized_epoch']
                n = len(recording_buffer)
                if fin < ep and MIN_CHUNKS <= n <= MAX_CHUNKS:
                    try:
                        rt_queue.put_nowait((ep, current_t0, list(recording_buffer)))
                    except queue.Full:
                        try:
                            rt_queue.get_nowait(); rt_queue.task_done()
                            rt_queue.put_nowait((ep, current_t0, list(recording_buffer)))
                        except: pass
                last_rt_update = time.time()

        except (OSError, IOError) as e:
            err = str(e)
            if err != last_error:
                logger.warning(f"Stream error: {err}. Retrying...")
            last_error = err
            time.sleep(0.5)
            try:
                stream.stop_stream()
                stream.close()
            except: pass
            try:
                stream = p_instance.open(
                    format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                    input_device_index=current_device_idx[0]
                )
                wakeword_buffer = np.array([], dtype=np.int16)
                ring_buffer.clear()
                recording_buffer.clear()
                is_recording = False
                last_error = ""
            except Exception as e2:
                logger.warning(f"[device] Recovery failed: {e2}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected: {e}")
            break

    import os
    logger.info("Shutting down...")
    try: stream.stop_stream(); stream.close()
    except: pass
    try:
        if wakeword_engine is not None:
            wakeword_engine.delete()
    except: pass
    try: feedback.shutdown()
    except: pass
    try: p_instance.terminate()
    except: pass
    if not final_queue.empty():
        logger.info("Waiting for final queue to drain...")
        final_queue.join()
    dispatch({"type": "system", "event": "shutdown"})
    logger.info("Done.")
    os._exit(0)


if __name__ == '__main__':
    main()
