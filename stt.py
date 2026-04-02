"""
convocortex-stt — headless speech-to-text engine

Dual-model pipeline: fast CPU model for realtime partials,
accurate GPU model for final results after silence.
Output via configurable local handlers. NATS optional.
"""

import copy
import _thread
import math
import sys
import time
import queue
import re
import threading
import logging
import os
from collections import deque
from pathlib import Path

import numpy as np
import pyaudio
import pvporcupine

from audio_devices import (
    ALIAS_DEVICE_NAMES,
    available_input_devices,
    available_output_devices,
    describe_device,
    host_api_names,
    load_profiles,
    matches_any_profile,
    open_input_session,
    order_devices_by_profiles,
    probe_output_device,
    profile_matches,
    save_profiles,
)
import config
from model_backends import load_backend


def _bootstrap_owned_console():
    if os.name != "nt":
        return
    if os.environ.get("CONVOCORTEX_OWN_CONSOLE", "").strip() != "1":
        return
    try:
        import ctypes
    except Exception:
        return

    kernel32 = ctypes.windll.kernel32
    try:
        if int(kernel32.GetConsoleWindow()):
            return
    except Exception:
        return

    try:
        if not kernel32.AllocConsole():
            return
    except Exception:
        return

    startup_mode = os.environ.get("CONVOCORTEX_CONSOLE_STARTUP_MODE", "").strip().lower()
    if startup_mode == "background":
        try:
            hwnd = int(kernel32.GetConsoleWindow())
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)
        except Exception:
            pass

    try:
        sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
    except Exception:
        pass
    try:
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
    except Exception:
        pass


_bootstrap_owned_console()


def _normalize_command_phrase(value: str) -> str:
    return str(value).strip().lower().strip(".,!?;:")


def _normalize_ignored_transcript(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,!?;:'\"()[]{}")


def _normalize_disfluency_phrase(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,!?;:'\"()[]{}")


def _phrase_pattern(phrase: str) -> str:
    parts = [re.escape(part) for part in phrase.split() if part]
    return r"(?:[\s-]+)".join(parts)


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


cfg = config.load()
FILTERS_CFG = cfg.get("filters", {})

# ── Logging ───────────────────────────────────────────────────────────────────
_log_cfg = cfg.get("logging", {})
DEBUG_LOGGING_ENABLED = bool(_log_cfg.get("debug", False))
DEBUG_LOG_FILE = str(_log_cfg.get("file", "stt-debug.log")).strip()
THIRD_PARTY_DEBUG_LOGGING = bool(_log_cfg.get("third_party_debug", False))
DEBUG_HEARTBEAT_SECONDS = max(1.0, float(_log_cfg.get("heartbeat_seconds", 5.0)))


def _setup_logging():
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if DEBUG_LOGGING_ENABLED else logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    if DEBUG_LOGGING_ENABLED and DEBUG_LOG_FILE:
        file_handler = logging.FileHandler(DEBUG_LOG_FILE, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(file_handler)

    fw_level = logging.DEBUG if DEBUG_LOGGING_ENABLED and THIRD_PARTY_DEBUG_LOGGING else logging.WARNING
    logging.getLogger("faster_whisper").setLevel(fw_level)
    logging.getLogger("ctranslate2").setLevel(fw_level)
    logging.getLogger("nemo").setLevel(fw_level)
    logging.getLogger("lightning").setLevel(fw_level)

    configured_logger = logging.getLogger("STT")
    if DEBUG_LOGGING_ENABLED:
        target = DEBUG_LOG_FILE or "<disabled>"
        configured_logger.debug(f"[debug] enabled file={target} third_party={THIRD_PARTY_DEBUG_LOGGING}")
    return configured_logger


logger = _setup_logging()


class WindowsConsoleController:
    SW_HIDE = 0
    SW_SHOW = 5
    SW_RESTORE = 9

    def __init__(self):
        self._lock = threading.RLock()
        self._user32 = None
        self._kernel32 = None
        self._available = False
        if os.name != "nt":
            return
        try:
            import ctypes

            self._user32 = ctypes.windll.user32
            self._kernel32 = ctypes.windll.kernel32
            self._available = bool(self._console_hwnd())
        except Exception as exc:
            logger.warning(f"[console] Windows console control unavailable: {exc}")

    def _console_hwnd(self) -> int:
        if not self._kernel32:
            return 0
        try:
            return int(self._kernel32.GetConsoleWindow())
        except Exception:
            return 0

    @property
    def available(self) -> bool:
        return bool(self._available and self._console_hwnd())

    def is_visible(self) -> bool:
        hwnd = self._console_hwnd()
        if not hwnd or not self._user32:
            return False
        return bool(self._user32.IsWindowVisible(hwnd))

    def show(self, reason: str = "") -> bool:
        with self._lock:
            hwnd = self._console_hwnd()
            if not hwnd or not self._user32:
                logger.warning("[console] No console window available to show.")
                return False
            self._user32.ShowWindow(hwnd, self.SW_RESTORE)
            self._user32.ShowWindow(hwnd, self.SW_SHOW)
            self._user32.SetForegroundWindow(hwnd)
            logger.info(f"[console] Shown" + (f" ({reason})" if reason else ""))
            return True

    def hide(self, reason: str = "") -> bool:
        with self._lock:
            hwnd = self._console_hwnd()
            if not hwnd or not self._user32:
                logger.warning("[console] No console window available to hide.")
                return False
            self._user32.ShowWindow(hwnd, self.SW_HIDE)
            logger.info(f"[console] Hidden" + (f" ({reason})" if reason else ""))
            return True

    def toggle(self, reason: str = "") -> bool:
        if self.is_visible():
            return self.hide(reason=reason)
        return self.show(reason=reason)


class WindowsTrayIcon:
    WMAPP_NOTIFYCALLBACK = 0x8001
    CMD_TOGGLE_CONSOLE = 1001
    CMD_EXIT = 1002

    def __init__(self, console_controller: WindowsConsoleController):
        self._console_controller = console_controller
        self._thread = None
        self._ready = threading.Event()
        self._hwnd = None
        self._class_name = f"ConvocortexSTTTray-{os.getpid()}"
        self._available = False
        self._win32api = None
        self._win32con = None
        self._win32gui = None

        if os.name != "nt":
            return
        try:
            import win32api
            import win32con
            import win32gui

            self._win32api = win32api
            self._win32con = win32con
            self._win32gui = win32gui
            self._available = True
        except ImportError:
            logger.warning("[tray] pywin32 not installed, tray icon disabled.")
        except Exception as exc:
            logger.warning(f"[tray] Windows tray icon unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> bool:
        if not self._available:
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._thread = threading.Thread(target=self._run, name="tray-icon", daemon=True)
        self._thread.start()
        return self._ready.wait(timeout=5.0)

    def stop(self, timeout: float = 2.0):
        if not self._available or not self._thread:
            return
        hwnd = self._hwnd
        if hwnd:
            try:
                self._win32gui.PostMessage(hwnd, self._win32con.WM_CLOSE, 0, 0)
            except Exception:
                pass
        self._thread.join(timeout=timeout)

    def _menu_label(self) -> str:
        return "Hide Console" if self._console_controller.is_visible() else "Show Console"

    def _toggle_console(self):
        self._console_controller.toggle(reason="tray")

    def _request_exit(self):
        logger.info("[tray] Exit requested.")
        _thread.interrupt_main()

    def _show_menu(self):
        if not self._hwnd:
            return
        menu = self._win32gui.CreatePopupMenu()
        try:
            self._win32gui.AppendMenu(menu, self._win32con.MF_STRING, self.CMD_TOGGLE_CONSOLE, self._menu_label())
            self._win32gui.AppendMenu(menu, self._win32con.MF_SEPARATOR, 0, "")
            self._win32gui.AppendMenu(menu, self._win32con.MF_STRING, self.CMD_EXIT, "Exit STT")
            pos = self._win32gui.GetCursorPos()
            self._win32gui.SetForegroundWindow(self._hwnd)
            self._win32gui.TrackPopupMenu(
                menu,
                self._win32con.TPM_LEFTALIGN | self._win32con.TPM_BOTTOMALIGN | self._win32con.TPM_RIGHTBUTTON,
                pos[0],
                pos[1],
                0,
                self._hwnd,
                None,
            )
            self._win32gui.PostMessage(self._hwnd, self._win32con.WM_NULL, 0, 0)
        finally:
            self._win32gui.DestroyMenu(menu)

    def _on_command(self, hwnd, msg, wparam, lparam):
        cmd = self._win32api.LOWORD(wparam)
        if cmd == self.CMD_TOGGLE_CONSOLE:
            self._toggle_console()
        elif cmd == self.CMD_EXIT:
            self._request_exit()
        return 0

    def _on_notify(self, hwnd, msg, wparam, lparam):
        if lparam == self._win32con.WM_LBUTTONUP:
            self._toggle_console()
        elif lparam in {self._win32con.WM_RBUTTONUP, self._win32con.WM_CONTEXTMENU}:
            self._show_menu()
        return 0

    def _remove_icon(self):
        if not self._hwnd:
            return
        try:
            self._win32gui.Shell_NotifyIcon(self._win32gui.NIM_DELETE, (self._hwnd, 0))
        except Exception:
            pass

    def _on_destroy(self, hwnd, msg, wparam, lparam):
        self._remove_icon()
        self._win32gui.PostQuitMessage(0)
        return 0

    def _run(self):
        try:
            message_map = {
                self._win32con.WM_COMMAND: self._on_command,
                self._win32con.WM_DESTROY: self._on_destroy,
                self.WMAPP_NOTIFYCALLBACK: self._on_notify,
            }
            window_class = self._win32gui.WNDCLASS()
            window_class.hInstance = self._win32api.GetModuleHandle(None)
            window_class.lpszClassName = self._class_name
            window_class.lpfnWndProc = message_map
            class_atom = self._win32gui.RegisterClass(window_class)
            self._hwnd = self._win32gui.CreateWindow(
                class_atom,
                self._class_name,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                window_class.hInstance,
                None,
            )
            icon_handle = self._win32gui.LoadIcon(0, self._win32con.IDI_APPLICATION)
            notify_id = (
                self._hwnd,
                0,
                self._win32gui.NIF_ICON | self._win32gui.NIF_MESSAGE | self._win32gui.NIF_TIP,
                self.WMAPP_NOTIFYCALLBACK,
                icon_handle,
                "Convocortex STT",
            )
            self._win32gui.Shell_NotifyIcon(self._win32gui.NIM_ADD, notify_id)
            logger.info("[tray] Ready.")
        except Exception as exc:
            logger.warning(f"[tray] Failed to initialize: {exc}")
        finally:
            self._ready.set()

        if not self._hwnd:
            return

        try:
            self._win32gui.PumpMessages()
        finally:
            self._remove_icon()

# ── Models ────────────────────────────────────────────────────────────────────
FINAL_MODEL        = cfg["models"]["final"]
FINAL_DEVICE       = cfg["models"]["final_device"]
FINAL_COMPUTE      = cfg["models"]["final_compute"]
FINAL_BACKEND      = str(cfg["models"].get("final_backend", "whisper")).strip().lower()
REALTIME_MODEL     = cfg["models"]["realtime"]
REALTIME_DEVICE    = cfg["models"]["realtime_device"]
REALTIME_BACKEND   = str(cfg["models"].get("realtime_backend", "whisper")).strip().lower()
LANGUAGE           = cfg["models"]["language"]
WHISPER_NO_SPEECH_THRESHOLD = float(cfg["models"].get("no_speech_threshold", 0.6))
WHISPER_LOG_PROB_THRESHOLD = float(cfg["models"].get("log_prob_threshold", -1.0))

# ── Audio ─────────────────────────────────────────────────────────────────────
RATE            = cfg["audio"]["rate"]
CHUNK           = cfg["audio"]["chunk"]
VAD_THRESHOLD   = cfg["audio"]["vad_threshold"]
VAD_END_THRESHOLD = float(cfg["audio"].get("vad_end_threshold", max(float(VAD_THRESHOLD) - 0.15, 0.01)))
MIN_SPEECH_DURATION_MS = float(cfg["audio"].get("min_speech_duration_ms", 250.0))
BUFFER_SECONDS  = cfg["audio"]["buffer_seconds"]
SILENCE_TIMEOUT = cfg["audio"]["silence_timeout"]
INPUT_STALL_RESET_SECONDS = float(cfg["audio"].get("input_stall_reset_seconds", 1.5))
PREFERRED_INPUT_DEVICE = str(cfg["audio"].get("input_device", "")).strip()
DEVICE_PROFILES_PATH = Path(str(cfg["audio"].get("device_profiles_file", "device-profiles.json")).strip() or "device-profiles.json")
DEVICE_SETUP_INITIALIZED = bool(cfg["audio"].get("device_setup_initialized", False))
_startup_cfg = cfg.get("startup", {})


def _normalize_startup_source(value, default: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"config", "state"}:
        return normalized
    return default


def _normalize_console_startup_mode(value, default: str = "foreground") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"foreground", "background"}:
        return normalized
    return default


def _normalize_silence_keepalive_mode(value, default: str = "always") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"off", "always"}:
        return normalized
    return default


INPUT_DEVICE_STARTUP_SOURCE = "config"
OUTPUT_DEVICE_STARTUP_SOURCE = "config"
OUTPUT_MODE_STARTUP_SOURCE = "config"
CONSOLE_STARTUP_MODE = _normalize_console_startup_mode(
    _startup_cfg.get("console_startup_mode", "foreground"),
    "foreground",
)

# ── Realtime ──────────────────────────────────────────────────────────────────
RT_CHECK_INTERVAL = cfg["realtime"]["check_interval"]
MIN_CHUNKS        = cfg["realtime"]["min_chunks"]
MAX_CHUNKS        = cfg["realtime"]["max_chunks"]
RT_QUEUE_SIZE     = cfg["realtime"]["queue_size"]

# ── Sleep / Wake ───────────────────────────────────────────────────────────────
_sw_cfg = cfg.get("sleep_wake", {})
SLEEP_START_SLEEPING = bool(_sw_cfg.get("start_sleeping", True))
SLEEP_HOTKEY_TOGGLE = str(cfg["hotkeys"].get("sleep_toggle", "")).strip()
SLEEP_HOTKEY_SUPPRESS = bool(cfg["hotkeys"].get("sleep_toggle_suppress", False))
TYPE_TOGGLE_HOTKEY = str(cfg["hotkeys"].get("typing_toggle", "")).strip()
OUTPUT_MODE_CYCLE_HOTKEY = str(cfg["hotkeys"].get("output_mode_cycle", "")).strip()
CONSOLE_TOGGLE_HOTKEY = str(cfg["hotkeys"].get("console_toggle", "")).strip()
SLEEP_STOP_WORDS = [
    _normalize_command_phrase(w)
    for w in _sw_cfg.get("stop_words", ["stop", "pause"])
    if _normalize_command_phrase(w)
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
FEEDBACK_STARTUP_SOUND = str(_fb_cfg.get("startup_sound", "sounds/startup.ogg")).strip()
FEEDBACK_FINAL_SOUND = str(_fb_cfg.get("final_sound", "sounds/final.ogg")).strip()
FEEDBACK_FINAL_SOUND_ENABLED = bool(_fb_cfg.get("final_sound_enabled", True))
FEEDBACK_SILENCE_KEEPALIVE_MODE = _normalize_silence_keepalive_mode(
    _fb_cfg.get("silence_keepalive_mode", "always"),
    "always",
)
FEEDBACK_ON_VOLUME = float(_fb_cfg.get("on_volume", 1.0))
FEEDBACK_OFF_VOLUME = float(_fb_cfg.get("off_volume", 1.0))
FEEDBACK_FINAL_VOLUME = float(_fb_cfg.get("final_volume", 1.0))
FEEDBACK_OUTPUT_DEVICE = str(_fb_cfg.get("output_device", "")).strip()
EXPECTED_CHUNK_MS = (CHUNK / RATE) * 1000.0
INPUT_PROBE_READ_LIMIT_MS = max(400.0, EXPECTED_CHUNK_MS * 8.0)
INPUT_PROBE_READS = 2
DEVICE_RETRY_COOLDOWN_SECONDS = 30.0
IGNORED_TRANSCRIPTS = {
    normalized
    for normalized in (
        _normalize_ignored_transcript(w)
        for w in FILTERS_CFG.get("ignored_exact_phrases", [])
    )
    if normalized
}
DISFLUENCY_PATTERNS = [
    re.compile(
        rf"(?ix)(?:(?<=^)|(?<=[\s(\[{{\"'])){_phrase_pattern(phrase)}"
        rf"(?:\s*[.,!?;:]+)?(?:(?=$)|(?=[\s)\]}}\"']))"
    )
    for phrase in sorted(
        {
            normalized
            for normalized in (
                _normalize_disfluency_phrase(w)
                for w in FILTERS_CFG.get("disfluency_words", [])
            )
            if normalized
        },
        key=len,
        reverse=True,
    )
]

# ── Voice commands (minimal) ──────────────────────────────────────────────────
VOICE_COMMANDS_ENABLED = bool(cfg.get("voice_commands", {}).get("enabled", False))


def _voice_command_words(name: str) -> list[str]:
    return [
        _normalize_command_phrase(w)
        for w in cfg.get("voice_commands", {}).get(name, {}).get("words", [])
        if _normalize_command_phrase(w)
    ]


def _voice_command_enabled(name: str, *, extra: bool = True) -> bool:
    return (
        VOICE_COMMANDS_ENABLED
        and extra
        and bool(cfg.get("voice_commands", {}).get(name, {}).get("enabled", False))
        and bool(_voice_command_words(name))
    )


ENTER_COMMAND_WORDS = [
    str(w).strip()
    for w in cfg.get("voice_commands", {}).get("enter", {}).get("words", [])
    if str(w).strip()
]
ENTER_COMMAND_WORDS_EXACT = [
    _normalize_command_phrase(w)
    for w in ENTER_COMMAND_WORDS
    if _normalize_command_phrase(w)
]
ENTER_COMMAND_ENABLED = (
    VOICE_COMMANDS_ENABLED
    and bool(cfg.get("voice_commands", {}).get("enter", {}).get("enabled", False))
    and bool(ENTER_COMMAND_WORDS)
    and bool(ENTER_COMMAND_WORDS_EXACT)
)
TYPE_TOGGLE_COMMAND_WORDS = _voice_command_words("type_at_cursor_toggle")
TYPE_TOGGLE_COMMAND_ENABLED = _voice_command_enabled("type_at_cursor_toggle")
REVERT_COMMAND_WORDS = _voice_command_words("rewind")
REVERT_COMMAND_ENABLED = _voice_command_enabled("rewind")
REPEAT_COMMAND_WORDS = _voice_command_words("repeat")
REPEAT_COMMAND_ENABLED = _voice_command_enabled("repeat")
INPUT_DEVICE_CYCLE_COMMAND_WORDS = _voice_command_words("input_device_cycle")
INPUT_DEVICE_CYCLE_COMMAND_ENABLED = _voice_command_enabled("input_device_cycle")
OUTPUT_DEVICE_CYCLE_COMMAND_WORDS = _voice_command_words("output_device_cycle")
OUTPUT_DEVICE_CYCLE_COMMAND_ENABLED = _voice_command_enabled("output_device_cycle")
OUTPUT_MODE_DEFAULT_COMMAND_WORDS = _voice_command_words("output_mode_default")
OUTPUT_MODE_DEFAULT_COMMAND_ENABLED = _voice_command_enabled("output_mode_default")
OUTPUT_MODE_CURSOR_COMMAND_WORDS = _voice_command_words("output_mode_cursor")
OUTPUT_MODE_CURSOR_COMMAND_ENABLED = _voice_command_enabled("output_mode_cursor")
OUTPUT_MODE_DRAFT_COMMAND_WORDS = _voice_command_words("output_mode_draft")
OUTPUT_MODE_DRAFT_COMMAND_ENABLED = _voice_command_enabled("output_mode_draft")
CONSOLE_SHOW_COMMAND_WORDS = _voice_command_words("console_show")
CONSOLE_SHOW_COMMAND_ENABLED = _voice_command_enabled("console_show")
CONSOLE_HIDE_COMMAND_WORDS = _voice_command_words("console_hide")
CONSOLE_HIDE_COMMAND_ENABLED = _voice_command_enabled("console_hide")
BUFFER_RELEASE_COMMAND_WORDS = _voice_command_words("buffer_release")
BUFFER_RELEASE_COMMAND_ENABLED = _voice_command_enabled(
    "buffer_release",
    extra=bool(cfg.get("output", {}).get("file_buffer", {}).get("enabled", False)),
)
BUFFER_CLEAR_COMMAND_WORDS = _voice_command_words("buffer_clear")
BUFFER_CLEAR_COMMAND_ENABLED = _voice_command_enabled(
    "buffer_clear",
    extra=bool(cfg.get("output", {}).get("file_buffer", {}).get("enabled", False)),
)
REVERT_COMMAND_WORDS_EXACT = REVERT_COMMAND_WORDS
REPEAT_COMMAND_WORDS_EXACT = REPEAT_COMMAND_WORDS
INPUT_DEVICE_CYCLE_COMMAND_WORDS_EXACT = INPUT_DEVICE_CYCLE_COMMAND_WORDS
OUTPUT_DEVICE_CYCLE_COMMAND_WORDS_EXACT = OUTPUT_DEVICE_CYCLE_COMMAND_WORDS
OUTPUT_MODE_DEFAULT_COMMAND_WORDS_EXACT = OUTPUT_MODE_DEFAULT_COMMAND_WORDS
OUTPUT_MODE_CURSOR_COMMAND_WORDS_EXACT = OUTPUT_MODE_CURSOR_COMMAND_WORDS
OUTPUT_MODE_DRAFT_COMMAND_WORDS_EXACT = OUTPUT_MODE_DRAFT_COMMAND_WORDS
CONSOLE_SHOW_COMMAND_WORDS_EXACT = CONSOLE_SHOW_COMMAND_WORDS
CONSOLE_HIDE_COMMAND_WORDS_EXACT = CONSOLE_HIDE_COMMAND_WORDS
BUFFER_RELEASE_COMMAND_WORDS_EXACT = BUFFER_RELEASE_COMMAND_WORDS
BUFFER_CLEAR_COMMAND_WORDS_EXACT = BUFFER_CLEAR_COMMAND_WORDS

OUTPUT_MODE_NAMES = [
    "direct-cursor",
    "draft-buffer",
]

def _clamp_volume(v: float) -> float:
    return max(0.0, min(2.0, float(v)))


_debug_timestamps: dict[str, float] = {}


def debug_log(message: str):
    if DEBUG_LOGGING_ENABLED:
        logger.debug(message)


def debug_log_every(key: str, seconds: float, message: str):
    if not DEBUG_LOGGING_ENABLED:
        return
    now = time.monotonic()
    if now - _debug_timestamps.get(key, 0.0) >= seconds:
        _debug_timestamps[key] = now
        logger.debug(message)



def _prompt_profile_selection(
    label: str,
    candidates: list[dict],
    existing_profiles: list[dict],
    *,
    recommended_profiles: list[dict] | None = None,
    allow_none: bool,
) -> list[dict]:
    existing_indexes = []
    for idx, candidate in enumerate(candidates, start=1):
        if any(profile_matches(candidate["info"], profile) for profile in existing_profiles):
            existing_indexes.append(str(idx))
    recommended_indexes = []
    for idx, candidate in enumerate(candidates, start=1):
        if any(profile_matches(candidate["info"], profile) for profile in (recommended_profiles or [])):
            recommended_indexes.append(str(idx))
    default_text = ",".join(existing_indexes or recommended_indexes)

    print()
    for idx, candidate in enumerate(candidates, start=1):
        status = "usable" if candidate.get("usable", True) else f"unusable: {candidate.get('reason', 'unknown')}"
        print(f"  {idx}. {candidate['description']} [{status}]")
    print(f"[device-setup] Select {label} devices in the order you want them used.")
    print("[device-setup] The first device becomes the startup default, and cycling follows the same order.")
    if allow_none:
        print("[device-setup] Press Enter to keep the current selection. Enter 'none' for no devices.")
    else:
        print("[device-setup] Press Enter to keep the current selection. Choose at least one device.")

    while True:
        prompt = f"{label} selection"
        if default_text:
            prompt += f" [{default_text}]"
        prompt += ": "
        try:
            raw = input(prompt)
        except EOFError:
            return existing_profiles
        raw = raw.strip()
        if not raw:
            if default_text:
                raw = default_text
            else:
                if allow_none:
                    print(f"[device-setup] No {label} devices selected.")
                    return []
                print(f"[device-setup] Select at least one {label} device.")
                continue
        if raw.lower() in {"none", "off", "disable"}:
            if allow_none:
                return []
            print(f"[device-setup] Select at least one {label} device.")
            continue
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        chosen = []
        seen = set()
        ok = True
        for part in parts:
            if not part.isdigit():
                ok = False
                break
            pos = int(part)
            if pos < 1 or pos > len(candidates):
                ok = False
                break
            if pos in seen:
                continue
            seen.add(pos)
            chosen.append(candidates[pos - 1]["profile"])
        if ok:
            return chosen
        print("[device-setup] Invalid selection. Use numbers like 1,3")


def _set_device_setup_initialized(value: bool):
    try:
        config_path = config.CONFIG_PATH
    except Exception:
        return
    try:
        text = Path(config_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[device-setup] Could not read config for initialization flag update: {e}")
        return

    lines = text.splitlines()
    in_audio = False
    updated = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_audio = stripped == "[audio]"
            continue
        if in_audio and stripped.startswith("device_setup_initialized"):
            lines[idx] = f"device_setup_initialized = {'true' if value else 'false'}"
            updated = True
            break
    if not updated:
        logger.warning("[device-setup] Could not find device_setup_initialized in config.toml")
        return
    try:
        Path(config_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning(f"[device-setup] Could not update initialization flag: {e}")


def _recommended_setup_profiles(
    candidates: list[dict],
    preferred_name: str,
    default_info: dict | None,
) -> list[dict]:
    for candidate in candidates:
        if not candidate.get("usable", True):
            continue
        if preferred_name and str(candidate["profile"].get("name", "")).strip() == preferred_name:
            return [candidate["profile"]]
    if default_info is not None:
        for candidate in candidates:
            if not candidate.get("usable", True):
                continue
            if int(candidate["info"].get("index", -1)) == int(default_info.get("index", -2)):
                return [candidate["profile"]]
    for candidate in candidates:
        if candidate.get("usable", True):
            return [candidate["profile"]]
    return []



def _maybe_run_device_setup() -> dict:
    profiles = load_profiles(DEVICE_PROFILES_PATH)
    needs_setup = (not DEVICE_SETUP_INITIALIZED) or (not DEVICE_PROFILES_PATH.exists())
    if not needs_setup:
        return profiles

    logger.info("[device-setup] Starting interactive device setup.")
    p = pyaudio.PyAudio()
    try:
        api_names = host_api_names(p)
        existing_inputs = profiles.get("inputs", [])
        existing_outputs = profiles.get("outputs", [])
        try:
            default_input_info = p.get_default_input_device_info()
        except Exception:
            default_input_info = None
        try:
            default_output_info = p.get_default_output_device_info()
        except Exception:
            default_output_info = None
        input_candidates = []
        for info in available_input_devices(p):
            reason = "ok"
            usable = True
            try:
                session, _, _ = open_input_session(
                    p,
                    info,
                    stt_rate=RATE,
                    chunk_frames=CHUNK,
                    probe_reads=1,
                    probe_limit_ms=INPUT_PROBE_READ_LIMIT_MS,
                )
                session.close()
            except Exception as exc:
                usable = False
                reason = str(exc)
            input_candidates.append({
                "info": info,
                "profile": {"name": str(info.get("name", "")).strip(), "host_api": info.get("hostApi")},
                "description": describe_device(info, api_names),
                "usable": usable,
                "reason": reason,
            })
        output_candidates = []
        for info in available_output_devices(p):
            usable, reason = probe_output_device(p, info)
            output_candidates.append({
                "info": info,
                "profile": {"name": str(info.get("name", "")).strip(), "host_api": info.get("hostApi")},
                "description": describe_device(info, api_names),
                "usable": usable,
                "reason": reason,
            })

        selected_inputs = _prompt_profile_selection(
            "input",
            input_candidates,
            existing_inputs,
            recommended_profiles=_recommended_setup_profiles(
                input_candidates,
                PREFERRED_INPUT_DEVICE,
                default_input_info,
            ),
            allow_none=False,
        )
        selected_outputs = _prompt_profile_selection(
            "output",
            output_candidates,
            existing_outputs,
            recommended_profiles=_recommended_setup_profiles(
                output_candidates,
                FEEDBACK_OUTPUT_DEVICE,
                default_output_info,
            ),
            allow_none=not FEEDBACK_ENABLED,
        )
        profiles = {
            "inputs": selected_inputs,
            "outputs": selected_outputs,
        }
        save_profiles(DEVICE_PROFILES_PATH, profiles)
        _set_device_setup_initialized(True)
        logger.info(f"[device-setup] Saved profiles to {DEVICE_PROFILES_PATH}")
        logger.info("[device-setup] The first approved device in each list is used at startup.")
        logger.info("[device-setup] Initialization complete. Set audio.device_setup_initialized = false to rerun.")
        return profiles
    finally:
        try:
            p.terminate()
        except Exception:
            pass


DEVICE_PROFILES = _maybe_run_device_setup()
APPROVED_INPUT_PROFILES = DEVICE_PROFILES.get("inputs", [])
APPROVED_OUTPUT_PROFILES = DEVICE_PROFILES.get("outputs", [])

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
    def __init__(
        self,
        enabled: bool,
        on_path: str,
        off_path: str,
        silence_path: str,
        startup_path: str,
        final_path: str,
        final_sound_enabled: bool,
        silence_keepalive_mode: str,
        on_volume: float,
        off_volume: float,
        final_volume: float,
        output_device: str,
        startup_source: str = "state",
        persisted_output_device_name: str = "",
        persisted_output_device_host_api: int | None = None,
    ):
        self.enabled = enabled
        self.on_path = on_path
        self.off_path = off_path
        self.silence_path = silence_path
        self.startup_path = startup_path
        self.final_path = final_path
        self.final_sound_enabled = final_sound_enabled
        self.silence_keepalive_mode = _normalize_silence_keepalive_mode(silence_keepalive_mode, "always")
        self.on_volume = _clamp_volume(on_volume)
        self.off_volume = _clamp_volume(off_volume)
        self.final_volume = _clamp_volume(final_volume)
        self.output_device = output_device
        self.startup_source = _normalize_startup_source(startup_source, "state")
        self._persisted_output_device_name = str(persisted_output_device_name or "").strip()
        self._persisted_output_device_host_api = persisted_output_device_host_api
        configured_output_name = str(output_device or "").strip()
        if self.startup_source == "config":
            self._preferred_output_device_name = configured_output_name
            self._preferred_output_device_host_api = None
        else:
            self._preferred_output_device_name = configured_output_name or self._persisted_output_device_name
            self._preferred_output_device_host_api = (
                None if configured_output_name else persisted_output_device_host_api
            )
        self._p_sfx = None
        self._p_silence = None
        self._output_device_index = None
        self._output_device_name = ""
        self._output_device_host_api = None
        self._probe_sr = None
        self._probe_channels = None
        self._sound_queue = queue.Queue()
        self._running = False
        self._clips = {}
        self._silence_thread = None
        self._sound_thread = None
        self._output_device_generation = 0
        self._lock = threading.Lock()

        if not self.enabled:
            return
        try:
            import torchaudio

            self._clips["on"] = self._load_clip(torchaudio, self.on_path, self.on_volume)
            self._clips["off"] = self._load_clip(torchaudio, self.off_path, self.off_volume)
            self._clips["silence"] = self._load_clip(torchaudio, self.silence_path)
            self._clips["startup"] = self._load_clip(torchaudio, self.startup_path)
            if self.final_sound_enabled:
                try:
                    self._clips["final"] = self._load_clip(torchaudio, self.final_path, self.final_volume)
                except Exception as e:
                    logger.warning(f"[feedback] final_sound disabled: {e}")
                    self.final_sound_enabled = False
            self._probe_sr = int(self._clips.get("on", {}).get("sr") or 0) or None
            self._probe_channels = int(self._clips.get("on", {}).get("channels") or 0) or None
            self._p_sfx = pyaudio.PyAudio()
            self._p_silence = pyaudio.PyAudio()
            self._set_initial_output_device()
            self._running = True
            self._sound_thread = threading.Thread(target=self._sound_worker, daemon=True, name="feedback-sound")
            self._sound_thread.start()
            if self.silence_keepalive_mode == "always":
                self._silence_thread = threading.Thread(
                    target=self._silence_worker,
                    daemon=True,
                    name="feedback-silence",
                )
                self._silence_thread.start()
            if self._output_device_name:
                logger.info(f"[feedback] Enabled (output={self._output_device_name})")
            else:
                logger.info("[feedback] Enabled (output=default)")
            logger.info(f"[feedback] silence_keepalive_mode={self.silence_keepalive_mode}")
        except Exception as e:
            logger.warning(f"[feedback] Disabled: {e}")
            self.enabled = False

    def _iter_output_devices(self, p_instance):
        for i in range(p_instance.get_device_count()):
            info = p_instance.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) > 0:
                yield info

    def _find_output_device_by_name(self, p_instance, name: str, host_api: int | None = None):
        target = name.strip().lower()
        if not target:
            return None
        for info in self._iter_output_devices(p_instance):
            if host_api is not None and info.get("hostApi") != host_api:
                continue
            if str(info.get("name", "")).strip().lower() == target:
                return info
        return None

    def _supports_output_format(self, p_instance, device_index: int) -> bool:
        if not self._probe_sr or not self._probe_channels:
            return True
        try:
            p_instance.is_format_supported(
                self._probe_sr,
                output_device=int(device_index),
                output_channels=int(self._probe_channels),
                output_format=pyaudio.paFloat32,
            )
            return True
        except Exception:
            return False

    def _set_output_device_state(self, info: dict | None):
        if not info:
            self._output_device_index = None
            self._output_device_name = ""
            self._output_device_host_api = None
            self._output_device_generation += 1
            return
        self._output_device_index = int(info["index"])
        self._output_device_name = str(info.get("name", "")).strip()
        self._output_device_host_api = info.get("hostApi")
        self._output_device_generation += 1

    def _set_initial_output_device(self):
        if not self.enabled or self._p_sfx is None:
            return

        desired_name = str(self._preferred_output_device_name or "").strip()
        desired_host_api = self._preferred_output_device_host_api
        source = "config" if self.startup_source == "config" else "persisted"

        info = None
        if desired_name:
            info = self._find_output_device_by_name(self._p_sfx, desired_name, desired_host_api)
            if not info:
                info = self._find_output_device_by_name(self._p_sfx, desired_name)
            if info and not self._supports_output_format(self._p_sfx, int(info["index"])):
                info = None
            if info:
                logger.info(f"[feedback] Startup output from {source}: {info['name']}")
            else:
                logger.warning(f"[feedback] {source} output unavailable: {desired_name}")

        if info is None:
            try:
                info = self._p_sfx.get_default_output_device_info()
            except Exception:
                info = None

        self._set_output_device_state(info)

    def get_output_device_state(self) -> dict:
        return {
            "idx": self._output_device_index,
            "name": self._output_device_name,
            "host_api": self._output_device_host_api,
        }

    def get_output_device_preference(self) -> dict:
        preferred_name = str(self._preferred_output_device_name or "").strip()
        if preferred_name:
            return {
                "name": preferred_name,
                "host_api": self._preferred_output_device_host_api,
            }
        return {
            "name": self._output_device_name,
            "host_api": self._output_device_host_api,
        }

    def get_probe_format(self) -> tuple[int, int] | None:
        if not self._probe_sr or not self._probe_channels:
            return None
        return (int(self._probe_sr), int(self._probe_channels))

    def try_set_output_device_index(
        self,
        device_index: int,
        device_name: str,
        host_api: int | None = None,
        *,
        update_preference: bool = True,
    ) -> bool:
        if not self.enabled or self._p_sfx is None:
            return False
        if not self._supports_output_format(self._p_sfx, int(device_index)):
            return False
        if self._probe_sr and self._probe_channels:
            stream = None
            try:
                open_kwargs = dict(
                    format=pyaudio.paFloat32,
                    channels=int(self._probe_channels),
                    rate=int(self._probe_sr),
                    output=True,
                    output_device_index=int(device_index),
                    frames_per_buffer=256,
                )
                stream = self._p_sfx.open(**open_kwargs)
                stream.write((b"\x00\x00\x00\x00") * int(self._probe_channels) * 256)
            except Exception:
                return False
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
        with self._lock:
            self._output_device_index = int(device_index)
            self._output_device_name = str(device_name or "").strip()
            self._output_device_host_api = host_api
            self._output_device_generation += 1
            if update_preference:
                self._preferred_output_device_name = self._output_device_name
                self._preferred_output_device_host_api = host_api
        return True

    def _load_clip(self, torchaudio, path: str, volume: float = 1.0):
        wav, sr = torchaudio.load(path)
        wav = wav.detach().cpu().numpy().T.astype(np.float32)
        vol = _clamp_volume(volume)
        if vol != 1.0:
            wav *= vol
            np.clip(wav, -1.0, 1.0, out=wav)
        if wav.ndim == 1:
            wav = wav[:, np.newaxis]
        return {"data": np.ascontiguousarray(wav), "sr": int(sr), "channels": int(wav.shape[1])}

    def _clip_open_kwargs(self, clip: dict, *, device_index: int | None):
        open_kwargs = dict(
            format=pyaudio.paFloat32,
            channels=clip["channels"],
            rate=clip["sr"],
            output=True,
        )
        if device_index is not None:
            open_kwargs["output_device_index"] = int(device_index)
        return open_kwargs

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
            with self._lock:
                out_idx = self._output_device_index
            open_kwargs = self._clip_open_kwargs(clip, device_index=out_idx)
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

    def play_final(self):
        if self.enabled and self.final_sound_enabled and "final" in self._clips:
            self._sound_queue.put("final")

    def play_startup(self):
        if self.enabled and "startup" in self._clips:
            self._sound_queue.put("startup")

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


def main(args=None):
    import signal
    def _sigterm(_s, _f):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    console_controller = WindowsConsoleController()
    tray_icon = WindowsTrayIcon(console_controller)
    background_mode = CONSOLE_STARTUP_MODE == "background"
    if background_mode:
        if console_controller.available:
            console_controller.hide(reason="startup-init")
        else:
            logger.warning("[console] Background mode requested but no Windows console is available.")
    if tray_icon.available and not tray_icon.start():
        logger.warning("[tray] Failed to start tray icon thread.")

    hotkey_targets: dict[str, list] = {}
    hotkey_not_ready_logged: set[str] = set()

    def set_hotkey_target(name: str, func):
        if name not in hotkey_targets:
            hotkey_targets[name] = [None]
        hotkey_targets[name][0] = func
        hotkey_not_ready_logged.discard(name)

    def invoke_hotkey_target(name: str):
        target_ref = hotkey_targets.get(name)
        target = target_ref[0] if target_ref else None
        if target is None:
            if name not in hotkey_not_ready_logged:
                hotkey_not_ready_logged.add(name)
                logger.info(f"[hotkey] {name} ignored during startup (not ready yet).")
            return
        target()

    try:
        import keyboard as kb
    except ImportError:
        kb = None
        logger.warning("[hotkeys] keyboard not installed, hotkeys disabled.")

    def register_hotkeys():
        if kb is None:
            return

        hk = cfg["hotkeys"]
        input_cycle_hotkey = str(hk.get("input_device_cycle", "")).strip()
        if input_cycle_hotkey:
            kb.add_hotkey(input_cycle_hotkey, lambda: invoke_hotkey_target("input_device_cycle"))
            logger.info(f"[hotkey] input_device_cycle = {input_cycle_hotkey}")

        output_cycle_hotkey = str(hk.get("output_device_cycle", "")).strip()
        if output_cycle_hotkey:
            kb.add_hotkey(output_cycle_hotkey, lambda: invoke_hotkey_target("output_device_cycle"))
            logger.info(f"[hotkey] output_device_cycle = {output_cycle_hotkey}")

        if SLEEP_HOTKEY_TOGGLE:
            kb.add_hotkey(
                SLEEP_HOTKEY_TOGGLE,
                lambda: invoke_hotkey_target("sleep_toggle"),
                suppress=SLEEP_HOTKEY_SUPPRESS,
            )
            logger.info(
                f"[hotkey] sleep_toggle = {SLEEP_HOTKEY_TOGGLE}"
                f" suppress={SLEEP_HOTKEY_SUPPRESS}"
            )

        if TYPE_TOGGLE_HOTKEY:
            kb.add_hotkey(TYPE_TOGGLE_HOTKEY, lambda: invoke_hotkey_target("typing_toggle"))
            logger.info(f"[hotkey] typing_toggle = {TYPE_TOGGLE_HOTKEY}")

        if OUTPUT_MODE_CYCLE_HOTKEY:
            kb.add_hotkey(OUTPUT_MODE_CYCLE_HOTKEY, lambda: invoke_hotkey_target("output_mode_cycle"))
            logger.info(f"[hotkey] output_mode_cycle = {OUTPUT_MODE_CYCLE_HOTKEY}")

        if CONSOLE_TOGGLE_HOTKEY:
            if console_controller.available:
                kb.add_hotkey(CONSOLE_TOGGLE_HOTKEY, lambda: invoke_hotkey_target("console_toggle"))
                logger.info(f"[hotkey] console_toggle = {CONSOLE_TOGGLE_HOTKEY}")
            else:
                logger.warning("[hotkey] console_toggle requested but no Windows console is available.")

    set_hotkey_target("console_toggle", lambda: console_controller.toggle(reason="hotkey"))
    register_hotkeys()

    print(f"Initializing STT (Realtime: {REALTIME_DEVICE}, Final: {FINAL_DEVICE})...")
    sys.stdout.flush()
    debug_log(
        "[startup] "
        f"rate={RATE} chunk={CHUNK} vad_start_threshold={VAD_THRESHOLD} "
        f"vad_end_threshold={VAD_END_THRESHOLD} min_speech_duration_ms={MIN_SPEECH_DURATION_MS} "
        f"silence_timeout={SILENCE_TIMEOUT} rt_interval={RT_CHECK_INTERVAL} "
        f"rt_queue_size={RT_QUEUE_SIZE} start_sleeping={SLEEP_START_SLEEPING} "
        f"final_backend={FINAL_BACKEND} realtime_backend={REALTIME_BACKEND} "
        f"whisper_no_speech_threshold={WHISPER_NO_SPEECH_THRESHOLD} "
        f"whisper_log_prob_threshold={WHISPER_LOG_PROB_THRESHOLD}"
    )

    resources = {}
    load_errors = []
    load_errors_lock = threading.Lock()
    warmup_audio = np.zeros(RATE, dtype=np.float32)
    shared_model_signature = (
        str(FINAL_BACKEND).strip().lower(),
        str(FINAL_MODEL).strip(),
        str(FINAL_DEVICE).strip().lower(),
        str(FINAL_COMPUTE).strip().lower(),
        float(WHISPER_NO_SPEECH_THRESHOLD),
        float(WHISPER_LOG_PROB_THRESHOLD),
    )
    realtime_model_signature = (
        str(REALTIME_BACKEND).strip().lower(),
        str(REALTIME_MODEL).strip(),
        str(REALTIME_DEVICE).strip().lower(),
        str("default" if REALTIME_DEVICE == "cpu" else FINAL_COMPUTE).strip().lower(),
        float(WHISPER_NO_SPEECH_THRESHOLD),
        float(WHISPER_LOG_PROB_THRESHOLD),
    )
    reuse_realtime_model = bool(REALTIME_MODEL) and realtime_model_signature == shared_model_signature

    def record_load_error(message: str):
        with load_errors_lock:
            load_errors.append(message)

    # ── Async model loading ───────────────────────────────────────────────────

    def load_final():
        try:
            logger.info(f"Loading final {FINAL_BACKEND} model {FINAL_MODEL} ({FINAL_DEVICE})...")
            m = load_backend(
                backend_name=FINAL_BACKEND,
                model_name=FINAL_MODEL,
                device=FINAL_DEVICE,
                compute_type=FINAL_COMPUTE,
                no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            )
            m.warmup(warmup_audio)
            resources['final_model'] = m
            logger.info("Final model ready.")
        except Exception as e:
            record_load_error(f"Final load error: {e}")

    def load_realtime():
        if not REALTIME_MODEL:
            logger.info("Realtime model disabled (no model configured). Partials will not fire.")
            resources['realtime_model'] = None
            return
        if reuse_realtime_model:
            logger.info("Realtime model reusing final model configuration; skipping duplicate load.")
            return
        try:
            logger.info(f"Loading realtime {REALTIME_BACKEND} model {REALTIME_MODEL} ({REALTIME_DEVICE})...")
            compute = "default" if REALTIME_DEVICE == "cpu" else FINAL_COMPUTE
            m = load_backend(
                backend_name=REALTIME_BACKEND,
                model_name=REALTIME_MODEL,
                device=REALTIME_DEVICE,
                compute_type=compute,
                no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            )
            m.warmup(warmup_audio)
            resources['realtime_model'] = m
            logger.info("Realtime model ready.")
        except Exception as e:
            record_load_error(f"Realtime load error: {e}")

    def load_audio_stack():
        import torch
        try:
            logger.info("Loading Silero VAD...")
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                force_reload=False, onnx=True
            )
            resources['vad'] = SileroVAD(model)

            logger.info("Initializing PyAudio...")
            p = pyaudio.PyAudio()
            resources['p_instance'] = p
            api_names = host_api_names(p)

            def _find_device_by_name(name: str, host_api: int | None = None):
                target = str(name or "").strip().lower()
                if not target:
                    return None
                for candidate_info in available_inputs:
                    if host_api is not None and candidate_info.get("hostApi") != host_api:
                        continue
                    if str(candidate_info.get("name", "")).strip().lower() == target:
                        return candidate_info
                return None

            available_inputs = list(available_input_devices(p, include_alias=True))
            logger.debug("[device] Available input devices:")
            for available_info in available_inputs:
                logger.debug(f"[device]   {describe_device(available_info, api_names)}")

            try:
                default_info = p.get_default_input_device_info()
            except Exception:
                default_info = None
            info = None
            input_session = None
            startup_candidates = []
            startup_failures = []
            seen_candidate_indices = set()

            def _push_candidate(source: str, candidate: dict | None, missing_name: str = ""):
                if candidate is None:
                    if missing_name:
                        logger.warning(f"[device] {source} startup device unavailable: {missing_name}")
                    return
                idx = int(candidate["index"])
                if idx in seen_candidate_indices:
                    return
                seen_candidate_indices.add(idx)
                startup_candidates.append((source, candidate))

            if APPROVED_INPUT_PROFILES:
                approved_infos = order_devices_by_profiles(available_inputs, APPROVED_INPUT_PROFILES)
                for approved_info in approved_infos:
                    _push_candidate("Profile", approved_info)
            if PREFERRED_INPUT_DEVICE:
                _push_candidate("Configured", _find_device_by_name(PREFERRED_INPUT_DEVICE), PREFERRED_INPUT_DEVICE)
            _push_candidate("Default", default_info)
            for extra_info in available_inputs:
                _push_candidate("Fallback", extra_info)

            for source, candidate in startup_candidates:
                try:
                    candidate_session, probe_read_ms, probe_rms = open_input_session(
                        p,
                        candidate,
                        stt_rate=RATE,
                        chunk_frames=CHUNK,
                        probe_reads=INPUT_PROBE_READS,
                        probe_limit_ms=INPUT_PROBE_READ_LIMIT_MS,
                    )
                    input_session = candidate_session
                    info = candidate
                    capture_note = ""
                    if int(candidate_session.capture_rate) != int(RATE):
                        capture_note = f" capture_rate={candidate_session.capture_rate}"
                    logger.info(
                        f"[device] Startup from {source.lower()}: {candidate['name']} "
                        f"(probe_read_ms={probe_read_ms:.0f}, rms={probe_rms}{capture_note})"
                    )
                    break
                except Exception as probe_error:
                    logger.warning(f"[device] {source} startup device rejected ({candidate['name']}): {probe_error}")
                    startup_failures.append((source, candidate, str(probe_error)))

            if info is None or input_session is None:
                summary_parts = []
                for source, candidate, reason in startup_failures[:8]:
                    summary_parts.append(
                        f"{source.lower()} {describe_device(candidate, api_names)} -> {reason}"
                    )
                if len(startup_failures) > 8:
                    summary_parts.append(f"... {len(startup_failures) - 8} more")
                if startup_failures and all(
                    ("Unanticipated host error" in reason) or ("Invalid device" in reason)
                    for _, _, reason in startup_failures
                ):
                    summary_parts.append(
                        "Windows is exposing input devices but PortAudio cannot open any of them; "
                        "check microphone privacy permissions, device busy/exclusive-mode conflicts, "
                        "Bluetooth hands-free routing, or reconnect the device."
                    )
                raise RuntimeError(
                    "No usable input device found at startup. "
                    + ("; ".join(summary_parts) if summary_parts else "No startup candidates were available.")
                )

            resources['input_session'] = input_session
            resources['stream'] = input_session.stream
            resources['dev_name'] = info['name']
            resources['dev_idx'] = int(info['index'])
            resources['host_api'] = info['hostApi']
            resources['capture_rate'] = int(input_session.capture_rate)
            logger.info(
                f"Audio: {info['name']} (capture_rate={int(input_session.capture_rate)}Hz -> stt_rate={RATE}Hz)"
            )
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

    if reuse_realtime_model:
        resources['realtime_model'] = resources['final_model']
        logger.info("Realtime model ready (shared with final model).")

    final_model = resources['final_model']
    rt_model    = resources['realtime_model']
    vad         = resources['vad']
    input_session = resources['input_session']
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
        startup_path=FEEDBACK_STARTUP_SOUND,
        final_path=FEEDBACK_FINAL_SOUND,
        final_sound_enabled=FEEDBACK_FINAL_SOUND_ENABLED,
        silence_keepalive_mode=FEEDBACK_SILENCE_KEEPALIVE_MODE,
        on_volume=FEEDBACK_ON_VOLUME,
        off_volume=FEEDBACK_OFF_VOLUME,
        final_volume=FEEDBACK_FINAL_VOLUME,
        output_device=FEEDBACK_OUTPUT_DEVICE,
        startup_source=OUTPUT_DEVICE_STARTUP_SOURCE,
        persisted_output_device_name="",
        persisted_output_device_host_api=None,
    )

    if getattr(feedback, "enabled", False) and APPROVED_OUTPUT_PROFILES:
        approved_output_infos = order_devices_by_profiles(
            available_output_devices(p_instance, include_alias=True),
            APPROVED_OUTPUT_PROFILES,
        )
        for candidate_info in approved_output_infos:
            if feedback.try_set_output_device_index(
                int(candidate_info["index"]),
                str(candidate_info.get("name", "")),
                candidate_info.get("hostApi"),
                update_preference=False,
            ):
                logger.info(f"[feedback] Startup output selected from device profiles: {candidate_info['name']}")
                break

    # ── State ─────────────────────────────────────────────────────────────────
    final_queue   = queue.Queue()
    rt_queue      = queue.Queue(maxsize=RT_QUEUE_SIZE)
    state_lock    = threading.Lock()
    state         = {'speech_epoch': 0, 'finalized_epoch': -1}
    rt_cancel     = threading.Event()
    mode_lock     = threading.Lock()
    mode_state    = {"sleeping": False}
    typing_lock   = threading.Lock()
    default_typing_enabled = bool(cfg["output"]["type_at_cursor"]["enabled"])
    typing_state = {
        "enabled": default_typing_enabled
    }
    output_mode_lock = threading.Lock()
    configured_output_mode = (
        "draft-buffer"
        if bool(cfg["output"]["file_buffer"]["enabled"]) and not bool(cfg["output"]["type_at_cursor"]["enabled"])
        else "direct-cursor"
    )
    output_mode_state = {"name": configured_output_mode}
    voice_command_lock = threading.Lock()
    voice_command_seen_by_epoch: dict[int, set[str]] = {}
    command_consumed_epochs: set[int] = set()
    last_replayable_final_event = [None]

    type_set_enabled = handler_extras.get("type_at_cursor_set_enabled")
    type_is_enabled = handler_extras.get("type_at_cursor_is_enabled")
    if type_set_enabled:
        type_set_enabled(typing_state["enabled"])
    else:
        typing_state["enabled"] = False

    file_buffer_set_enabled = handler_extras.get("file_buffer_set_enabled")
    file_buffer_is_enabled = handler_extras.get("file_buffer_is_enabled")
    output_mode_profiles = {
        "direct-cursor": {
            "type_at_cursor": True,
            "file_buffer": False,
        },
        "draft-buffer": {
            "type_at_cursor": False,
            "file_buffer": True,
        },
    }

    # Keep the user's selected device durable even if startup had to fall back.
    preferred_input_device_state = [{"name": "", "host_api": None}]
    if PREFERRED_INPUT_DEVICE:
        preferred_input_device_state[0] = {"name": PREFERRED_INPUT_DEVICE, "host_api": None}
    else:
        preferred_input_device_state[0] = {"name": dev_name, "host_api": resources['host_api']}

    # ── State persistence helper ──────────────────────────────────────────────
    current_device_state = [{"name": dev_name, "host_api": resources['host_api']}]
    current_output_device_state = [
        feedback.get_output_device_state() if getattr(feedback, "enabled", False) else {"idx": None, "name": "", "host_api": None}
    ]

    def persist():
        return

    def apply_output_mode(mode_name: str, *, reason: str = "", announce: bool = True) -> str:
        normalized = str(mode_name or "").strip().lower()
        if normalized not in output_mode_profiles:
            normalized = "direct-cursor"
        profile = output_mode_profiles[normalized]

        if file_buffer_set_enabled:
            file_buffer_set_enabled(profile["file_buffer"])
        if type_set_enabled:
            type_set_enabled(profile["type_at_cursor"])
            with typing_lock:
                typing_state["enabled"] = bool(profile["type_at_cursor"])

        with output_mode_lock:
            previous = output_mode_state["name"]
            output_mode_state["name"] = normalized

        persist()

        if announce and previous != normalized:
            logger.info(f"[output_mode] {normalized}" + (f" ({reason})" if reason else ""))
            dispatch({"type": "system", "event": "output_mode_changed", "mode": normalized})
        return normalized

    def cycle_output_mode(reason: str = "hotkey") -> str:
        with output_mode_lock:
            current = output_mode_state["name"]
        try:
            pos = OUTPUT_MODE_NAMES.index(current)
        except ValueError:
            pos = -1
        next_mode = OUTPUT_MODE_NAMES[(pos + 1) % len(OUTPUT_MODE_NAMES)]
        return apply_output_mode(next_mode, reason=reason)

    def replay_last_final_event(epoch: int, t: float, inf_ms: int, source: str) -> bool:
        event = last_replayable_final_event[0]
        if not event:
            logger.info(f"[{source.upper()} +{t:.2f}s] → CMD repeat ignored (no replayable final event)")
            return False
        replay = copy.deepcopy(event)
        replay["epoch"] = epoch
        replay["t"] = round(t, 3)
        replay["inference_ms"] = inf_ms
        replay["replayed"] = True
        logger.info(f"[{source.upper()} +{t:.2f}s] → CMD repeat ({inf_ms}ms)")
        dispatch(replay)
        return True

    def set_sleeping(sleeping: bool, reason: str = "", *, play_feedback: bool = True):
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
            if play_feedback:
                feedback.play_off()
        else:
            if play_feedback:
                feedback.play_on()
        logger.info(f"[mode] {'Sleeping' if sleeping else 'Working'}" + (f" ({reason})" if reason else ""))
        dispatch({"type": "status", "value": "sleeping" if sleeping else "working"})
        persist()

    def trigger_wake_feedback(reason: str = ""):
        feedback.play_startup()
        logger.info(f"[wake] feedback" + (f" ({reason})" if reason else ""))

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
        text = final_text.strip()
        actions = {}

        def consume_command(words: list[str]) -> tuple[str, bool]:
            if not words:
                return text, False
            ordered = sorted(words, key=len, reverse=True)
            consumed_text = text
            matched = False

            for trigger in ordered:
                pat = re.compile(
                    rf"(?:^|[\s\.,!?;:]+){re.escape(trigger)}(?:[\s\.,!?;:]+)?$",
                    re.IGNORECASE
                )
                m = pat.search(consumed_text)
                if m:
                    consumed_text = consumed_text[:m.start()].strip()
                    matched = True
                    break

            return consumed_text, matched

        file_buffer_active = bool(file_buffer_is_enabled and file_buffer_is_enabled())

        if ENTER_COMMAND_ENABLED and ENTER_COMMAND_WORDS:
            text, enter_matched = consume_command(ENTER_COMMAND_WORDS)
            if enter_matched:
                if file_buffer_active:
                    actions["release_file_buffer"] = True
                    actions["press_enter_after"] = True
                else:
                    actions["press_enter_after"] = True

        if file_buffer_active and BUFFER_RELEASE_COMMAND_ENABLED and BUFFER_RELEASE_COMMAND_WORDS:
            text, buffer_matched = consume_command(BUFFER_RELEASE_COMMAND_WORDS)
            if buffer_matched:
                actions["release_file_buffer"] = True

        return text, actions

    def _mark_voice_command_seen(epoch: int, key: str) -> bool:
        with voice_command_lock:
            seen = voice_command_seen_by_epoch.setdefault(epoch, set())
            if key in seen:
                return False
            seen.add(key)
            stale_epochs = [e for e in voice_command_seen_by_epoch if e < epoch - 8]
            for e in stale_epochs:
                del voice_command_seen_by_epoch[e]
            stale_consumed = [e for e in command_consumed_epochs if e < epoch - 8]
            for e in stale_consumed:
                command_consumed_epochs.discard(e)
            return True

    def _mark_epoch_consumed_by_partial_command(epoch: int, source: str):
        if source != "partial":
            return
        with voice_command_lock:
            command_consumed_epochs.add(epoch)

    def _is_epoch_consumed_by_partial_command(epoch: int) -> bool:
        with voice_command_lock:
            return epoch in command_consumed_epochs

    def apply_exact_voice_command(text: str, epoch: int, t: float, inf_ms: int, source: str) -> bool:
        normalized = _normalize_command_phrase(text)
        if not normalized:
            return False

        if normalized in SLEEP_STOP_WORDS:
            if _mark_voice_command_seen(epoch, f"sleep:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                set_sleeping(True, reason=f"voice:{normalized}")
            return True

        if TYPE_TOGGLE_COMMAND_ENABLED and normalized in TYPE_TOGGLE_COMMAND_WORDS:
            if _mark_voice_command_seen(epoch, f"typing_toggle:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                toggle_typing_enabled(reason=f"voice:{normalized}")
            return True

        if REVERT_COMMAND_ENABLED and normalized in REVERT_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"rewind:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                revert_actions = {}
                if file_buffer_is_enabled and file_buffer_is_enabled():
                    revert_actions["restore_last_released_buffer"] = True
                    revert_actions["undo_type_at_cursor"] = True
                else:
                    revert_actions["undo_type_at_cursor"] = True
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD rewind ({inf_ms}ms)")
                dispatch({
                    "type": "final",
                    "text": "",
                    "actions": revert_actions,
                    "epoch": epoch,
                    "t": round(t, 3),
                    "inference_ms": inf_ms,
                })
                feedback.play_on()
            return True

        if REPEAT_COMMAND_ENABLED and normalized in REPEAT_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"repeat:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                if file_buffer_is_enabled and file_buffer_is_enabled():
                    logger.info(f"[{source.upper()} +{t:.2f}s] → CMD repeat ({inf_ms}ms)")
                    dispatch({
                        "type": "final",
                        "text": "",
                        "actions": {"repeat_last_released_buffer": True},
                        "epoch": epoch,
                        "t": round(t, 3),
                        "inference_ms": inf_ms,
                    })
                    feedback.play_on()
                    return True
                if replay_last_final_event(epoch, t, inf_ms, source):
                    feedback.play_on()
            return True

        if INPUT_DEVICE_CYCLE_COMMAND_ENABLED and normalized in INPUT_DEVICE_CYCLE_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"input_device_cycle:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD input_device_cycle ({inf_ms}ms)")
                cycle_input_device()
                feedback.play_on()
            return True

        if OUTPUT_DEVICE_CYCLE_COMMAND_ENABLED and normalized in OUTPUT_DEVICE_CYCLE_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"output_device_cycle:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD output_device_cycle ({inf_ms}ms)")
                cycle_output_device()
                feedback.play_on()
            return True

        if OUTPUT_MODE_DEFAULT_COMMAND_ENABLED and normalized in OUTPUT_MODE_DEFAULT_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"output_mode_default:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD output_mode={configured_output_mode} ({inf_ms}ms)")
                apply_output_mode(configured_output_mode, reason=f"voice:{normalized}")
                feedback.play_on()
            return True

        if OUTPUT_MODE_CURSOR_COMMAND_ENABLED and normalized in OUTPUT_MODE_CURSOR_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"output_mode_cursor:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD output_mode=direct-cursor ({inf_ms}ms)")
                apply_output_mode("direct-cursor", reason=f"voice:{normalized}")
                feedback.play_on()
            return True

        if OUTPUT_MODE_DRAFT_COMMAND_ENABLED and normalized in OUTPUT_MODE_DRAFT_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"output_mode_draft:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD output_mode=draft-buffer ({inf_ms}ms)")
                apply_output_mode("draft-buffer", reason=f"voice:{normalized}")
                feedback.play_on()
            return True

        if CONSOLE_SHOW_COMMAND_ENABLED and normalized in CONSOLE_SHOW_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"console_show:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD console_show ({inf_ms}ms)")
                if console_controller.show(reason=f"voice:{normalized}"):
                    feedback.play_on()
            return True

        if CONSOLE_HIDE_COMMAND_ENABLED and normalized in CONSOLE_HIDE_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"console_hide:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD console_hide ({inf_ms}ms)")
                if console_controller.hide(reason=f"voice:{normalized}"):
                    feedback.play_on()
            return True

        if ENTER_COMMAND_ENABLED and normalized in ENTER_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"press_enter:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD enter ({inf_ms}ms)")
                actions = {"press_enter_after": True}
                if file_buffer_is_enabled and file_buffer_is_enabled():
                    actions["release_file_buffer"] = True
                dispatch({
                    "type": "final",
                    "text": "",
                    "actions": actions,
                    "epoch": epoch,
                    "t": round(t, 3),
                    "inference_ms": inf_ms,
                })
                feedback.play_on()
            return True

        if BUFFER_RELEASE_COMMAND_ENABLED and normalized in BUFFER_RELEASE_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"release_buffer:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD buffer_release ({inf_ms}ms)")
                dispatch({
                    "type": "final",
                    "text": "",
                    "actions": {
                        "release_file_buffer": True,
                    },
                    "epoch": epoch,
                    "t": round(t, 3),
                    "inference_ms": inf_ms,
                })
                feedback.play_on()
            return True

        if BUFFER_CLEAR_COMMAND_ENABLED and normalized in BUFFER_CLEAR_COMMAND_WORDS_EXACT:
            if _mark_voice_command_seen(epoch, f"clear_buffer:{normalized}"):
                _mark_epoch_consumed_by_partial_command(epoch, source)
                logger.info(f"[{source.upper()} +{t:.2f}s] → CMD buffer_clear ({inf_ms}ms)")
                dispatch({
                    "type": "final",
                    "text": "",
                    "actions": {
                        "clear_file_buffer": True,
                    },
                    "epoch": epoch,
                    "t": round(t, 3),
                    "inference_ms": inf_ms,
                })
                feedback.play_on()
            return True

        return False

    def should_ignore_transcript(text: str) -> bool:
        raw = str(text).strip()
        if not raw:
            return False
        normalized = _normalize_ignored_transcript(raw)
        if not normalized:
            return True
        if not IGNORED_TRANSCRIPTS:
            return False
        return normalized in IGNORED_TRANSCRIPTS

    def strip_disfluencies(text: str) -> str:
        cleaned = str(text).strip()
        if not cleaned or not DISFLUENCY_PATTERNS:
            return cleaned

        for pattern in DISFLUENCY_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)

        return _cleanup_transcript_text(cleaned)

    def final_worker():
        while True:
            try:
                epoch, t0, audio_data = final_queue.get()
                debug_log(
                    f"[final_worker] dequeue epoch={epoch} chunks={len(audio_data)} "
                    f"pending_final={final_queue.qsize()} pending_rt={rt_queue.qsize()}"
                )
                with mode_lock:
                    if mode_state["sleeping"]:
                        debug_log(f"[final_worker] skip epoch={epoch} reason=sleeping")
                        continue
                if _is_epoch_consumed_by_partial_command(epoch):
                    debug_log(f"[final_worker] skip epoch={epoch} reason=partial_command_consumed")
                    continue
                audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
                logger.info(f"[FINAL] Processing {len(audio)/RATE:.2f}s audio...")

                t_start = time.time()
                raw_text = final_model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=5,
                ).text
                inf_ms = int((time.time() - t_start) * 1000)
                t = time.time() - t0

                if apply_exact_voice_command(raw_text, epoch, t, inf_ms, source="final"):
                    continue
                if should_ignore_transcript(raw_text):
                    continue

                cleaned_text = strip_disfluencies(raw_text)
                if cleaned_text and apply_exact_voice_command(cleaned_text, epoch, t, inf_ms, source="final"):
                    continue

                text, actions = apply_voice_commands(cleaned_text)
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
                    if text:
                        last_replayable_final_event[0] = copy.deepcopy(ev)
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
                        if text:
                            feedback.play_final()
                        if actions.get("press_enter_after"):
                            feedback.play_on()
                else:
                    debug_log(f"[FINAL +{t:.2f}s] drop empty ({inf_ms}ms)")

                drained = 0
                with state_lock:
                    if state['finalized_epoch'] < epoch:
                        state['finalized_epoch'] = epoch
                        rt_cancel.set()
                try:
                    while True:
                        rt_queue.get_nowait(); rt_queue.task_done()
                        drained += 1
                except queue.Empty:
                    pass
                if drained:
                    debug_log(f"[final_worker] cleared_rt_queue epoch={epoch} dropped={drained}")
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
                debug_log(
                    f"[realtime_worker] dequeue epoch={epoch} chunks={len(audio_data)} "
                    f"pending_rt={rt_queue.qsize()}"
                )

                with state_lock:
                    if (epoch != state['speech_epoch']
                            or state['finalized_epoch'] >= epoch
                            or rt_cancel.is_set()):
                        debug_log(f"[realtime_worker] skip epoch={epoch} reason=stale")
                        continue
                with mode_lock:
                    if mode_state["sleeping"]:
                        debug_log(f"[realtime_worker] skip epoch={epoch} reason=sleeping")
                        continue

                if not (MIN_CHUNKS <= len(audio_data) <= MAX_CHUNKS):
                    debug_log(
                        f"[realtime_worker] skip epoch={epoch} reason=chunk_window "
                        f"chunks={len(audio_data)}"
                    )
                    continue

                audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
                t_start = time.time()
                text = rt_model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=1,
                ).text
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

                if apply_exact_voice_command(text, epoch, t, inf_ms, source="partial"):
                    continue
                if should_ignore_transcript(text):
                    continue

                cleaned_text = strip_disfluencies(text)
                if cleaned_text and apply_exact_voice_command(cleaned_text, epoch, t, inf_ms, source="partial"):
                    continue

                if cleaned_text:
                    ev = {"type": "partial", "text": text, "epoch": epoch,
                          "t": round(t, 3), "inference_ms": inf_ms}
                    ev["text"] = cleaned_text
                    logger.info(f"[PARTIAL +{t:.2f}s] → OUT  '{cleaned_text}' ({inf_ms}ms)")
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

    def _sleep_toggle():
        with mode_lock:
            sleeping = mode_state["sleeping"]
        set_sleeping(not sleeping, reason="hotkey")

    apply_output_mode(output_mode_state["name"], reason="startup", announce=False)
    set_hotkey_target("sleep_toggle", _sleep_toggle)
    set_hotkey_target("typing_toggle", lambda: toggle_typing_enabled(reason="hotkey"))
    set_hotkey_target("output_mode_cycle", lambda: cycle_output_mode(reason="hotkey"))
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
                elif cmd == "output_mode_cycle":
                    cycle_output_mode(reason="nats:output_mode_cycle")
                elif cmd == "status_query":
                    with mode_lock:
                        sleeping = mode_state["sleeping"]
                    with typing_lock:
                        typing_enabled = typing_state["enabled"]
                    with output_mode_lock:
                        output_mode_name = output_mode_state["name"]
                    reply = {
                        "sleeping": sleeping,
                        "mode": "sleeping" if sleeping else "working",
                        "typing_at_cursor_enabled": typing_enabled,
                        "output_mode": output_mode_name,
                    }
                    if msg.reply:
                        await nc.publish(msg.reply, json.dumps(reply).encode())
                elif cmd == "input_device_cycle":
                    cycle_input_device()
                elif cmd == "output_device_cycle":
                    cycle_output_device()
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
    current_capture_rate = [resources.get('capture_rate', RATE)]
    _pending_device    = [None]   # (idx, name, info) set by hotkey, consumed by main loop
    _bad_devices       = {}       # (name, host_api) -> retry_at monotonic timestamp
    _bad_output_devices = {}
    _pending_output_device = [None]  # (idx, name, info) set by hotkey, consumed by main loop

    def _device_temporarily_blocked(bad_devices: dict, key: tuple[str, int | None]) -> bool:
        retry_at = bad_devices.get(key)
        if retry_at is None:
            return False
        if time.monotonic() >= retry_at:
            bad_devices.pop(key, None)
            return False
        return True

    def _enumerate_input_devices():
        """Read-only enumeration — safe to call from any thread."""
        infos = available_input_devices(p_instance, include_alias=True)
        if APPROVED_INPUT_PROFILES:
            infos = order_devices_by_profiles(infos, APPROVED_INPUT_PROFILES)
        devices = []
        seen = set()
        for info in infos:
            key = (str(info.get('name', '')).strip(), info.get('hostApi'))
            if _device_temporarily_blocked(_bad_devices, key):
                continue
            if not APPROVED_INPUT_PROFILES and key[0] in ALIAS_DEVICE_NAMES:
                continue
            if key in seen:
                continue
            seen.add(key)
            devices.append((int(info['index']), key[0], info))
        return devices

    def _enumerate_output_devices():
        """Read-only enumeration — safe to call from any thread."""
        if not getattr(feedback, 'enabled', False):
            return []
        infos = available_output_devices(p_instance, include_alias=True)
        if APPROVED_OUTPUT_PROFILES:
            infos = order_devices_by_profiles(infos, APPROVED_OUTPUT_PROFILES)
        devices = []
        seen = set()
        probe_format = feedback.get_probe_format() if hasattr(feedback, 'get_probe_format') else None
        for info in infos:
            key = (str(info.get('name', '')).strip(), info.get('hostApi'))
            if _device_temporarily_blocked(_bad_output_devices, key):
                continue
            if not APPROVED_OUTPUT_PROFILES and key[0] in ALIAS_DEVICE_NAMES:
                continue
            if key in seen:
                continue
            usable, _ = probe_output_device(p_instance, info, probe_format)
            if not usable:
                continue
            seen.add(key)
            devices.append((int(info['index']), key[0], info))
        return devices

    def _open_and_probe_runtime_input(info: dict):
        return open_input_session(
            p_instance,
            info,
            stt_rate=RATE,
            chunk_frames=CHUNK,
            probe_reads=INPUT_PROBE_READS,
            probe_limit_ms=INPUT_PROBE_READ_LIMIT_MS,
        )

    def cycle_input_device():
        """Hotkey handler — only signals the main loop, never touches the stream."""
        if _pending_device[0] is not None:
            return
        devices = _enumerate_input_devices()
        if len(devices) <= 1:
            logger.info('[device] Only one input device available.')
            return
        idxs = [d[0] for d in devices]
        try:
            pos = idxs.index(current_device_idx[0])
        except ValueError:
            pos = -1
        _pending_device[0] = devices[(pos + 1) % len(devices)]

    def cycle_output_device():
        """Hotkey handler — only signals the main loop, never touches PyAudio."""
        if not getattr(feedback, 'enabled', False):
            logger.info('[feedback] Output device cycle ignored (feedback disabled).')
            return
        if _pending_output_device[0] is not None:
            return
        devices = _enumerate_output_devices()
        if len(devices) <= 1:
            logger.info('[feedback] Only one output device available.')
            return
        current_name = str(current_output_device_state[0].get('name', '')).strip()
        current_host_api = current_output_device_state[0].get('host_api')
        pos = -1
        if current_name:
            for i, (_, candidate_name, candidate_info) in enumerate(devices):
                if (
                    str(candidate_name).strip() == current_name
                    and candidate_info.get('hostApi') == current_host_api
                ):
                    pos = i
                    break
        if pos < 0:
            preferred = feedback.get_output_device_preference() if hasattr(feedback, 'get_output_device_preference') else {}
            preferred_name = str(preferred.get('name', '')).strip()
            preferred_host_api = preferred.get('host_api')
            if preferred_name:
                for i, (_, candidate_name, candidate_info) in enumerate(devices):
                    if (
                        str(candidate_name).strip() == preferred_name
                        and candidate_info.get('hostApi') == preferred_host_api
                    ):
                        pos = i
                        break
        if pos < 0:
            try:
                default_info = p_instance.get_default_output_device_info()
                default_name = str(default_info.get('name', '')).strip()
                default_host_api = default_info.get('hostApi')
                for i, (_, candidate_name, candidate_info) in enumerate(devices):
                    if (
                        str(candidate_name).strip() == default_name
                        and candidate_info.get('hostApi') == default_host_api
                    ):
                        pos = i
                        break
            except Exception:
                pass
        _pending_output_device[0] = devices[(pos + 1) % len(devices)]

    set_hotkey_target("input_device_cycle", cycle_input_device)
    set_hotkey_target("output_device_cycle", cycle_output_device)

    # ── Main loop ─────────────────────────────────────────────────────────────
    ring_buffer      = deque(maxlen=int(BUFFER_SECONDS * RATE / CHUNK))
    recording_buffer = []
    is_recording     = False
    speech_counter   = 0
    min_speech_chunks = max(1, int(math.ceil((MIN_SPEECH_DURATION_MS / 1000.0) * RATE / CHUNK)))
    silence_counter  = 0
    silence_limit    = int(SILENCE_TIMEOUT * (RATE / CHUNK))
    last_rt_update   = 0.0
    current_t0       = None
    last_error       = ""
    drop_chunks_after_wake = 0
    vad_prev_active = False
    wakeword_prev_detected = False

    dispatch({"type": "system", "event": "startup",
              "device": dev_name,
              "models": {
                  "final": FINAL_MODEL,
                  "final_backend": FINAL_BACKEND,
                  "realtime": REALTIME_MODEL,
                  "realtime_backend": REALTIME_BACKEND,
              },
              "output_mode": output_mode_state["name"]})
    dispatch({"type": "status", "value": "sleeping" if mode_state["sleeping"] else "working"})
    print("STT Ready!")
    sys.stdout.flush()
    feedback.play_startup()

    def reset_active_utterance(reason: str):
        nonlocal recording_buffer, is_recording, speech_counter, silence_counter, current_t0, last_rt_update
        had_audio = bool(is_recording or recording_buffer)
        ring_buffer.clear()
        recording_buffer = []
        is_recording = False
        speech_counter = 0
        silence_counter = 0
        current_t0 = None
        last_rt_update = 0.0
        rt_cancel.set()
        with state_lock:
            state["speech_epoch"] += 1
            state["finalized_epoch"] = state["speech_epoch"]
            epoch = state["speech_epoch"]
        if had_audio:
            logger.info(f"[speech] reset active utterance epoch={epoch} reason={reason}")
            dispatch({"type": "status", "value": "idle"})
        try:
            while True:
                rt_queue.get_nowait()
                rt_queue.task_done()
        except queue.Empty:
            pass
        try:
            while True:
                final_queue.get_nowait()
                final_queue.task_done()
        except queue.Empty:
            pass

    while True:
        try:
            data, read_ms = input_session.read_chunk()
            if read_ms > max(250.0, EXPECTED_CHUNK_MS * 4.0):
                logger.warning(f"[audio] stream.read blocked for {read_ms:.0f}ms")
            if is_recording and read_ms >= INPUT_STALL_RESET_SECONDS * 1000.0:
                reset_active_utterance(reason=f"input_stall:{read_ms/1000.0:.2f}s")
            if last_error:
                logger.info("Recovered.")
                last_error = ""

            # ── Device switch (signalled by hotkey thread) ────────────────────
            pending = _pending_device[0]
            if pending is not None:
                _pending_device[0] = None
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
                    next_idx, next_name, next_info = _all_devices[target_pos]
                    if next_idx in _tried:
                        continue
                    _tried.add(next_idx)
                    prev_idx = current_device_idx[0]
                    prev_info = None
                    try:
                        prev_info = p_instance.get_device_info_by_index(int(prev_idx))
                    except Exception:
                        prev_info = None
                    try:
                        input_session.close()
                        input_session, probe_read_ms, rms = _open_and_probe_runtime_input(next_info)
                        stream = input_session.stream
                        current_device_idx[0] = next_idx
                        current_capture_rate[0] = int(input_session.capture_rate)
                        current_device_state[0] = {"name": next_name, "host_api": next_info.get("hostApi")}
                        preferred_input_device_state[0] = dict(current_device_state[0])
                        wakeword_buffer = np.array([], dtype=np.int16)
                        ring_buffer.clear(); recording_buffer.clear(); is_recording = False
                        logger.info(
                            f"[device] Cycled to: {next_name} "
                            f"(probe_read_ms={probe_read_ms:.0f}, rms={rms}, capture_rate={input_session.capture_rate})"
                        )
                        dispatch({"type": "system", "event": "device_changed", "device": next_name})
                        if target_pos == 0:
                            feedback.play_off()
                        else:
                            feedback.play_on()
                        persist()
                        break
                    except Exception as e:
                        _bad_devices[(str(next_name), next_info.get("hostApi"))] = (
                            time.monotonic() + DEVICE_RETRY_COOLDOWN_SECONDS
                        )
                        logger.warning(f"[device] Skipping {next_name}: {e}")
                        try:
                            if prev_info is not None:
                                input_session, _, _ = _open_and_probe_runtime_input(prev_info)
                                stream = input_session.stream
                                current_device_idx[0] = prev_idx
                                current_capture_rate[0] = int(input_session.capture_rate)
                        except Exception:
                            pass
                continue

            # ── Feedback output device switch (signalled by hotkey thread) ────
            pending_out = _pending_output_device[0]
            if pending_out is not None:
                _pending_output_device[0] = None
                if not getattr(feedback, "enabled", False):
                    continue
                _all_devices = _enumerate_output_devices()
                if not _all_devices:
                    logger.warning("[feedback] No output devices available to cycle.")
                    continue
                try:
                    _start_pos = next(
                        (i for i, d in enumerate(_all_devices) if d[0] == pending_out[0]), 0
                    )
                except Exception:
                    _start_pos = 0

                switched = False
                for _attempt in range(len(_all_devices)):
                    target_pos = (_start_pos + _attempt) % len(_all_devices)
                    next_idx, next_name, next_info = _all_devices[target_pos]
                    next_host_api = next_info.get("hostApi") if next_info else None
                    ok = False
                    try:
                        ok = feedback.try_set_output_device_index(int(next_idx), str(next_name), next_host_api)
                    except Exception:
                        ok = False
                    if ok:
                        current_output_device_state[0] = {"idx": int(next_idx), "name": str(next_name), "host_api": next_host_api}
                        logger.info(f"[feedback] Cycled output to: {next_name}")
                        dispatch({"type": "system", "event": "output_device_changed", "device": str(next_name)})
                        if target_pos == 0:
                            feedback.play_off()
                        else:
                            feedback.play_on()
                        persist()
                        switched = True
                        break
                    _bad_output_devices[(str(next_name), next_host_api)] = (
                        time.monotonic() + DEVICE_RETRY_COOLDOWN_SECONDS
                    )
                    logger.warning(f"[feedback] Skipping output {next_name}")
                if not switched:
                    logger.warning("[feedback] Output device cycle failed (all candidates rejected).")
                continue

            chunk = np.frombuffer(data, dtype=np.int16)
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) if len(chunk) else 0.0

            wake_detected = detect_wakeword(chunk)
            if wake_detected and not wakeword_prev_detected:
                debug_log(f"[wake] detected word={WAKEWORD} rms={chunk_rms:.1f}")
                trigger_wake_feedback(reason=f"wake:{WAKEWORD}")
            wakeword_prev_detected = wake_detected

            with mode_lock:
                sleeping = mode_state["sleeping"]
            if sleeping:
                if is_recording or recording_buffer:
                    debug_log("[audio] clearing active buffers while sleeping")
                    reset_active_utterance(reason="sleeping")
                if wake_detected:
                    set_sleeping(False, reason=f"wake:{WAKEWORD}", play_feedback=False)
                    drop_chunks_after_wake = int(max(0.0, WAKEWORD_DROP_SECONDS) * RATE / CHUNK)
                    reset_active_utterance(reason="wake_transition")
                    debug_log(f"[wake] dropping_chunks_after_wake={drop_chunks_after_wake}")
                else:
                    debug_log_every(
                        "sleeping_heartbeat",
                        DEBUG_HEARTBEAT_SECONDS,
                        f"[heartbeat] mode=sleeping rms={chunk_rms:.1f} read_ms={read_ms:.1f}"
                    )
                continue

            if drop_chunks_after_wake > 0:
                drop_chunks_after_wake -= 1
                debug_log_every(
                    "wake_drop_heartbeat",
                    1.0,
                    f"[wake] dropping post-wake audio remaining={drop_chunks_after_wake}"
                )
                continue

            prob  = vad.is_speech(chunk.astype(np.float32) / 32768.0)
            vad_start_active = prob >= VAD_THRESHOLD
            vad_end_active = prob >= VAD_END_THRESHOLD
            vad_active = vad_start_active if not is_recording else vad_end_active
            if vad_active != vad_prev_active:
                debug_log(
                    f"[vad] transition active={vad_active} prob={prob:.3f} "
                    f"start_threshold={VAD_THRESHOLD:.3f} end_threshold={VAD_END_THRESHOLD:.3f} "
                    f"rms={chunk_rms:.1f}"
                )
                vad_prev_active = vad_active

            if vad_active:
                silence_counter = 0
                if not is_recording:
                    ring_buffer.append(chunk)
                    speech_counter += 1
                    if speech_counter >= min_speech_chunks:
                        with state_lock:
                            state['speech_epoch'] += 1
                            state['finalized_epoch'] = -1
                            rt_cancel.clear()
                            epoch = state['speech_epoch']
                        current_t0 = time.time()
                        recording_buffer = list(ring_buffer)
                        ring_buffer.clear()
                        is_recording = True
                        speech_counter = 0
                        debug_log(
                            f"[speech] start epoch={epoch} preroll_chunks={len(recording_buffer)} "
                            f"min_speech_chunks={min_speech_chunks} rt_queue={rt_queue.qsize()} "
                            f"final_queue={final_queue.qsize()}"
                        )
                        dispatch({"type": "status", "value": "recording"})
                else:
                    recording_buffer.append(chunk)
            else:
                speech_counter = 0
                if is_recording:
                    recording_buffer.append(chunk)
                    silence_counter += 1
                    if silence_counter > silence_limit:
                        with state_lock: epoch = state['speech_epoch']
                        is_recording = False
                        final_queue.put((epoch, current_t0, list(recording_buffer)))
                        debug_log(
                            f"[speech] finalize epoch={epoch} chunks={len(recording_buffer)} "
                            f"silence_counter={silence_counter} final_queue={final_queue.qsize()}"
                        )
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
                        debug_log(
                            f"[realtime_queue] enqueue epoch={ep} chunks={n} pending_rt={rt_queue.qsize()}"
                        )
                    except queue.Full:
                        try:
                            rt_queue.get_nowait(); rt_queue.task_done()
                            rt_queue.put_nowait((ep, current_t0, list(recording_buffer)))
                            debug_log(
                                f"[realtime_queue] drop_oldest_and_enqueue epoch={ep} "
                                f"chunks={n} pending_rt={rt_queue.qsize()}"
                            )
                        except Exception as queue_error:
                            debug_log(f"[realtime_queue] enqueue_failed epoch={ep} error={queue_error}")
                else:
                    debug_log_every(
                        "realtime_window_skip",
                        1.0,
                        f"[realtime_queue] skip epoch={ep} fin={fin} chunks={n} "
                        f"window={MIN_CHUNKS}-{MAX_CHUNKS}"
                    )
                last_rt_update = time.time()

            debug_log_every(
                "working_heartbeat",
                DEBUG_HEARTBEAT_SECONDS,
                f"[heartbeat] mode=working read_ms={read_ms:.1f} rms={chunk_rms:.1f} "
                f"vad={prob:.3f} recording={is_recording} ring={len(ring_buffer)} "
                f"recording_chunks={len(recording_buffer)} silence_counter={silence_counter} "
                f"rt_queue={rt_queue.qsize()} final_queue={final_queue.qsize()}"
            )

        except (OSError, IOError) as e:
            err = str(e)
            if err != last_error:
                logger.warning(f"Stream error: {err}. Retrying...")
            last_error = err
            if is_recording or recording_buffer:
                reset_active_utterance(reason=f"stream_error:{err}")
            time.sleep(0.5)
            try:
                input_session.close()
            except Exception:
                pass
            try:
                recovery_info = p_instance.get_device_info_by_index(int(current_device_idx[0]))
                input_session, _, _ = _open_and_probe_runtime_input(recovery_info)
                stream = input_session.stream
                wakeword_buffer = np.array([], dtype=np.int16)
                ring_buffer.clear()
                recording_buffer.clear()
                is_recording = False
                silence_counter = 0
                current_t0 = None
                last_error = ""
            except Exception as e2:
                logger.warning(f"[device] Recovery failed: {e2}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected: {e}")
            break

    logger.info("Shutting down...")
    try:
        input_session.close()
    except Exception:
        pass
    try:
        if wakeword_engine is not None:
            wakeword_engine.delete()
    except: pass
    try: tray_icon.stop()
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
