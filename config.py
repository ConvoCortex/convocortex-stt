import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.toml"

_DEFAULTS = {
    "models": {
        "final": "large-v3-turbo",
        "final_device": "cuda",
        "final_compute": "float16",
        "realtime": "tiny.en",
        "realtime_device": "cpu",
        "language": "en",
    },
    "audio": {
        "rate": 16000,
        "chunk": 512,
        "vad_threshold": 0.4,
        "buffer_seconds": 1.0,
        "silence_timeout": 0.8,
    },
    "realtime": {
        "check_interval": 0.1,
        "min_chunks": 40,
        "max_chunks": 100,
        "queue_size": 10,
    },
    "output": {
        "file_append":         {"enabled": True,  "path": "output.txt"},
        "file_overwrite":      {"enabled": False, "path": "watched.txt"},
        "clipboard_replace":   {"enabled": False},
        "clipboard_accumulate":{"enabled": False, "separator": " "},
        "type_at_cursor":      {"enabled": False},
        "trailing_char":       {"enabled": False, "char": " "},
    },
    "emission_gate": {
        "enabled": True,
        "default_open": True,
        "start_words": ["listen", "start"],
        "stop_words": ["stop", "pause"],
    },
    "hotkeys": {
        "mute_toggle": "`",
        "emission_gate_open": "shift+f5",
        "emission_gate_close": "shift+f6",
        "push_to_talk": "shift+f7",
        "device_cycle": "shift+f8",
    },
    "nats": {
        "enabled": False,
        "url": "nats://localhost:4222",
        "subject_emit": "stt",
        "subject_control": "stt.control",
    },
    "state": {
        "file": "state.json",
    },
}


def _deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load() -> dict:
    cfg = _DEFAULTS
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "rb") as f:
                user = tomllib.load(f)
            cfg = _deep_merge(_DEFAULTS, user)
        except Exception as e:
            print(f"[config] Failed to parse config.toml: {e}", file=sys.stderr)
            print("[config] Using defaults.", file=sys.stderr)
    else:
        print(f"[config] config.toml not found, using defaults.", file=sys.stderr)
    return cfg
