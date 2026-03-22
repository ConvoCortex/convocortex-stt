"""
State persistence for convocortex-stt.

Read once at startup. Write atomically on every state change.
The file is a crash recovery artifact — not a control mechanism.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("STT")

_path: Path | None = None

_DEFAULTS = {
    "sleeping": True,
    "type_at_cursor_enabled": True,
    "output_mode": "config-default",
    "last_input_device_name": "",
    "last_input_device_host_api": None,
    "last_output_device_name": "",
    "last_output_device_host_api": None,
    "approved_input_devices": [],
    "approved_output_devices": [],
    "device_setup_completed": False,
}


def init(cfg: dict):
    global _path
    _path = Path(cfg["state"]["file"])


def load() -> dict:
    if _path and _path.exists():
        try:
            with open(_path, encoding="utf-8") as f:
                data = json.load(f)
            merged = {**_DEFAULTS, **data}
            logger.info(f"[state] Restored from {_path}: {merged}")
            return merged
        except Exception as e:
            logger.warning(f"[state] Could not read {_path}: {e}. Using defaults.")
    return dict(_DEFAULTS)


def save(state: dict):
    if not _path:
        return
    tmp = _path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, _path)
    except Exception as e:
        logger.warning(f"[state] Could not save {_path}: {e}")
