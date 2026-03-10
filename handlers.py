"""
Local output handlers for convocortex-stt.

Each handler is a callable(event: dict). Only "final" events produce output —
partials and status events are ignored by all local handlers.
Handlers are registered at startup based on config.
"""

import logging
import threading

logger = logging.getLogger("STT")


def _final_text(event: dict, cfg: dict) -> str | None:
    """Return text to output, with optional trailing char, or None to skip."""
    if event.get("type") != "final":
        return None
    text = event.get("text", "").strip()
    if not text:
        return None
    if cfg["output"]["trailing_char"]["enabled"]:
        text = text + cfg["output"]["trailing_char"]["char"]
    return text


# ── File append ───────────────────────────────────────────────────────────────

def make_file_append(cfg: dict):
    path = cfg["output"]["file_append"]["path"]
    lock = threading.Lock()

    def file_append(event: dict):
        label = "second" if event["type"] == "final" else "first"
        if event["type"] == "status":
            line = f"[second] [STATUS:{event['value'].upper()}]\n"
        elif event["type"] in ("final", "partial"):
            text = event.get("text", "").strip()
            if not text:
                return
            line = f"[{label}] {text}\n"
        else:
            return
        with lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
                f.flush()

    file_append.__name__ = "file_append"
    return file_append


# ── File overwrite ────────────────────────────────────────────────────────────

def make_file_overwrite(cfg: dict):
    path = cfg["output"]["file_overwrite"]["path"]
    lock = threading.Lock()

    def file_overwrite(event: dict):
        text = _final_text(event, cfg)
        if text is None:
            return
        with lock:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
                f.flush()

    file_overwrite.__name__ = "file_overwrite"
    return file_overwrite


# ── Clipboard replace ─────────────────────────────────────────────────────────

def make_clipboard_replace(cfg: dict):
    try:
        import win32clipboard
    except ImportError:
        logger.warning("[clipboard_replace] pywin32 not installed, handler disabled.")
        return None

    def clipboard_replace(event: dict):
        text = _final_text(event, cfg)
        if text is None:
            return
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()

    clipboard_replace.__name__ = "clipboard_replace"
    return clipboard_replace


# ── Clipboard accumulate ──────────────────────────────────────────────────────

def make_clipboard_accumulate(cfg: dict):
    try:
        import win32clipboard
    except ImportError:
        logger.warning("[clipboard_accumulate] pywin32 not installed, handler disabled.")
        return None

    sep = cfg["output"]["clipboard_accumulate"]["separator"]

    def clipboard_accumulate(event: dict):
        text = _final_text(event, cfg)
        if text is None:
            return
        try:
            win32clipboard.OpenClipboard()
            try:
                existing = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            except Exception:
                existing = ""
            combined = (existing + sep + text).lstrip(sep) if existing else text
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(combined, win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()

    clipboard_accumulate.__name__ = "clipboard_accumulate"
    return clipboard_accumulate


# ── Type at cursor ────────────────────────────────────────────────────────────

def make_type_at_cursor(cfg: dict):
    try:
        import keyboard
    except ImportError:
        logger.warning("[type_at_cursor] keyboard not installed, handler disabled.")
        return None

    def type_at_cursor(event: dict):
        text = _final_text(event, cfg)
        if text is None:
            return
        keyboard.write(text, delay=0)

    type_at_cursor.__name__ = "type_at_cursor"
    return type_at_cursor


# ── Registration ──────────────────────────────────────────────────────────────

def register_all(cfg: dict, register):
    out = cfg["output"]

    if out["file_append"]["enabled"]:
        register(make_file_append(cfg))
        logger.info(f"[handler] file_append -> {out['file_append']['path']}")

    if out["file_overwrite"]["enabled"]:
        register(make_file_overwrite(cfg))
        logger.info(f"[handler] file_overwrite -> {out['file_overwrite']['path']}")

    if out["clipboard_replace"]["enabled"]:
        fn = make_clipboard_replace(cfg)
        if fn:
            register(fn)
            logger.info("[handler] clipboard_replace")

    if out["clipboard_accumulate"]["enabled"]:
        fn = make_clipboard_accumulate(cfg)
        if fn:
            register(fn)
            logger.info("[handler] clipboard_accumulate")

    if out["type_at_cursor"]["enabled"]:
        fn = make_type_at_cursor(cfg)
        if fn:
            register(fn)
            logger.info("[handler] type_at_cursor")
