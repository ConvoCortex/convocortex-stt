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
        text = _final_text(event, cfg)
        if text is None:
            return
        with lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(text + "\n")
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

    def clipboard_accumulate_reset():
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
        finally:
            win32clipboard.CloseClipboard()
        logger.info("[clipboard_accumulate] reset")

    clipboard_accumulate.__name__ = "clipboard_accumulate"
    return clipboard_accumulate, clipboard_accumulate_reset


# ── Type at cursor ────────────────────────────────────────────────────────────

def make_type_at_cursor(cfg: dict):
    try:
        import keyboard
    except ImportError:
        logger.warning("[type_at_cursor] keyboard not installed, handler disabled.")
        return None

    def type_at_cursor(event: dict):
        text = _final_text(event, cfg)
        actions = event.get("actions", {}) or {}
        if text is not None:
            keyboard.write(text, delay=0)
        if actions.get("press_enter_after"):
            keyboard.press_and_release("enter")

    type_at_cursor.__name__ = "type_at_cursor"
    return type_at_cursor


# ── NATS emit ─────────────────────────────────────────────────────────────────

def make_nats_publisher(cfg: dict):
    try:
        import nats
    except ImportError:
        logger.warning("[nats] nats-py not installed, NATS disabled.")
        return None

    import asyncio
    import json

    ncfg    = cfg["nats"]
    url     = ncfg["url"]
    subject = ncfg["subject_emit"]

    loop   = asyncio.new_event_loop()
    nc_ref = [None]

    async def _connect():
        try:
            nc_ref[0] = await nats.connect(url)
            logger.info(f"[nats] Connected to {url}")
        except Exception as e:
            logger.warning(f"[nats] Could not connect to {url}: {e}")

    async def _publish(subj: str, data: bytes):
        if nc_ref[0] and nc_ref[0].is_connected:
            await nc_ref[0].publish(subj, data)

    def _run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_connect())
        loop.run_forever()

    t = threading.Thread(target=_run_loop, daemon=True, name="nats-loop")
    t.start()

    def nats_publisher(event: dict):
        etype = event.get("type", "system")
        subj  = f"{subject}.{etype}"
        data  = json.dumps(event).encode()
        asyncio.run_coroutine_threadsafe(_publish(subj, data), loop)

    nats_publisher.__name__ = "nats_publisher"
    return nats_publisher


# ── Registration ──────────────────────────────────────────────────────────────

def register_all(cfg: dict, register) -> dict:
    """Register handlers and return extras dict with optional callables (e.g. resets)."""
    out = cfg["output"]
    extras = {}

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
        result = make_clipboard_accumulate(cfg)
        if result:
            fn, reset = result
            register(fn)
            extras["clipboard_accumulate_reset"] = reset
            logger.info("[handler] clipboard_accumulate")

    if out["type_at_cursor"]["enabled"]:
        fn = make_type_at_cursor(cfg)
        if fn:
            register(fn)
            logger.info("[handler] type_at_cursor")

    if cfg["nats"]["enabled"]:
        fn = make_nats_publisher(cfg)
        if fn:
            register(fn)
            logger.info(f"[handler] nats_publisher -> {cfg['nats']['url']}")

    return extras
