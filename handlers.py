"""
Local output handlers for convocortex-stt.

Each handler is a callable(event: dict). Only "final" events produce output —
partials and status events are ignored by all local handlers.
Handlers are registered at startup based on config.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path

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


# ── File buffer ───────────────────────────────────────────────────────────────

def make_file_buffer(cfg: dict):
    bcfg = cfg["output"]["file_buffer"]
    path = Path(bcfg["path"])
    sep = bcfg["separator"]
    clear_after_release = bool(bcfg.get("clear_after_release", True))
    reset_after_each_message = bool(bcfg.get("reset_after_each_message", False))
    undo_history_limit = max(0, int(bcfg.get("undo_history_limit", 10)))
    release_method = str(bcfg.get("release_method", "type_keys")).strip().lower()
    clipboard_restore_delay_ms = max(0, int(bcfg.get("clipboard_restore_delay_ms", 250)))
    clipboard_open_retry_count = max(1, int(bcfg.get("clipboard_open_retry_count", 8)))
    clipboard_open_retry_delay_ms = max(1, int(bcfg.get("clipboard_open_retry_delay_ms", 25)))
    post_paste_enter_delay_ms = max(0, int(bcfg.get("post_paste_enter_delay_ms", 80)))
    lock = threading.Lock()
    enabled_lock = threading.Lock()
    enabled_state = [bool(bcfg.get("enabled", False))]
    undo_history = deque(maxlen=undo_history_limit) if undo_history_limit > 0 else None
    release_lock = threading.Lock()
    last_released_content = [""]

    try:
        import keyboard
    except ImportError:
        keyboard = None
        logger.warning("[file_buffer] keyboard not installed, voice-triggered release disabled.")

    win32clipboard = None
    win32con = None
    if release_method == "paste_preserve_clipboard":
        try:
            import win32clipboard as _win32clipboard
            import win32con as _win32con
            win32clipboard = _win32clipboard
            win32con = _win32con
        except ImportError:
            logger.warning(
                "[file_buffer] pywin32 not installed, falling back to type_keys release method."
            )
            release_method = "type_keys"

    def _read_buffer() -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_buffer(content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _remember_undo(previous: str, next_content: str):
        if undo_history is None or previous == next_content:
            return
        undo_history.append(previous)

    def _set_buffer(content: str):
        with lock:
            previous = _read_buffer()
            _remember_undo(previous, content)
            _write_buffer(content)

    def _append_buffer(text: str):
        with lock:
            existing = "" if reset_after_each_message else _read_buffer()
            combined = (existing + sep + text) if existing else text
            _remember_undo(existing, combined)
            _write_buffer(combined)

    def set_enabled(enabled: bool):
        with enabled_lock:
            enabled_state[0] = bool(enabled)

    def is_enabled() -> bool:
        with enabled_lock:
            return enabled_state[0]

    def _clear_buffer():
        _set_buffer("")
        logger.info("[file_buffer] cleared")

    def _undo_buffer():
        if undo_history is None:
            logger.info("[file_buffer] Undo ignored (undo_history_limit=0).")
            return
        with lock:
            if not undo_history:
                logger.info("[file_buffer] Undo ignored (history empty).")
                return
            restored = undo_history.pop()
            _write_buffer(restored)
        logger.info(f"[file_buffer] undo restored {len(restored)} chars")

    def _open_clipboard_with_retry():
        assert win32clipboard is not None
        last_error = None
        for _ in range(clipboard_open_retry_count):
            try:
                win32clipboard.OpenClipboard()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(clipboard_open_retry_delay_ms / 1000.0)
        raise RuntimeError(f"failed to open clipboard after retries: {last_error}")

    def _capture_clipboard_state() -> tuple[str | None, list[tuple[int, object]]]:
        assert win32clipboard is not None
        text_value = None
        try:
            text_value = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        except Exception:
            text_value = None
        formats: list[tuple[int, object]] = []
        fmt = 0
        while True:
            fmt = int(win32clipboard.EnumClipboardFormats(fmt))
            if not fmt:
                break
            try:
                data = win32clipboard.GetClipboardData(fmt)
            except Exception:
                continue
            formats.append((fmt, data))
        return text_value, formats

    def _restore_clipboard_state(inserted_text: str, saved_text: str | None, saved_formats: list[tuple[int, object]]):
        if win32clipboard is None:
            return
        time.sleep(clipboard_restore_delay_ms / 1000.0)
        try:
            _open_clipboard_with_retry()
            try:
                current_text = None
                try:
                    current_text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                except Exception:
                    current_text = None
                if current_text != inserted_text:
                    logger.info("[file_buffer] Clipboard changed after paste; skipping restore.")
                    return
                win32clipboard.EmptyClipboard()
                restored = False
                if saved_text is not None:
                    try:
                        win32clipboard.SetClipboardText(saved_text, win32con.CF_UNICODETEXT)
                        restored = True
                    except Exception:
                        restored = False
                if not restored:
                    for fmt, data in saved_formats:
                        try:
                            win32clipboard.SetClipboardData(fmt, data)
                            restored = True
                        except Exception:
                            continue
                if not restored:
                    logger.warning("[file_buffer] Clipboard restore failed: no clipboard formats could be restored.")
            finally:
                win32clipboard.CloseClipboard()
        except Exception as exc:
            logger.warning(f"[file_buffer] Clipboard restore failed: {exc}")

    def _release_via_typing(content: str, press_enter_after: bool):
        assert keyboard is not None
        keyboard.write(content, delay=0)
        if press_enter_after:
            keyboard.press_and_release("enter")

    def _release_via_clipboard(content: str, press_enter_after: bool):
        assert keyboard is not None
        assert win32clipboard is not None and win32con is not None

        with release_lock:
            _open_clipboard_with_retry()
            try:
                saved_text, saved_formats = _capture_clipboard_state()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(content, win32con.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()

            keyboard.press_and_release("ctrl+v")
            if press_enter_after:
                if post_paste_enter_delay_ms > 0:
                    time.sleep(post_paste_enter_delay_ms / 1000.0)
                keyboard.press_and_release("enter")

            restore_thread = threading.Thread(
                target=_restore_clipboard_state,
                args=(content, saved_text, saved_formats),
                daemon=True,
            )
            restore_thread.start()

    def _release_buffer(press_enter_after: bool = False):
        if keyboard is None:
            logger.warning("[file_buffer] Release ignored: keyboard not installed.")
            return
        with lock:
            content = _read_buffer()
        if not content:
            logger.info("[file_buffer] Release ignored: buffer is empty.")
            return
        if release_method == "paste_preserve_clipboard":
            _release_via_clipboard(content, press_enter_after)
        else:
            _release_via_typing(content, press_enter_after)
        last_released_content[0] = content
        if clear_after_release:
            with lock:
                current = _read_buffer()
                _remember_undo(current, "")
                _write_buffer("")
        logger.info(
            f"[file_buffer] released {len(content)} chars via {release_method}"
            + (" + enter" if press_enter_after else "")
        )

    def _repeat_last_released_buffer(press_enter_after: bool = False):
        content = last_released_content[0]
        if not content:
            logger.info("[file_buffer] Repeat ignored: no released buffer in memory.")
            return
        if release_method == "paste_preserve_clipboard":
            _release_via_clipboard(content, press_enter_after)
        else:
            _release_via_typing(content, press_enter_after)
        logger.info(
            f"[file_buffer] repeated {len(content)} chars via {release_method}"
            + (" + enter" if press_enter_after else "")
        )

    def _restore_last_released_buffer():
        content = last_released_content[0]
        if not content:
            logger.info("[file_buffer] Revert ignored: no released buffer in memory.")
            return
        _set_buffer(content)
        logger.info(f"[file_buffer] restored last released buffer ({len(content)} chars)")

    def file_buffer(event: dict):
        actions = event.get("actions", {}) or {}
        if actions.get("undo_file_buffer"):
            if not is_enabled():
                logger.info("[file_buffer] Undo ignored (disabled).")
                return
            _undo_buffer()
            return
        if actions.get("restore_last_released_buffer"):
            if not is_enabled():
                logger.info("[file_buffer] Revert ignored (disabled).")
                return
            _restore_last_released_buffer()
            return
        if actions.get("repeat_last_released_buffer"):
            if not is_enabled():
                logger.info("[file_buffer] Repeat ignored (disabled).")
                return
            _repeat_last_released_buffer(bool(actions.get("press_enter_after")))
            return
        if actions.get("clear_file_buffer"):
            if not is_enabled():
                logger.info("[file_buffer] Clear ignored (disabled).")
                return
            _clear_buffer()
            return
        if actions.get("release_file_buffer"):
            if not is_enabled():
                logger.info("[file_buffer] Release ignored (disabled).")
                return
            text = _final_text(event, cfg)
            if text:
                _append_buffer(text)
            _release_buffer(bool(actions.get("press_enter_after")))
            return
        if not is_enabled():
            return
        text = _final_text(event, cfg)
        if text is None:
            return
        _append_buffer(text)

    file_buffer.__name__ = "file_buffer"
    return file_buffer, _clear_buffer, set_enabled, is_enabled


# ── Clipboard replace ─────────────────────────────────────────────────────────

def make_clipboard_replace(cfg: dict):
    try:
        import win32clipboard
    except ImportError:
        logger.warning("[clipboard_replace] pywin32 not installed, handler disabled.")
        return None

    enabled_lock = threading.Lock()
    enabled_state = [bool(cfg["output"]["clipboard_replace"].get("enabled", False))]

    def set_enabled(enabled: bool):
        with enabled_lock:
            enabled_state[0] = bool(enabled)

    def is_enabled() -> bool:
        with enabled_lock:
            return enabled_state[0]

    def clipboard_replace(event: dict):
        if not is_enabled():
            return
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
    return clipboard_replace, set_enabled, is_enabled


# ── Clipboard accumulate ──────────────────────────────────────────────────────

def make_clipboard_accumulate(cfg: dict):
    try:
        import win32clipboard
    except ImportError:
        logger.warning("[clipboard_accumulate] pywin32 not installed, handler disabled.")
        return None

    sep = cfg["output"]["clipboard_accumulate"]["separator"]
    enabled_lock = threading.Lock()
    enabled_state = [bool(cfg["output"]["clipboard_accumulate"].get("enabled", False))]

    def set_enabled(enabled: bool):
        with enabled_lock:
            enabled_state[0] = bool(enabled)

    def is_enabled() -> bool:
        with enabled_lock:
            return enabled_state[0]

    def clipboard_accumulate(event: dict):
        if not is_enabled():
            return
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
        if not is_enabled():
            logger.info("[clipboard_accumulate] reset ignored (disabled)")
            return
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
        finally:
            win32clipboard.CloseClipboard()
        logger.info("[clipboard_accumulate] reset")

    clipboard_accumulate.__name__ = "clipboard_accumulate"
    return clipboard_accumulate, clipboard_accumulate_reset, set_enabled, is_enabled


# ── Type at cursor ────────────────────────────────────────────────────────────

def make_type_at_cursor(cfg: dict):
    try:
        import keyboard
    except ImportError:
        logger.warning("[type_at_cursor] keyboard not installed, handler disabled.")
        return None

    tcfg = cfg["output"]["type_at_cursor"]
    enabled_lock = threading.Lock()
    enabled_state = [bool(tcfg["enabled"])]
    revert_mode = str(tcfg.get("revert_mode", tcfg.get("undo_mode", "off"))).strip().lower()
    if revert_mode not in {"off", "ctrl+z", "backspace"}:
        logger.warning(f"[type_at_cursor] Invalid revert_mode={revert_mode!r}; using 'off'.")
        revert_mode = "off"
    revert_backspace_count = max(1, int(tcfg.get("revert_backspace_count", tcfg.get("undo_backspace_count", 24))))

    def set_enabled(enabled: bool):
        with enabled_lock:
            enabled_state[0] = bool(enabled)

    def toggle_enabled() -> bool:
        with enabled_lock:
            enabled_state[0] = not enabled_state[0]
            return enabled_state[0]

    def is_enabled() -> bool:
        with enabled_lock:
            return enabled_state[0]

    def revert_last():
        if revert_mode == "off":
            logger.info("[type_at_cursor] Revert ignored (revert_mode=off).")
            return
        if revert_mode == "ctrl+z":
            keyboard.press_and_release("ctrl+z")
            logger.info("[type_at_cursor] Revert sent via ctrl+z")
            return
        for _ in range(revert_backspace_count):
            keyboard.press_and_release("backspace")
        logger.info(f"[type_at_cursor] Revert sent via {revert_backspace_count} backspaces")

    def type_at_cursor(event: dict):
        actions = event.get("actions", {}) or {}
        if actions.get("undo_type_at_cursor"):
            revert_last()
            return
        if not is_enabled():
            return
        if actions.get("release_file_buffer"):
            return
        text = _final_text(event, cfg)
        if text is not None:
            keyboard.write(text, delay=0)
        if actions.get("press_enter_after"):
            keyboard.press_and_release("enter")

    type_at_cursor.__name__ = "type_at_cursor"
    return type_at_cursor, set_enabled, toggle_enabled, is_enabled


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

    result = make_file_buffer(cfg)
    if result:
        fn, reset, set_enabled, is_enabled = result
        register(fn)
        extras["file_buffer_clear"] = reset
        extras["file_buffer_set_enabled"] = set_enabled
        extras["file_buffer_is_enabled"] = is_enabled
        logger.info(f"[handler] file_buffer -> {out['file_buffer']['path']}")

    result = make_clipboard_replace(cfg)
    if result:
        fn, set_enabled, is_enabled = result
        register(fn)
        extras["clipboard_replace_set_enabled"] = set_enabled
        extras["clipboard_replace_is_enabled"] = is_enabled
        logger.info("[handler] clipboard_replace")

    result = make_clipboard_accumulate(cfg)
    if result:
        fn, reset, set_enabled, is_enabled = result
        register(fn)
        extras["clipboard_accumulate_reset"] = reset
        extras["clipboard_accumulate_set_enabled"] = set_enabled
        extras["clipboard_accumulate_is_enabled"] = is_enabled
        logger.info("[handler] clipboard_accumulate")

    result = make_type_at_cursor(cfg)
    if result:
        fn, set_enabled, toggle_enabled, is_enabled = result
        register(fn)
        extras["type_at_cursor_set_enabled"] = set_enabled
        extras["type_at_cursor_toggle"] = toggle_enabled
        extras["type_at_cursor_is_enabled"] = is_enabled
        logger.info("[handler] type_at_cursor")

    if cfg["nats"]["enabled"]:
        fn = make_nats_publisher(cfg)
        if fn:
            register(fn)
            logger.info(f"[handler] nats_publisher -> {cfg['nats']['url']}")

    return extras
