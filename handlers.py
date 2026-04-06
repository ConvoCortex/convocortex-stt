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


def _now_iso8601_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_envelope(*, source: str, payload: dict, utterance_id: str = "", correlation_id: str = "") -> dict:
    event_id = correlation_id or utterance_id or f"stt-{int(time.time() * 1000)}"
    return {
        "schema_version": "v1alpha1",
        "event_id": event_id,
        "correlation_id": correlation_id or utterance_id,
        "workflow": "basic-companion",
        "source": source,
        "published_at": _now_iso8601_utc(),
        "payload": payload,
    }


def _release_common_modifiers(keyboard):
    for key_name in (
        "shift",
        "left shift",
        "right shift",
        "ctrl",
        "left ctrl",
        "right ctrl",
        "alt",
        "left alt",
        "right alt",
        "alt gr",
        "windows",
        "left windows",
        "right windows",
    ):
        try:
            keyboard.release(key_name)
        except Exception:
            continue


def _load_keyboard():
    try:
        import keyboard
        return keyboard
    except ImportError:
        return None


def _load_win32_clipboard():
    try:
        import win32clipboard
        import win32con
        return win32clipboard, win32con
    except ImportError:
        return None, None


def _open_clipboard_with_retry(win32clipboard, retry_count: int, retry_delay_ms: int):
    last_error = None
    for _ in range(max(1, int(retry_count))):
        try:
            win32clipboard.OpenClipboard()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(max(1, int(retry_delay_ms)) / 1000.0)
    raise RuntimeError(f"failed to open clipboard after retries: {last_error}")


def _capture_clipboard_state(win32clipboard, win32con) -> tuple[str | None, list[tuple[int, object]]]:
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


def _restore_clipboard_state(
    win32clipboard,
    win32con,
    inserted_text: str,
    saved_text: str | None,
    saved_formats: list[tuple[int, object]],
    restore_delay_ms: int,
    retry_count: int,
    retry_delay_ms: int,
    logger_label: str,
):
    time.sleep(max(0, int(restore_delay_ms)) / 1000.0)
    try:
        _open_clipboard_with_retry(win32clipboard, retry_count, retry_delay_ms)
        try:
            current_text = None
            try:
                current_text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            except Exception:
                current_text = None
            if current_text != inserted_text:
                logger.info(f"[{logger_label}] Clipboard changed after paste; skipping restore.")
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
                logger.warning(f"[{logger_label}] Clipboard restore failed: no clipboard formats could be restored.")
        finally:
            win32clipboard.CloseClipboard()
    except Exception as exc:
        logger.warning(f"[{logger_label}] Clipboard restore failed: {exc}")


def _emit_text_via_typing(keyboard, content: str, press_enter_after: bool):
    _release_common_modifiers(keyboard)
    try:
        keyboard.write(content, delay=0)
        if press_enter_after:
            keyboard.press_and_release("enter")
    finally:
        _release_common_modifiers(keyboard)


def _emit_text_via_clipboard(
    keyboard,
    win32clipboard,
    win32con,
    content: str,
    press_enter_after: bool,
    *,
    clipboard_restore_delay_ms: int,
    clipboard_open_retry_count: int,
    clipboard_open_retry_delay_ms: int,
    post_paste_enter_delay_ms: int,
    logger_label: str,
):
    _release_common_modifiers(keyboard)
    _open_clipboard_with_retry(win32clipboard, clipboard_open_retry_count, clipboard_open_retry_delay_ms)
    try:
        saved_text, saved_formats = _capture_clipboard_state(win32clipboard, win32con)
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(content, win32con.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()

    try:
        keyboard.press_and_release("ctrl+v")
        if press_enter_after:
            if post_paste_enter_delay_ms > 0:
                time.sleep(post_paste_enter_delay_ms / 1000.0)
            keyboard.press_and_release("enter")
    finally:
        _release_common_modifiers(keyboard)

    restore_thread = threading.Thread(
        target=_restore_clipboard_state,
        args=(
            win32clipboard,
            win32con,
            content,
            saved_text,
            saved_formats,
            clipboard_restore_delay_ms,
            clipboard_open_retry_count,
            clipboard_open_retry_delay_ms,
            logger_label,
        ),
        daemon=True,
    )
    restore_thread.start()


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
    rewind_history_limit = max(0, int(bcfg.get("rewind_history_limit", 10)))
    release_method = str(bcfg.get("release_method", "type_keys")).strip().lower()
    clipboard_restore_delay_ms = max(0, int(bcfg.get("clipboard_restore_delay_ms", 250)))
    clipboard_open_retry_count = max(1, int(bcfg.get("clipboard_open_retry_count", 8)))
    clipboard_open_retry_delay_ms = max(1, int(bcfg.get("clipboard_open_retry_delay_ms", 25)))
    post_paste_enter_delay_ms = max(0, int(bcfg.get("post_paste_enter_delay_ms", 140)))
    lock = threading.Lock()
    enabled_lock = threading.Lock()
    enabled_state = [False]
    rewind_history = deque(maxlen=rewind_history_limit) if rewind_history_limit > 0 else None
    release_lock = threading.Lock()
    last_released_content = [""]

    keyboard = _load_keyboard()
    if keyboard is None:
        logger.warning("[file_buffer] keyboard not installed, voice-triggered release disabled.")

    win32clipboard, win32con = None, None
    if release_method == "paste_preserve_clipboard":
        win32clipboard, win32con = _load_win32_clipboard()
        if win32clipboard is None or win32con is None:
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

    def _remember_rewind(previous: str, next_content: str):
        if rewind_history is None or previous == next_content:
            return
        rewind_history.append(previous)

    def _set_buffer(content: str):
        with lock:
            previous = _read_buffer()
            _remember_rewind(previous, content)
            _write_buffer(content)

    def _append_buffer(text: str):
        with lock:
            existing = "" if reset_after_each_message else _read_buffer()
            combined = (existing + sep + text) if existing else text
            _remember_rewind(existing, combined)
            _write_buffer(combined)

    def _append_external_text(text: str):
        content = str(text or "")
        if not content:
            logger.info("[file_buffer] External append ignored: empty text.")
            return
        _append_buffer(content)
        logger.info(f"[file_buffer] appended external text ({len(content)} chars)")

    def set_enabled(enabled: bool):
        with enabled_lock:
            enabled_state[0] = bool(enabled)

    def is_enabled() -> bool:
        with enabled_lock:
            return enabled_state[0]

    def _clear_buffer():
        _set_buffer("")
        logger.info("[file_buffer] cleared")

    def _rewind_buffer():
        if rewind_history is None:
            return
        with lock:
            if not rewind_history:
                return
            restored = rewind_history.pop()
            _write_buffer(restored)
        logger.info(f"[file_buffer] rewind restored {len(restored)} chars")

    def _release_via_clipboard(content: str, press_enter_after: bool):
        assert keyboard is not None
        assert win32clipboard is not None and win32con is not None

        with release_lock:
            _emit_text_via_clipboard(
                keyboard,
                win32clipboard,
                win32con,
                content,
                press_enter_after,
                clipboard_restore_delay_ms=clipboard_restore_delay_ms,
                clipboard_open_retry_count=clipboard_open_retry_count,
                clipboard_open_retry_delay_ms=clipboard_open_retry_delay_ms,
                post_paste_enter_delay_ms=post_paste_enter_delay_ms,
                logger_label="file_buffer",
            )

    def _compose_release_content(inline_text: str = "") -> str:
        with lock:
            existing = "" if reset_after_each_message else _read_buffer()
        text = str(inline_text or "")
        if text:
            return (existing + sep + text) if existing else text
        return existing

    def _commit_released_content(content: str, *, clear_buffer: bool | None = None):
        final_content = str(content or "")
        should_clear = clear_after_release if clear_buffer is None else bool(clear_buffer)
        with lock:
            current = _read_buffer()
            if current != final_content:
                _remember_rewind(current, final_content)
                if not should_clear:
                    _write_buffer(final_content)
            if should_clear:
                persisted = final_content if current != final_content else current
                _remember_rewind(persisted, "")
                _write_buffer("")
        last_released_content[0] = final_content

    def _release_buffer(press_enter_after: bool = False):
        if keyboard is None:
            logger.warning("[file_buffer] Release ignored: keyboard not installed.")
            return
        content = _compose_release_content()
        if not content:
            if press_enter_after:
                keyboard.press_and_release("enter")
                logger.info("[file_buffer] buffer empty; sent enter only")
            return
        if release_method == "paste_preserve_clipboard":
            _release_via_clipboard(content, press_enter_after)
        else:
            _emit_text_via_typing(keyboard, content, press_enter_after)
        _commit_released_content(content)
        logger.info(
            f"[file_buffer] released {len(content)} chars via {release_method}"
            + (" + enter" if press_enter_after else "")
        )

    def _repeat_last_released_buffer(press_enter_after: bool = False):
        content = last_released_content[0]
        if not content:
            return
        if release_method == "paste_preserve_clipboard":
            _release_via_clipboard(content, press_enter_after)
        else:
            _emit_text_via_typing(keyboard, content, press_enter_after)
        logger.info(
            f"[file_buffer] repeated {len(content)} chars via {release_method}"
            + (" + enter" if press_enter_after else "")
        )

    def _restore_last_released_buffer():
        content = last_released_content[0]
        if not content:
            return
        _set_buffer(content)
        logger.info(f"[file_buffer] restored last released buffer ({len(content)} chars)")

    def file_buffer(event: dict):
        actions = event.get("actions", {}) or {}
        if actions.get("rewind_file_buffer"):
            _rewind_buffer()
            return
        if actions.get("restore_last_released_buffer"):
            _restore_last_released_buffer()
            return
        if actions.get("repeat_last_released_buffer"):
            _repeat_last_released_buffer(bool(actions.get("press_enter_after")))
            return
        if actions.get("clear_file_buffer"):
            _clear_buffer()
            return
        if actions.get("release_file_buffer"):
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

    def _get_buffer_content() -> str:
        with lock:
            return _read_buffer()

    file_buffer.__name__ = "file_buffer"
    return (
        file_buffer,
        _clear_buffer,
        set_enabled,
        is_enabled,
        _get_buffer_content,
        _append_external_text,
        _compose_release_content,
        _commit_released_content,
    )


# ── Type at cursor ────────────────────────────────────────────────────────────

def make_type_at_cursor(cfg: dict):
    keyboard = _load_keyboard()
    if keyboard is None:
        logger.warning("[type_at_cursor] keyboard not installed, handler disabled.")
        return None

    tcfg = cfg["output"]["type_at_cursor"]
    enabled_lock = threading.Lock()
    enabled_state = [False]
    release_method = str(tcfg.get("release_method", "paste_preserve_clipboard")).strip().lower()
    clipboard_restore_delay_ms = max(0, int(tcfg.get("clipboard_restore_delay_ms", 1000)))
    clipboard_open_retry_count = max(1, int(tcfg.get("clipboard_open_retry_count", 8)))
    clipboard_open_retry_delay_ms = max(1, int(tcfg.get("clipboard_open_retry_delay_ms", 25)))
    post_paste_enter_delay_ms = max(0, int(tcfg.get("post_paste_enter_delay_ms", 0)))
    revert_mode = str(tcfg.get("revert_mode", "off")).strip().lower()
    if revert_mode not in {"off", "ctrl+z", "backspace"}:
        logger.warning(f"[type_at_cursor] Invalid revert_mode={revert_mode!r}; using 'off'.")
        revert_mode = "off"
    revert_backspace_count = max(1, int(tcfg.get("revert_backspace_count", 24)))
    win32clipboard, win32con = None, None
    if release_method == "paste_preserve_clipboard":
        win32clipboard, win32con = _load_win32_clipboard()
        if win32clipboard is None or win32con is None:
            logger.warning("[type_at_cursor] pywin32 not installed, falling back to type_keys release method.")
            release_method = "type_keys"

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
            keyboard.press_and_release("ctrl+z")
            logger.info("[type_at_cursor] Revert sent via ctrl+z fallback")
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
        if actions.get("rewind_type_at_cursor"):
            revert_last()
            return
        if not is_enabled():
            return
        if actions.get("release_file_buffer"):
            return
        text = _final_text(event, cfg)
        if text is not None:
            if release_method == "paste_preserve_clipboard":
                _emit_text_via_clipboard(
                    keyboard,
                    win32clipboard,
                    win32con,
                    text,
                    False,
                    clipboard_restore_delay_ms=clipboard_restore_delay_ms,
                    clipboard_open_retry_count=clipboard_open_retry_count,
                    clipboard_open_retry_delay_ms=clipboard_open_retry_delay_ms,
                    post_paste_enter_delay_ms=post_paste_enter_delay_ms,
                    logger_label="type_at_cursor",
                )
            else:
                _emit_text_via_typing(keyboard, text, False)
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

    def _normalize_event_for_nats(event: dict) -> tuple[str, dict] | None:
        if event.get("suppress_nats"):
            return None
        etype = str(event.get("type", "system") or "system").strip()
        if etype in {"partial", "final", "buffer_released"}:
            text = str(event.get("text") or "")
            if not text.strip():
                return None
            normalized_type = "final" if etype == "buffer_released" else etype
            utterance_id = ""
            if event.get("epoch") is not None:
                utterance_id = str(int(event["epoch"]))
            payload = {
                "utterance_id": utterance_id,
                "text": text,
            }
            speaker = str(event.get("speaker") or "").strip()
            if speaker:
                payload["speaker"] = speaker
            if event.get("replayed"):
                payload["replayed"] = True
            if event.get("released_by_recognize"):
                payload["released_by_recognize"] = True
            return normalized_type, _canonical_envelope(
                source="convocortex-stt",
                utterance_id=utterance_id,
                correlation_id=utterance_id,
                payload=payload,
            )
        if etype == "status":
            value = str(event.get("value") or "").strip()
            if not value:
                return None
            return etype, _canonical_envelope(
                source="convocortex-stt",
                payload={"status": value},
            )
        if etype == "system":
            normalized = {}
            for key in ("event", "device", "mode", "models", "output_mode"):
                if key in event:
                    normalized[key] = event[key]
            if not normalized:
                return None
            return etype, _canonical_envelope(
                source="convocortex-stt",
                payload=normalized,
            )
        return None

    def nats_publisher(event: dict):
        normalized = _normalize_event_for_nats(event)
        if normalized is None:
            return
        etype, payload = normalized
        subj  = f"{subject}.{etype}"
        data  = json.dumps(payload).encode()
        logger.info(f"[nats] publish {subj} {json.dumps(payload, ensure_ascii=False)}")
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
        (
            fn,
            reset,
            set_enabled,
            is_enabled,
            get_content,
            append_external_text,
            compose_release_content,
            commit_released_content,
        ) = result
        register(fn)
        extras["file_buffer_clear"] = reset
        extras["file_buffer_set_enabled"] = set_enabled
        extras["file_buffer_is_enabled"] = is_enabled
        extras["file_buffer_get_content"] = get_content
        extras["file_buffer_append_text"] = append_external_text
        extras["file_buffer_compose_release_content"] = compose_release_content
        extras["file_buffer_commit_released_content"] = commit_released_content
        logger.info(f"[handler] file_buffer -> {out['file_buffer']['path']}")

    result = make_type_at_cursor(cfg)
    if result:
        fn, set_enabled, toggle_enabled, is_enabled = result
        register(fn)
        extras["cursor_output_set_enabled"] = set_enabled
        extras["cursor_output_toggle"] = toggle_enabled
        extras["cursor_output_is_enabled"] = is_enabled
        logger.info("[handler] type_at_cursor")

    if cfg["nats"]["enabled"]:
        fn = make_nats_publisher(cfg)
        if fn:
            register(fn)
            logger.info(f"[handler] nats_publisher -> {cfg['nats']['url']}")

    return extras
