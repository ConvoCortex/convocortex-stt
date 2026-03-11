"""
convocortex-stt — headless speech-to-text engine

Dual-model pipeline: fast CPU model for realtime partials,
accurate GPU model for final results after silence.
Output via configurable local handlers. NATS optional.
"""

import sys
import time
import queue
import threading
import logging
from collections import deque

import numpy as np
import pyaudio
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

# ── Realtime ──────────────────────────────────────────────────────────────────
RT_CHECK_INTERVAL = cfg["realtime"]["check_interval"]
MIN_CHUNKS        = cfg["realtime"]["min_chunks"]
MAX_CHUNKS        = cfg["realtime"]["max_chunks"]
RT_QUEUE_SIZE     = cfg["realtime"]["queue_size"]

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
#   {"type": "status",  "value": str}   # "recording" | "idle" | "muted" | "unmuted"
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

# ── Emission gate ─────────────────────────────────────────────────────────────
# Gates whether final transcription output is forwarded to handlers.
# Trigger word matches on final text only — matching utterance is consumed, not output.
# Partials and status/system events always pass through.

class EmissionGate:
    def __init__(self, cfg: dict):
        ecfg = cfg["emission_gate"]
        self.enabled     = ecfg["enabled"]
        self.open        = ecfg["default_open"]
        self.start_words = [w.lower() for w in ecfg["start_words"]]
        self.stop_words  = [w.lower() for w in ecfg["stop_words"]]
        self._lock       = threading.Lock()

    def process(self, event: dict) -> bool:
        """Return True if event should be forwarded to handlers."""
        if not self.enabled:
            return True
        etype = event.get("type")
        if etype == "partial":
            with self._lock:
                return self.open
        if etype == "final":
            text = event.get("text", "").strip().lower().strip(".,!?;:")
            with self._lock:
                if text in self.stop_words:
                    self.open = False
                    logger.info(f"[gate] Closed ('{text}')")
                    return False
                if text in self.start_words:
                    self.open = True
                    logger.info(f"[gate] Opened ('{text}')")
                    return False
                return self.open
        return True  # status/system always pass

    def open_gate(self):
        with self._lock:
            self.open = True
            logger.info("[gate] Opened")

    def close_gate(self):
        with self._lock:
            self.open = False
            logger.info("[gate] Closed")

# ── VAD wrapper ───────────────────────────────────────────────────────────────
class SileroVAD:
    def __init__(self, model):
        self.model = model

    def is_speech(self, audio_chunk):
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(audio_chunk).unsqueeze(0)
            return self.model(tensor, RATE).item()


def main():
    import signal
    def _sigterm(_s, _f):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    print(f"Initializing STT (Realtime: {REALTIME_DEVICE}, Final: {FINAL_DEVICE})...")
    sys.stdout.flush()

    resources = {}
    warmup_audio = np.zeros(RATE, dtype=np.float32)

    # ── Async model loading ───────────────────────────────────────────────────

    def load_final():
        try:
            logger.info(f"Loading {FINAL_MODEL} ({FINAL_DEVICE})...")
            m = WhisperModel(FINAL_MODEL, device=FINAL_DEVICE, compute_type=FINAL_COMPUTE)
            m.transcribe(warmup_audio, beam_size=1)
            resources['final_model'] = m
            logger.info("Final model ready.")
        except Exception as e:
            logger.error(f"Final load error: {e}")
            sys.exit(1)

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
            logger.error(f"Realtime load error: {e}")
            sys.exit(1)

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
            info = p.get_default_input_device_info()
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, frames_per_buffer=CHUNK,
                input_device_index=info['index']
            )
            resources['stream'] = stream
            resources['dev_name'] = info['name']
            resources['dev_idx'] = info['index']
            logger.info(f"Audio: {info['name']}")
        except Exception as e:
            logger.error(f"Audio stack error: {e}")
            sys.exit(1)

    threads = [
        threading.Thread(target=load_final),
        threading.Thread(target=load_realtime),
        threading.Thread(target=load_audio_stack),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    final_model = resources['final_model']
    rt_model    = resources['realtime_model']
    vad         = resources['vad']
    stream      = resources['stream']
    p_instance  = resources['p_instance']
    dev_name    = resources['dev_name']

    # ── Register handlers ─────────────────────────────────────────────────────
    import handlers as h
    handler_extras = h.register_all(cfg, register_handler)

    # ── Emission gate ─────────────────────────────────────────────────────────
    gate = EmissionGate(cfg)
    gate.open = _persisted['gate_open']  # restore persisted gate state

    # ── State ─────────────────────────────────────────────────────────────────
    final_queue   = queue.Queue()
    rt_queue      = queue.Queue(maxsize=RT_QUEUE_SIZE)
    state_lock    = threading.Lock()
    state         = {'speech_epoch': 0, 'finalized_epoch': -1}
    rt_cancel     = threading.Event()

    control_lock  = threading.Lock()
    control_state = {'is_muted': _persisted['is_muted']}

    # ── Workers ───────────────────────────────────────────────────────────────

    def final_worker():
        while True:
            try:
                epoch, t0, audio_data = final_queue.get()
                audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
                logger.info(f"[FINAL] Processing {len(audio)/RATE:.2f}s audio...")

                t_start = time.time()
                segs, _ = final_model.transcribe(audio, language=LANGUAGE, beam_size=5)
                text = " ".join(s.text for s in segs).strip()
                inf_ms = int((time.time() - t_start) * 1000)
                t = time.time() - t0

                if text:
                    ev = {"type": "final", "text": text, "epoch": epoch,
                          "t": round(t, 3), "inference_ms": inf_ms}
                    if gate.process(ev):
                        logger.info(f"[FINAL +{t:.2f}s] → OUT  '{text}' ({inf_ms}ms)")
                        dispatch(ev)
                    else:
                        logger.info(f"[FINAL +{t:.2f}s] → GATED '{text}' ({inf_ms}ms)")
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

                if text:
                    ev = {"type": "partial", "text": text, "epoch": epoch,
                          "t": round(t, 3), "inference_ms": inf_ms}
                    if gate.process(ev):
                        logger.info(f"[PARTIAL +{t:.2f}s] → OUT  '{text}' ({inf_ms}ms)")
                        dispatch(ev)
                    else:
                        logger.info(f"[PARTIAL +{t:.2f}s] → GATED '{text}' ({inf_ms}ms)")
            except Exception as e:
                logger.error(f"[realtime_worker] {e}")
            finally:
                try: rt_queue.task_done()
                except: pass

    threading.Thread(target=final_worker, daemon=True).start()
    if rt_model:
        threading.Thread(target=realtime_worker, daemon=True).start()

    # ── State persistence helper ──────────────────────────────────────────────

    def persist():
        with control_lock: muted = control_state['is_muted']
        with gate._lock:   go    = gate.open
        state_store.save({"is_muted": muted, "gate_open": go})

    # ── Hotkeys ───────────────────────────────────────────────────────────────

    def setup_hotkeys():
        try:
            import keyboard as kb
        except ImportError:
            logger.warning("[hotkeys] keyboard not installed, hotkeys disabled.")
            return

        hk = cfg["hotkeys"]

        if hk["emission_gate_open"]:
            kb.add_hotkey(hk["emission_gate_open"], lambda: (gate.open_gate(), persist()))
            logger.info(f"[hotkey] emission_gate_open = {hk['emission_gate_open']}")

        if hk["emission_gate_close"]:
            kb.add_hotkey(hk["emission_gate_close"], lambda: (gate.close_gate(), persist()))
            logger.info(f"[hotkey] emission_gate_close = {hk['emission_gate_close']}")

        if hk.get("clipboard_accumulate_cycle") and "clipboard_accumulate_reset" in handler_extras:
            kb.add_hotkey(hk["clipboard_accumulate_cycle"], handler_extras["clipboard_accumulate_reset"])
            logger.info(f"[hotkey] clipboard_accumulate_cycle = {hk['clipboard_accumulate_cycle']}")

        if hk.get("device_cycle"):
            kb.add_hotkey(hk["device_cycle"], lambda: cycle_input_device())
            logger.info(f"[hotkey] device_cycle = {hk['device_cycle']}")

        if hk["mute_toggle"]:
            def _mute_toggle():
                with control_lock:
                    control_state['is_muted'] = not control_state['is_muted']
                    muted = control_state['is_muted']
                logger.info(f"[hotkey] {'Muted' if muted else 'Unmuted'}")
                dispatch({"type": "status", "value": "muted" if muted else "unmuted"})
                persist()
            kb.add_hotkey(hk["mute_toggle"], _mute_toggle)
            logger.info(f"[hotkey] mute_toggle = {hk['mute_toggle']}")


    setup_hotkeys()

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

                if cmd == "mute":
                    with control_lock: control_state['is_muted'] = True
                    dispatch({"type": "status", "value": "muted"})
                    persist()
                elif cmd == "unmute":
                    with control_lock: control_state['is_muted'] = False
                    dispatch({"type": "status", "value": "unmuted"})
                    persist()
                elif cmd == "open_emission_gate":
                    gate.open_gate()
                    persist()
                elif cmd == "close_emission_gate":
                    gate.close_gate()
                    persist()
                elif cmd == "status_query":
                    with control_lock: muted = control_state['is_muted']
                    with gate._lock: gate_open = gate.open
                    reply = {"muted": muted, "gate_open": gate_open,
                             "gate_enabled": gate.enabled}
                    if msg.reply:
                        await nc.publish(msg.reply, json.dumps(reply).encode())
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

    # ── Device cycling ────────────────────────────────────────────────────────
    # IMPORTANT: stream.close()/open() must only happen in the main loop thread.
    # The hotkey thread only writes to _pending_device; the main loop consumes it.

    current_device_idx = [resources['dev_idx']]
    _pending_device    = [None]   # (idx, name) set by hotkey, consumed by main loop
    _ALIAS_NAMES       = {'Primary Sound Capture Driver', 'Microsoft Sound Mapper'}
    _bad_devices       = set()    # device indices that failed to open; skipped in cycle

    def _enumerate_input_devices():
        """Read-only enumeration — safe to call from any thread."""
        devices = []
        seen_names = set()
        for i in range(p_instance.get_device_count()):
            info = p_instance.get_device_info_by_index(i)
            if info['maxInputChannels'] <= 0:
                continue
            name = info['name']
            if name in _ALIAS_NAMES or name in seen_names:
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

    dispatch({"type": "system", "event": "startup",
              "device": dev_name,
              "models": {"final": FINAL_MODEL, "realtime": REALTIME_MODEL}})
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
                next_idx, next_name = pending
                prev_idx = current_device_idx[0]
                _open_kwargs = dict(
                    format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK,
                )
                try:
                    stream.stop_stream()
                    stream.close()
                    stream = p_instance.open(input_device_index=next_idx, **_open_kwargs)
                    # Verify with a real read — catches devices that open but don't deliver audio
                    vdata = stream.read(CHUNK, exception_on_overflow=False)
                    vchunk = np.frombuffer(vdata, dtype=np.int16).astype(np.float32)
                    rms = int(np.sqrt(np.mean(vchunk ** 2)))
                    current_device_idx[0] = next_idx
                    ring_buffer.clear()
                    recording_buffer.clear()
                    is_recording = False
                    logger.info(f"[device] Cycled to: {next_name} (rms={rms})")
                    dispatch({"type": "system", "event": "device_changed", "device": next_name})
                except Exception as e:
                    logger.error(f"[device] Failed to switch to {next_name}: {e}")
                    # Reopen previous device so the main loop isn't left with a dead stream
                    try:
                        stream.stop_stream(); stream.close()
                    except: pass
                    try:
                        stream = p_instance.open(input_device_index=prev_idx, **_open_kwargs)
                        ring_buffer.clear(); recording_buffer.clear(); is_recording = False
                        logger.info(f"[device] Reverted to previous device")
                    except Exception as e2:
                        logger.error(f"[device] Revert failed: {e2}")
                continue

            with control_lock:
                if control_state['is_muted']:
                    continue

            chunk = np.frombuffer(data, dtype=np.int16)
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
