# convocortex-stt

Headless speech-to-text engine designed for hands-free prolonged engagement. Intended to run in the background and stay out of your way — VAD triggers transcription automatically on pause, so you never need to press anything. Say a stop word to gate output off, say a start word to bring it back. No hands required.

Self-sufficient standalone. Participates in NATS as a peer when available.

## How it works

Dual-model pipeline: a fast CPU model produces **partial** results during speech, an accurate GPU model produces **final** results after silence. Silero VAD gates all inference so nothing runs while you are not speaking.

**Final** results are high-accuracy transcriptions produced once a pause in speech is detected. This is the primary output — written to file, clipboard, typed at cursor, or published over NATS.

**Partial** results are low-accuracy transcriptions fired every ~100ms while you are still speaking. Useful for detecting start/stop words before the utterance ends, or for live feedback displays. Not intended as reliable text output.

## Features

- Headless always — no GUI, no TUI
- VAD-triggered — silence ends an utterance, no push-to-talk required
- Start/stop words gate output on and off hands-free
- High-accuracy final transcriptions after each pause
- Realtime partial transcriptions during speech
- Local output handlers: file append, file overwrite, clipboard, type at cursor
- Hotkeys available but not required for normal use
- NATS event emit and control surface (optional — works fully without NATS)
- State persistence across restarts (mute state, gate state)
- Device reconnect on audio loss

## Prerequisites

### CUDA 12.1 (Windows, GPU acceleration)

Download: https://developer.nvidia.com/cuda-12-1-0-download-archive

Use **custom installation** and select only the CUDA toolkit — uncheck the driver and everything else if you already have a driver installed.

Add to PATH (adjust if you installed to a non-default location):
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp
```

To run on CPU only, skip CUDA entirely and set `final_device = "cpu"` in config.toml. Expect significantly slower final transcriptions.

### Python 3.10+

### uv

```
pip install uv
```

## Installation

```bash
git clone https://github.com/you/convocortex-stt
cd convocortex-stt
uv sync
```

On Windows (enables clipboard handlers):

```bash
uv sync --extra windows
```

**Hotkeys on Linux** require either running as root or adding your user to the `input` group:

```bash
sudo usermod -aG input $USER  # then log out and back in
```

**Hotkeys on macOS** require accessibility permissions granted in System Settings → Privacy & Security → Accessibility.

With NATS support:

```bash
uv sync --extra nats
```

Combined:

```bash
uv sync --extra windows --extra nats
```

## Running

```bash
uv run python stt.py
```

## Configuration

All settings live in `config.toml`. The file is read once at startup — restart to apply changes.

### Models

```toml
[models]
final           = "large-v3-turbo"   # Whisper model for final transcriptions
final_device    = "cuda"             # "cuda" or "cpu"
final_compute   = "float16"          # "float16", "int8", etc.
realtime        = "tiny.en"          # Whisper model for partial results
realtime_device = "cpu"
language        = "en"
```

### Audio

```toml
[audio]
vad_threshold   = 0.4    # Silero VAD sensitivity 0–1 (higher = less sensitive)
silence_timeout = 0.8    # Seconds of silence before finalizing an utterance
buffer_seconds  = 1.0    # Pre-speech ring buffer duration
```

### Output handlers

All default-on. Disable when a NATS consumer replaces them.

```toml
[output.file_append]
enabled = true
path    = "output.txt"       # Appends every partial and final result

[output.file_overwrite]
enabled = false
path    = "watched.txt"      # Overwrites with latest final result only

[output.clipboard_replace]
enabled = false              # Replaces clipboard with each final result

[output.clipboard_accumulate]
enabled   = false            # Appends each final result to clipboard
separator = " "

[output.type_at_cursor]
enabled = false              # Types final result at current cursor position

[output.trailing_char]
enabled = false
char    = " "                # Appended to every output (e.g. " " or ".")
```

### Emission gate

Trigger words gate whether output is forwarded. Not a voice command engine — just two word lists controlling a boolean. The matching utterance is consumed (not output).

```toml
[emission_gate]
enabled      = false
default_open = true          # Start forwarding at startup
start_words  = ["listen", "start"]
stop_words   = ["stop", "pause"]
```

### Hotkeys

```toml
[hotkeys]
emission_gate_toggle = "ctrl+shift+space"   # Toggle gate open/closed
push_to_talk         = ""                   # Hold to record, empty = disabled
mute_toggle          = ""                   # Toggle mute, empty = disabled
```

### NATS

```toml
[nats]
enabled         = false
url             = "nats://localhost:4222"
subject_emit    = "stt"         # Events published to stt.final, stt.partial, etc.
subject_control = "stt.control" # Control commands subscribed here
```

Requires `uv sync --extra nats`.

#### Event schema

```json
{"type": "partial", "text": "like I can just be", "epoch": 4, "t": 1.23, "inference_ms": 140}
{"type": "final",   "text": "like I can just be ranting", "epoch": 4, "t": 2.81, "inference_ms": 340}
{"type": "status",  "value": "recording"}
{"type": "status",  "value": "idle"}
{"type": "status",  "value": "muted"}
{"type": "system",  "event": "startup", "device": "Microphone (USB)", "models": {...}}
{"type": "system",  "event": "device_changed", "device": "Microphone (USB)"}
{"type": "system",  "event": "shutdown"}
```

#### Control commands

Send JSON to `stt.control`:

```json
{"cmd": "mute"}
{"cmd": "unmute"}
{"cmd": "push_to_talk_start"}
{"cmd": "push_to_talk_end"}
{"cmd": "open_emission_gate"}
{"cmd": "close_emission_gate"}
{"cmd": "status_query"}
{"cmd": "shutdown"}
```

## Stdin control

Send `MUTE` or `UNMUTE` to stdin at runtime.

## License

AGPL-3.0. See [LICENSE](LICENSE).

[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) by KoljaB is used as a dependency under the MIT license.
