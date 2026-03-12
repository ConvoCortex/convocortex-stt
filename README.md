# convocortex-stt

Self-sufficient NATS-native hands-free ergonomic speech-to-text engine.

Designed for ambient operation. VAD triggers transcription automatically on pause. Say a stop word to silence output, say a start word to resume. Hotkey use is optional.

Self-sufficient standalone, but intended to integrate via NATS.

## How it works

Dual-model pipeline: a fast model produces **partial** results during speech, an accurate GPU model produces **final** results after silence. Silero VAD gates all inference so nothing runs while you are not speaking.

**Final** results are high-accuracy transcriptions produced once a pause in speech is detected. This is the primary output — written to file, clipboard, typed at cursor, or published over NATS.

**Partial** results are low-accuracy transcriptions fired every ~100ms while you are still speaking. Useful for reducing latency of voice commands and detecting start/stop words before the utterance ends. Not intended as accurate text output.

## Features

- No GUI
- VAD-triggered — silence ends an utterance, no push-to-talk required
- Start/stop words gate output on and off hands-free
- High-accuracy final transcriptions after each pause
- Realtime partial transcriptions during speech
- Local output handlers: file append, file overwrite, clipboard, type at cursor
- Hotkeys with additional features available but not required for normal use
- NATS event emit and control surface (optional — works fully without NATS)
- State persistence across restarts (mute state, gate state, last input device)
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

cuDNN may not be required — ctranslate2 installed via pip bundles some CUDA libraries. If you get CUDA errors at runtime, install cuDNN separately and place the files in the CUDA `bin` directory.

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

With NATS support:

```bash
uv sync --extra nats
```

**Hotkeys on Linux** require either running as root or adding your user to the `input` group:

```bash
sudo usermod -aG input $USER  # then log out and back in
```

**Hotkeys on macOS** require accessibility permissions granted in System Settings → Privacy & Security → Accessibility.

## Running

```bash
uv run python stt.py
```

## Configuration

All settings live in `config.toml`. Every option is documented there with comments. The file is read once at startup — restart to apply changes.

Audio startup selection order:
- `audio.input_device` (if set to an exact device name)
- remembered last-used input device from `state.json`
- OS default input device

Minimal voice command behavior:
- Enable `voice_commands.enabled = true` and `voice_commands.enter.enabled = true`.
- If an utterance starts or ends with a configured `voice_commands.enter.words` trigger, STT removes that trigger from typed text and sends Enter after typing.
- If you say only the trigger word (e.g. `enter`), it sends Enter without typing text.
- `voice_commands.tts*` settings are currently placeholders and are not implemented yet.

## NATS

Requires `uv sync --extra nats` and `nats.enabled = true` in config.toml.

### Event schema

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

### Control commands

Send JSON to the subject configured in `nats.subject_control`:

```json
{"cmd": "mute"}
{"cmd": "unmute"}
{"cmd": "open_emission_gate"}
{"cmd": "close_emission_gate"}
{"cmd": "device_cycle"}
{"cmd": "status_query"}
{"cmd": "shutdown"}
```

## License

convocortex-stt is licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0). See [LICENSE](LICENSE) for the full text.

AGPL-3.0 means: you may use, modify, and distribute this software freely,
including running it as a network service — but any modifications must be
released under the same license.

### Commercial license

If you need to use convocortex-stt in a proprietary product or service without
the AGPL-3.0 obligations, a commercial license is available. Contact the
maintainer to discuss terms.

### Contributing

Contributions require signing the [Individual Contributor License Agreement](CLA.md).
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
