# convocortex-stt

Hands-free speech-to-text for ambient use with voice commands, wake-word, VAD, feedback, realtime partials and NATS integration.

It is designed to run continuously in the background, transcribe on pause, and expose events/controls over NATS so you can plug it into a larger voice system.

## How it actually works

### 1) Hands-free transcription on pause

In working mode, you speak naturally. Silero VAD tracks speech vs silence:
- speech starts/continues an utterance buffer
- silence timeout closes the utterance
- final transcription runs after that pause
- if audio capture stalls mid-utterance (for example device mute/disconnect), the active utterance is dropped instead of being resumed later

This is why it is hands-free: no key press required to start/stop an utterance.

### 2) Wake word is mode control, not push-to-talk

The runtime has two modes:
- `sleeping`: listens only for wake word
- `working`: normal STT behavior

Wake word exists to move from sleeping -> working.
Stop words (or hotkey/NATS command) move from working -> sleeping.

So wake word is not "say phrase to start one transcription". It controls whether STT is active at all.

### 3) VAD prevents constant noise chewing

When in working mode, inference is driven by VAD-detected speech windows instead of blindly processing endless noise.

### 4) Dual-model pipeline

- Realtime model emits **partial** text while you are speaking (low latency)
- Final model emits **final** text after pause/silence (higher accuracy)

Partials are primarily for responsiveness and command latency, not for high-accuracy text output.

## Voice commands

Built-in voice commands are intentionally minimal and convenience-oriented.

Current built-in command types:
- sleep/stop words
- type-at-cursor toggle words
- enter trigger words
- file-buffer release words
- file-buffer clear words

Commands are matched from partials (fast path) and finals (reliable path), using normalized exact matching for configured command phrases.

For a proper, richer voice command engine, use the emitted NATS events and implement command logic externally.

## Features

- No GUI
- Hands-free transcription on pause
- Sleep/working mode with wake word + stop words
- Realtime partial transcriptions during speech
- High-accuracy final transcriptions after pauses
- Built-in audio feedback sounds (on/off/final + silence loop for bluetooth audio issues)
- Simple built-in voice command actions
- Local output handlers:
  - append file
  - overwrite file
  - file buffer accumulate/release
  - clipboard replace
  - clipboard accumulate
  - type at cursor
- Hotkeys:
  - sleep toggle
  - typing toggle
  - input device cycle
  - output device cycle
  - clipboard-accumulate reset cycle
- NATS integration:
  - event stream (`partial`, `final`, `status`, `system`)
  - control surface (`sleep`, `wake`, `typing_*`, device cycle, etc.)
- Persisted runtime state across restarts
- Input/output device switching and reconnect behavior

## Performance behavior

Defaults favor practical command responsiveness over nonstop partial spam:
- partials are rate-limited by `realtime.check_interval`
- partial jobs are constrained by `realtime.min_chunks` / `realtime.max_chunks`

You can tune these if you want denser realtime partial output.

## Prerequisites

Primary tested target:
- Windows
- NVIDIA GPU with CUDA 12.1 (recommended)

Linux/macOS may work but are currently untested in this repository.

### CUDA 12.1 (Windows)

Download: https://developer.nvidia.com/cuda-12-1-0-download-archive

Use custom install and keep CUDA toolkit. If runtime CUDA errors occur, verify driver/runtime and related CUDA libs.

Optional PATH entries (adjust to your install path):

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp
```

CPU-only is supported by setting `models.final_device = "cpu"` in `config.toml` (slower finals).

### Python and uv

- Python 3.10+
- `uv` installed (`pip install uv`)

## Installation

```bash
git clone https://github.com/ConvoCortex/convocortex-stt
cd convocortex-stt
uv sync
```

NATS Python client dependency is included by default.

Create your local config from the example:

```bash
cp config.example.toml config.toml
```

On Windows PowerShell:

```powershell
Copy-Item config.example.toml config.toml
```

## Running

```bash
uv run python stt.py
```

To start normally and hide the console once the runtime is ready:

```bash
uv run python stt.py --background
```

On Windows, the default `hotkeys.console_toggle = "shift+f8"` hides/shows that same console window. When you bring it back, `Ctrl+C` kills the process as usual.
If you want the default backtick sleep hotkey to be intercepted instead of typed into the active app, set `hotkeys.sleep_toggle_suppress = true`.

## Windows Startup Shortcut

If you want this to come up automatically in the background after you log in on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-startup-shortcut.ps1
```

That creates a shortcut in:

```text
C:\Users\user\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

The shortcut launches:

```text
.venv\Scripts\python.exe stt.py --background
```

This is the simpler Windows logon path and fits this app better because it relies on an interactive desktop session for global hotkeys and for restoring the console window to the foreground.

To remove the shortcut:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall-startup-shortcut.ps1
```

## Configuration

All runtime settings live in `config.toml` and are loaded at startup.

Important areas:
- `models`: realtime/final models and device choices
- `audio`: VAD behavior + silence timeout + preferred input
- `realtime`: partial cadence/window limits
- `filters`: exact junk-phrase suppression before partial/final output is shown or dispatched
- `logging`: toggleable debug traces written to a file for freeze/debug analysis
- `output`: handler toggles and paths, including editable `file_buffer`
- `voice_commands`: built-in convenience command words
- `sleep_wake`: wake word backend + stop words + startup mode
- `hotkeys`: optional runtime controls
- `nats`: URL/subjects + enable toggle

### File buffer workflow

If `output.file_buffer.enabled = true`, each final transcription is appended to `output.file_buffer.path`.

This gives you a plain text working buffer you can keep open in an editor, adjust manually, and then inject into the active cursor later with an exact voice command such as `buffer`.

Relevant settings:
- `output.file_buffer.separator`: separator inserted between finalized utterances
- `output.file_buffer.clear_after_release`: clear the file after the `buffer` command types it
- `output.file_buffer.reset_after_each_message`: clear the existing buffer before writing each new finalized utterance
- `voice_commands.buffer_release.words`: exact phrases that trigger release
- `voice_commands.buffer_release.press_enter_after`: optionally press Enter after typing the buffer
- `voice_commands.buffer_clear.words`: exact phrases that clear the buffer without typing it

### Debug logging

When you need to investigate intermittent stalls or missed speech, enable:

```toml
[logging]
debug = true
file = "stt-debug.log"
third_party_debug = false
heartbeat_seconds = 5.0
```

With `debug = true`, the runtime keeps normal console output but also appends detailed traces to the log file, including:
- long `stream.read()` blocks
- VAD speech/silence transitions with scores
- speech start/finalization boundaries
- realtime queue enqueue/drop behavior
- periodic heartbeats showing mode, queue depth, buffer sizes, and current VAD state

This project is not currently using the `realtimestt` package at runtime for the main loop; the useful debug points are in [`stt.py`](stt.py) around PyAudio, Silero VAD, and faster-whisper.

## NATS

Enable by setting:
- `nats.enabled = true`

NATS is intended as the integration boundary to a larger app:
- consume emitted events
- issue control commands
- implement advanced voice command logic outside this process

### Event schema

```json
{"type": "partial", "text": "like I can just be", "epoch": 4, "t": 1.23, "inference_ms": 140}
{"type": "final",   "text": "like I can just be ranting", "epoch": 4, "t": 2.81, "inference_ms": 340}
{"type": "status",  "value": "recording"}
{"type": "status",  "value": "idle"}
{"type": "status",  "value": "sleeping"}
{"type": "status",  "value": "working"}
{"type": "system",  "event": "startup", "device": "Microphone (USB)", "models": {...}}
{"type": "system",  "event": "device_changed", "device": "Microphone (USB)"}
{"type": "system",  "event": "output_device_changed", "device": "Speakers (USB)"}
{"type": "system",  "event": "shutdown"}
```

### Control commands

Send JSON to `nats.subject_control`:

```json
{"cmd": "sleep"}
{"cmd": "wake"}
{"cmd": "sleep_toggle"}
{"cmd": "typing_toggle"}
{"cmd": "typing_enable"}
{"cmd": "typing_disable"}
{"cmd": "input_device_cycle"}
{"cmd": "output_device_cycle"}
{"cmd": "status_query"}
{"cmd": "shutdown"}
```

## Platform notes

- Windows is the primary tested path.
- Linux hotkeys may require input permissions (group/root setup).
- macOS hotkeys may require Accessibility permissions.

## Acknowledgments

This project builds on top of the RealTimeSTT ecosystem and uses faster-whisper/ctranslate2 under the hood.

## License

convocortex-stt is licensed under **GNU Affero General Public License v3.0** (AGPL-3.0).
See [LICENSE](LICENSE).

Commercial licensing is available for proprietary use; contact the maintainer.

## Contributing

Contributions require signing the [Individual Contributor License Agreement](CLA.md).
See [CONTRIBUTING.md](CONTRIBUTING.md).




