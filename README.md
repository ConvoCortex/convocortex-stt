# convocortex-stt

Ambient speech-to-text for desktop use: leave it running, wake it when needed, speak naturally, and have final text routed to files, clipboard, a live buffer, NATS, or directly into the active cursor.

It is built for daily background use rather than one-shot dictation: wake/sleep modes, VAD-gated utterances, realtime partials, accurate finals, device cycling, audio feedback, hotkeys, and a simple integration boundary over NATS.

## Quick start

```bash
git clone https://github.com/ConvoCortex/convocortex-stt
cd convocortex-stt
uv sync
uv run python stt.py
```

On Windows, if you want it to start hidden once the runtime is ready:

```bash
uv run python stt.py --background
```

Before first real use, open `config.toml` and set the bits you actually care about:
- `models.final_device` / `models.realtime_device`
- `sleep_wake.wake_word`
- `output.type_at_cursor.enabled`
- `hotkeys.sleep_toggle`

Focused output-mode presets live in:
- `presets/output-modes/direct-cursor.toml`
- `presets/output-modes/draft-buffer.toml`
- `presets/output-modes/cursor-with-clipboard-last.toml`

## Why this repo is useful

- It is hands-free. VAD starts and ends utterances from speech and silence, not button presses.
- It separates low-latency partials from higher-accuracy finals, which makes it usable both for responsiveness and for actual text output.
- It already handles the ugly desktop details: wake/sleep state, audio feedback, hotkeys, device switching, reconnect behavior, and persisted runtime state.
- It works as a standalone local tool and also as a component inside a larger voice system through NATS.

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
- type-at-cursor undo words
- input device cycle words
- output device cycle words
- output mode selection words
- console show/hide words
- enter trigger words
- file-buffer release words
- file-buffer clear words

Commands are matched from partials (fast path) and finals (reliable path), using normalized exact matching for configured command phrases.
If a command has no configured `words`, it is treated as disabled and is not loaded.

For a proper, richer voice command engine, use the emitted NATS events and implement command logic externally.

### Practical device workflow

One useful pattern is to keep two approved input devices and switch between them by voice:

- use the desk microphone while you are at the PC, because it can sound better and it does not force bluetooth headphones into hands-free mic mode
- say `input` when you want to walk away and switch to the headset microphone
- say `input` again when you come back and want the desk microphone again

This is a good fit for the built-in input-device cycle command because it lets you move between a higher-quality stationary mic and a mobile headset mic without touching the keyboard.

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
  - output mode cycle
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

Edit `config.toml` directly, or replace it with one of the included preset configs and then adjust it.

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
- `models.no_speech_threshold` / `models.log_prob_threshold`: Whisper-side silence / low-confidence rejection for reducing spurious transcripts
- `audio`: VAD behavior + silence timeout + preferred input
- `startup`: whether startup restores last runtime state or uses config defaults for output mode and devices
- `realtime`: partial cadence/window limits
- `filters.ignored_exact_phrases`: drops matching utterances entirely before partial/final output is shown or dispatched
- `filters.disfluency_words`: strips configured filler/profanity words from otherwise valid transcripts, then cleans spacing/punctuation
- `logging`: toggleable debug traces written to a file for freeze/debug analysis
- `output`: handler toggles and paths, including editable `file_buffer`
- `voice_commands`: built-in convenience command words
- `sleep_wake`: wake word backend + stop words + startup mode
- `hotkeys`: optional runtime controls
- `nats`: URL/subjects + enable toggle

### Output workflow sets

The repo currently has three output workflows that make sense as first-class sets:

- `direct-cursor`: finalized text types into the active app immediately. Buffering and clipboard mirroring are off.
- `draft-buffer`: keep the buffer workflow explicit. Typical loop is `typing` to disable direct typing, `clear`, speak, `buffer`, repeat.
- `cursor-with-clipboard-last`: type immediately, but also mirror the last finalized utterance into the clipboard so you can manually paste/recover it.

Those focused presets are provided as small snippets under `presets/output-modes/`. They describe the built-in runtime output modes.

At runtime, output mode is treated as state, not as a config rewrite. Startup behavior is controlled by `startup.output_mode_source`:
- hotkey `shift+f9` cycles output modes
- exact voice commands `default mode`, `cursor mode`, `draft mode`, and `clipboard mode` select output modes directly

Startup source controls:
- `startup.output_mode_source = "state"` restores the last selected output mode from `state.json`
- `startup.output_mode_source = "config"` always starts in `config-default`
- `startup.input_device_source = "state"` restores the last runtime input device when possible
- `startup.input_device_source = "config"` starts from `audio.input_device`, otherwise OS default input
- `startup.output_device_source = "state"` restores the last runtime feedback output device when possible
- `startup.output_device_source = "config"` starts from `feedback.output_device`, otherwise OS default output

The built-in runtime modes are:
- `config-default`: whatever `config.toml` says for the output handlers
- `direct-cursor`
- `draft-buffer`
- `cursor-with-clipboard-last`

`clipboard_accumulate` is still available, but it is more niche and less coherent as a default workflow than the three sets above.

### Type-at-cursor undo

If cursor typing is the active output path, the built-in exact voice command `undo` can send a best-effort undo action to the active app.

Relevant settings:
- `voice_commands.undo.words`: exact phrases that trigger the undo action
- `output.type_at_cursor.undo_mode`: `ctrl+z`, `backspace`, or `off`
- `output.type_at_cursor.undo_backspace_count`: number of backspaces to send when using `backspace` mode

This is intentionally app-dependent. `ctrl+z` works in many GUI text fields, while terminals and some apps may require `backspace` mode or `off`.

### File buffer workflow

If `output.file_buffer.enabled = true`, each final transcription is appended to `output.file_buffer.path`.

This gives you a plain text working buffer you can keep open in an editor, adjust manually, and then inject into the active cursor later with an exact voice command such as `buffer`.

Relevant settings:
- `output.file_buffer.separator`: separator inserted between finalized utterances
- `output.file_buffer.clear_after_release`: clear the file after the `buffer` command types it
- `output.file_buffer.reset_after_each_message`: clear the existing buffer before writing each new finalized utterance
- `output.file_buffer.undo_history_limit`: in-memory cap for one-step-at-a-time buffer undos triggered by the exact `undo` voice command while file-buffer mode is active
- `voice_commands.buffer_release.words`: exact phrases that trigger release
- `voice_commands.buffer_release.press_enter_after`: optionally press Enter after typing the buffer
- `voice_commands.buffer_clear.words`: exact phrases that clear the buffer without typing it

In `draft-buffer` mode, `undo` rewinds `buffer.txt` one saved state at a time instead of sending a cursor undo. The default history cap is `10` buffer states and is not persisted across restarts.

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
{"cmd": "output_mode_cycle"}
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

This repository uses the same general stack as the RealTimeSTT ecosystem, but the runtime loop in this project is custom and lives in `stt.py`. The main transcription path here is built around PyAudio, Silero VAD, and faster-whisper/ctranslate2.

## Responsible use

This project is a general-purpose speech-to-text tool for local and networked desktop workflows.

You are responsible for using it lawfully, including complying with rules that may apply to:
- audio recording or monitoring
- notice and consent
- handling of personal, confidential, or sensitive data

Do not use this software to violate privacy, confidentiality, platform terms, or applicable law.

This README is not legal advice. Warranty and liability terms for the software itself are governed by the project license.

## License

convocortex-stt is licensed under **GNU Affero General Public License v3.0** (AGPL-3.0).
See [LICENSE](LICENSE).

Commercial licensing is available for proprietary use; contact the maintainer.

## Contributing

Contributions require signing the [Individual Contributor License Agreement](CLA.md).
See [CONTRIBUTING.md](CONTRIBUTING.md).




