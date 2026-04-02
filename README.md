# convocortex-stt

Ambient speech-to-text for desktop use: leave it running, wake it when needed, speak naturally, and route text into a working buffer, the active cursor, files, or NATS.

It is built for daily hands-free use rather than one-shot dictation: wake/sleep modes, VAD-gated utterances, pluggable realtime/final backends, device cycling, audio feedback, hotkeys, and a simple integration boundary over NATS.

## Quick start

```bash
git clone https://github.com/ConvoCortex/convocortex-stt
cd convocortex-stt
uv sync
uv run python stt.py
```

On Windows, console visibility during and after startup is controlled by
`startup.console_startup_mode` in `config.toml`.


Before first real use, open `config.toml` and set the bits you actually care about:
- `models.final_backend` / `models.realtime_backend`
- `models.final_device` / `models.realtime_device`
- `sleep_wake.wake_word`
- `output.file_buffer.enabled`
- `hotkeys.sleep_toggle`

On the first real startup, the app runs interactive device setup so you can
pick the input and output devices that should participate in runtime cycling.

Focused output-mode presets live in:
- `presets/output-modes/direct-cursor.toml`
- `presets/output-modes/draft-buffer.toml`

## Recommended workflow

The intended workflow is `draft-buffer`.

Keep [`buffer.txt`](buffer.txt) open in an editor such as VS Code, speak naturally into the buffer, adjust the text if needed, and then say `enter` when you want to release the buffer into the active app and submit it.

That gives you a practical loop:
- speak into the live buffer
- glance at or edit `buffer.txt`
- say `enter` to paste the buffer and press Enter

If you want immediate direct typing instead, switch to `direct-cursor` mode at runtime with the output-mode hotkey or the exact voice command `cursor mode`.

## Why this repo is useful

- It is hands-free. VAD starts and ends utterances from speech and silence, not button presses.
- It supports separate realtime and final backends, but the intended path here is NVIDIA Parakeet for both.
- Its buffer-first workflow is practical for real desktop work instead of forcing immediate text emission all the time.
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

### 4) Dual-backend pipeline

- Realtime backend emits **partial** text while you are speaking
- Final backend emits **final** text after pause/silence

Each side can use a different backend and model. In practice, Parakeet is the recommended path here for both realtime and final transcription. It is accurate enough on partials that voice commands can trigger almost immediately, while finals still wait for pause detection before emission.

Partials are primarily for responsiveness and command latency, not for high-accuracy text output.

## Voice commands

Built-in voice commands are intentionally minimal and convenience-oriented.

Current built-in command types:
- sleep/stop words
- type-at-cursor toggle words
- rewind/repeat words
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

One useful pattern is to keep a small approved input cycle and switch it by voice:

- local desk microphone
- optional bluetooth/headset microphone
- optional virtual input such as `B1` from a remote-audio bridge

Then say `input` to move between those options without touching the keyboard.

This is especially useful when you want to compare microphones in the same environment or swap between local-PC mode and a remote-phone microphone path.

## Features

- No GUI
- Hands-free transcription on pause
- Sleep/working mode with wake word + stop words
- Realtime partial transcriptions during speech
- High-accuracy final transcriptions after pauses
- Independent realtime/final backend selection with faster-whisper and Parakeet
- Built-in audio feedback sounds (on/off/final + silence loop for bluetooth audio issues)
- Simple built-in voice command actions
- Draft-buffer workflow for reviewing/editing before release
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
- Friendly to virtual-audio and remote-audio bridge setups

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

Parakeet support is included in the normal `uv sync` install.

Example `config.toml` model section for Parakeet:

```toml
[models]
final_backend    = "parakeet"
realtime_backend = "parakeet"
final            = "nvidia/parakeet-tdt-0.6b-v3"
realtime         = "nvidia/parakeet-tdt-0.6b-v2"
final_device     = "cuda"
realtime_device  = "cuda"
language         = ""
```

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


On Windows, the default `hotkeys.console_toggle = "shift+f8"` hides/shows that same console window. A tray icon is also created so you can show/hide the console or exit STT without relying on the taskbar state of the console host. When you bring the console back, `Ctrl+C` kills the process as usual.
If you want the default backtick sleep hotkey to be intercepted instead of typed into the active app, set `hotkeys.sleep_toggle_suppress = true`.

## Windows Startup Shortcut

If you want this to come up automatically after you log in on Windows, using the
console mode from `config.toml`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-startup-shortcut.ps1
```

That creates a shortcut in:

```text
C:\Users\user\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

The shortcut launches the repo's Windows launcher, which reads
`startup.console_startup_mode` from `config.toml` and then starts `stt.py`
accordingly.

This is the simpler Windows logon path and fits this app better because it relies on an interactive desktop session for global hotkeys and for restoring the console window to the foreground.

To remove the shortcut:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall-startup-shortcut.ps1
```

## Configuration

All runtime settings live in `config.toml` and are loaded at startup.

Important areas:
- `models`: backend choice plus realtime/final models and device choices
- `models.no_speech_threshold` / `models.log_prob_threshold`: Whisper-only silence / low-confidence rejection for reducing spurious transcripts
- `audio`: VAD behavior + silence timeout + preferred input, especially `vad_threshold`, `vad_end_threshold`, and `min_speech_duration_ms` for speech-vs-noise gating
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

`feedback.silence_keepalive_mode = "always"` keeps `sounds/silence.ogg` looping continuously on the feedback output device. Set it to `"off"` to disable that keepalive.

### Output workflow sets

The repo currently has two output workflows that make sense as first-class sets:

- `draft-buffer`: the recommended mode. Keep a visible working buffer, then explicitly release it. Typical loop is `clear`, speak, edit if needed, `enter`.
- `direct-cursor`: finalized text types into the active app immediately. Buffering and clipboard mirroring are off.

Those focused presets are provided as small snippets under `presets/output-modes/`. They describe the built-in runtime output modes.

At runtime, output mode is treated as state, not as a config rewrite. Startup behavior is controlled by `startup.output_mode_source`:
- hotkey `shift+f9` cycles output modes
- exact voice commands `default mode`, `cursor mode`, and `draft mode` select output modes directly

Startup source controls:
- `startup.output_mode_source = "state"` restores the last selected output mode from `state.json`
- `startup.output_mode_source = "config"` always starts in the configured default mode (`direct-cursor` or `draft-buffer`)
- `startup.input_device_source = "state"` restores the last runtime input device when possible
- `startup.input_device_source = "config"` starts from `audio.input_device`, otherwise OS default input
- `startup.output_device_source = "state"` restores the last runtime feedback output device when possible
- `startup.output_device_source = "config"` starts from `feedback.output_device`, otherwise OS default output
- `startup.console_startup_mode = "foreground"` keeps the console visible during and after startup; `"background"` hides it during and after startup

The built-in runtime modes are:
- `direct-cursor`
- `draft-buffer`

### Rewind And Repeat

If cursor typing is the active output path, the built-in exact voice command `rewind` can send a best-effort revert action to the active app.

Relevant settings:
- `voice_commands.rewind.words`: exact phrases that trigger the rewind action
- `voice_commands.repeat.words`: exact phrases that repeat the last emitted chunk
- `output.type_at_cursor.revert_mode`: `ctrl+z`, `backspace`, or `off`
- `output.type_at_cursor.revert_backspace_count`: number of backspaces to send when using `backspace` mode

This is intentionally app-dependent. `ctrl+z` works in many GUI text fields, while terminals and some apps may require `backspace` mode or `off`. In `draft-buffer` mode, `rewind` restores the last released buffer into the current buffer and also sends the configured cursor revert action. `repeat` re-sends the last released buffer without requiring it to still be present in `buffer.txt`.

### File buffer workflow

If `output.file_buffer.enabled = true`, each final transcription is appended to `output.file_buffer.path`.

This gives you a plain text working buffer you can keep open in an editor, adjust manually, and then inject into the active cursor later with an exact voice command such as `buffer` or `enter`.

Relevant settings:
- `output.file_buffer.separator`: separator inserted between finalized utterances
- `output.file_buffer.clear_after_release`: clear the file after the `buffer` command releases it
- `output.file_buffer.reset_after_each_message`: clear the existing buffer before writing each new finalized utterance
- `output.file_buffer.undo_history_limit`: in-memory cap for one-step-at-a-time internal buffer history entries
- `output.file_buffer.release_method`: `paste_preserve_clipboard` for fast paste with clipboard restore, or `type_keys` for the older direct typing path
- `output.file_buffer.clipboard_restore_delay_ms`: how long to wait after paste before restoring clipboard contents
- `output.file_buffer.clipboard_open_retry_count` / `output.file_buffer.clipboard_open_retry_delay_ms`: retry behavior for busy clipboard cases
- `output.file_buffer.post_paste_enter_delay_ms`: brief wait between paste and Enter when a buffer release also submits
- `voice_commands.buffer_release.words`: exact phrases that trigger buffer release, and are also recognized at the end of an utterance while draft-buffer mode is active
- `voice_commands.buffer_clear.words`: exact phrases that clear the buffer without typing it

If the file buffer is active, the built-in exact voice command `enter` releases the current buffer and then presses Enter. This makes `draft-buffer` practical for chat boxes, terminals, and other submit-oriented text fields.

The default history cap is `10` buffer states and is not persisted across restarts.

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

The useful debug points are in [`stt.py`](stt.py) around PyAudio, Silero VAD, and the transcription backend adapter layer.

## Remote audio bridge workflow

This project also works well with a remote-audio bridge setup based on:
- a routed or overlay network that gives the remote device access to the PC
- a Mumble-compatible voice link
- virtual audio devices such as Voicemeeter

That lets you treat a phone or remote client as another microphone/speaker endpoint while keeping STT local on the desktop.

See [docs/remote-audio-bridge.md](docs/remote-audio-bridge.md).

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
{"type": "system",  "event": "startup", "device": "Microphone (USB)", "models": {"final": "large-v3-turbo", "final_backend": "whisper", "realtime": "tiny.en", "realtime_backend": "whisper"}}
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
- Parakeet models depend on NVIDIA NeMo. NVIDIA’s Parakeet model cards list Linux as the supported OS, so Windows use is still best-effort even though it works well enough for this repo's workflow.
- Linux hotkeys may require input permissions (group/root setup).
- macOS hotkeys may require Accessibility permissions.

## Acknowledgments

This repository is built around PyAudio, Silero VAD, pluggable transcription backends, and a local-first desktop workflow.

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





