# convocortex-stt

Speech-to-text engine with real-time partial results and final high-accuracy transcription. Uses a dual-model approach: a fast CPU model for live partial results during speech, and a GPU model for accurate final output after silence.

Output is written to a file and Windows named pipes. NATS event output is planned.

## How it works

- **Silero VAD** gates audio — inference only runs when speech is detected
- **tiny.en (CPU)** transcribes partial results every ~100ms during speech
- **large-v3-turbo (GPU)** transcribes the full utterance after silence
- A ring buffer captures ~1s of audio before speech starts so no words are missed
- An epoch system prevents stale partial results appearing after a final result

## Prerequisites

### 1. CUDA 12.1

Download: https://developer.nvidia.com/cuda-12-1-0-download-archive

Install with **custom installation**, selecting only the CUDA toolkit — no driver, no other components.

Add to your system PATH:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp
```

### 2. Python 3.10+

### 3. uv

```
pip install uv
```

## Installation

```bash
git clone https://github.com/you/convocortex-stt.git
cd convocortex-stt
uv sync
```

`uv sync` creates a `.venv` and installs all dependencies including the correct CUDA-enabled PyTorch build.

## Usage

```bash
uv run python stt.py
```

Or activate the venv and run directly:

```bash
.venv\Scripts\activate
python stt.py
```

### Mute control

Send `MUTE` or `UNMUTE` to stdin to control the microphone at runtime.

## Configuration

Edit the constants at the top of `stt.py`:

| Variable | Default | Description |
|---|---|---|
| `OUTPUT_FILE` | `E:\Projects\STT\output.txt` | Path for transcript output |
| `FINAL_MODEL` | `large-v3-turbo` | Whisper model for final transcription |
| `REALTIME_MODEL` | `tiny.en` | Whisper model for partial results |
| `VAD_THRESHOLD` | `0.4` | Silero VAD sensitivity (0–1) |
| `SILENCE_TIMEOUT` | `0.8` | Seconds of silence before finalizing |

## License

AGPL-3.0. See [LICENSE](LICENSE).

RealtimeSTT by [KoljaB](https://github.com/KoljaB/RealtimeSTT) is used as a dependency under the MIT license.
