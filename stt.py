# py/stt/stt.py
# HYBRID STT — CPU tiny (Realtime), GPU final (Large-v3-turbo)
# Features: Total Async Loading, Stdin Mute Control, OS Default Device.
# NOTE: Device switching requires process restart.

import sys
import time
import queue
import threading
import struct
import logging
from collections import deque

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

OUTPUT_FILE = r"E:\Projects\STT\output.txt"

# ----------------------------
# Configuration
# ----------------------------
# Models
FINAL_MODEL = "large-v3-turbo"
FINAL_DEVICE = "cuda"
FINAL_COMPUTE_TYPE = "float16"

REALTIME_MODEL = "tiny.en"
REALTIME_DEVICE = "cpu"
REALTIME_CPU_COMPUTE_TYPE = "default" 

# Audio IO
RATE = 16000
CHUNK = 512
VAD_THRESHOLD = 0.4
BUFFER_SECONDS = 1.0
SILENCE_TIMEOUT = 0.8

# Realtime windowing
REALTIME_CHECK_INTERVAL = 0.1
MIN_CHUNKS_FOR_PARTIAL = 40
MAX_CHUNKS_FOR_PARTIAL = 100
REALTIME_QUEUE_SIZE = 10

# Pipes
PIPE_NAMES = {
    'first': r'\\.\pipe\FirstSTTtoMain',
    'second': r'\\.\pipe\SecondSTTtoMain'
}
MESSAGE_SIZE = 8192

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("STT")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("ctranslate2").setLevel(logging.WARNING)

# ----------------------------
# Pipe helpers
# ----------------------------
def get_pipe(name):
    try:
        return win32file.CreateFile(
            PIPE_NAMES[name],
            win32file.GENERIC_WRITE,
            0, None,
            win32file.OPEN_EXISTING,
            0, None
        )
    except Exception:
        return None

def send_to_pipe(name, text):
    if not text: return
    pipe = get_pipe(name)
    if pipe:
        try:
            encoded = text.encode('utf-8')[:MESSAGE_SIZE]
            padded = encoded + b'\x00' * (MESSAGE_SIZE - len(encoded))
            win32file.WriteFile(pipe, struct.pack(f'{MESSAGE_SIZE}s', padded))
        except Exception:
            pass

# ----------------------------
# Silero VAD Wrapper
# ----------------------------
class SileroVAD:
    def __init__(self, model):
        self.model = model

    def is_speech(self, audio_chunk):
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(audio_chunk).unsqueeze(0)
            return self.model(tensor, 16000).item()

# ----------------------------
# Main
# ----------------------------
def main():
    print(f"Initializing Hybrid STT (Realtime: {REALTIME_DEVICE}, Final: {FINAL_DEVICE})...")
    sys.stdout.flush()

    # Containers for results
    resources = {}
    warmup_audio = np.zeros(16000, dtype=np.float32)

    # ------------------------
    # Async Loading Functions
    # ------------------------
    def load_final():
        try:
            logger.info(f"Loading {FINAL_MODEL} ({FINAL_DEVICE})...")
            m = WhisperModel(FINAL_MODEL, device=FINAL_DEVICE, compute_type=FINAL_COMPUTE_TYPE)
            m.transcribe(warmup_audio, beam_size=1)
            resources['final_model'] = m
            logger.info(f"Final model ready.")
        except Exception as e:
            logger.error(f"Final Load Error: {e}")
            sys.exit(1)

    def load_realtime():
        try:
            logger.info(f"Loading {REALTIME_MODEL} ({REALTIME_DEVICE})...")
            rt_compute_type = REALTIME_CPU_COMPUTE_TYPE if REALTIME_DEVICE == "cpu" else FINAL_COMPUTE_TYPE
            m = WhisperModel(REALTIME_MODEL, device=REALTIME_DEVICE, compute_type=rt_compute_type)
            m.transcribe(warmup_audio, beam_size=1)
            resources['realtime_model'] = m
            logger.info(f"Realtime model ready.")
        except Exception as e:
            logger.error(f"RT Load Error: {e}")
            sys.exit(1)

    def load_audio_stack():
        # Load VAD and PyAudio in parallel with models
        import torch
        try:
            logger.info("Loading Silero VAD...")
            try:
                model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='local', onnx=True)
            except:
                model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
            
            resources['vad'] = SileroVAD(model)
            
            logger.info("Initializing PyAudio...")
            p_instance = pyaudio.PyAudio()
            resources['p_instance'] = p_instance

            # Find Default Device
            try:
                info = p_instance.get_default_input_device_info()
                target_idx = info['index']
                target_name = info['name']
            except: 
                raise Exception("No valid input device found")

            stream = p_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=target_idx
            )
            resources['stream'] = stream
            resources['dev_name'] = target_name
            logger.info(f"Audio Connected: {target_name}")
            
        except Exception as e:
            logger.error(f"Audio Stack Error: {e}")
            sys.exit(1)

    # ------------------------
    # Launch Threads
    # ------------------------
    threads = [
        threading.Thread(target=load_final),
        threading.Thread(target=load_realtime),
        threading.Thread(target=load_audio_stack)
    ]
    
    for t in threads: t.start()
    for t in threads: t.join()

    # Unpack Resources
    final_model = resources['final_model']
    rt_model = resources['realtime_model']
    vad = resources['vad']
    stream = resources['stream']
    p_instance = resources['p_instance']
    
    # Queues & State
    final_queue = queue.Queue()
    realtime_queue = queue.Queue(maxsize=REALTIME_QUEUE_SIZE)
    state_lock = threading.Lock()
    state = { 'speech_epoch': 0, 'finalized_epoch': -1 }
    rt_cancel = threading.Event()

    # ------------------------
    # Workers
    # ------------------------
    def final_worker():
        while True:
            try:
                epoch, t0, audio_data = final_queue.get()
                audio_float = np.concatenate(audio_data).astype(np.float32) / 32768.0
                dur_audio = len(audio_float) / RATE
                logger.info(f"[FINAL @ +{(time.time()-t0):.2f}s] Processing {dur_audio:.2f}s audio...")
                
                start = time.time()
                segments, _ = final_model.transcribe(audio_float, language="en", beam_size=5)
                text = " ".join([s.text for s in segments]).strip()
                dur_proc = time.time() - start
                t_done = time.time() - t0

                if text:
                    logger.info(f"[FINAL RESULT @ +{t_done:.2f}s] -> '{text}' (Inference: {dur_proc:.2f}s)")
                    send_to_pipe('second', text)
                else:
                    logger.info(f"[FINAL RESULT @ +{t_done:.2f}s] -> <EMPTY> (Inference: {dur_proc:.2f}s)")

                with state_lock:
                    if state['finalized_epoch'] < epoch:
                        state['finalized_epoch'] = epoch
                        rt_cancel.set()
                
                try:
                    while True:
                        realtime_queue.get_nowait()
                        realtime_queue.task_done()
                except queue.Empty: pass
            except Exception as e:
                logger.error(f"[FINAL] ERROR: {e}")
            finally:
                try: final_queue.task_done()
                except: pass

    def realtime_worker():
        while True:
            try:
                job = realtime_queue.get()
                if job is None:
                    continue

                epoch, t0, audio_data = job
                
                with state_lock:
                    if epoch != state['speech_epoch'] or state['finalized_epoch'] >= epoch or rt_cancel.is_set():
                        continue

                chunks_len = len(audio_data)
                if not (MIN_CHUNKS_FOR_PARTIAL <= chunks_len <= MAX_CHUNKS_FOR_PARTIAL):
                    continue

                audio_float = np.concatenate(audio_data).astype(np.float32) / 32768.0
                dur_audio = len(audio_float) / RATE
                logger.info(f"[REALTIME @ +{(time.time()-t0):.2f}s] Processing {dur_audio:.2f}s audio...")

                start = time.time()
                segments, _ = rt_model.transcribe(audio_float, language="en", beam_size=1)
                text = " ".join([s.text for s in segments]).strip()
                proc_time = time.time() - start
                t_done = time.time() - t0

                with state_lock:
                    if epoch != state['speech_epoch'] or state['finalized_epoch'] >= epoch or rt_cancel.is_set():
                        continue

                if text:
                    logger.info(f"[REALTIME RESULT @ +{t_done:.2f}s] -> '{text}' (Inference: {proc_time:.3f}s)")
                    send_to_pipe('first', text)
                else:
                    logger.info(f"[REALTIME RESULT @ +{t_done:.2f}s] -> <EMPTY> (Inference: {proc_time:.3f}s)")

            except Exception as e:
                logger.error(f"[REALTIME] ERROR: {e}")
            finally:
                try: realtime_queue.task_done()
                except: pass

    threading.Thread(target=final_worker, daemon=True).start()
    threading.Thread(target=realtime_worker, daemon=True).start()

    # ------------------------
    # Control Setup
    # ------------------------
    ring_buffer = deque(maxlen=int(BUFFER_SECONDS * RATE / CHUNK))
    recording_buffer = []
    
    control_lock = threading.Lock()
    control_state = { 'is_muted': False }

    def input_listener():
        while True:
            try:
                line = sys.stdin.readline()
                if not line: break
                cmd = line.strip().upper()
                if cmd == "MUTE":
                    with control_lock: control_state['is_muted'] = True
                    logger.info("Microphone MUTED via command")
                elif cmd == "UNMUTE":
                    with control_lock: control_state['is_muted'] = False
                    logger.info("Microphone UNMUTED via command")
            except: break
    
    threading.Thread(target=input_listener, daemon=True).start()

    # Recording Loop State
    is_recording = False
    silence_counter = 0
    silence_limit = int(SILENCE_TIMEOUT * (RATE / CHUNK))
    last_realtime_update = 0.0
    current_t0 = None
    last_error_msg = ""

    # Reconnect Logic (Basic fallback)
    def reset_stream_sync():
        nonlocal stream, p_instance
        if stream:
            try: stream.stop_stream(); stream.close()
            except: pass
        if p_instance:
            try: p_instance.terminate()
            except: pass
        
        p_instance = pyaudio.PyAudio()
        try:
             info = p_instance.get_default_input_device_info()
             idx = info['index']
             name = info['name']
        except: raise Exception("No device")

        stream = p_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=idx
        )
        return name

    print("STT Ready!")
    sys.stdout.flush()

    # ------------------------
    # Main Loop
    # ------------------------
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            if last_error_msg:
                logger.info("Recovered.")
                last_error_msg = ""

            # Check Mute State
            with control_lock:
                if control_state['is_muted']:
                    continue

            chunk = np.frombuffer(data, dtype=np.int16)
            prob = vad.is_speech(chunk.astype(np.float32) / 32768.0)

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
                    print(">>> [T=0.00s] SPEECH START")
                    send_to_pipe('second', "[STATUS:RECORDING]")
                recording_buffer.append(chunk)
            else:
                if is_recording:
                    recording_buffer.append(chunk)
                    silence_counter += 1
                    if silence_counter > silence_limit:
                        with state_lock: epoch = state['speech_epoch']
                        is_recording = False
                        t_end = time.time() - (current_t0 or time.time())
                        print(f"<<< [T={t_end:.2f}s] SPEECH END")
                        final_queue.put((epoch, current_t0, list(recording_buffer)))
                        recording_buffer = []
                        current_t0 = None
                else:
                    ring_buffer.append(chunk)

            if is_recording and (time.time() - last_realtime_update > REALTIME_CHECK_INTERVAL):
                with state_lock:
                    ep = state['speech_epoch']
                    fin = state['finalized_epoch']
                
                chunks_len = len(recording_buffer)
                if fin < ep and MIN_CHUNKS_FOR_PARTIAL <= chunks_len <= MAX_CHUNKS_FOR_PARTIAL:
                    try:
                        realtime_queue.put_nowait((ep, current_t0, list(recording_buffer)))
                    except queue.Full:
                        try:
                            realtime_queue.get_nowait()
                            realtime_queue.task_done()
                            realtime_queue.put_nowait((ep, current_t0, list(recording_buffer)))
                        except: pass
                last_realtime_update = time.time()

        except (OSError, IOError) as e:
            err_str = str(e)
            if err_str != last_error_msg:
                logger.warning(f"Stream error: {err_str}. Retrying...")
                last_error_msg = err_str
            
            time.sleep(0.5)
            try:
                dev_name = reset_stream_sync()
                logger.info(f"Switched to: {dev_name}")
                last_error_msg = ""
            except: pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break

    if stream: stream.stop_stream(); stream.close()
    if p_instance: p_instance.terminate()

if __name__ == '__main__':
    main()
