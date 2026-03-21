import audioop
import json
import time
from pathlib import Path

import numpy as np
import pyaudio

ALIAS_DEVICE_NAMES = {
    'Primary Sound Capture Driver',
    'Microsoft Sound Mapper',
    'Microsoft Sound Mapper - Input',
    'Microsoft Sound Mapper - Output',
}


def _normalize_name(value: str) -> str:
    return str(value or '').strip().lower()


def _clean_profile(profile: dict | None) -> dict | None:
    if not isinstance(profile, dict):
        return None
    name = str(profile.get('name') or '').strip()
    if not name:
        return None
    host_api = profile.get('host_api')
    if host_api is not None:
        try:
            host_api = int(host_api)
        except Exception:
            host_api = None
    return {'name': name, 'host_api': host_api}


def load_profiles(path: str | Path) -> dict:
    result = {'inputs': [], 'outputs': []}
    profile_path = Path(path)
    if not profile_path.exists():
        return result
    try:
        data = json.loads(profile_path.read_text(encoding='utf-8'))
    except Exception:
        return result
    for key in ('inputs', 'outputs'):
        items = []
        for item in data.get(key, []):
            cleaned = _clean_profile(item)
            if cleaned:
                items.append(cleaned)
        result[key] = items
    return result


def save_profiles(path: str | Path, profiles: dict):
    profile_path = Path(path)
    cleaned = {
        'inputs': [p for p in (_clean_profile(x) for x in profiles.get('inputs', [])) if p],
        'outputs': [p for p in (_clean_profile(x) for x in profiles.get('outputs', [])) if p],
    }
    tmp = profile_path.with_suffix(profile_path.suffix + '.tmp')
    tmp.write_text(json.dumps(cleaned, indent=2), encoding='utf-8')
    tmp.replace(profile_path)


def profile_from_info(info: dict) -> dict:
    return {
        'name': str(info.get('name', '')).strip(),
        'host_api': info.get('hostApi'),
    }


def profile_matches(info: dict, profile: dict) -> bool:
    cleaned = _clean_profile(profile)
    if not cleaned:
        return False
    if _normalize_name(info.get('name', '')) != _normalize_name(cleaned['name']):
        return False
    host_api = cleaned.get('host_api')
    if host_api is not None and info.get('hostApi') != host_api:
        return False
    return True


def matches_any_profile(info: dict, profiles: list[dict]) -> bool:
    return any(profile_matches(info, profile) for profile in profiles)


def host_api_names(p_instance) -> dict[int, str]:
    result = {}
    try:
        for i in range(p_instance.get_host_api_count()):
            api_info = p_instance.get_host_api_info_by_index(i)
            result[i] = str(api_info.get('name', '')).strip()
    except Exception:
        return {}
    return result


def describe_device(info: dict, api_names: dict[int, str] | None = None) -> str:
    if not info:
        return '<missing>'
    api_names = api_names or {}
    name = str(info.get('name', '')).strip() or '<unnamed>'
    index = info.get('index', '?')
    host_api = info.get('hostApi')
    host_label = api_names.get(int(host_api), '') if host_api is not None else ''
    host_text = f'{host_label}#{host_api}' if host_label else str(host_api)
    default_rate = info.get('defaultSampleRate')
    if default_rate:
        return f'{name} [idx={index} host={host_text} default_rate={default_rate:.0f}]'
    return f'{name} [idx={index} host={host_text}]'


def available_input_devices(p_instance, *, include_alias: bool = False) -> list[dict]:
    devices = []
    seen = set()
    for i in range(p_instance.get_device_count()):
        info = p_instance.get_device_info_by_index(i)
        if info.get('maxInputChannels', 0) <= 0:
            continue
        name = str(info.get('name', '')).strip()
        key = (name, info.get('hostApi'))
        if not include_alias and name in ALIAS_DEVICE_NAMES:
            continue
        if key in seen:
            continue
        seen.add(key)
        devices.append(info)
    return devices


def available_output_devices(p_instance, *, include_alias: bool = False) -> list[dict]:
    devices = []
    seen = set()
    for i in range(p_instance.get_device_count()):
        info = p_instance.get_device_info_by_index(i)
        if info.get('maxOutputChannels', 0) <= 0:
            continue
        name = str(info.get('name', '')).strip()
        key = (name, info.get('hostApi'))
        if not include_alias and name in ALIAS_DEVICE_NAMES:
            continue
        if key in seen:
            continue
        seen.add(key)
        devices.append(info)
    return devices


def order_devices_by_profiles(infos: list[dict], profiles: list[dict]) -> list[dict]:
    ordered = []
    seen = set()
    for profile in profiles:
        for info in infos:
            if profile_matches(info, profile):
                idx = int(info.get('index', -1))
                if idx not in seen:
                    seen.add(idx)
                    ordered.append(info)
                break
    return ordered


def _candidate_rates(info: dict, stt_rate: int) -> list[int]:
    rates = []
    for value in (stt_rate, info.get('defaultSampleRate')):
        try:
            rate = int(round(float(value)))
        except Exception:
            continue
        if rate > 0 and rate not in rates:
            rates.append(rate)
    return rates or [int(stt_rate)]


class InputSession:
    def __init__(self, p_instance, info: dict, stream, capture_rate: int, stt_rate: int, chunk_frames: int):
        self.p_instance = p_instance
        self.info = info
        self.stream = stream
        self.capture_rate = int(capture_rate)
        self.stt_rate = int(stt_rate)
        self.chunk_frames = max(1, int(chunk_frames))
        self._capture_frames = max(1, int(round(self.chunk_frames * self.capture_rate / self.stt_rate)))
        self._rate_state = None
        self._pending = b''

    def read_chunk(self) -> tuple[bytes, float]:
        target_bytes = self.chunk_frames * 2
        max_read_ms = 0.0
        while len(self._pending) < target_bytes:
            started = time.perf_counter()
            raw = self.stream.read(self._capture_frames, exception_on_overflow=False)
            read_ms = (time.perf_counter() - started) * 1000.0
            max_read_ms = max(max_read_ms, read_ms)
            if self.capture_rate != self.stt_rate:
                converted, self._rate_state = audioop.ratecv(
                    raw,
                    2,
                    1,
                    self.capture_rate,
                    self.stt_rate,
                    self._rate_state,
                )
            else:
                converted = raw
            self._pending += converted
            if not raw and not converted:
                break
        if len(self._pending) < target_bytes:
            raise RuntimeError(
                f'input underrun: expected {target_bytes} bytes after resample, got {len(self._pending)}'
            )
        data = self._pending[:target_bytes]
        self._pending = self._pending[target_bytes:]
        return data, max_read_ms

    def close(self):
        try:
            self.stream.stop_stream()
        except Exception:
            pass
        try:
            self.stream.close()
        except Exception:
            pass


def open_input_session(
    p_instance,
    info: dict,
    *,
    stt_rate: int,
    chunk_frames: int,
    probe_reads: int,
    probe_limit_ms: float,
) -> tuple[InputSession, float, int]:
    failures = []
    for rate in _candidate_rates(info, stt_rate):
        try:
            p_instance.is_format_supported(
                rate,
                input_device=int(info['index']),
                input_channels=1,
                input_format=pyaudio.paInt16,
            )
        except Exception as exc:
            failures.append(f'{rate}Hz unsupported: {exc}')
            continue
        stream = None
        try:
            stream = p_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=int(rate),
                input=True,
                frames_per_buffer=max(1, int(round(chunk_frames * int(rate) / int(stt_rate)))),
                input_device_index=int(info['index']),
            )
            session = InputSession(
                p_instance,
                info,
                stream,
                capture_rate=int(rate),
                stt_rate=int(stt_rate),
                chunk_frames=int(chunk_frames),
            )
            max_read_ms = 0.0
            last_chunk = b''
            for _ in range(max(1, int(probe_reads))):
                probe_data, read_ms = session.read_chunk()
                max_read_ms = max(max_read_ms, read_ms)
                last_chunk = probe_data
            rms = 0
            if last_chunk:
                probe_chunk = np.frombuffer(last_chunk, dtype=np.int16).astype(np.float32)
                if len(probe_chunk):
                    rms = int(np.sqrt(np.mean(probe_chunk ** 2)))
            if max_read_ms > float(probe_limit_ms):
                raise RuntimeError(
                    f'probe read too slow ({max_read_ms:.0f}ms > {float(probe_limit_ms):.0f}ms)'
                )
            return session, max_read_ms, rms
        except Exception as exc:
            failures.append(f'{rate}Hz failed: {exc}')
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
    raise RuntimeError('; '.join(failures) if failures else 'no usable capture format found')


def probe_output_device(p_instance, info: dict, probe_format: tuple[int, int] | None = None) -> tuple[bool, str]:
    probe_rate = int(probe_format[0]) if probe_format else 44100
    probe_channels = int(probe_format[1]) if probe_format else 2
    try:
        p_instance.is_format_supported(
            probe_rate,
            output_device=int(info['index']),
            output_channels=probe_channels,
            output_format=pyaudio.paFloat32,
        )
    except Exception as exc:
        return False, str(exc)
    stream = None
    try:
        stream = p_instance.open(
            format=pyaudio.paFloat32,
            channels=probe_channels,
            rate=probe_rate,
            output=True,
            output_device_index=int(info['index']),
            frames_per_buffer=256,
        )
        stream.write((b'\x00\x00\x00\x00') * probe_channels * 32)
        return True, 'ok'
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
