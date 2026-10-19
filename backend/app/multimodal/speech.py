"""Local ASR only. No chat, tools, retrieval, or command execution lives here."""
from __future__ import annotations

import asyncio
import io
import logging
import multiprocessing as mp
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings


class SpeechError(Exception):
    def __init__(self, code: str, status: int = 422):
        self.code, self.status = code, status
        super().__init__(code)


FORMATS = {
    "audio/webm": "matroska",
    "audio/ogg": "ogg",
    "audio/mp4": "mov",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


def validate_audio(data: bytes, mime: str) -> str:
    """Do not trust a browser-supplied MIME type or filename."""
    fmt = FORMATS.get(mime)
    if fmt is None:
        raise SpeechError("UNSUPPORTED_AUDIO_FORMAT", 415)
    if not data:
        raise SpeechError("EMPTY_AUDIO")
    signatures = {
        "matroska": data.startswith(b"\x1aE\xdf\xa3"),
        "ogg": data.startswith(b"OggS"),
        "mov": len(data) >= 12 and data[4:8] == b"ftyp",
        "wav": data.startswith(b"RIFF") and data[8:12] == b"WAVE",
    }
    if not signatures[fmt]:
        raise SpeechError("INVALID_AUDIO")
    return fmt


def decode_audio(data: bytes, fmt: str, max_seconds: int):
    """Decode incrementally, bounding PCM samples even if metadata lies.

    Force a local container demuxer; never accept playlists, paths, or URLs.
    Browser WebM often omits duration metadata, so count resampled samples.
    """
    import av
    import numpy as np

    limit = max_seconds * 16000
    chunks, samples = [], 0
    try:
        with av.open(io.BytesIO(data), mode="r", format=fmt) as container:
            if len(container.streams.audio) != 1 or container.streams.video:
                raise SpeechError("INVALID_AUDIO")
            stream = container.streams.audio[0]
            if stream.codec_context.channels > 2 or stream.codec_context.sample_rate > 96000:
                raise SpeechError("INVALID_AUDIO")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

            def collect(frames):
                nonlocal samples
                for frame in frames:
                    array = frame.to_ndarray().reshape(-1)
                    samples += array.size
                    if samples > limit:
                        raise SpeechError("AUDIO_TOO_LONG")
                    chunks.append(array)

            for frame in container.decode(audio=0):
                frame.pts = None
                collect(resampler.resample(frame))
            collect(resampler.resample(None))
    except SpeechError:
        raise
    except Exception:
        raise SpeechError("INVALID_AUDIO") from None
    if samples < 1600:
        raise SpeechError("EMPTY_AUDIO")
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    if not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 0.001:
        raise SpeechError("NO_SPEECH")
    return audio


def speech_worker(connection, model_path: str, language: str, cpu_threads: int, max_seconds: int):
    """Spawn-safe entry point. Model is lazy and reused until shutdown/timeout."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    logging.getLogger("faster_whisper").disabled = True
    model = None
    try:
        while True:
            data, fmt = connection.recv()
            started = time.perf_counter()
            try:
                audio = decode_audio(data, fmt, max_seconds)
                init_seconds = 0.0
                if model is None:
                    if not (Path(model_path) / "model.bin").is_file():
                        raise SpeechError("ASR_MODEL_UNAVAILABLE", 503)
                    try:
                        from faster_whisper import WhisperModel

                        init_started = time.perf_counter()
                        model = WhisperModel(
                            model_path, device="cpu", compute_type="int8",
                            cpu_threads=cpu_threads, num_workers=1, local_files_only=True,
                        )
                        init_seconds = time.perf_counter() - init_started
                    except Exception:
                        raise SpeechError("ASR_MODEL_UNAVAILABLE", 503) from None
                inference_started = time.perf_counter()
                segments, _ = model.transcribe(
                    audio, language=language, task="transcribe", beam_size=1,
                    condition_on_previous_text=False, vad_filter=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                if not text:
                    raise SpeechError("NO_SPEECH")
                if len(text) > 50000:
                    raise SpeechError("TRANSCRIPT_TOO_LONG")
                result = {
                    "text": text, "language": language, "duration_seconds": len(audio) / 16000,
                    "model_init_seconds": init_seconds,
                    "transcription_seconds": time.perf_counter() - inference_started,
                    "processing_seconds": time.perf_counter() - started,
                }
                connection.send((True, result))
            except SpeechError as exc:
                connection.send((False, (exc.code, exc.status)))
            except ImportError:
                connection.send((False, ("ASR_MODEL_UNAVAILABLE", 503)))
            except Exception:
                # Neither input audio, transcript, nor exception details go into logs/HTTP.
                connection.send((False, ("TRANSCRIPTION_FAILED", 503)))
            finally:
                data = None
                audio = None
                result = None
                text = None
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class SpeechService:
    def __init__(self, settings):
        self.settings = settings
        self._busy = threading.Lock()
        self._process = None
        self._connection = None

    def close(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            self._process.close()
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _run(self, data: bytes, fmt: str):
        try:
            if self._process is None or not self._process.is_alive():
                self.close()
                context = mp.get_context("spawn")
                self._connection, child = context.Pipe()
                self._process = context.Process(
                    target=speech_worker,
                    args=(child, str(self.settings.voice_model_path.resolve()),
                          self.settings.voice_language, self.settings.voice_cpu_threads,
                          self.settings.voice_max_duration_seconds),
                    daemon=True,
                )
                self._process.start()
                child.close()
            self._connection.send((data, fmt))
            if not self._connection.poll(self.settings.voice_timeout_seconds):
                self.close()
                raise SpeechError("TRANSCRIPTION_TIMEOUT", 504)
            success, result = self._connection.recv()
            if not success:
                raise SpeechError(*result)
            return result
        except SpeechError:
            raise
        except Exception:
            self.close()
            raise SpeechError("TRANSCRIPTION_FAILED", 503) from None
        finally:
            self._busy.release()

    async def transcribe(self, data: bytes, mime: str):
        fmt = validate_audio(data, mime)
        if not self._busy.acquire(blocking=False):
            raise SpeechError("VOICE_BUSY", 429)
        # The thread retains the lock after HTTP cancellation until work actually ends.
        return await asyncio.to_thread(self._run, data, fmt)


@lru_cache(maxsize=1)
def get_speech_service() -> SpeechService:
    return SpeechService(get_settings())
