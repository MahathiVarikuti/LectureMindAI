"""Speech-to-text with faster-whisper (CTranslate2; much faster than openai-whisper on CPU)."""
import subprocess
import time
from functools import lru_cache

import numpy as np

from app.config import settings
from app.errors import AppError


@lru_cache(maxsize=1)
def get_whisper_model():
    from faster_whisper import WhisperModel

    # int8 quantisation keeps memory and CPU time low on machines without a GPU.
    return WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")


SAMPLE_RATE = 16000  # Whisper's native rate


def decode_audio(path: str) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 with ffmpeg.

    faster-whisper would otherwise decode with PyAV, whose newest releases raise
    `open() got an unexpected keyword argument 'metadata_errors'` with faster-whisper 1.2.x.
    ffmpeg is already a hard requirement of the project, so use it directly.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
            capture_output=True, timeout=settings.whisper_timeout,
        )
    except FileNotFoundError as exc:
        raise AppError(500, "ffmpeg is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise AppError(504, "Decoding the audio timed out; try a shorter recording.") from exc
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode(errors="replace").strip()[:200] or "no audio stream found"
        raise AppError(400, f"Could not read audio from that file: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe_audio(path: str) -> list[dict]:
    """Return [{text, start, end}]. Enforces a wall-clock budget (WHISPER_TIMEOUT): the
    segment generator does the real work lazily, so we check the deadline as we iterate."""
    deadline = time.monotonic() + settings.whisper_timeout
    try:
        # beam_size=1 (greedy) is ~2-3x faster on CPU with a small quality cost.
        audio = decode_audio(path)
        segments, _info = get_whisper_model().transcribe(audio, beam_size=1, vad_filter=True)
        results = []
        for seg in segments:
            if time.monotonic() > deadline:
                raise AppError(504, "Transcription timed out; try a shorter recording or a smaller "
                                    "WHISPER_MODEL.")
            text = seg.text.strip()
            if text:
                results.append({"text": text, "start": float(seg.start), "end": float(seg.end)})
        return results
    except AppError:
        raise
    except Exception as exc:  # decoding / model-load failures
        raise AppError(500, f"Transcription failed: {exc}") from exc
