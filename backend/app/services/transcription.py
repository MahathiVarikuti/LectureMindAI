"""Speech-to-text with faster-whisper (CTranslate2; much faster than openai-whisper on CPU)."""
import time
from functools import lru_cache

from app.config import settings
from app.errors import AppError


@lru_cache(maxsize=1)
def get_whisper_model():
    from faster_whisper import WhisperModel

    # int8 quantisation keeps memory and CPU time low on machines without a GPU.
    return WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")


def transcribe_audio(path: str) -> list[dict]:
    """Return [{text, start, end}]. Enforces a wall-clock budget (WHISPER_TIMEOUT): the
    segment generator does the real work lazily, so we check the deadline as we iterate."""
    deadline = time.monotonic() + settings.whisper_timeout
    try:
        # beam_size=1 (greedy) is ~2-3x faster on CPU with a small quality cost.
        segments, _info = get_whisper_model().transcribe(str(path), beam_size=1, vad_filter=True)
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
