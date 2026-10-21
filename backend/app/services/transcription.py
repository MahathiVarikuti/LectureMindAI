"""Audio transcription via faster-whisper."""

from faster_whisper import WhisperModel

from app.config import settings

_model: WhisperModel | None = None


def get_whisper_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(file_path: str) -> list[dict]:
    """Return segments: [{text, start, end}]."""
    model = get_whisper_model()
    segments, _ = model.transcribe(file_path, beam_size=5)
    results: list[dict] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            results.append({"text": text, "start": seg.start, "end": seg.end})
    return results
