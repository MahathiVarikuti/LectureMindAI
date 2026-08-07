"""Overlap-based text chunking.

Trade-off: larger chunk_size gives more context per retrieval hit but may
include irrelevant text; overlap helps preserve continuity across boundaries.
"""

from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    if overlap >= size:
        overlap = size // 4
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_segments(
    segments: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Chunk transcript segments respecting boundaries where possible."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    if overlap >= size:
        overlap = size // 4

    results: list[dict] = []
    buffer = ""
    buf_start: float | None = None
    buf_end: float | None = None

    def flush():
        nonlocal buffer, buf_start, buf_end
        if buffer.strip():
            results.append(
                {
                    "text": buffer.strip(),
                    "timestamp_start": buf_start,
                    "timestamp_end": buf_end,
                }
            )
        buffer = ""
        buf_start = None
        buf_end = None

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        seg_start = seg.get("start")
        seg_end = seg.get("end")
        candidate = f"{buffer} {seg_text}".strip() if buffer else seg_text
        if len(candidate) <= size:
            buffer = candidate
            buf_start = buf_start if buf_start is not None else seg_start
            buf_end = seg_end
        else:
            if buffer:
                flush()
            if len(seg_text) <= size:
                buffer = seg_text
                buf_start = seg_start
                buf_end = seg_end
            else:
                sub_chunks = chunk_text(seg_text, size, overlap)
                for i, sc in enumerate(sub_chunks):
                    results.append(
                        {
                            "text": sc,
                            "timestamp_start": seg_start,
                            "timestamp_end": seg_end,
                        }
                    )
    flush()
    return results
