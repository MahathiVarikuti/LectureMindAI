import re

from app import config

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def clean_text(text):
    """Normalise raw PDF text: join hyphenated line breaks, collapse whitespace."""
    text = text.replace("\x00", "").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "gradi-\nent" -> "gradient"
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text, chunk_size, separators):
    """Recursively split text into pieces no longer than chunk_size, preferring
    paragraph, then line, then sentence, then word boundaries."""
    if len(text) <= chunk_size:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        pieces = []
        for j, part in enumerate(parts):
            # keep the separator attached so sentences keep their punctuation
            piece = part + sep if j < len(parts) - 1 else part
            if len(piece) > chunk_size:
                pieces.extend(_split(piece, chunk_size, separators[i + 1:]))
            elif piece:
                pieces.append(piece)
        return pieces
    # no separator left: hard cut
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text(text, chunk_size=None, overlap=None, min_chars=None):
    """Structure-aware chunking: packs paragraphs/sentences into chunks of at most
    chunk_size characters, carrying roughly `overlap` characters of trailing
    sentences into the next chunk. Never cuts in the middle of a word."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    min_chars = config.MIN_CHUNK_CHARS if min_chars is None else min_chars

    text = clean_text(text)
    if not text:
        return []

    pieces = _split(text, chunk_size, _SEPARATORS)
    chunks, current = [], []
    current_len = 0
    for piece in pieces:
        if current and current_len + len(piece) > chunk_size:
            chunks.append("".join(current).strip())
            # start the next chunk with the tail of the current one
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev)
            current, current_len = tail, tail_len
        current.append(piece)
        current_len += len(piece)
    if current:
        chunks.append("".join(current).strip())

    return [c for c in chunks if len(c) >= min_chars]


def chunk_text_fixed(text, chunk_size=1000, overlap=200):
    """The original V1 character-window chunker, kept for evaluation baselines."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
