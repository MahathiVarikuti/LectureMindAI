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


def _pack(pieces, chunk_size, overlap):
    """Greedily pack pieces into chunks <= chunk_size, carrying ~`overlap` chars of trailing
    pieces into the next chunk."""
    chunks, current, current_len = [], [], 0
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
    return chunks


_BULLET = ("●", "•", "▪", "-", "*", "–")


def _is_heading(line):
    """A short ALL-CAPS line such as EDUCATION or TECHNICAL SKILLS (not an acronym like SQL)."""
    line = line.strip()
    letters = [c for c in line if c.isalpha()]
    return (len(letters) >= 4 and line.isupper() and len(line.split()) <= 6
            and not line.endswith((".", ",", ";")) and not line.startswith(_BULLET))


def _split_sections(text):
    """[(heading|None, body)] split at heading lines. Documents without headings return
    a single (None, text) section, which keeps the original chunking behaviour."""
    sections, heading, body = [], None, []
    for line in text.split("\n"):
        if _is_heading(line):
            if "\n".join(body).strip() or heading:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line.strip(), []
        else:
            body.append(line)
    sections.append((heading, "\n".join(body).strip()))
    return [(h, b) for h, b in sections if b]


def _units(body):
    """Split a section into logical entries so one entry is never cut in half: a title line
    (plus its stack/date lines) stays with the bullets that follow it, and wrapped bullet
    lines stay with their bullet."""
    units, current, seen_bullet = [], [], False
    for line in body.split("\n"):
        is_bullet = line.lstrip().startswith(_BULLET)
        if (not line.strip()) or (seen_bullet and not is_bullet and not line.startswith(" ")
                                  and current and _starts_new_entry(current, line)):
            if current:
                units.append("\n".join(current))
            current, seen_bullet = [], False
            if not line.strip():
                continue
        current.append(line)
        seen_bullet = seen_bullet or is_bullet
    if current:
        units.append("\n".join(current))
    return units


def _starts_new_entry(current, line):
    # After bullets, a non-bullet line is a continuation of the last bullet when the bullet
    # does not look finished (PDF line wrapping); otherwise it starts a new entry (a title).
    last = current[-1].rstrip()
    return last.endswith((".", ")", ":", "%")) and line[:1].isupper()


def chunk_text(text, chunk_size=None, overlap=None, min_chars=None):
    """Structure-aware chunking: packs paragraphs/sentences into chunks of at most
    chunk_size characters, carrying roughly `overlap` characters of trailing
    sentences into the next chunk. Never cuts in the middle of a word.

    Documents with section headings (EDUCATION, PROJECTS, ... or lecture headings) are
    split per section, entries inside a section (e.g. one project) stay together, and every
    chunk is prefixed with its section heading so "what are her projects?" matches."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    min_chars = config.MIN_CHUNK_CHARS if min_chars is None else min_chars

    text = clean_text(text)
    if not text:
        return []

    sections = _split_sections(text)
    if len(sections) == 1 and sections[0][0] is None:
        chunks = _pack(_split(text, chunk_size, _SEPARATORS), chunk_size, overlap)
        return [c for c in chunks if len(c) >= min_chars]

    chunks = []
    for heading, body in sections:
        prefix = f"{heading}\n" if heading else ""
        budget = max(200, chunk_size - len(prefix))
        pieces = []
        for unit in _units(body):
            pieces.extend(_split(unit + "\n", budget, _SEPARATORS))
        chunks.extend(prefix + c for c in _pack(pieces, budget, overlap) if c)
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


def chunk_segments(segments, chunk_size=None, overlap=None):
    """Chunk timestamped transcript segments [{text, start, end}] without splitting a segment.

    Segments are packed up to chunk_size characters; the next chunk restarts with the last
    segments of the previous one (up to `overlap` characters) so context carries across
    chunk boundaries. Each chunk keeps the start of its first and end of its last segment.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks, current, size = [], [], 0

    def flush():
        if current:
            chunks.append({
                "text": " ".join(s["text"] for s in current).strip(),
                "timestamp_start": current[0]["start"],
                "timestamp_end": current[-1]["end"],
            })

    for seg in segments:
        if current and size + len(seg["text"]) > chunk_size:
            flush()
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev["text"]) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev["text"]) + 1
            current, size = tail, tail_len
        current.append(seg)
        size += len(seg["text"]) + 1
    flush()
    return [c for c in chunks if len(c["text"]) >= 1]
