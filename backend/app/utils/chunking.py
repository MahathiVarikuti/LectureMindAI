"""Overlap-based text chunking.

Trade-off: larger chunk_size gives more context per retrieval hit but may
include irrelevant text; overlap helps preserve continuity across boundaries.

For resume-like PDFs, the text usually contains repeated section headers such as
Education, Experience, Skills, Projects, and Certifications. We preserve those
section boundaries so retrieval can answer category-level questions without
pulling unrelated sections into the same chunk.
"""

import re

from app.config import settings

# Match section headers - typically all-caps or followed by colon.
# Be careful with multi-word contexts: "Languages:" within skills section shouldn't
# trigger a new section. Only match clear section headers like "EDUCATION", "TECHNICAL SKILLS", etc.
SECTION_HEADER_RE = re.compile(
    r"^(education|experience|projects|certifications|technical\s+skills|"
    r"leadership\s+&\s+activities|awards|summary|contact|"
    r"skills|certification)\s*(?::|$)",
    re.IGNORECASE,
)


def _split_sections(text: str) -> list[str]:
    """Split text by section headers, keeping headers with their content.
    
    Handles cases where headers are directly followed by content with no
    blank line separator.
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    sections: list[str] = []
    i = 0

    # Collect intro text before first section header
    intro_lines = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if SECTION_HEADER_RE.match(line):
            break
        intro_lines.append(lines[i])
        i += 1

    if intro_lines:
        sections.append("\n".join(intro_lines).strip())

    # Collect sections
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if SECTION_HEADER_RE.match(line):
            # Start a new section with this header
            section_lines = [lines[i]]
            i += 1

            # Collect content until next section header
            while i < len(lines):
                line = lines[i].strip()
                if line and SECTION_HEADER_RE.match(line):
                    # Found next section header
                    break
                section_lines.append(lines[i])
                i += 1

            section_text = "\n".join(section_lines).strip()
            if section_text:
                sections.append(section_text)
        else:
            i += 1

    return sections


def _chunk_section(section: str, size: int, overlap: int) -> list[str]:
    """Chunk a single section while keeping its header with each chunk."""
    if len(section) <= size:
        return [section]

    # Extract section header (first line if it matches section pattern)
    lines = section.split("\n")
    header = ""
    content_start = 0

    if lines and SECTION_HEADER_RE.match(lines[0].strip()):
        header = lines[0]
        content_start = 1

    # Build chunks from content, prefixing each with the header
    content_lines = lines[content_start:]
    if not content_lines:
        return [section]

    # Group content by paragraph
    paragraphs = []
    current = []
    for line in content_lines:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))

    # Build chunks by accumulating paragraphs
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = len(header) + 2

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_len = len(para)
        # If adding this paragraph exceeds size, flush the buffer
        if buffer and buffer_len + para_len + 2 > size:
            chunk_text = header + "\n" + "\n\n".join(buffer)
            chunks.append(chunk_text.strip())
            buffer = [para]
            buffer_len = len(header) + 2 + para_len
        else:
            buffer.append(para)
            buffer_len += para_len + 2

    # Flush remaining
    if buffer:
        chunk_text = header + "\n" + "\n\n".join(buffer)
        chunks.append(chunk_text.strip())

    # If a single chunk is still too large, apply character-level split
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= size:
            final.append(chunk)
        else:
            # Character-level split with header preservation
            content_only = chunk[len(header):].strip()
            start = 0
            while start < len(content_only):
                end = min(start + (size - len(header) - 50), len(content_only))
                piece = header + "\n" + content_only[start:end].strip()
                final.append(piece.strip())
                if end >= len(content_only):
                    break
                start = end - overlap

    return [c for c in final if c.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Chunk text and return dicts with text + section metadata.
    
    Returns: list of {"text": str, "section": str} dicts
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    if overlap >= size:
        overlap = size // 4
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= size:
        return [{"text": text, "section": _detect_section(text)}]

    sections = _split_sections(text)
    if len(sections) > 1:
        chunks: list[dict] = []
        for section in sections:
            section_name = _detect_section(section)
            for chunk_text_str in _chunk_section(section, size, overlap):
                chunks.append({"text": chunk_text_str, "section": section_name})
        return chunks

    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append({"text": chunk, "section": _detect_section(chunk)})
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _detect_section(text: str) -> str:
    """Detect the section type from chunk content."""
    text_lower = text.lower()
    # Check for section headers
    if re.search(r"^education\b", text_lower, re.MULTILINE):
        return "education"
    if re.search(r"^experience\b", text_lower, re.MULTILINE):
        return "experience"
    if re.search(r"^projects\b", text_lower, re.MULTILINE):
        return "projects"
    if re.search(r"^technical\s+skills\b", text_lower, re.MULTILINE):
        return "skills"
    if re.search(r"^certifications?\b", text_lower, re.MULTILINE):
        return "certifications"
    if re.search(r"^leadership\s+&\s+activities\b", text_lower, re.MULTILINE):
        return "leadership"
    # Default to generic if no clear section
    return "other"


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
                    "section": "transcript",
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
                for i, sc_dict in enumerate(sub_chunks):
                    # chunk_text returns dicts with {"text": str, "section": str}
                    results.append(
                        {
                            "text": sc_dict["text"],
                            "timestamp_start": seg_start,
                            "timestamp_end": seg_end,
                            "section": sc_dict.get("section", "transcript"),
                        }
                    )
    flush()
    return results
