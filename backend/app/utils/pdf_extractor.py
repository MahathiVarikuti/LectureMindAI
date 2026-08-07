"""PDF text extraction via PyMuPDF."""

import fitz


def extract_pdf_pages(file_path: str) -> list[dict]:
    """Return list of {page: int (1-based), text: str}."""
    doc = fitz.open(file_path)
    pages: list[dict] = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    finally:
        doc.close()
    return pages
