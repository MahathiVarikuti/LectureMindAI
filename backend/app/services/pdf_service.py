import fitz

from app.utils.chunker import clean_text


def extract_text_from_pdf(pdf_path, document_id=None, filename=None):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            # sort=True reads blocks top-to-bottom, left-to-right, which fixes
            # jumbled text on multi-column slides and papers.
            text = clean_text(page.get_text("text", sort=True))
            pages.append({
                "page": page_num + 1,
                "text": text,
                "document_id": document_id,
                "filename": filename,
            })
    return pages
