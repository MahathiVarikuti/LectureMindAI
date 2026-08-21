from pathlib import Path

from app.utils.chunking import chunk_text
from app.utils.pdf_extractor import extract_pdf_pages

p = Path(r"C:\Users\mahat\LectureMindAI-V2\backend\data\test_runtime\runtime_audit.pdf")
print("EXISTS", p.exists())
if p.exists():
    pages = extract_pdf_pages(str(p))
    print("PAGES", len(pages))
    for page in pages:
        print("PAGE", page["page"])
        print(page["text"])
        print("--- CHUNKS ---")
        for i, c in enumerate(chunk_text(page["text"], chunk_size=1000, chunk_overlap=200)):
            print(f"CHUNK {i}:", repr(c[:250]))
        print("===")
