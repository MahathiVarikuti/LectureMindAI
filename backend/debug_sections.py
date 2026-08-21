#!/usr/bin/env python3
"""Debug section splitting."""

import sys
from pathlib import Path
from app.utils.pdf_extractor import extract_pdf_pages
from app.utils.chunking import _split_sections

resume_pdf = Path(
    r"C:\Users\mahat\LectureMindAI-V2\backend\data\uploads\4d8212e4-87f8-4e2b-9919-556465eadbab\c3013338-e4a1-4572-a5a6-a61390f46e93_VarikutiMahathiReddyResume (1).pdf"
)

pages = extract_pdf_pages(str(resume_pdf))
all_text = "\n".join([p["text"] for p in pages])

sections = _split_sections(all_text)
print(f"Total sections: {len(sections)}\n")

for i, section in enumerate(sections):
    first_line = section.split("\n")[0][:80] if section.split("\n") else "(empty)"
    last_line = section.split("\n")[-1][:80] if section.split("\n") else "(empty)"
    print(f"\nSECTION {i} (len={len(section)})")
    print(f"  First line: {repr(first_line)}")
    print(f"  Last line:  {repr(last_line)}")
    print(f"  Full:\n{section}")
