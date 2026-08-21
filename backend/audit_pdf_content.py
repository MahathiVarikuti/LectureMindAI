#!/usr/bin/env python3
"""Audit the actual resume PDF content and chunking."""

import sys
from pathlib import Path
from app.utils.pdf_extractor import extract_pdf_pages
from app.utils.chunking import chunk_text

# Use the real resume PDF
resume_pdf = Path(
    r"C:\Users\mahat\LectureMindAI-V2\backend\data\uploads\4d8212e4-87f8-4e2b-9919-556465eadbab\c3013338-e4a1-4572-a5a6-a61390f46e93_VarikutiMahathiReddyResume (1).pdf"
)

# Redirect stdout to file to avoid encoding issues
output_file = open("audit_output.txt", "w", encoding="utf-8")
sys.stdout = output_file

print(f"PDF exists: {resume_pdf.exists()}")
if not resume_pdf.exists():
    print("ERROR: PDF not found")
    exit(1)

print("\n" + "="*80)
print("EXTRACTED PAGES")
print("="*80)

pages = extract_pdf_pages(str(resume_pdf))
print(f"Total pages: {len(pages)}")

all_text = ""
for page in pages:
    print(f"\n--- PAGE {page['page']} ---")
    text = page["text"]
    all_text += text + "\n"
    # Print text with repr to see actual content
    print(repr(text))

print("\n" + "="*80)
print("CHUNKING WITH DEFAULT SETTINGS")
print("="*80)

chunks = chunk_text(all_text, chunk_size=1000, chunk_overlap=200)
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"\n{'='*80}")
    print(f"CHUNK {i} (len={len(chunk)})")
    print(f"{'='*80}")
    print(chunk)
    print()

print("\n" + "="*80)
print("SECTION IDENTIFICATION")
print("="*80)

section_keywords = ["education", "experience", "skills", "projects", "certifications", "awards"]
for keyword in section_keywords:
    found_in = []
    for i, chunk in enumerate(chunks):
        if keyword.lower() in chunk.lower():
            found_in.append(i)
    print(f"{keyword.upper()}: chunks {found_in}")

output_file.close()
print("\n✓ Audit results written to audit_output.txt", file=sys.__stdout__)
