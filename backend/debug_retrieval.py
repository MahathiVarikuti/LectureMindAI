#!/usr/bin/env python3
"""Detailed debugging of what chunks are retrieved for each query."""

import httpx
import json
import time
import uuid
from pathlib import Path

BACKEND_URL = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())
RESUME_PATH = Path(
    r"C:\Users\mahat\LectureMindAI-V2\backend\data\uploads\4d8212e4-87f8-4e2b-9919-556465eadbab"
) / "c3013338-e4a1-4572-a5a6-a61390f46e93_VarikutiMahathiReddyResume (1).pdf"

client = httpx.Client(timeout=60.0)

# Health check
resp = client.get(f"{BACKEND_URL}/health")
assert resp.status_code == 200, "Backend not running"
print("Backend running\n")

# Upload
with open(RESUME_PATH, "rb") as f:
    files = {"file": (RESUME_PATH.name, f, "application/pdf")}
    resp = client.post(f"{BACKEND_URL}/upload", files=files, data={"session_id": SESSION_ID})
assert resp.status_code == 200, f"Upload failed: {resp.text}"
doc_id = resp.json().get("document_id")
print(f"Uploaded: {doc_id}\n")

time.sleep(1)

# Test PROJECT query specifically
query = "What are her projects?"
print(f"QUERY: {query}\n")

resp = client.post(
    f"{BACKEND_URL}/query",
    json={"query": query, "session_id": SESSION_ID, "selected_documents": [doc_id]},
)

data = resp.json()
answer = data.get("answer", "")
sources = data.get("sources", [])
context = data.get("context", "")

print("=" * 80)
print("ANSWER:")
print(answer)

print("\n" + "=" * 80)
print(f"CONTEXT ({len(context)} chars):")
print(context)

print("\n" + "=" * 80)
print(f"SOURCES ({len(sources)} items):")
for src in sources:
    print(f"  - {src.get('filename')} (page {src.get('page')})")
    snippet = src.get("snippet", "")
    print(f"    {repr(snippet[:100])}")

print("\n" + "=" * 80)
print("ANALYSIS:")
context_lower = context.lower()

# Check for actual project names
actual_projects = ["data quality", "facefi", "lecturemind"]
found_projects = []
for proj in actual_projects:
    if proj in context_lower:
        found_projects.append(proj)

print(f"Actual projects found in context: {found_projects}")

# Check for skills contamination
skills_terms = ["python", "java", "scala", "fastapi"]
found_skills = []
for skill in skills_terms:
    if skill in context_lower:
        found_skills.append(skill)

print(f"Skills terms found in context: {found_skills}")

# Check for experience contamination
exp_terms = ["signature", "intern", "data pipeline"]
found_exp = []
for exp in exp_terms:
    if exp in context_lower:
        found_exp.append(exp)

print(f"Experience terms found in context: {found_exp}")
