#!/usr/bin/env python3
"""
Live retrieval audit: Upload real resume and test all 7 queries.
Inspect retrieved context and verify answer grounding.
"""

import httpx
import json
import time
import uuid
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())  # Must be a valid UUID
RESUME_PATH = Path(
    r"C:\Users\mahat\LectureMindAI-V2\backend\data\uploads\4d8212e4-87f8-4e2b-9919-556465eadbab"
) / "c3013338-e4a1-4572-a5a6-a61390f46e93_VarikutiMahathiReddyResume (1).pdf"

QUERIES = [
    "What are her projects?",
    "What is her education and CGPA?",
    "What skills does she have?",
    "What certifications does she have?",
    "Where did she study?",
    "What is her work experience?",
    "What is her favorite color?",
]

client = httpx.Client(timeout=60.0)


def health_check():
    """Verify backend is running."""
    try:
        resp = client.get(f"{BACKEND_URL}/health")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False


def upload_resume():
    """Upload the resume PDF."""
    if not RESUME_PATH.exists():
        print(f"❌ Resume file not found: {RESUME_PATH}")
        return None

    print(f"\n📄 Uploading resume from: {RESUME_PATH}")
    with open(RESUME_PATH, "rb") as f:
        files = {"file": (RESUME_PATH.name, f, "application/pdf")}
        resp = client.post(
            f"{BACKEND_URL}/upload",
            files=files,
            data={"session_id": SESSION_ID},
        )

    if resp.status_code != 200:
        print(f"❌ Upload failed: {resp.status_code}")
        print(f"   Response: {resp.text}")
        return None

    doc_id = resp.json().get("document_id")
    print(f"✓ Uploaded successfully: {doc_id}")
    return doc_id


def test_query(query: str, doc_id: str):
    """Test a single query and inspect the retrieved context."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    resp = client.post(
        f"{BACKEND_URL}/query",
        json={
            "query": query,
            "session_id": SESSION_ID,
            "selected_documents": [doc_id],
        },
    )

    if resp.status_code != 200:
        print(f"❌ Query failed: {resp.status_code}")
        print(f"   Response: {resp.text}")
        return

    data = resp.json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    context = data.get("context", "")

    print(f"\n📝 ANSWER:")
    print(answer)

    print(f"\n🔍 RETRIEVED CONTEXT ({len(context)} chars):")
    # Print first 500 chars of context
    context_preview = context[:600] if len(context) > 600 else context
    print(context_preview)
    if len(context) > 600:
        print(f"\n... (truncated, full length: {len(context)} chars)")

    print(f"\n📌 SOURCES ({len(sources)} items):")
    for i, src in enumerate(sources):
        snippet_preview = src.get("snippet", "")[:100] if src.get("snippet") else ""
        print(f"  {i+1}. {src.get('filename')} (page {src.get('page', 'N/A')})")
        print(f"     Snippet: {repr(snippet_preview)}...")

    # Simple grounding check: does answer contain words from context?
    answer_lower = answer.lower()
    context_lower = context.lower()

    key_terms = []
    if "project" in answer_lower:
        key_terms.extend(["project", "platform", "data", "flask", "react"])
    if "skill" in answer_lower or "python" in answer_lower:
        key_terms.extend(["python", "fastapi", "django", "sql"])
    if "education" in answer_lower or "cgpa" in answer_lower:
        key_terms.extend(["education", "btech", "cgpa", "manipal"])
    if "certification" in answer_lower:
        key_terms.extend(["certification", "ibm", "deloitte"])
    if "experience" in answer_lower or "intern" in answer_lower:
        key_terms.extend(["experience", "intern", "signature"])
    if "favorite color" in answer_lower:
        # This should be NOT_FOUND
        if "could not find" in answer_lower or "not found" in answer_lower:
            print(f"✓ Correctly rejected unsupported question")
        else:
            print(f"⚠️  Expected NOT_FOUND response for unsupported question")
        return

    grounded_terms = [t for t in key_terms if t in context_lower]
    grounding_ratio = len(grounded_terms) / len(key_terms) if key_terms else 0

    print(f"\n✓ Grounding check: {len(grounded_terms)}/{len(key_terms)} key terms found in context")
    if grounding_ratio < 0.5:
        print(f"⚠️  WARNING: Low grounding ratio ({grounding_ratio:.1%})")


def main():
    print("=" * 80)
    print("LIVE RETRIEVAL AUDIT")
    print("=" * 80)
    print(f"Backend: {BACKEND_URL}")
    print(f"Session: {SESSION_ID}")

    # Check backend health
    if not health_check():
        print("❌ Backend is not running. Start the backend first:")
        print(f"   cd backend && ./venv/Scripts/python.exe -m uvicorn app.main:app --reload")
        return

    print("✓ Backend is running")

    # Upload resume
    doc_id = upload_resume()
    if not doc_id:
        return

    # Test all queries
    time.sleep(1)  # Give backend time to process upload
    for query in QUERIES:
        test_query(query, doc_id)

    print(f"\n{'='*80}")
    print("✓ AUDIT COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
