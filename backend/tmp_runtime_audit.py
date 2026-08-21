import json
import uuid
from pathlib import Path

import fitz
import requests

base = Path(r"C:\Users\mahat\LectureMindAI-V2\backend\data\test_runtime")
base.mkdir(exist_ok=True)
pdf_path = base / "runtime_audit.pdf"

doc = fitz.open()
page = doc.new_page()
page.insert_text(
    (72, 72),
    """Resume

Name: Aisha Khan
Education: B.S. Computer Science, Lahore University of Management Sciences (LUMS), 2021
Experience: Senior Product Engineer at BrightLabs, 2022-2024
Skills: Python, FastAPI, SQL, machine learning
Projects: LectureMindAI, an AI assistant for lecture notes.
Contact: aisha@example.com
""",
)
doc.save(pdf_path)
doc.close()

session_id = str(uuid.uuid4())
with open(pdf_path, "rb") as fh:
    upload = requests.post(
        "http://localhost:8000/upload",
        files={"file": (pdf_path.name, fh, "application/pdf")},
        data={"session_id": session_id},
        timeout=180,
    )
print("UPLOAD", upload.status_code)
print(upload.text)

doc_id = upload.json()["document_id"]
for q in [
    "Where did Aisha Khan study?",
    "What was her favorite color?",
    "What happened at UET?",
]:
    r = requests.post(
        "http://localhost:8000/query",
        json={"query": q, "session_id": session_id},
        timeout=180,
    )
    print("QUERY", q)
    print(r.status_code)
    print(json.dumps(r.json(), ensure_ascii=False))
    print("---")

r = requests.post(
    "http://localhost:8000/query",
    json={"query": "Where did she study?", "session_id": session_id, "selected_documents": [doc_id]},
    timeout=180,
)
print("SCOPED")
print(r.status_code)
print(json.dumps(r.json(), ensure_ascii=False))
