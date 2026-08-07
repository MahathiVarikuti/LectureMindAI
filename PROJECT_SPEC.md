# LectureMindAI — Complete Build Specification for Cursor AI

You are my project architect and lead engineer. Build this project completely, following every specification below exactly. Where a decision is already made, implement it as specified — do not substitute your own default choice (e.g., do not swap SQLite for Postgres, do not swap ChromaDB for Pinecone, do not introduce LangChain/LlamaIndex). Where something is marked "your judgment," use standard production best practices and explain the choice in a code comment.

---

## 1. Project Overview

**Name:** LectureMindAI

**What it is:** A full-stack, multimodal Retrieval-Augmented Generation (RAG) system that lets a user upload lecture material — PDFs, audio recordings, video recordings, or a URL to their own/openly-licensed audio/video content — and ask natural-language questions about it. The system retrieves relevant content and generates grounded answers using a locally-hosted LLM, with source citations (page numbers for PDFs, timestamps for audio/video).

**Core constraint — read this first:** Do NOT use LangChain, LlamaIndex, or any RAG orchestration framework. Every part of the pipeline (chunking, embedding, retrieval, prompt construction, session handling) must be implemented directly, using base libraries only. This is a deliberate architectural choice, not an oversight.

**Cost constraint:** Every tool/library/service used must be free and open-source, runnable entirely on local/free-tier infrastructure. No paid APIs, no paid vector databases, no paid LLM inference.

**Hardware constraint:** Target machine is CPU-only (no CUDA/NVIDIA GPU available). All model choices and defaults must be selected accordingly (see Tech Stack section for specific model sizes).

---

## 2. Features

### 2.1 Core Features (must-have, V1)
- Upload and ingest PDF documents (text extraction, chunking, embedding, storage)
- Upload and ingest audio files (transcription via Whisper, chunking, embedding, storage)
- Upload and ingest video files (audio track → transcription pipeline; sampled frames → CLIP embedding, stored separately)
- Ingest audio/video from a user-provided URL via `yt-dlp` (scoped to the user's own recordings or openly-licensed content — include a disclaimer in the UI, do not market as a general downloader)
- Natural-language chat interface with conversational follow-up support (query rewriting using chat history)
- Grounded answer generation using a locally-hosted LLM (Ollama), refusing to answer when retrieved context doesn't cover the question
- Source citation display: page number for PDF sources, timestamp for audio/video sources
- Document management: list uploaded documents, delete a document (and its vector entries and its file on disk)
- Document selection: user can scope a query to specific uploaded documents
- Session-based usage (no login required) with all data (chat history, document ownership) scoped to a per-browser session identifier
- Persistent chat history across page reloads (backed by SQLite, not in-memory)
- Cross-modal retrieval: a single query can retrieve and cite both text chunks and video frames, merged into one coherent context for the LLM

### 2.2 Explicitly Out of Scope for V1 (do not build these)
- User accounts, login/signup, password auth, JWT-based authentication
- Hybrid search (BM25 + vector), reranking models, query expansion
- Streaming LLM responses (use `stream: False` for all Ollama calls)
- Multi-worker horizontal scaling, managed cloud databases (Postgres, managed Redis)
- Payment/billing of any kind

---

## 3. Tech Stack

### Frontend
- Next.js (App Router, latest stable)
- TypeScript
- Tailwind CSS
- Axios for HTTP requests
- `react-markdown` for rendering LLM responses safely

### Backend
- Python 3.11+
- FastAPI + Uvicorn (ASGI)
- Pydantic for request/response validation
- `pydantic-settings` for environment-based configuration

### Database / Storage
- **SQLite** for chat history and session metadata (NOT in-memory dict, NOT Redis, NOT Postgres)
- **ChromaDB** (persistent local client) for vector storage — TWO separate collections:
  - `text_chunks` collection — for PDF text and audio/video transcripts (SentenceTransformers embeddings)
  - `image_frames` collection — for video frame embeddings (CLIP embeddings)
  - Do NOT mix these into one collection. See Design Decisions section for why.

### AI/ML
- **Text embeddings:** `sentence-transformers`, model `all-MiniLM-L6-v2`
- **LLM inference:** Ollama, model `qwen2.5:1.5b`, called via local HTTP API (`http://localhost:11434/api/generate`), `stream: False`
- **Audio transcription:** `faster-whisper` (NOT `openai-whisper` — faster-whisper is CTranslate2-based and meaningfully faster on CPU). Default model size: `base` (configurable via env to `tiny` for faster/lower-quality or `small`/`medium` for slower/higher-quality)
- **Video frame embeddings:** CLIP via `open_clip` or `transformers`, model `openai/clip-vit-base-patch32` (smaller variant, chosen for CPU inference speed)
- **PDF text extraction:** PyMuPDF (`fitz`)
- **Video/audio processing:** `ffmpeg` (system dependency) for audio track extraction and frame sampling
- **URL-based media acquisition:** `yt-dlp`

### DevOps / Infra
- Docker (separate `Dockerfile` for backend and frontend)
- `docker-compose.yml` for local multi-container orchestration
- `.env` / `.env.example` for all configuration (no hardcoded secrets, URLs, or paths)
- pytest for backend testing
- Git with Conventional Commits

---

## 4. System Architecture & Design Decisions

Implement these exactly — each was decided deliberately with a specific rationale.

### 4.1 Session Model (no login)
- On first frontend load, generate a random UUID client-side, store in `localStorage` as the session identifier.
- Every request to the backend (`/upload`, `/query`, `/documents`) includes this `session_id`.
- **All resources are scoped to `session_id`**: a session can only see, query, and delete its own documents and its own chat history. This is a hard requirement — implement it as a `WHERE session_id = ?` / `where={"session_id": ...}` filter on every relevant query, not as a UI-only restriction.
- Document the known limitation clearly in code comments and README: clearing browser storage or switching browsers loses access to prior documents/history. This is an accepted trade-off for V1 (no login system) — do not silently add authentication to "fix" this.

### 4.2 Multimodal Embedding Architecture
- Text (PDF text + audio/video transcripts) → SentenceTransformers → `text_chunks` collection.
- Video frames → CLIP → `image_frames` collection (separate from text).
- **Do NOT** attempt to force everything into a single shared CLIP embedding space. Text-to-text retrieval quality matters more than architectural purity for this text-dominant use case.
- At query time: query both collections independently, then merge results. **Do not merge by raw similarity score directly** — scores from CLIP and SentenceTransformers are not on comparable scales. Normalize each collection's scores independently (e.g., min-max normalize within each result set) before combining/ranking, or present them as separate labeled sections in the context rather than a single interleaved ranked list. Comment the reasoning in code.

### 4.3 Chat Memory
- SQLite database file (e.g., `chat_history.db`), NOT an in-memory Python dict.
- Schema (minimum): a `sessions` table (or implicit via session_id foreign key), a `messages` table with columns: `id`, `session_id`, `role` (`user`/`assistant`), `content`, `created_at`. Foreign key `session_id` on every relevant table.
- Trim/limit history sent to the LLM for query rewriting (e.g., last 4-10 messages), but persist full history in SQLite.

### 4.4 Metadata Schema for Vector Store
Every entry in `text_chunks` must include at minimum:
```
{
  "session_id": str,
  "document_id": str,
  "filename": str,
  "modality": "pdf" | "audio" | "video_transcript",
  "chunk_id": int,
  "page": int | None,           # for PDF
  "timestamp_start": float | None,  # for audio/video
  "timestamp_end": float | None,
  "uploaded_at": ISO8601 string
}
```
Every entry in `image_frames` must include:
```
{
  "session_id": str,
  "document_id": str,
  "filename": str,
  "modality": "video_frame",
  "timestamp": float,
  "uploaded_at": ISO8601 string
}
```

### 4.5 Ingestion Pipelines
- **PDF:** extract text per page (PyMuPDF) → chunk with overlap → embed (SentenceTransformers) → store with `page` metadata.
- **Audio:** transcribe with `faster-whisper` (returns segments with start/end timestamps) → chunk transcript text (respecting segment boundaries where possible) → embed → store with `timestamp_start`/`timestamp_end` metadata.
- **Video:** split into (a) audio track → run through the Audio pipeline above, and (b) sampled frames (default: 1 frame per 7 seconds, configurable via env) → CLIP embed each frame → store in `image_frames`.
- **URL input:** use `yt-dlp` to download audio (or video, if frame extraction is requested) to a temp file, then feed into the pipelines above as if locally uploaded. Clean up temp files after processing.

### 4.6 Chunking
- Overlap-based chunking (configurable `chunk_size`, `chunk_overlap` via env, sensible defaults e.g. 1000/200 characters). Comment on the trade-off of chunk size vs. overlap in code.

### 4.7 Query Pipeline
1. Receive `{ query, session_id, selected_documents? }`.
2. Pull recent chat history for this session from SQLite.
3. If history exists, rewrite the query into a standalone question via the LLM. Skip this call if there's no history (save latency).
4. Embed the rewritten query (SentenceTransformers) and query `text_chunks` (filtered by `session_id`, optionally by `selected_documents`).
5. If the query is plausibly about visual content, also embed the query via CLIP and query `image_frames` similarly (your judgment on how to detect this, or always query both and let low relevance naturally exclude irrelevant frames).
6. Merge/normalize results per Section 4.2.
7. Assemble a context string clearly labeling each source (filename + page/timestamp).
8. If no relevant chunks retrieved, return a clear "not found in uploaded material" response WITHOUT calling the LLM (save cost/latency).
9. Call the LLM with the assembled context + rewritten query, `stream: False`.
10. Persist the exchange to SQLite (trim to a reasonable history length).
11. Return `{ answer, sources, used_documents }` to the frontend.

---

## 5. API Specification

All endpoints require `session_id` (query param, form field, or JSON body field depending on endpoint — your judgment, but be consistent).

- `POST /upload` — multipart form: `file`, `session_id`. Detects file type (PDF/audio/video) by extension/MIME and routes to the correct ingestion pipeline. Returns `{ document_id, filename, chunks, modality, message }`.
- `POST /upload-url` — JSON: `{ url, session_id }`. Downloads via `yt-dlp`, routes into ingestion. Same response shape as `/upload`.
- `POST /query` — JSON: `{ query, session_id, selected_documents?: string[] }`. Returns `{ answer, sources, used_documents, context }`.
- `GET /documents?session_id=...` — returns list of documents belonging to this session only: `[{ document_id, filename, modality, chunk_count, uploaded_at }]`.
- `DELETE /documents/{document_id}?session_id=...` — verifies the document belongs to this session before deleting; removes vector entries from BOTH collections (if applicable) AND the file from disk. 404 if not found or not owned by this session (do not leak existence of other sessions' documents — same 404 either way).
- `GET /health` — basic health check, no auth needed.

Every endpoint must return proper HTTP status codes (400 for bad input, 404 for not found, 413 for payload too large, 429 for rate limited, 500 only for genuine unexpected errors) — never a bare 500 for expected failure modes like "file too large" or "unsupported format."

---

## 6. Security Requirements

Implement all of the following — these are not optional:

### 6.1 Authorization
- Every read/write to documents and chat history MUST be filtered by `session_id`. No endpoint should ever return or modify data belonging to a different session. Write a test that explicitly verifies session A cannot see or delete session B's documents.

### 6.2 Rate Limiting
- Apply rate limiting to `/upload`, `/upload-url`, and `/query` (the expensive endpoints) using a free/open-source library (e.g., `slowapi`). Reasonable default: e.g., 10 requests/minute per session_id or IP for `/query`, lower for `/upload`. Make limits configurable via `.env`.

### 6.3 Secrets Management
- All configuration (Ollama URL, model names, storage paths, CORS origins, rate limit values, resource limits) must live in `.env`, loaded via `pydantic-settings`. Never hardcode. Provide a complete `.env.example` with safe defaults and comments explaining each variable. Ensure `.env` is in `.gitignore`.

### 6.4 Access Control / Resource Limits
Implement and enforce (configurable via `.env`, with these defaults):
- Max PDF file size: 25MB — reject with 413 if exceeded.
- Max audio/video duration: 30 minutes — reject with 413/400 if exceeded (check duration before running expensive transcription).
- Max total storage per session: 500MB–1GB — reject new uploads once exceeded.
- Max documents per session: 20 — reject new uploads once exceeded.
- Validate file type/extension server-side (do not trust client-provided MIME type alone).

### 6.5 Token/Session Security
- `session_id` is an unsigned, non-guessable UUID (v4). Note in code comments that this is NOT a security boundary against a determined attacker who obtains the UUID (e.g., via browser inspection by someone with physical device access) — it is a lightweight isolation mechanism appropriate for a no-login demo, not a substitute for real authentication. Document this honestly in `docs/security.md`.

### 6.6 Resilience
- Enforce all limits in Section 6.4 BEFORE expensive processing begins (check file size/duration first, fail fast).
- Set reasonable timeouts on Ollama and Whisper calls; handle timeout/failure gracefully with a proper error response, not a hang or crash.
- CORS: in development, permissive is acceptable; production config must restrict `allow_origins` to the actual deployed frontend origin(s) only — implement this as an environment-driven setting, not hardcoded `*`.
- Wrap all external calls (Ollama, ffmpeg, yt-dlp, file I/O) in proper error handling with meaningful error messages returned to the client (never a raw stack trace).

---

## 7. Database Schema (SQLite)

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_messages_session ON messages(session_id);
```
(Extend with additional tables/columns as needed for session metadata, but keep this as the minimum baseline.)

---

## 8. Build Order (Milestones)

Build in this exact sequence — later steps depend on earlier ones being correct:

1. Project scaffolding: repo structure, `.gitignore`, `README.md` skeleton, Git init.
2. Backend skeleton: FastAPI app, CORS middleware, `/health` endpoint, `requirements.txt`.
3. PDF upload + text extraction (PyMuPDF), saved to disk, no embeddings yet — verify extraction correctness first.
4. Chunking utility (configurable size/overlap).
5. Embedding service (SentenceTransformers) + ChromaDB `text_chunks` collection wiring for PDFs.
6. Ollama integration: `generate_answer` and `rewrite_query` functions, tested standalone.
7. `/query` endpoint: full text-only RAG loop working end-to-end for PDFs.
8. Audio ingestion: `faster-whisper` transcription, timestamp segmentation, chunking, embedding into `text_chunks` with `modality: "audio"`.
9. Video ingestion: ffmpeg audio extraction (→ step 8 pipeline) + frame sampling + CLIP embedding into `image_frames`.
10. URL-based ingestion via `yt-dlp`, feeding into steps 8/9.
11. Cross-modal merged retrieval in `/query` (query both collections, normalize, merge).
12. SQLite chat history (replace any temporary in-memory approach), wire into `/query`.
13. `/documents` (list) and `DELETE /documents/{id}` endpoints, session-scoped, with correct file deletion (verify the actual file is removed from disk, not just the vector entries).
14. Frontend scaffold: Next.js + Tailwind + Axios client.
15. Upload UI: multi-modal (PDF/audio/video/URL input), loading/error states.
16. Chat UI: message history, Markdown rendering, source citations (modality-aware: page vs. timestamp).
17. Document sidebar: list from `/documents`, selection checkboxes wired into `/query`'s `selected_documents`, delete button.
18. Backend tests (pytest): chunking correctness, upload validation, session isolation (critical test), deletion correctness, empty-context guard.
19. Security hardening: rate limiting, `.env`/config externalization, resource limit enforcement, CORS tightening.
20. Documentation: full README, `docs/architecture.md`, `docs/setup.md`, `docs/security.md` (with the honest limitations noted in 6.5).
21. Docker: `Dockerfile` for backend and frontend, `docker-compose.yml`.
22. Deployment to a free-tier host (document the real constraint: verify whether Ollama can run within the target host's resource limits; if not, document a fallback plan clearly rather than silently failing).

---

## 9. Testing Strategy

- **Unit tests:** chunking boundaries and overlap correctness; resource-limit validation logic; metadata schema correctness.
- **Integration tests (FastAPI `TestClient`):** full upload → query round trip for each modality; session isolation (session A cannot access session B's data — this test is mandatory, not optional); deletion actually removes both vector entries and the file from disk; empty/irrelevant query returns the "not found" response without calling the LLM.
- **Do NOT** write tests asserting exact LLM output text — LLM outputs are non-deterministic and testing exact phrasing is brittle and wrong. Test the plumbing around the LLM call instead (correct context assembly, correct error handling, correct fallback behavior).
- **Manual/exploratory testing:** short (30-90s) audio/video clips during development given CPU-only constraints; full-length content tested occasionally once pipeline correctness is confirmed.

---

## 10. Deliverables

- Complete, working source code in the structure implied above (`backend/app/...`, `frontend/src/...`).
- `README.md` covering: what the project does, architecture diagram/explanation, setup instructions (including Ollama model pull, ffmpeg install), environment variables, how to run locally, how to run tests.
- `docs/architecture.md`: explains the dual-collection vector store design, session-based isolation model, and RAG pipeline flow.
- `docs/security.md`: honest documentation of the session-based auth model and its limitations, plus what would need to change to add real authentication later.
- `.env.example` with every configurable value documented.
- Passing test suite (`pytest`).
- Dockerized setup that runs via `docker-compose up`.
- Git history with Conventional Commit messages reflecting the build order above (one logical commit per milestone, not one giant commit).

---

## 11. Explicit Non-Goals (do not implement, do not suggest adding)

- No user accounts / login / password auth / JWT.
- No LangChain, LlamaIndex, or any RAG framework.
- No paid vector database (Pinecone, Weaviate Cloud, etc.) — ChromaDB local only.
- No paid LLM API — Ollama local only.
- No streaming responses.
- No multi-worker/horizontal scaling setup.