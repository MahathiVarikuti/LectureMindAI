# LectureMindAI

A full-stack, multimodal Retrieval-Augmented Generation (RAG) app. Upload lecture material (PDFs, audio, video, or a URL to your own / openly-licensed recording), ask questions in natural language, and get answers grounded in that material with **citations**: page numbers for PDFs, timestamps for audio/video.

Everything is free and local: a local LLM (Ollama), local embeddings, local vector store (ChromaDB), SQLite. No LangChain / LlamaIndex: chunking, retrieval, prompting and session handling are written directly.

## Features
- PDF, audio (Whisper), video (transcript **and** sampled frames via CLIP) and URL (yt-dlp) ingestion
- Chat with follow-up questions (query rewriting from history), grounded answers, refusal when nothing relevant is found (the LLM is not even called)
- Cross-modal retrieval: one query can cite text chunks and video frames
- Per-browser sessions (no login); documents and chat history are scoped to the session and persisted (SQLite)
- Document list, selection (scope a query to chosen documents) and delete (vectors + file)
- Rate limiting, upload/duration/storage limits, restricted CORS, env-based config

## Architecture
```
Next.js (Tailwind, Axios, react-markdown)
        |  session_id (UUID in localStorage) on every request
        v
FastAPI  --- /upload /upload-url /query /documents /history /health
   |-- ingestion: PyMuPDF | ffmpeg + faster-whisper | CLIP (frames) | yt-dlp
   |-- ChromaDB   text_chunks (MiniLM) | image_frames (CLIP)   <- separate collections
   |-- SQLite     chat history + document ownership
   `-- Ollama     qwen2.5:1.5b  (/api/generate, stream: false)
```
Details: [docs/architecture.md](docs/architecture.md). Security model and its limits: [docs/security.md](docs/security.md).

## Setup
Prerequisites: Python 3.11+, Node 20+, [ffmpeg](https://ffmpeg.org/) on PATH (`winget install ffmpeg` / `apt install ffmpeg` / `brew install ffmpeg`), [Ollama](https://ollama.com/).

```bash
ollama pull qwen2.5:1.5b

# backend
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env                                  # optional: every value has a default
uvicorn app.main:app --port 8000

# frontend (new terminal)
cd frontend
npm ci
npm run dev                                              # http://localhost:3000
```
The first upload/query downloads the embedding model (and Whisper/CLIP when you ingest media), so it is slow once.

**Low-RAM machines:** the defaults are the lightest configuration (dense retrieval only, no reranker, no NLI check, one ingestion at a time, Whisper `base`). Use `WHISPER_MODEL=tiny` for faster transcription.

### Docker
```bash
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:1.5b      # first run only
```
Frontend: http://localhost:3000, API: http://localhost:8000.

## Configuration
All settings are environment variables (see [.env.example](.env.example) for the documented list): Ollama URL/model/timeouts, model names, chunk size/overlap, storage paths, resource limits (25 MB PDF, 30 min media, 500 MB and 20 documents per session), rate limits, CORS origins.

## Tests
```bash
cd backend
pip install -r requirements-dev.txt
pytest
```
The suite fakes the embedder, LLM, Whisper and CLIP, so it needs no model downloads. It covers chunking, upload limits, every modality's round trip, **session isolation**, deletion (vectors + file), the empty-context guard, rate limiting and error mapping. It never asserts exact LLM wording.

## Evaluation
Retrieval, abstention-threshold, hallucination and model-size experiments (QASPER, SQuAD v2, CoQA, RAGTruth, RAGBench) live in `backend/eval/`; see [docs/evaluation.md](docs/evaluation.md). They also power the optional hybrid search, reranker and NLI grounding check (`USE_HYBRID`, `USE_RERANKER`, `USE_GROUNDING_CHECK`, all off by default).

## Known limitations
- Sessions are browser-local UUIDs: clearing site data or switching browsers loses access to earlier documents and history. This is an accepted V1 trade-off, not real authentication.
- Scanned PDFs (no text layer) are rejected; there is no OCR.
- yt-dlp import is for your own or openly-licensed content only.
