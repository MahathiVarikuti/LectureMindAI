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
Details: [docs/architecture.md](docs/architecture.md). Security model and its limits: [docs/security.md](docs/security.md). Full code walkthrough with known issues and interview notes: [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md).

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

## Tuning for a CPU-only laptop
Environment variables (PowerShell: `$env:NAME="value"` before `uvicorn`):

| Variable | Suggested | Why |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:3b` for better answers, `qwen2.5:1.5b` (default) for speed | the 1.5b model sometimes answers with a bare heading or over-refuses |
| `USE_HYBRID` | `1` | adds keyword (BM25) search; helps section words like "projects" |
| `TOP_K` | `4`-`6` | chunks sent to the LLM |
| `LLM_NUM_CTX` | `4096` | smaller context = faster on CPU |
| `OLLAMA_TIMEOUT` | `300` | first request after a restart loads the model and can be slow |

Warm the model first (`ollama run qwen2.5:3b "hi"`). Re-upload documents after changing chunking or the embedding model.

## How chunking works
PDF text is split at section headings (short ALL-CAPS lines such as `EDUCATION` or `PROJECTS`), entries inside a section (for example one project) stay together, and each chunk is prefixed with its heading. Documents without headings use overlapping paragraph/sentence chunks (1000 characters, 200 overlap). Audio/video transcripts are chunked by segment so each chunk keeps its start/end timestamps.

## Configuration
All settings are environment variables (see [.env.example](.env.example) for the documented list): Ollama URL/model/timeouts, model names, chunk size/overlap, storage paths, resource limits (25 MB PDF, 30 min media, 500 MB and 20 documents per session), rate limits, CORS origins.

## Tests
```bash
cd backend
pip install -r requirements-dev.txt
pytest
```
The suite fakes the embedder, LLM, Whisper and CLIP, so it needs no model downloads. It covers chunking, upload limits, every modality's round trip, **session isolation**, deletion (vectors + file), the empty-context guard, rate limiting and error mapping. It never asserts exact LLM wording.

## Evaluation and fine-tuning
Retrieval, abstention-threshold, hallucination and model-size experiments (QASPER, SQuAD v2, CoQA, RAGTruth, RAGBench) live in `backend/eval/`; see [docs/evaluation.md](docs/evaluation.md). They also power the optional hybrid search, reranker and NLI grounding check (`USE_HYBRID`, `USE_RERANKER`, `USE_GROUNDING_CHECK`, all off by default).

Fine-tuning scripts (CPU-friendly; weights are written to the git-ignored `backend/models/`):
```bash
cd backend
python -m eval.finetune_grounding --pairs 2000 --epochs 2   # hallucination checker, RAGBench train
python -m eval.finetune_embedder  --pairs 3000 --epochs 2   # embedder, QASPER train + hard negatives
python -m eval.finetune_reranker  --pairs 3000 --epochs 2   # reranker, QASPER train
python -m eval.run hallucination --source ragtruth --nli-model models/nli-deberta-lecturemind
```
The fine-tuned grounding checker raises hallucination-detection AUROC on held-out RAGTruth from 0.65 to 0.74 (best F1 0.36 to 0.52) but shows no gain on RAGBench test, so it is not the default (details in `docs/evaluation.md`). Embedder and reranker fine-tunes are still to be evaluated.

## Known limitations
- Audio, video and URL ingestion are implemented and tested with fakes; the PDF path is the one exercised end to end with real models.
- With `USE_HYBRID=1` the relevance threshold is weaker (keyword-only matches get a neutral score), so the LLM's own refusal does more of the work. See the walkthrough, section 4.
- The `model` field of `/query` is client-controlled; do not expose the API publicly without restricting it.
- Sessions are browser-local UUIDs: clearing site data or switching browsers loses access to earlier documents and history. This is an accepted V1 trade-off, not real authentication.
- Scanned PDFs (no text layer) are rejected; there is no OCR.
- yt-dlp import is for your own or openly-licensed content only.
