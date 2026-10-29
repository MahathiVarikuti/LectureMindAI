# LectureMindAI

**Ask questions about your lectures and get answers with citations.** Upload PDFs, audio or video, then chat with the material. Every answer points back to its source: a page number for PDFs, a timestamp for audio and video.

Everything runs locally and for free: a local LLM (Ollama), local embeddings, a local vector store (ChromaDB) and SQLite. There is no LangChain or LlamaIndex. Chunking, retrieval, prompting and session handling are written directly.

## Overview
Lecture material is spread across slides, recordings and notes, and finding one explanation means scrubbing through an hour of video or searching dozens of pages. General chat models can answer, but they may invent facts that are not in your course.

LectureMindAI retrieves the relevant parts of *your* material first, hands only those to the language model, and refuses to answer when nothing relevant is found.

- Lecture content is spread across PDFs, audio and video
- Answers must be traceable to a page or a timestamp
- A small local model can hallucinate or over-refuse, so retrieval quality and thresholds matter
- The system has to run on an ordinary CPU laptop

## Core workflow
```
Upload PDF / audio / video
        ↓
Extract text (PyMuPDF) or transcribe (Whisper) and sample frames (CLIP)
        ↓
Chunk the content and embed it into ChromaDB
        ↓
Ask a question (follow-ups are rewritten using chat history)
        ↓
Retrieve the best chunks (and video frames)
        ↓
Relevance threshold: nothing relevant → "not found", LLM not called
        ↓
Local LLM answers from the retrieved context only
        ↓
Answer with citations (page numbers / timestamps)
```

## Key Features

### Multimodal ingestion
| Input | How it is processed |
|---|---|
| PDF | Text extracted with PyMuPDF, split at section headings, cited by page |
| Audio | Transcribed with faster-whisper, chunked by segment, cited by timestamp |
| Video | Audio transcript **and** frames sampled every 7 s and embedded with CLIP |

### Chat with grounded answers
- Follow-up questions are rewritten into standalone questions using chat history ("what about its cost?" becomes a full question)
- If the best chunk scores below the relevance threshold, the API answers "I could not find that information" without calling the LLM
- Cross-modal retrieval: one query can cite both text chunks and video frames

### Sessions and document management
- No login: each browser gets a UUID session; documents and chat history are scoped to it and persisted in SQLite
- Document list, scope a query to chosen documents, and delete (removes vectors and the file)

### Safety and limits
- Rate limiting per IP, restricted CORS, environment-based configuration
- Upload size, media duration, per-session storage and document-count limits are checked *before* any expensive work
- Heavy ingestion runs one at a time to protect RAM

### Optional quality features (off by default)
Added from the evaluation work, switched on with environment variables:
- `USE_HYBRID=1`: adds BM25 keyword search to dense retrieval
- `USE_RERANKER=1`: cross-encoder reranker for the retrieved candidates
- `USE_GROUNDING_CHECK=1`: NLI model that flags answer sentences not supported by the retrieved text

## Architecture
```
Next.js (Tailwind, Axios, react-markdown)
        |  session_id (UUID in localStorage) on every request
        v
FastAPI  --- /upload /query /documents /history /health
   |-- ingestion: PyMuPDF | ffmpeg + faster-whisper | CLIP (frames)
   |-- ChromaDB   text_chunks (MiniLM) | image_frames (CLIP)   <- separate collections
   |-- SQLite     chat history + document ownership
   `-- Ollama     qwen2.5:1.5b  (/api/generate, stream: false)
```
Details: [docs/architecture.md](docs/architecture.md). Security model and its limits: [docs/security.md](docs/security.md). Code walkthrough with known issues: [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md).

## Technology Stack
| Layer | Technology |
|---|---|
| Frontend | Next.js, React, Tailwind CSS, Axios, react-markdown |
| Backend | Python, FastAPI, Uvicorn, Pydantic Settings, SlowAPI (rate limiting) |
| LLM | Ollama, `qwen2.5:1.5b` (configurable) |
| Text embeddings | sentence-transformers, `all-MiniLM-L6-v2` |
| Image embeddings | CLIP, `openai/clip-vit-base-patch32` |
| Speech to text | faster-whisper (`base`) with ffmpeg |
| PDF parsing | PyMuPDF |
| Vector store | ChromaDB (two collections) |
| Chat history | SQLite |
| Deployment | Docker Compose (frontend, backend, Ollama) |
| Testing and CI | pytest, GitHub Actions |

### Where each library is used
| Library | Used in | Purpose |
|---|---|---|
| PyMuPDF | `services/pdf_service.py` | extract page text from PDFs |
| faster-whisper, ffmpeg | `services/transcription.py`, `utils/media.py` | extract the audio track and transcribe it with timestamps |
| CLIP (transformers) | `services/clip_service.py`, `frame_retrieval.py` | embed video frames and match text queries to them |
| sentence-transformers | `services/embedding_service.py` | embed chunks and queries; also the reranker and NLI cross-encoders |
| ChromaDB | `db/chroma.py`, `services/retrieval_service.py` | store and search vectors |
| rank-bm25 | `services/retrieval_service.py` | optional keyword search for hybrid retrieval |
| SlowAPI | `limiter.py` | per-IP rate limits |
| Ollama (HTTP) | `services/llm_service.py` | answer generation and query rewriting |

## How retrieval works

**Chunking.** PDF text is split at section headings (short ALL-CAPS lines such as `EDUCATION` or `PROJECTS`). Entries inside a section stay together and each chunk is prefixed with its heading. Documents without headings use overlapping paragraph and sentence chunks (1000 characters, 200 overlap). Audio and video transcripts are chunked by segment so every chunk keeps its start and end timestamps.

**Query pipeline.**
1. Rewrite the question using recent chat history
2. Retrieve candidates: dense search (optionally + BM25, optionally reranked)
3. Compare the best score to the relevance threshold; abstain if it is lower
4. Send the top chunks to the LLM with an instruction to answer only from them and cite
5. Optionally check each answer sentence against the context (NLI) and flag unsupported ones

**Thresholds.**

| Setting | Default | Meaning |
|---|---|---|
| Dense relevance threshold | 0.15 cosine | best chunk below this → "not found" |
| Reranker threshold | 0.125 (stock reranker), 0.30 (fine-tuned) | the same gate on the reranker's probability scale |
| Grounding threshold | 0.6 | fraction of supported sentences below which an answer is flagged |

The benchmark-optimal dense threshold is 0.30, but real lecture PDFs often score the right chunk only 0.15-0.3, so 0.15 is the shipped default to avoid false refusals. Set `RELEVANCE_THRESHOLD=0.30` for stricter abstention. Reasoning and sweeps are in [docs/evaluation.md](docs/evaluation.md).

## Evaluation and Fine-tuning
An evaluation harness in `backend/eval/` measures retrieval, abstention thresholds, hallucination detection, model size and query rewriting on public benchmarks. Training and evaluation use separate splits.

| Dataset | Used for |
|---|---|
| QASPER | retrieval evaluation, embedder and reranker fine-tuning |
| SQuAD v2 | abstention threshold, answer generation |
| CoQA | follow-up question rewriting |
| RAGTruth, RAGBench | hallucination detector training and evaluation |

### Retrieval results (QASPER validation, 100 questions)
| Configuration | hit@1 | hit@5 | hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| MiniLM, fixed chunks (baseline) | 0.06 | 0.20 | 0.30 | 0.12 |
| BGE dense | 0.13 | 0.25 | 0.35 | 0.18 |
| BGE hybrid | 0.08 | 0.26 | 0.36 | 0.16 |
| BGE hybrid + reranker | 0.13 | 0.34 | 0.43 | 0.22 |
| BGE hybrid + **fine-tuned reranker** | 0.18 | 0.42 | - | 0.27 |
| **MiniLM hybrid + fine-tuned reranker (recommended)** | **0.19** | **0.45** | - | **0.29** |

Retrieval is still the weakest part, but fine-tuning the reranker on QASPER lifted hit@5 from 0.34 to 0.45 (best configuration, about 1.3 s per query). The numbers come from scientific papers, not lecture material, and n=100, so treat them as indicative.

### Hallucination detector (fine-tuned NLI model)
| Test set (n=300) | Original AUROC | Fine-tuned AUROC |
|---|---:|---:|
| RAGTruth (held out, never trained on) | 0.65 | 0.74 |
| RAGBench test | 0.61 | 0.60 |

The fine-tune helps on RAGTruth but gives no gain on RAGBench test, so it is **not** the default. See [docs/evaluation.md](docs/evaluation.md) section 9 for the retrieval fine-tuning results and the recommended profile.

### Small vs larger LLM (SQuAD v2, n=120)
| | qwen2.5:1.5b | qwen2.5:3b |
|---|---:|---:|
| Answer recall | 0.90 | 0.82 |
| Refuses answerable questions | 5% | 13% |
| Refuses unanswerable questions | 20% | 60% |
| p95 latency | 17 s | 27 s |

Larger models refuse unanswerable questions more reliably but also refuse more answerable ones and are slower.

### Reproduce
```bash
cd backend
python -m eval.run retrieval     --dataset qasper -n 100
python -m eval.run threshold     --dataset squad_v2 -n 100
python -m eval.run hallucination --source ragtruth
python -m eval.finetune_grounding --pairs 2000 --epochs 2   # hallucination checker, RAGBench train
python -m eval.finetune_embedder  --pairs 3000 --epochs 2   # embedder, QASPER train + hard negatives
python -m eval.finetune_reranker  --pairs 3000 --epochs 2   # reranker, QASPER train
```
Trained weights are written to the git-ignored `backend/models/`.

## API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload` | upload a PDF, audio or video file |
| POST | `/query` | ask a question (optionally scoped to chosen documents) |
| GET | `/documents` | list the session's documents |
| DELETE | `/documents/{id}` | delete a document (vectors and file) |
| GET | `/history` | load the session's chat history |
| DELETE | `/history` | clear the chat history |
| GET | `/health` | health check |

## Running the Project

Prerequisites: Python 3.11+, Node 20+, [ffmpeg](https://ffmpeg.org/) on PATH (`winget install ffmpeg` / `apt install ffmpeg` / `brew install ffmpeg`) and [Ollama](https://ollama.com/).

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
The first upload or query downloads the embedding model (and Whisper/CLIP when you ingest media), so it is slow once.

### Docker
```bash
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:1.5b      # first run only
```
Frontend: http://localhost:3000, API: http://localhost:8000.

### Best-quality profile
For better retrieval and meaningful match scores, put this in `backend/.env` (see section 9 of [docs/evaluation.md](docs/evaluation.md); the fine-tuned reranker is trained locally with `python -m eval.finetune_reranker`):
```
USE_HYBRID=1
USE_RERANKER=1
RERANKER_MODEL=models/reranker-lecturemind
RELEVANCE_THRESHOLD=0.30
```
With the reranker on, the "match" percentage shown on each source is the reranker's relevance probability, and the sources listed are only the ones the answer cites.

### Tuning for a CPU-only laptop
The defaults are the lightest configuration: dense retrieval only, no reranker, no NLI check, one ingestion at a time and Whisper `base`.

| Variable | Suggested | Why |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:3b` for better answers, `qwen2.5:1.5b` (default) for speed | the 1.5b model sometimes answers with a bare heading or over-refuses |
| `USE_HYBRID` | `1` | adds keyword search; helps section words like "projects" |
| `TOP_K` | `4`-`6` | chunks sent to the LLM |
| `LLM_NUM_CTX` | `4096` | smaller context is faster on CPU |
| `OLLAMA_TIMEOUT` | `300` | the first request after a restart loads the model and can be slow |
| `WHISPER_MODEL` | `tiny` | faster transcription, lower quality |

Re-upload documents after changing chunking or the embedding model.

## Configuration
All settings are environment variables; see [.env.example](.env.example) for the documented list: Ollama URL, model and timeouts, model names, chunk size and overlap, storage paths, resource limits (25 MB PDF, 30 min media, 500 MB and 20 documents per session), rate limits and CORS origins. Do not commit real secrets to the repository.

## Tests
```bash
cd backend
pip install -r requirements-dev.txt
pytest
```
The suite fakes the embedder, LLM, Whisper and CLIP, so it needs no model downloads. It covers chunking, upload limits, every modality's round trip, **session isolation**, deletion (vectors and file), the empty-context guard, rate limiting and error mapping. It never asserts exact LLM wording.

## Project Structure
```
LectureMindAI/
├── backend/
│   ├── app/
│   │   ├── routes/        # upload, query, documents
│   │   ├── services/      # ingestion, retrieval, llm, grounding, transcription, clip
│   │   ├── db/            # chroma.py, sqlite.py
│   │   ├── utils/         # chunker, media helpers
│   │   └── config.py      # all settings
│   ├── eval/              # evaluation harness and fine-tuning scripts
│   └── tests/
├── frontend/src/          # Next.js app (chat page, upload panel, citations)
├── docs/                  # architecture, security, evaluation, deployment, walkthrough
├── docker-compose.yml
├── PROJECT_SPEC.md
└── TASKS.md
```

## Known Limitations
- Audio and video ingestion are implemented and tested with fakes; the PDF path is the one exercised end to end with real models
- With `USE_HYBRID=1` the relevance threshold is weaker (keyword-only matches get a neutral score), so the LLM's own refusal does more of the work
- The `model` field of `/query` is client-controlled; do not expose the API publicly without restricting it
- Sessions are browser-local UUIDs: clearing site data or switching browsers loses access to earlier documents and history. This is a deliberate V1 trade-off, not real authentication
- Scanned PDFs (no text layer) are rejected; there is no OCR
- Retrieval quality is still limited (hit@5 of 0.45 on QASPER at best), and the hallucination flag is only moderately reliable, so treat it as a warning, not a guarantee

## What I Learned
- Building a RAG pipeline end to end without a framework: chunking, embeddings, retrieval, prompting and sessions
- Why retrieval quality, not the language model, is usually the bottleneck
- Calibrating abstention thresholds with AUROC and sweeps instead of guessing
- Measuring hallucination detection honestly: a fine-tune can win on one dataset and not on another
- Fine-tuning embedders, rerankers and NLI models on a CPU-only laptop within a few GB of RAM
- Multimodal ingestion (PDF, speech, video frames) and cross-modal retrieval with CLIP
- Session isolation, resource limits and rate limiting for an app that accepts uploads
- Testing LLM applications by testing the plumbing around the model, not its exact output

## Future Improvements
- Ship the fine-tuned reranker weights (currently local only) and verify the gain on real lecture material
- A stronger hallucination detector (more training data, MiniCheck or HHEM)
- Hand-labelled lecture questions for an in-domain metric
- Lecture-specific benchmarks (slides plus speech) and long-document PDF benchmarks
- OCR for scanned PDFs
- Real user accounts instead of browser-local sessions
- Larger-n evaluation (n ≥ 500) with confidence intervals and 7B/8B model comparisons
- Streaming answers in the chat UI

## Conclusion
LectureMindAI turns a pile of lecture PDFs and recordings into a chat interface whose answers cite their sources. The main engineering focus was keeping each part small and measurable: ingestion, chunking, retrieval, thresholds and hallucination checks are separate pieces, each with an evaluation behind it, so the system's weak spots (retrieval recall, a moderate hallucination detector) are documented rather than hidden.
