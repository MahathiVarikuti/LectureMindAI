# Real-Time Multimodal RAG Study Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system built from scratch to understand the internals of modern AI retrieval systems without relying on frameworks such as LangChain or LlamaIndex.

The project enables users to upload lecture PDFs and ask natural-language questions about the content. It performs semantic retrieval using vector embeddings and generates grounded responses using a locally hosted Large Language Model (LLM).

## Features

### Implemented (V1)

* PDF upload and ingestion
* Text extraction using PyMuPDF
* Configurable text chunking with overlap
* Semantic embeddings using BAAI/bge-small-en-v1.5
* Vector storage and similarity search using ChromaDB
* Context-aware question answering using Qwen 2.5 via Ollama
* Source metadata tracking
* Document identifiers for scalable multi-document support
* Conversational query rewriting for follow-up questions
* FastAPI backend APIs
* Next.js frontend interface

### Implemented (V2 — retrieval quality & hallucination control)

* **Structure-aware chunking** that packs whole sentences/paragraphs instead of
  cutting mid-word, with de-hyphenation of PDF line breaks ([chunker.py](backend/app/utils/chunker.py))
* **Hybrid retrieval**: dense BGE embeddings + BM25 keyword search, fused with
  Reciprocal Rank Fusion, so exact terms (acronyms, symbols, names) are no
  longer lost to embedding similarity alone ([retrieval_service.py](backend/app/services/retrieval_service.py))
* **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) re-scores the fused
  candidates by reading query + chunk together — far more accurate than
  comparing two independent vectors, and it produces a calibrated relevance
  score the system can threshold on
* **Calibrated abstention**: if the best chunk's relevance score is below a
  threshold (calibrated on SQuAD 2.0's unanswerable questions, see
  [Evaluation](#evaluation--datasets)), the system refuses *before* calling the
  LLM — no hallucinated answer from an empty retrieval
* **Grounding / hallucination check**: after generation, every answer sentence
  is checked against the retrieved chunks with a small NLI model
  (`cross-encoder/nli-deberta-v3-small`); low-support answers are flagged with
  the unsupported sentences shown to the user ([grounding_service.py](backend/app/services/grounding_service.py))
* **Grounded, cited answers**: the LLM prompt requires `[n]` citations per
  claim and an explicit refusal line, enforced with `num_ctx` large enough
  that the system prompt is never silently truncated
* **Smarter query rewriting**: only rewrites questions that look like
  follow-ups (pronouns, short questions), with a few-shot prompt and
  sanity-checked output, instead of rewriting every turn
* **Duplicate-upload detection** by file hash, document-filtered search,
  session reset, and an index that's kept in sync (BM25 + Chroma) on
  upload/delete
* **Evaluation harness** (`backend/eval/`) against four public RAG benchmarks
  to measure retrieval quality, calibrate the abstention threshold, compare
  LLM sizes, and validate the hallucination detector — see below

### Example Queries

* What is gradient descent?
* Explain backpropagation in simple terms.
* Summarize the key concepts discussed in this document.
* How does it differ from stochastic gradient descent?

## Architecture

```text
PDF Upload
    ↓
Text Extraction (PyMuPDF, column-aware)
    ↓
Structure-aware Chunking (sentence/paragraph packing)
    ↓
Embeddings (BGE) ──────────────┐
    ↓                          │
ChromaDB (dense)      BM25 index (sparse)
    ↓                          │
    └──────────┬───────────────┘
               ↓
     Reciprocal Rank Fusion
               ↓
     Cross-Encoder Reranking
               ↓
     Relevance Threshold ──→ (below threshold) → Abstain, no LLM call
               ↓
     Query Rewriting (follow-ups only)
               ↓
     Context Assembly + Citations
               ↓
     Qwen 2.5 (Ollama)
               ↓
     NLI Grounding Check ──→ (low support) → Flag possible hallucination
               ↓
     Grounded, Cited Answer
```

## Tech Stack

### Frontend

* Next.js 15
* TypeScript
* Tailwind CSS
* Axios

### Backend

* FastAPI
* Uvicorn

### AI / ML

* Embeddings: `BAAI/bge-small-en-v1.5` (SentenceTransformers), optionally fine-tuned
  on QASPER with `eval/finetune_embedder.py`
* Sparse retrieval: BM25 (`rank-bm25`)
* Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Hallucination detection: `cross-encoder/nli-deberta-v3-small` (NLI)
* LLM: Qwen 2.5 (1.5B / 3B / 7B — pick per `eval/run.py generation`) via Ollama

### Database

* ChromaDB

### Document Processing

* PyMuPDF (fitz)

## Project Structure

```text
backend/
│
├── app/
│   ├── routes/            # upload, query, documents (HTTP layer only)
│   ├── services/          # embedding, retrieval, grounding, LLM, pdf, chat memory
│   ├── db/                 # Chroma client/collection
│   ├── utils/              # chunker
│   ├── config.py           # every tunable, overridable via env vars
│   └── main.py
│
├── eval/                   # evaluation harness (see below) — not shipped at runtime
│   ├── datasets.py          # loaders for qasper/squad_v2/coqa/ragtruth
│   ├── metrics.py           # EM/F1, retrieval hit/MRR/nDCG, AUROC, threshold sweep
│   ├── run.py                # retrieval / threshold / generation / hallucination / conversation
│   └── finetune_embedder.py  # fine-tunes the embedding model on QASPER
│
├── uploads/
└── chroma_db/

frontend/
│
└── src/
    ├── app/
    ├── components/
    └── lib/
```

## How It Works

### 1. Document Ingestion

When a PDF is uploaded:

* The file is hashed; an identical file already indexed is rejected as a duplicate
* Text is extracted using PyMuPDF (column-aware ordering) and cleaned (de-hyphenation, whitespace normalisation)
* Text is split into chunks that respect paragraph/sentence boundaries, with overlap
* Embeddings are generated using BGE
* Chunks and metadata are stored in ChromaDB, and the in-memory BM25 index is invalidated so it rebuilds with the new chunks

### 2. Query Processing

When a user asks a question:

* If the question looks like a follow-up (pronouns, very short), conversation history is used to rewrite it into a standalone query; otherwise the LLM rewrite step is skipped entirely
* The query is searched with **both** a dense (embedding) and a sparse (BM25) retriever; the two ranked lists are fused with Reciprocal Rank Fusion
* The fused candidates are reranked by a cross-encoder, which produces the final relevance scores
* If the best score is below `RELEVANCE_THRESHOLD`, the system **abstains without calling the LLM**
* Otherwise, the top chunks are assembled into a numbered context block and passed to Qwen through Ollama, with a system prompt that requires `[n]` citations and an explicit refusal line
* The generated answer is checked sentence-by-sentence against the retrieved chunks with an NLI model; if too few sentences are supported, the answer is flagged as a possible hallucination (shown in the UI, not hidden)

### 3. Metadata Storage

Each chunk stores metadata such as:

```json
{
  "document_id": "...",
  "filename": "...",
  "stored_filename": "...",
  "file_hash": "...",
  "page": 1,
  "chunk_index": 0,
  "uploaded_at": "..."
}
```

This enables:

* Multi-document retrieval and document-filtered search (`selected_documents`)
* Source citations with per-chunk relevance scores
* Page-level references
* Duplicate-upload rejection by content hash

## Running Locally

### Backend

Create a virtual environment and install dependencies (CPU-only PyTorch by default; remove the `--extra-index-url` line in `requirements.txt` if you have a CUDA GPU):

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

> The embedding, reranker and NLI models are downloaded from Hugging Face on first run (~500 MB total) and cached afterward. **Documents must be re-uploaded after upgrading from V1**: the embedding model and chunking changed, so the old `lecture_notes` collection is incompatible (the app now writes to a separate `lecture_notes_v2` collection, so old data isn't deleted — just unused).

### Ollama

Install Ollama and pull a model (any size works; see [Evaluation](#evaluation--datasets) for the EM/F1/latency trade-off between sizes):

```bash
ollama pull qwen2.5:1.5b   # fastest, used by default
ollama pull qwen2.5:3b     # better quality, still CPU-friendly
ollama pull qwen2.5:7b     # best quality, slowest on CPU
```

Run the model:

```bash
ollama run qwen2.5:1.5b
```

Override the model per-request with `"model": "qwen2.5:7b"` in the `/query` body, or set `LLM_MODEL=qwen2.5:7b` as an environment variable for the whole server.

### Frontend

Install dependencies:

```bash
npm install
```

Run the application:

```bash
npm run dev
```

## API Endpoints

### Upload PDF

```http
POST /upload
```

Uploads and indexes a PDF document. Rejects non-PDFs, PDFs with no extractable text, and exact duplicate files (by content hash — returns `"duplicate": true` instead of re-indexing).

### List / Delete Documents

```http
GET /documents
DELETE /documents/{document_id}
```

### Ask Question

```http
POST /query
```

Request:

```json
{
  "query": "How does it differ from stochastic gradient descent?",
  "session_id": "uuid-from-the-client",
  "selected_documents": [],
  "model": "qwen2.5:1.5b"
}
```

`selected_documents` restricts retrieval to those document IDs (empty = search everything). `model` is optional and overrides `LLM_MODEL` for this request.

Response:

```json
{
  "answer": "SGD estimates the gradient from a mini-batch instead of the whole dataset [2]. ...",
  "rewritten_query": "What is the difference between SGD and batch gradient descent?",
  "abstained": false,
  "abstain_reason": null,
  "top_score": 0.98,
  "flagged": false,
  "grounding": {"groundedness": 1.0, "unsupported": []},
  "sources": [
    {"chunk_id": "...", "document_id": "...", "filename": "lecture3.pdf", "page": 2, "text": "...", "score": 0.98}
  ],
  "timings": {"rewrite": 0.0, "retrieval": 0.3, "generation": 1.2, "grounding": 0.2, "total": 1.7}
}
```

`abstained: true` means the system refused to answer (either retrieval found nothing relevant, or the LLM itself said so); `flagged: true` means the answer was generated but the grounding check found unsupported claims — `grounding.unsupported` lists them so the UI can warn the user instead of silently trusting the model.

### Reset a Chat Session

```http
DELETE /session/{session_id}
```

## Evaluation & Datasets

The project previously had no way to measure retrieval quality, pick a hallucination threshold, or compare LLM sizes — changes were "vibes-based." `backend/eval/` fixes that using four public benchmarks downloaded on demand from the Hugging Face Hub (curated from [Awesome-RAG-Datasets](https://github.com/jinbo0906/Awesome-RAG-Datasets)):

| Dataset | Why this one | Used for |
|---|---|---|
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | Long, technical, section-structured documents — the closest public analogue to lecture notes/slides; has unanswerable questions | Retrieval ablation, embedding fine-tuning |
| [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) | 1/3 of questions are deliberately unanswerable from the given passage — ideal for calibrating when the system should refuse | Abstention-threshold calibration |
| [CoQA](https://huggingface.co/datasets/stanfordnlp/coqa) | Multi-turn conversations where follow-ups use pronouns ("it", "this") | Query-rewriting evaluation |
| [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) | LLM responses with human-annotated, word-level hallucination labels | Hallucination-detector validation |

Run from `backend/` (same venv as the server):

```bash
# 1. Does the new retrieval (hybrid + rerank) actually beat the V1 pipeline?
python -m eval.run retrieval --dataset qasper -n 300

# 2. What RELEVANCE_THRESHOLD should the system abstain below?
python -m eval.run threshold --dataset squad_v2 -n 300

# 3. End-to-end quality/latency/hallucination-rate, across LLM sizes
python -m eval.run generation --dataset squad_v2 --models qwen2.5:1.5b qwen2.5:3b qwen2.5:7b -n 120

# 4. Does the NLI grounding check actually catch hallucinations?
python -m eval.run hallucination -n 300

# 5. Does query rewriting help retrieval on multi-turn conversations?
python -m eval.run conversation -n 150
```

Each command prints a summary table and writes full results + per-question predictions to `eval/results/*.json(l)`.

### What the numbers said (QASPER/SQuAD 2.0, 300 questions, this machine)

**Retrieval** — hit@5 (is a truly relevant chunk in the top 5?):

| Pipeline | hit@5 | mrr@10 |
|---|---|---|
| V1 (fixed-size chunks, MiniLM, dense only) | 0.196 | 0.123 |
| + structure-aware chunking, BGE embeddings | 0.259 | 0.157 |
| + BM25 hybrid search | 0.277 | 0.166 |
| + cross-encoder reranking | **0.366** | **0.236** |

Each change compounds: better chunking and embeddings nearly double hit@5 over V1, hybrid search recovers exact-term matches the embedding model misses, and reranking gives the single largest jump by actually reading the query against each candidate.

**Abstention threshold** (SQuAD 2.0, 300 in-scope vs. 300 out-of-scope questions): the reranked relevance score separates the two classes almost perfectly (AUROC 0.994 — in-scope median score 0.996, out-of-scope median 0.000). `RELEVANCE_THRESHOLD = 0.125` answers 97% of in-scope questions while correctly refusing 97% of out-of-scope ones.

**Hallucination detector** (RAGTruth, 300 QA responses, ~16% hallucinated): the NLI groundedness score reaches AUROC 0.65-0.74 at separating hallucinated from clean responses — far from perfect, but at the current `GROUNDEDNESS_THRESHOLD = 0.6` it catches 92% of hallucinated responses (recall) at the cost of flagging some clean ones too (precision ~0.21). For a student-facing tool, that trade-off is intentional: a false "this might be wrong" warning costs little, a missed hallucination costs trust. Re-run `eval.run hallucination` after any prompt or chunking change — the threshold is a knob, not a law.

**LLM size** (SQuAD 2.0, 120 questions: 60 answerable, 30 unanswerable, 30 out-of-scope, same retrieval pipeline for both):

| Model | EM | F1 | answer recall | false refusal | groundedness | hallucination-flag rate | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:1.5b | 0.017 | 0.350 | **0.895** | **0.050** | 0.434 | 0.580 | **7.4s** | **17.0s** |
| qwen2.5:3b | 0.000 | 0.346 | 0.818 | 0.133 | 0.438 | 0.578 | 10.0s | 27.1s |

(EM/F1 are near-zero for both because SQuAD's gold answers are short extractive spans — "France" — while the system is prompted to answer in full cited sentences; **answer recall**, the fraction of gold-answer words present in the generated answer, is the metric that actually reflects whether the content is right.) On this CPU laptop, the larger model was not better: lower answer recall, more than double the false-refusal rate on answerable questions, no real change in groundedness, and 35-60% higher latency. The counter-intuitive takeaway: **for a retrieval-grounded assistant, the retrieval/reranking stack does far more for answer quality than LLM size does** — the 1.5B model is the better default here. The `--models` flag makes this a two-minute check whenever a new model is worth trying; a GPU-backed deployment may shift the trade-off.

The hallucination-flag rate (58%) looks alarming but is partly an artifact of this test: SQuAD passages are short and the LLM often adds reasonable connective phrasing the strict sentence-level NLI model doesn't find literally entailed. Compare against the RAGTruth numbers above, which measure the detector against *actual* human-labelled hallucinations rather than stylistic paraphrase — that's the number to trust for tuning `GROUNDEDNESS_THRESHOLD`.

**Query rewriting** (CoQA, 150 pronoun-laden follow-up questions like "How does it differ from this?"):

| Query sent to retrieval | hit@1 | hit@5 | mrr@5 |
|---|---|---|---|
| Raw follow-up question, unchanged | 0.480 | 0.680 | 0.558 |
| Naive: previous question + question | 0.647 | 0.813 | 0.717 |
| LLM rewrite (this project's approach) | **0.667** | **0.833** | **0.735** |

Rewriting follow-ups into standalone questions nearly doubles the hit@1 rate over sending the raw pronoun-laden question to the retriever, and it also beats the "just concatenate the previous turn" shortcut — confirming the LLM rewrite step is worth its latency cost, which is why it's skipped for questions that are already self-contained (see `needs_rewrite` in [`llm_service.py`](backend/app/services/llm_service.py)).

Re-run these commands after changing the embedding model, chunk size, or thresholds — the numbers above are a snapshot, not a guarantee.

### Fine-tuning the embedding model

`eval/finetune_embedder.py` fine-tunes `BAAI/bge-small-en-v1.5` on question/evidence pairs mined from QASPER's **train** split (evaluation always uses the **validation** split, so there's no leakage). The training data also mines same-paper hard negatives for future experiments; the current `MultipleNegativesRankingLoss` uses in-batch negatives because the installed sentence-transformers API accepts anchor/positive pairs:

```bash
python -m eval.finetune_embedder --pairs 3000 --epochs 2
# then compare it against the base model:
python -m eval.run retrieval --dataset qasper --embed-model models/bge-small-lecturemind
```

This keeps the fine-tuning command compatible with current sentence-transformers releases. If hard-negative triplet training is added later, compare it against this pair-loss baseline on the held-out QASPER validation split rather than assuming it improves retrieval.

### Configuration

Every threshold lives in [`app/config.py`](backend/app/config.py) and can be overridden with an environment variable of the same name (e.g. `RELEVANCE_THRESHOLD=0.2`, `LLM_MODEL=qwen2.5:7b`, `EMBEDDING_MODEL=models/bge-small-lecturemind`). Recalibrate `RELEVANCE_THRESHOLD` and `GROUNDEDNESS_THRESHOLD` with the `eval` commands above whenever the embedding model, chunk size, or LLM changes.

## Roadmap

### V2 — Multi-Document Knowledge Base ✅

* Multiple PDF support
* Document filtering
* Cross-document retrieval
* Source citations
* Structure-aware chunking, hybrid retrieval, reranking (pulled forward from V6)
* Calibrated abstention + NLI-based hallucination flagging
* Evaluation harness against public RAG benchmarks

### V2.5 — Remaining Gaps

* OCR fallback for scanned PDFs (currently rejected with a 422 if no text is extractable)
* Query expansion / context compression (the rest of V6)
* Persistent chat memory (sessions are in-memory and lost on restart — move to SQLite/Redis)
* Automatic re-indexing when `EMBEDDING_MODEL` changes, instead of a manual re-upload
* Frontend: surface `abstain_reason`/latency breakdown, a document-management page using `/documents`

### V3 — Audio RAG

* Whisper transcription
* Timestamp-aware retrieval
* Lecture audio support

### V4 — Video Processing

* Video ingestion
* Frame extraction
* Timestamp retrieval

### V5 — Multimodal RAG

* Unified retrieval across PDFs, audio, and video
* Cross-modal search
* Context fusion

### V6 — Advanced Retrieval

* Hybrid Search (BM25 + Vector Search) ✅ — see V2
* Reranking ✅ — see V2
* Query Expansion
* Context Compression

### V7 — Real-Time Processing

* Streaming ingestion
* WebSockets
* Live querying

### V8 — AI Study Assistant

* Summaries
* Flashcards
* Quiz generation
* Topic extraction

### V9 — Production Deployment

* Authentication
* User accounts
* Background jobs
* Monitoring and analytics

## Why No LangChain or LlamaIndex?

This project intentionally avoids abstraction frameworks in order to:

* Understand RAG internals deeply
* Learn retrieval engineering concepts
* Gain debugging experience
* Build AI systems from first principles

The goal is to master the underlying architecture before introducing higher-level frameworks.

## License

MIT License
