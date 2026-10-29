# Architecture

## Components
- **Frontend** (Next.js App Router, TypeScript, Tailwind, Axios): upload panel (file or URL), document sidebar (select / delete), chat with Markdown answers and modality-aware citations. A random UUID in `localStorage` is the session id sent with every request.
- **Backend** (FastAPI): routes in `app/routes/`, pipeline code in `app/services/`, storage in `app/db/`.
- **SQLite** (`chat_history.db`): `messages` (chat history) and `documents` (ownership, file path, size, hash). Every query filters on `session_id`.
- **ChromaDB** (persistent): two collections.
- **Ollama**: `qwen2.5:1.5b` over `/api/generate` with `stream: false` and a timeout.

## Dual-collection vector store
| collection | content | embedder | metadata |
|---|---|---|---|
| `text_chunks` | PDF text, audio transcripts, video transcripts | `all-MiniLM-L6-v2` (384-d) | session_id, document_id, filename, modality (`pdf`/`audio`/`video_transcript`), chunk_id, page *or* timestamp_start/end, uploaded_at |
| `image_frames` | video frames, one per 7 s | CLIP `clip-vit-base-patch32` (512-d) | session_id, document_id, filename, modality=`video_frame`, timestamp, uploaded_at |

They are separate on purpose. Forcing text and images into a single CLIP space would hurt text-to-text retrieval, the dominant use case. Optional metadata (page / timestamps) is omitted rather than set to `None` because Chroma rejects null values.

**Merging scores.** CLIP text-to-image similarities (~0.2-0.35) and MiniLM cosine similarities (~0.3-0.8) are not comparable. We therefore never rank them in one list: each collection is queried independently and thresholded on its own scale (`RELEVANCE_THRESHOLD`, `IMAGE_MIN_SCORE`); frames then appear as a separate labelled section in the LLM context and as separate citations. CLIP is only loaded when the session actually has frames.

## Session isolation
Documents and chat history are scoped by `session_id` in three places: SQLite `WHERE session_id = ?`, Chroma `where={"session_id": ...}` on every query/delete, and ownership checks in the routes (`selected_documents` must belong to the session; deleting another session's document returns the same 404 as an unknown id). `tests/test_api.py::TestSessionIsolation` verifies it.

## Ingestion
- **PDF**: PyMuPDF per page -> structure-aware chunking with overlap -> embed -> store with `page`.
- **Audio**: ffprobe duration check (before any heavy work) -> faster-whisper (CPU, int8) -> segment-respecting chunking (`chunk_segments`) -> embed -> store with `timestamp_start/end`.
- **Video**: duration check -> ffmpeg extracts the audio track (-> audio pipeline, modality `video_transcript`) and samples frames (`FRAME_SAMPLE_INTERVAL`, default 7 s) -> CLIP -> `image_frames`.

All limits (size, duration, storage, document count, concurrent ingestions) are enforced before expensive work; failures purge partial vectors and files.

## Query pipeline (`app/services/rag_pipeline.py`)
1. Load recent history from SQLite; if any, rewrite the question to be standalone (skipped without history).
2. Embed the query and search `text_chunks` filtered by session (and selected documents).
3. If the session has video frames, embed the query with CLIP and search `image_frames`.
4. Apply the relevance threshold. If no text chunk passes: return "not found" **without calling the LLM** (if only frames match, list those moments, also without the LLM).
5. Build a labelled context (`[n] (filename, page N | mm:ss-mm:ss)` plus a frames section) and call the LLM.
6. Persist the exchange and return `{answer, sources, used_documents, context, ...}`.

Optional, off by default (they come from the evaluation work, see `evaluation.md`): hybrid BM25 + dense fusion, cross-encoder reranking, NLI grounding check.
