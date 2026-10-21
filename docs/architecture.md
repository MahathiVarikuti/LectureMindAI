# LectureMindAI Architecture

## Overview

LectureMindAI is a multimodal RAG system with a Next.js frontend and FastAPI backend. All AI inference runs locally (Ollama, SentenceTransformers, faster-whisper, CLIP).

## Dual-Collection Vector Store

ChromaDB stores embeddings in two separate collections:

| Collection     | Content                        | Embedding Model          |
|----------------|--------------------------------|--------------------------|
| `text_chunks`  | PDF text, audio/video transcripts | all-MiniLM-L6-v2      |
| `image_frames` | Sampled video frames           | CLIP ViT-B/32            |

Text and image embeddings live in different semantic spaces. At query time each collection is searched independently; scores are min-max normalized within each result set before merging (scores are not directly comparable across collections).

## Session Model

- Client generates a UUID v4 on first load, stored in `localStorage`.
- Every API call includes `session_id`.
- SQLite and ChromaDB queries filter by `session_id`.
- No login — clearing browser storage loses access to prior data.

## RAG Pipeline

```
User query
  → Load chat history (SQLite)
  → Rewrite query via Ollama (if history exists)
  → Embed query (SentenceTransformers + CLIP)
  → Query text_chunks + image_frames (session-scoped)
  → Normalize scores independently
  → Assemble labeled context
  → If no relevant chunks: return "not found" (skip LLM)
  → Generate answer via Ollama
  → Persist exchange to SQLite
  → Return answer + citations
```

## Ingestion Pipelines

- **PDF:** PyMuPDF page extraction → chunk → embed → `text_chunks`
- **Audio:** faster-whisper → segment-aware chunking → embed → `text_chunks`
- **Video:** ffmpeg audio extract → transcript pipeline + frame sampling → CLIP → `image_frames`
- **URL:** yt-dlp download → route to audio/video pipeline
