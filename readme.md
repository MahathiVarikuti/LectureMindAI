# LectureMindAI

Multimodal RAG system for lecture materials. Upload PDFs, audio, video, or URLs and ask natural-language questions with grounded answers and source citations.

## Features

- PDF, audio, video upload and URL ingestion (yt-dlp)
- Local LLM answers via Ollama (`qwen2.5:1.5b`)
- Source citations: page numbers (PDF) and timestamps (audio/video)
- Cross-modal retrieval: text chunks + video frames
- Session-based usage (no login), persistent chat history in SQLite
- Document management: list, scope queries, delete

## Architecture

```
┌─────────────┐     HTTP      ┌──────────────┐
│  Next.js    │ ────────────► │   FastAPI    │
│  Frontend   │               │   Backend    │
└─────────────┘               └──────┬───────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
         SQLite (chat)          ChromaDB              File storage
                            text_chunks + image_frames
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
         Ollama LLM          SentenceTransformers      faster-whisper + CLIP
```

See [docs/architecture.md](docs/architecture.md) for details.

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 18+
- [Ollama](https://ollama.ai/) with `qwen2.5:1.5b` pulled
- [ffmpeg](https://ffmpeg.org/) on PATH

```bash
ollama pull qwen2.5:1.5b
```

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

Full setup: [docs/setup.md](docs/setup.md)

## Environment Variables

Copy `.env.example` to `.env` and adjust. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | LLM model name |
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `CHUNK_SIZE` | `1000` | Text chunk size (chars) |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed frontend origins |

## Tests

```bash
cd backend
pytest -v
```

## Docker

```bash
docker-compose up --build
```

Ollama must run on the host (or a reachable container). Set `OLLAMA_URL` accordingly.

## Deployment

Free-tier deployment options (Render, Railway, Fly.io) can host the FastAPI backend and Next.js frontend, but **Ollama requires significant CPU/RAM** (~2GB+ for qwen2.5:1.5b). Most free tiers cannot run Ollama alongside the app.

**Recommended approach:**
1. Run Ollama on a local machine or a dedicated VPS with ≥4GB RAM
2. Deploy backend + frontend to a free host, pointing `OLLAMA_URL` to your Ollama instance
3. If Ollama is unavailable, queries will fail gracefully with an error message

Alternative: run everything locally via `docker-compose` with Ollama installed on the host.

## Security

Session-based isolation without login. See [docs/security.md](docs/security.md).

## License

MIT
