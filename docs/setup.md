# Setup

1. **Install**: Python 3.11+, Node 20+, ffmpeg (provides `ffmpeg` and `ffprobe`), Ollama.
2. **Pull the LLM**: `ollama pull qwen2.5:1.5b`.
3. **Backend**: in `backend/`, create a venv, `pip install -r requirements.txt`, then `uvicorn app.main:app --port 8000`. Optionally copy `.env.example` to `backend/.env`.
4. **Frontend**: in `frontend/`, `npm ci` then `npm run dev`. Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the API is not on `http://localhost:8000`.
5. **Check**: open http://localhost:3000, upload a small PDF, ask a question. `GET http://localhost:8000/health` returns `{"status":"ok"}`.

## Troubleshooting
- *"The language model (Ollama) is unavailable"* (503): start Ollama and pull the model.
- *"ffmpeg is not installed"*: install ffmpeg and make sure `ffmpeg -version` works in the same terminal.
- *Slow first request*: models are downloaded on first use (embedder ~90 MB, Whisper base ~150 MB, CLIP ~600 MB).
- *Machine slows down*: keep `USE_RERANKER=0` and `USE_GROUNDING_CHECK=0`, use `WHISPER_MODEL=tiny`, close other heavy apps, and upload short clips first.
- *CORS error in the browser*: add the frontend origin to `CORS_ORIGINS`.

## Tests
`cd backend && pip install -r requirements-dev.txt && pytest` (no model downloads needed).
