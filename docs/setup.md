# Setup Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai/) installed and running
- [ffmpeg](https://ffmpeg.org/) on PATH (`ffmpeg` and `ffprobe`)
- CPU-only machine supported (no GPU required)

## Ollama Model

```bash
ollama pull qwen2.5:1.5b
```

Verify: `curl http://localhost:11434/api/tags`

## Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Copy env config
cp ../.env.example ../.env   # edit as needed

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First run downloads embedding/Whisper/CLIP models (may take several minutes on CPU).

## Frontend

```bash
cd frontend
npm install
cp ../.env.example .env.local   # set NEXT_PUBLIC_API_URL
npm run dev
```

Open http://localhost:3000

## Tests

```bash
cd backend
pytest -v
```

## Docker

```bash
docker-compose up --build
```

Note: Ollama must be reachable from the backend container. By default `OLLAMA_URL=http://host.docker.internal:11434` on Docker Desktop.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Ollama connection refused | Start Ollama; verify `OLLAMA_URL` |
| ffmpeg not found | Install ffmpeg and add to PATH |
| Slow transcription | Set `WHISPER_MODEL=tiny` in `.env` |
| CORS errors | Set `CORS_ORIGINS` to your frontend URL |
