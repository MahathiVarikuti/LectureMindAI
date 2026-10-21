# Deployment Notes

## Free-Tier Constraints

LectureMindAI depends on locally-hosted models:

| Component | RAM (approx) | CPU |
|-----------|-------------|-----|
| Ollama qwen2.5:1.5b | ~1.5 GB | Heavy during inference |
| SentenceTransformers | ~100 MB | Moderate |
| faster-whisper base | ~150 MB | Heavy during transcription |
| CLIP ViT-B/32 | ~350 MB | Moderate |

Total backend footprint can exceed 2 GB during video ingestion.

## Can Ollama Run on Free-Tier Hosts?

Most free-tier PaaS (Render free, Railway trial, Vercel serverless) **cannot** run Ollama reliably:

- Render free: 512 MB RAM — insufficient
- Railway: may work on hobby plan with 8 GB but not free
- Fly.io free: 256 MB shared — insufficient

## Recommended Deployment Patterns

### Pattern A: Fully Local (simplest)

```bash
ollama serve
docker-compose up --build
```

All services on one machine. Best for development and demos.

### Pattern B: Split Architecture

1. **Ollama host:** Local PC, home server, or cheap VPS (Hetzner CX22 ~€4/mo, 4 GB RAM)
2. **Backend:** Render/Railway with `OLLAMA_URL=https://your-ollama-host:11434`
3. **Frontend:** Vercel with `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`

Ensure Ollama host firewall allows inbound 11434 only from backend IP, or use Tailscale.

### Pattern C: Backend Only on VPS

Deploy backend + Ollama on a single 4 GB VPS:

```bash
# On VPS
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:1.5b
docker-compose up -d backend
```

Frontend on Vercel pointing to VPS backend URL.

## Fallback When Ollama Unavailable

The backend returns a clear error if Ollama is unreachable:

```
LLM request failed: Connection refused
```

No silent degradation — users see an explicit error. To add a fallback, you would need a different LLM provider (out of scope for V1 per spec).

## Docker Production Checklist

- [ ] Set `CORS_ORIGINS` to production frontend URL (not `*`)
- [ ] Mount persistent volume for `data/` (uploads, ChromaDB, SQLite)
- [ ] Set resource limits in docker-compose
- [ ] Use HTTPS reverse proxy (nginx/Caddy)
- [ ] Do not expose Ollama port publicly without auth
