# Deployment

## Local / single machine
`docker compose up --build`, then `docker compose exec ollama ollama pull qwen2.5:1.5b`. Data (uploads, vectors, SQLite, model cache) lives in the `backend_data` volume; Ollama models in `ollama_data`.

## Free-tier hosting: the real constraint
Ollama must run somewhere. `qwen2.5:1.5b` needs roughly 1.5-2 GB of RAM, plus ~1 GB for the embedder, more for Whisper/CLIP during ingestion. Most free tiers (512 MB-1 GB RAM, no persistent disk, sleeping instances) **cannot** run the backend together with Ollama, and free instances lose local disk on restart, which wipes SQLite, Chroma and uploads.

Recommended fallback plan:
1. Host the **frontend** on a free static/Node host (for example Vercel) with `NEXT_PUBLIC_API_URL` pointing at the backend.
2. Host the **backend + Ollama** on a machine you control (a laptop or a free-tier VM with at least 4 GB RAM, for example an Oracle Cloud Always Free Ampere instance) using `docker compose`, behind HTTPS (Caddy/Cloudflare Tunnel).
3. Set `CORS_ORIGINS` to the deployed frontend origin only, and keep the rate limits.
4. If no host can run Ollama, the backend still serves uploads, documents and retrieval; `/query` then returns HTTP 503 "language model unavailable" instead of failing silently. Do not swap in a paid API (non-goal).
