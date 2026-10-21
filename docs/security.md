# Security model

## Session-based isolation (no login)
There are no accounts. On first load the browser generates a random UUID v4 and keeps it in `localStorage`; every request carries it. The backend validates it (UUID v4, else HTTP 400) and scopes **all** reads and writes to it: SQLite `WHERE session_id = ?`, Chroma `where={"session_id": ...}`, and ownership checks before delete/select. Another session's document id returns the same 404 as an unknown id, so existence is not leaked. A test (`TestSessionIsolation`) enforces this.

### Honest limitations
- The session id is an **unsigned bearer token**, not authentication. Anyone who obtains it (for example from the browser's storage on a shared computer) has full access to that session's documents and chat history. It is a lightweight isolation mechanism suitable for a no-login demo, not a security boundary against a determined attacker.
- Clearing browser storage or switching browsers loses access to earlier documents and history (the data stays on the server until removed). This is an accepted V1 trade-off.
- Rate limits are per client IP; behind a reverse proxy, forward the real client address or all users share one bucket.
- Uploaded files are stored unencrypted on the server disk.

### What real authentication would require
Accounts with password hashing or OAuth; signed, expiring tokens (for example JWT) issued at login and verified on every request; a `user_id` foreign key replacing `session_id` in `messages` and `documents` and in Chroma metadata; migration of anonymous sessions; HTTPS-only cookies and CSRF protection. The query paths already filter on one identifier everywhere, so swapping `session_id` for the verified `user_id` is localised to `validate_session_id` and the metadata field.

## Other controls
- **Rate limiting** (slowapi): `/query` and `/upload*` per IP, configurable (`RATE_LIMIT_QUERY`, `RATE_LIMIT_UPLOAD`); HTTP 429.
- **Resource limits**, checked before expensive work: PDF 25 MB, media size 500 MB, media duration 30 min, 500 MB and 20 documents per session; HTTP 413. Uploads are streamed to disk in 1 MB blocks, never read whole into RAM. Heavy ingestion runs one job at a time.
- **File validation** is server-side: extension allow-list, `%PDF-` signature check, ffprobe must be able to parse media. Client MIME types are ignored. Filenames are sanitised and stored as `<session>/<uuid>_<name>`; deletes are confined to the session's folder.
- **URL import**: http(s) only; hosts resolving to private/loopback addresses are rejected (basic SSRF guard); live streams rejected; size and duration capped; temp files always removed. For your own or openly-licensed content only.
- **Secrets/config**: everything in environment variables (`.env` is git-ignored); nothing hardcoded.
- **CORS**: restricted to `CORS_ORIGINS` (use `*` only in development).
- **Errors**: external calls (Ollama, ffmpeg, yt-dlp, Whisper) have timeouts and map to 4xx/5xx JSON messages; unexpected errors return a generic 500 without a stack trace.
- **Rendering**: model output is rendered with `react-markdown` (no raw HTML).
