# Security

## Session-Based Isolation (Not Authentication)

LectureMindAI uses a client-generated UUID v4 stored in `localStorage` as the session identifier. This provides lightweight data isolation between browser sessions but is **not** a security boundary against a determined attacker who obtains the UUID (e.g., via browser dev tools or physical device access).

### Limitations

- No user accounts, passwords, or JWT tokens
- Session IDs are unsigned and transmitted in API requests
- Anyone with the UUID can access that session's data
- Clearing browser storage permanently loses access to documents and chat history
- Suitable for local/demo use, not multi-tenant production

### What Would Change for Real Authentication

1. Add user registration/login with password hashing (bcrypt/argon2)
2. Issue signed, expiring session tokens (JWT or server-side sessions)
3. Bind documents and messages to `user_id`, not client-supplied UUID
4. Enforce authorization server-side on every endpoint
5. Add HTTPS in production
6. Consider rate limiting by authenticated user, not just session/IP

## Implemented Controls

| Control | Implementation |
|---------|----------------|
| Session scoping | All DB/Chroma queries filter by `session_id` |
| Rate limiting | slowapi on `/upload`, `/upload-url`, `/query` |
| File validation | Extension check server-side |
| Size limits | Max PDF 25MB, media 30 min, 500MB/session, 20 docs |
| CORS | Configurable via `CORS_ORIGINS` env var |
| Error handling | No raw stack traces returned to client |
| Secrets | All config in `.env`, not hardcoded |

## Resource Limits (defaults)

- `MAX_PDF_SIZE_MB=25`
- `MAX_MEDIA_DURATION_MINUTES=30`
- `MAX_STORAGE_PER_SESSION_MB=500`
- `MAX_DOCUMENTS_PER_SESSION=20`
- `RATE_LIMIT_QUERY=10/min`
- `RATE_LIMIT_UPLOAD=5/min`

Limits are enforced before expensive processing (transcription, embedding).

## URL Ingestion Disclaimer

The UI warns users to only ingest their own recordings or openly-licensed content. yt-dlp is used for convenience, not as a general-purpose downloader.
