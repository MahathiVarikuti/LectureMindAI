import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.db import sqlite
from app.errors import AppError
from app.limiter import limiter
from app.routes.documents import router as documents_router
from app.routes.query import router as query_router
from app.routes.upload import router as upload_router

logger = logging.getLogger("lecturemind")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sqlite.init_db()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    from app.services.ingestion import reconcile_orphans

    reconcile_orphans()
    yield


app = FastAPI(title="LectureMindAI", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # -> HTTP 429

# Production: set CORS_ORIGINS to the deployed frontend origin(s). "*" is for local dev only
# (and browsers refuse credentials with a wildcard origin, so credentials are disabled then).
_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)
