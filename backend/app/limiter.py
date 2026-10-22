"""Rate limiting with slowapi, keyed by client IP. (Behind a reverse proxy, configure it to
forward the real client address, otherwise all users share one bucket.)"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(key_func=get_remote_address)


def query_limit() -> str:
    return f"{settings.rate_limit_query}/minute"


def upload_limit() -> str:
    return f"{settings.rate_limit_upload}/minute"
