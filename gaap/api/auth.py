import logging
import os
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("gaap.api.auth")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self.api_key = api_key or os.environ.get("GAAP_API_KEY")
        self.enabled = bool(self.api_key)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        if request.url.path in ["/health", "/docs", "/openapi.json", "/"]:
            return await call_next(request)

        client_key = request.headers.get("X-API-Key")
        if not client_key:
            logger.warning(f"Missing API key from {request.client.host}")

        if client_key != self.api_key:
            logger.warning(f"Invalid API key from {request.client.host}")
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)


async def verify_api_key(api_key: str | None = API_KEY_HEADER) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is required")

    expected_key = os.environ.get("GAAP_API_KEY")
    if not expected_key:
        return api_key

    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


def require_auth(api_key: str | None = None):
    async def dependency():
        return await verify_api_key(api_key)

    return dependency
