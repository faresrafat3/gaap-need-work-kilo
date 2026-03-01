"""
Security Middleware for GAAP API
==================================
Provides security headers, rate limiting, and input validation.
"""

from fastapi import Request, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import re
from typing import Dict, List, Optional
import logging
import os
import redis.asyncio as redis

logger = logging.getLogger("gaap.security")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-based rate limiting with in-memory fallback."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis = False
        self._init_redis()

    def _init_redis(self):
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            try:
                self.redis_client = redis.from_url(
                    redis_url, encoding="utf-8", decode_responses=True
                )
                self.use_redis = True
                logger.info("Redis rate limiting enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory fallback: {e}")
                self.use_redis = False
        else:
            logger.warning("REDIS_URL not set, using in-memory rate limiting")

    async def _check_redis_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        key = f"ratelimit:{client_ip}"
        window = 60

        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = await pipe.execute()
            current_count = results[0]

            remaining = max(0, self.requests_per_minute - current_count)

            if current_count > self.requests_per_minute:
                return False, remaining
            return True, remaining
        except Exception as e:
            logger.error(f"Redis error, falling back to in-memory: {e}")
            self.use_redis = False
            return await self._check_memory_rate_limit(client_ip)

    async def _check_memory_rate_limit(self, client_ip: str) -> tuple[bool, int]:
        current_time = time.time()
        window = 60

        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip] if current_time - t < window
            ]

        current_count = len(self.requests.get(client_ip, []))
        remaining = max(0, self.requests_per_minute - current_count - 1)

        if current_count >= self.requests_per_minute:
            return False, 0

        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)

        return True, remaining

    async def dispatch(self, request: Request, call_next):
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        if isinstance(client_ip, str) and "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()

        if self.use_redis:
            allowed, remaining = await self._check_redis_rate_limit(client_ip)
        else:
            allowed, remaining = await self._check_memory_rate_limit(client_ip)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429, detail="Rate limit exceeded. Please try again later."
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize input."""

    # Patterns to detect common attacks
    SQLI_PATTERNS = [
        r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
        r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
        r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>[\s\S]*?</script>",
        r"javascript:",
        r"on\w+\s*=",
    ]

    async def dispatch(self, request: Request, call_next):
        # Skip for GET requests without body
        if request.method == "GET":
            return await call_next(request)

        # Check path for suspicious patterns
        path = request.url.path
        for pattern in self.SQLI_PATTERNS + self.XSS_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                logger.warning(f"Potential attack detected in path: {path}")
                raise HTTPException(status_code=400, detail="Invalid request")

        return await call_next(request)
