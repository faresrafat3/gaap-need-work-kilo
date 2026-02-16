import logging
import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("gaap.api.ratelimit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

        self._minute_requests: dict[str, list[float]] = defaultdict(list)
        self._hour_requests: dict[str, list[float]] = defaultdict(list)
        self._cleanup_interval = 3600
        self._last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        self._cleanup_old_requests(client_ip, current_time)

        minute_count = len(self._minute_requests[client_ip])
        if minute_count >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip} (minute)")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )

        hour_count = len(self._hour_requests[client_ip])
        if hour_count >= self.requests_per_hour:
            logger.warning(f"Rate limit exceeded for {client_ip} (hour)")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )

        self._minute_requests[client_ip].append(current_time)
        self._hour_requests[client_ip].append(current_time)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            self.requests_per_minute - minute_count - 1
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            self.requests_per_hour - hour_count - 1
        )

        return response

    def _cleanup_old_requests(self, client_ip: str, current_time: float) -> None:
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        self._minute_requests[client_ip] = [
            t for t in self._minute_requests[client_ip] if t > minute_ago
        ]

        self._hour_requests[client_ip] = [t for t in self._hour_requests[client_ip] if t > hour_ago]

        if current_time - self._last_cleanup > self._cleanup_interval:
            for ip in list(self._minute_requests.keys()):
                self._minute_requests[ip] = [t for t in self._minute_requests[ip] if t > minute_ago]
                if not self._minute_requests[ip]:
                    del self._minute_requests[ip]

            for ip in list(self._hour_requests.keys()):
                self._hour_requests[ip] = [t for t in self._hour_requests[ip] if t > hour_ago]
                if not self._hour_requests[ip]:
                    del self._hour_requests[ip]

            self._last_cleanup = current_time


def rate_limit(requests_per_minute: int = 60, requests_per_hour: int = 1000):
    def decorator(func: Callable):
        return func

    return decorator
