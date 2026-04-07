from __future__ import annotations

"""
Personal Health — request lifecycle middleware.

Adds (in order):
  1. request_id contextvar + X-Request-ID header
  2. structured access log
  3. security headers
  4. token-bucket rate limit (per IP, per route group)
  5. global exception handler producing sanitized JSON

Designed to be FastAPI-friendly via BaseHTTPMiddleware.
"""

import time
import traceback
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from logging_setup import get_logger, request_id_var
from metrics import observe_request

log = get_logger("middleware")


# ─── Token-bucket rate limiter ──────────────────────────────────────────────


class _TokenBucket:
    __slots__ = ("tokens", "updated_at")

    def __init__(self, tokens: float):
        self.tokens = tokens
        self.updated_at = time.monotonic()


class RateLimiter:
    """Per-IP per-route-group token bucket. In-process only."""

    def __init__(self, per_minute: int, burst: int):
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self.buckets: dict[str, _TokenBucket] = defaultdict(lambda: _TokenBucket(self.burst))

    def consume(self, key: str, cost: float = 1.0) -> tuple[bool, float, float]:
        bucket = self.buckets[key]
        now = time.monotonic()
        elapsed = now - bucket.updated_at
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
        bucket.updated_at = now
        if bucket.tokens >= cost:
            bucket.tokens -= cost
            remaining = bucket.tokens
            return True, remaining, 0.0
        # not enough tokens
        deficit = cost - bucket.tokens
        retry_after = deficit / self.rate
        return False, bucket.tokens, retry_after


_LIMITER = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)


_RATE_LIMIT_EXEMPT = {"/livez", "/readyz", "/metrics", "/", "/health", "/banner"}


# ─── Middleware ─────────────────────────────────────────────────────────────


class RequestLifecycleMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        # 1. request id
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(rid)

        # 2. rate limit (skip for exempt + websockets)
        path = request.url.path
        if request.scope.get("type") == "http" and path not in _RATE_LIMIT_EXEMPT:
            ip = request.client.host if request.client else "unknown"
            group = path.split("/")[1] if "/" in path else "root"
            key = f"{ip}:{group}"
            allowed, remaining, retry_after = _LIMITER.consume(key)
            if not allowed:
                log.warning("rate limited", extra={"ip": ip, "path": path, "retry_after": round(retry_after, 2)})
                resp = JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "error_id": rid,
                        "request_id": rid,
                        "retry_after_seconds": round(retry_after, 2),
                    },
                )
                resp.headers["Retry-After"] = str(int(retry_after) + 1)
                resp.headers["X-RateLimit-Remaining"] = "0"
                resp.headers["X-Request-ID"] = rid
                self._add_security_headers(resp)
                request_id_var.reset(token)
                return resp

        # 3. dispatch with global exception handler
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000
            log.error(
                "unhandled exception",
                extra={
                    "path": path,
                    "method": request.method,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                    "duration_ms": round(duration_ms, 2),
                },
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "error_id": rid,
                    "request_id": rid,
                    "message": "An internal error occurred. Reference this id when reporting.",
                },
            )

        duration_ms = (time.perf_counter() - start) * 1000

        # 4. headers + access log + metrics
        response.headers["X-Request-ID"] = rid
        self._add_security_headers(response)
        observe_request(request.method, path, response.status_code, duration_ms / 1000.0)
        log.info(
            "http",
            extra={
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "ip": request.client.host if request.client else None,
            },
        )
        request_id_var.reset(token)
        return response

    @staticmethod
    def _add_security_headers(response) -> None:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        if settings.is_prod:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )


def install_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestLifecycleMiddleware)
