from __future__ import annotations
import time
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from src.services.redis_client import get_redis

log = structlog.get_logger(__name__)

# (path_prefix, max_requests, window_seconds)
_LIMITS = [
    ("/chat",    60, 60),
    ("/booking", 10, 60),
]


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        for prefix, max_req, window in _LIMITS:
            if path.startswith(prefix):
                key = f"rl:{client_ip}:{prefix}"
                redis = get_redis()
                pipe = redis.pipeline()
                now = int(time.time())
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window)
                results = await pipe.execute()
                count = results[2]

                if count > max_req:
                    log.warning("rate_limit.exceeded", ip=client_ip, path=path, count=count)
                    return JSONResponse(
                        {"detail": "Rate limit exceeded. Please slow down."},
                        status_code=429,
                    )
                break

        return await call_next(request)
