from __future__ import annotations
import json
import hashlib
import redis.asyncio as aioredis
from src.config import settings

_redis: aioredis.Redis | None = None
_redis_binary: aioredis.Redis | None = None

CACHE_TTL = 3600  # 1 hour


def get_redis() -> aioredis.Redis:
    """String-mode client — for text key/value ops (retrieval cache, rate limiting)."""
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def get_redis_binary() -> aioredis.Redis:
    """Binary-mode client — for pickle blobs (BM25 index)."""
    global _redis_binary
    if _redis_binary is None:
        _redis_binary = aioredis.from_url(settings.redis_url, decode_responses=False)
    return _redis_binary


def cache_key(query: str, mode: str) -> str:
    digest = hashlib.sha256(f"{query}:{mode}".encode()).hexdigest()[:16]
    return f"retrieval:{digest}"


async def get_cached(query: str, mode: str) -> list | None:
    r = get_redis()
    raw = await r.get(cache_key(query, mode))
    if raw:
        return json.loads(raw)
    return None


async def set_cached(query: str, mode: str, chunks: list) -> None:
    r = get_redis()
    await r.setex(cache_key(query, mode), CACHE_TTL, json.dumps(chunks))


async def close_redis() -> None:
    global _redis, _redis_binary
    if _redis:
        await _redis.aclose()
        _redis = None
    if _redis_binary:
        await _redis_binary.aclose()
        _redis_binary = None
