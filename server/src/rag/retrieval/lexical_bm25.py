from __future__ import annotations
import asyncio
import pickle
import re
import asyncpg
import redis.asyncio as aioredis
from rank_bm25 import BM25Okapi

from src.models.retrieval import RetrievedChunk
from src.services.redis_client import get_redis_binary

_BM25_CACHE_KEY = "persona:bm25_index"
_BM25_TTL = 7200  # 2 hours


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    def __init__(self, doc_ids: list[str], sources: list[str], contents: list[str]):
        self._doc_ids = doc_ids
        self._sources = sources
        self._contents = contents
        corpus = [_tokenize(c) for c in contents]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Get indices sorted by score descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            # Normalise score to [0, 1] range approximately
            normalised = min(scores[idx] / 10.0, 1.0)
            results.append(RetrievedChunk(
                id=self._doc_ids[idx],
                content=self._contents[idx],
                source=self._sources[idx],
                score=normalised,
                metadata={},
            ))
        return results


async def _load_from_db(pool: asyncpg.Pool) -> BM25Index:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id::text, source, content FROM documents")
    doc_ids = [r["id"] for r in rows]
    sources = [r["source"] for r in rows]
    contents = [r["content"] for r in rows]
    return BM25Index(doc_ids, sources, contents)


async def get_or_build_index(
    pool: asyncpg.Pool,
    redis: aioredis.Redis,  # kept for API compat; we use binary client internally
) -> BM25Index:
    """
    Load BM25 index from Redis cache if available; otherwise build from DB and cache.
    Uses binary Redis client since pickle data is not UTF-8.
    """
    r = get_redis_binary()
    cached = await r.get(_BM25_CACHE_KEY)
    if cached:
        return pickle.loads(cached)

    index = await _load_from_db(pool)
    await r.setex(_BM25_CACHE_KEY, _BM25_TTL, pickle.dumps(index))
    return index


async def invalidate_index(redis: aioredis.Redis) -> None:
    """Call after re-ingestion to force a rebuild on next request."""
    r = get_redis_binary()
    await r.delete(_BM25_CACHE_KEY)
