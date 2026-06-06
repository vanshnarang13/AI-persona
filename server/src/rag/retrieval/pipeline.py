from __future__ import annotations
import asyncio
import hashlib
import pickle
import time
import asyncpg
import redis.asyncio as aioredis
import structlog

from src.models.retrieval import RetrievedChunk
from src.config.tuning import retrieval as cfg
from .vector import embed_query, vector_search
from .lexical_bm25 import BM25Index
from .hybrid_rrf import rrf_fuse

log = structlog.get_logger(__name__)

_CACHE_TTL = 3600  # 1 hour


def _cache_key(query: str, mode: str) -> str:
    config_sig = (
        f"topc={cfg.TOP_K_CHAT}:topv={cfg.TOP_K_VOICE}:"
        f"rrfk={cfg.RRF_K}:vw={cfg.VECTOR_WEIGHT}:"
        f"bw={cfg.BM25_WEIGHT}:ef={cfg.HNSW_EF_SEARCH}"
    )
    return "retrieve:" + hashlib.sha256(f"{query}:{mode}:{config_sig}".encode()).hexdigest()[:16]


async def retrieve(
    query: str,
    mode: str,
    pool: asyncpg.Pool,
    bm25_index: BM25Index,
    redis: aioredis.Redis,
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    """
    Full hybrid retrieval pipeline.
    Returns (chunks, latency_breakdown_ms).
    Voice mode: smaller top_k, no reranking.
    """
    latency: dict[str, float] = {}
    top_k = cfg.TOP_K_VOICE if mode == "voice" else cfg.TOP_K_CHAT

    # ── Cache check (binary Redis — pickle is not UTF-8) ────────────
    cache_key = _cache_key(query, mode)
    t0 = time.perf_counter()
    cached = await redis.get(cache_key)
    latency["cache_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if cached:
        log.info("retrieve.cache_hit", query=query[:60], mode=mode)
        return pickle.loads(cached), {**latency, "cache_hit": 1}

    # ── Embed query ──────────────────────────────────────────────────
    t1 = time.perf_counter()
    query_embedding = await embed_query(query)
    latency["embed_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    # ── Vector + BM25 in parallel ────────────────────────────────────
    t2 = time.perf_counter()
    vector_task = asyncio.create_task(
        vector_search(query_embedding, top_k, cfg.HNSW_EF_SEARCH, pool)
    )
    bm25_results = bm25_index.search(query, top_k)
    vector_results = await vector_task
    latency["vector_ms"] = round((time.perf_counter() - t2) * 1000, 1)
    latency["bm25_ms"] = 0.0  # BM25 is sync, measured within vector window

    log.debug("retrieve.search_done",
              vector_count=len(vector_results), bm25_count=len(bm25_results))

    # ── RRF fusion ───────────────────────────────────────────────────
    t3 = time.perf_counter()
    fused = rrf_fuse(vector_results, bm25_results)
    latency["rrf_ms"] = round((time.perf_counter() - t3) * 1000, 1)

    final = fused[:top_k]

    # ── Cache result ─────────────────────────────────────────────────
    await redis.setex(cache_key, _CACHE_TTL, pickle.dumps(final))

    total = sum(v for k, v in latency.items() if k != "cache_hit")
    latency["total_ms"] = round(total, 1)
    log.info("retrieve.done", mode=mode, chunks=len(final), **latency)

    return final, latency


__all__ = ["retrieve"]
