from __future__ import annotations
import asyncio
import time
from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


async def _background_warmup(request: Request) -> None:
    try:
        from src.rag.retrieval import retrieve
        from src.services.redis_client import get_redis_binary
        await retrieve(
            "background skills projects",
            "voice",
            request.app.state.pool,
            request.app.state.bm25,
            get_redis_binary(),
        )
    except Exception:
        pass


@router.get("/health")
async def health(request: Request):
    asyncio.create_task(_background_warmup(request))
    return {"status": "ok"}


@router.get("/health/detailed")
async def health_detailed(request: Request):
    results: dict = {"status": "ok", "services": {}}

    # DB ping
    try:
        t0 = time.perf_counter()
        async with request.app.state.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        results["services"]["database"] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000)}
    except Exception as e:
        results["services"]["database"] = {"ok": False, "error": str(e)}
        results["status"] = "degraded"

    # Redis ping
    try:
        from src.services.redis_client import get_redis
        t0 = time.perf_counter()
        await get_redis().ping()
        results["services"]["redis"] = {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000)}
    except Exception as e:
        results["services"]["redis"] = {"ok": False, "error": str(e)}
        results["status"] = "degraded"

    return results
