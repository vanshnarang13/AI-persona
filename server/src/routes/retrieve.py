from __future__ import annotations
from fastapi import APIRouter, Request
from src.models.retrieval import RetrievalRequest, RetrievalResponse
from src.rag.retrieval import retrieve
from src.services.redis_client import get_redis_binary

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_endpoint(request: Request, body: RetrievalRequest):
    """
    Hybrid retrieval endpoint — used by Vapi as a custom tool.
    Returns grounded context chunks for a query.
    """
    import time
    t0 = time.perf_counter()

    chunks, latency = await retrieve(
        query=body.query,
        mode=body.mode,
        pool=request.app.state.pool,
        bm25_index=request.app.state.bm25,
        redis=get_redis_binary(),
    )

    return RetrievalResponse(
        chunks=chunks,
        query=body.query,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
