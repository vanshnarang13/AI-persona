from __future__ import annotations
import asyncio
from functools import lru_cache
from src.models.retrieval import RetrievedChunk

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _RERANK_AVAILABLE = True
except ImportError:
    _RERANK_AVAILABLE = False


@lru_cache(maxsize=1)
def _get_model():
    return _CrossEncoder(_MODEL_NAME)


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    if not chunks or not _RERANK_AVAILABLE:
        return chunks

    model = await asyncio.get_event_loop().run_in_executor(None, _get_model)
    pairs = [(query, c.content) for c in chunks]

    scores = await asyncio.get_event_loop().run_in_executor(
        None, lambda: model.predict(pairs)
    )

    reranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [c for c, _ in reranked]
