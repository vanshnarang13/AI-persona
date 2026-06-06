from __future__ import annotations
import asyncio
from functools import lru_cache
from src.models.retrieval import RetrievedChunk

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_MODEL_NAME)


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Re-score chunks with a cross-encoder. Run in thread pool to avoid blocking the event loop.
    Only called on chat path when len(chunks) > top_k.
    """
    if not chunks:
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

    # Preserve original RRF/cosine scores — cross-encoder logits aren't calibrated
    # for threshold-based grounding checks. Reranking only changes the ORDER.
    return [c for c, _ in reranked]
