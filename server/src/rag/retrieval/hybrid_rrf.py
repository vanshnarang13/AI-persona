from __future__ import annotations
from src.models.retrieval import RetrievedChunk
from src.config.tuning import retrieval as cfg


def rrf_fuse(
    results_a: list[RetrievedChunk],
    results_b: list[RetrievedChunk],
    k: int | None = None,
    w_a: float | None = None,
    w_b: float | None = None,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion of two ranked lists.
    RRF score = w_a / (k + rank_a) + w_b / (k + rank_b)
    Chunks appearing in only one list still get a partial score.
    """
    k = k if k is not None else cfg.RRF_K
    w_a = w_a if w_a is not None else cfg.VECTOR_WEIGHT
    w_b = w_b if w_b is not None else cfg.BM25_WEIGHT

    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(results_a, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + w_a / (k + rank)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(results_b, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + w_b / (k + rank)
        if chunk.id not in chunk_map:
            chunk_map[chunk.id] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for doc_id, rrf_score in fused:
        chunk = chunk_map[doc_id]
        result.append(RetrievedChunk(
            id=chunk.id,
            content=chunk.content,
            source=chunk.source,
            score=rrf_score,
            metadata=chunk.metadata,
        ))
    return result
