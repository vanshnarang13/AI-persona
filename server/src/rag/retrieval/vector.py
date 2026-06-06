from __future__ import annotations
import json
import asyncpg
from src.models.retrieval import RetrievedChunk
from src.services.llm_client import embed as _embed


def _parse_metadata(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


async def embed_query(text: str) -> list[float]:
    return await _embed(text)


async def vector_search(
    query_embedding: list[float],
    top_k: int,
    ef_search: int,
    pool: asyncpg.Pool,
) -> list[RetrievedChunk]:
    """ANN search via HNSW. ef_search trades recall for speed — higher = better recall, slower."""
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    async with pool.acquire() as conn:
        await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        rows = await conn.fetch(
            """
            SELECT id::text, content, source, source_type, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM documents
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vec_str,
            top_k,
        )

    return [
        RetrievedChunk(
            id=str(r["id"]),
            content=r["content"],
            source=r["source"],
            score=float(r["score"]),
            metadata={**_parse_metadata(r["metadata"]), "cos_score": float(r["score"])},
        )
        for r in rows
    ]
