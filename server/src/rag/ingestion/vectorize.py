from __future__ import annotations
import json
import asyncpg
from .partition import Document


async def upsert_documents(chunks: list[Document], pool: asyncpg.Pool) -> int:
    """
    Idempotent upsert: delete existing rows for each unique source, then bulk insert.
    Returns number of rows inserted.
    """
    if not chunks:
        return 0

    # Collect unique sources in this batch
    sources = list({c.source for c in chunks})

    async with pool.acquire() as conn:
        # Delete existing chunks for these sources (enables re-ingestion)
        await conn.execute(
            "DELETE FROM documents WHERE source = ANY($1::text[])",
            sources,
        )

        # pgvector expects the vector as a string: "[0.1, 0.2, ...]"
        def _vec(emb: list[float]) -> str:
            return "[" + ",".join(str(v) for v in emb) + "]"

        rows = [
            (
                chunk.content,
                _vec(chunk.embedding),
                chunk.source,
                chunk.source_type,
                json.dumps(chunk.metadata),
            )
            for chunk in chunks
            if chunk.embedding is not None
        ]

        await conn.executemany(
            """
            INSERT INTO documents (content, embedding, source, source_type, metadata)
            VALUES ($1, $2::vector, $3, $4, $5::jsonb)
            """,
            rows,
        )

    return len(rows)
