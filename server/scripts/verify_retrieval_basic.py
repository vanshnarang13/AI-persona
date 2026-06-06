"""
Quick sanity check: run 5 representative queries against pgvector and print top-3 results.
Gate: every query must have at least one result with score > 0.35.

Run from server/:
    PYTHONPATH=. python scripts/verify_retrieval_basic.py
    or: make verify-retrieval
"""
from __future__ import annotations
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import asyncpg
from src.services.llm_client import get_client
from src.config import settings

TEST_QUERIES = [
    ("AtlasRAG architecture", "proj_atlasrag"),
    ("SteganoGAN steganography", "exp_noos_technologies"),
    ("what is your CGPA", "resume"),
    ("MCP servers TradeSmith trading", "proj_tradesmith"),
    ("book an interview schedule meeting", "fit_for_role"),
]

GREEN = "\033[92m✓\033[0m"
RED = "\033[91m✗\033[0m"


async def embed_query(text: str) -> list[float]:
    client = get_client()
    resp = await client.embeddings.create(input=text, model=settings.openai_embedding_model)
    return resp.data[0].embedding


async def vector_search(query: str, conn: asyncpg.Connection, top_k: int = 3):
    vec = await embed_query(query)
    vec_str = "[" + ",".join(str(v) for v in vec) + "]"
    rows = await conn.fetch(
        """
        SELECT source, LEFT(content, 120) as snippet,
               1 - (embedding <=> $1::vector) as score
        FROM documents
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        vec_str, top_k,
    )
    return rows


async def main():
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn, statement_cache_size=0, timeout=15)

    count = await conn.fetchval("SELECT COUNT(*) FROM documents")
    print(f"\nDocuments in DB: {count}")
    print("=" * 60)

    all_passed = True
    for query, expected_source_fragment in TEST_QUERIES:
        rows = await vector_search(query, conn)
        top_score = max(r["score"] for r in rows) if rows else 0.0
        passed = top_score >= 0.35

        status = GREEN if passed else RED
        print(f"\n{status} Query: \"{query}\"")
        print(f"   Top score: {top_score:.4f}  (gate: 0.35)  expected hint: {expected_source_fragment}")
        for i, r in enumerate(rows, 1):
            print(f"   [{i}] {r['score']:.4f}  {r['source']}")
            print(f"       {r['snippet'].strip()[:100]}...")

        if not passed:
            all_passed = False

    await conn.close()
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN} All queries passed the 0.35 threshold — retrieval is working.")
    else:
        print(f"{RED} Some queries failed. Check corpus content or chunking strategy.")


if __name__ == "__main__":
    asyncio.run(main())
