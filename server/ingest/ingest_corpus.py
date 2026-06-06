"""
Run from server/ directory:
    PYTHONPATH=. python ingest/ingest_corpus.py
    or: make ingest-all
"""
import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from src.rag.ingestion import run_ingestion
from src.rag.retrieval import invalidate_index
from src.services.redis_client import get_redis_binary, close_redis


async def _delete_retrieval_cache() -> int:
    redis = get_redis_binary()
    deleted = 0
    async for key in redis.scan_iter(match="retrieve:*", count=500):
        await redis.delete(key)
        deleted += 1
    return deleted


async def main() -> None:
    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    corpus_dir = Path(__file__).parent.parent / "data" / "corpus"

    print(f"Connecting to database...")
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5, statement_cache_size=0)

    print(f"Ingesting corpus from: {corpus_dir}")
    stats = await run_ingestion(corpus_dir, pool, source_type="corpus")
    await invalidate_index(get_redis_binary())
    deleted_cache = await _delete_retrieval_cache()

    print("\n── Ingestion Complete ──────────────────────")
    print(f"  Files processed : {stats['files']}")
    print(f"  Raw sections    : {stats['raw_docs']}")
    print(f"  Chunks created  : {stats['chunks']}")
    print(f"  Stored in DB    : {stats['stored']}")
    print(f"  Duration        : {stats['duration_ms']}ms")
    print(f"  Retrieval cache : cleared {deleted_cache} entries")
    print("────────────────────────────────────────────")

    await pool.close()
    await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
