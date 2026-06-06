from __future__ import annotations
import asyncio
import time
from pathlib import Path

import asyncpg
import structlog

from .partition import Document, partition_corpus, partition_file
from .chunk import chunk_documents
from .embed import embed_batch
from .vectorize import upsert_documents

log = structlog.get_logger(__name__)


async def run_ingestion(
    corpus_dir: Path,
    pool: asyncpg.Pool,
    source_type: str = "corpus",
) -> dict:
    """
    Full pipeline: partition → chunk → embed → vectorize.
    Returns stats dict: {files, raw_docs, chunks, stored, duration_ms}
    """
    t0 = time.perf_counter()

    # 1. Partition all markdown files
    raw_docs = partition_corpus(corpus_dir, source_type=source_type)
    log.info("ingestion.partitioned", files=len({d.metadata["file_path"] for d in raw_docs}), docs=len(raw_docs))

    # 2. Chunk large sections
    chunks = chunk_documents(raw_docs)
    log.info("ingestion.chunked", chunks=len(chunks))

    # 3. Embed all chunks
    texts = [c.content for c in chunks]
    embeddings = await embed_batch(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    log.info("ingestion.embedded", count=len(embeddings))

    # 4. Upsert into pgvector
    stored = await upsert_documents(chunks, pool)
    log.info("ingestion.stored", stored=stored)

    duration_ms = round((time.perf_counter() - t0) * 1000)
    stats = {
        "files": len({d.metadata["file_path"] for d in raw_docs}),
        "raw_docs": len(raw_docs),
        "chunks": len(chunks),
        "stored": stored,
        "duration_ms": duration_ms,
    }
    log.info("ingestion.complete", **stats)
    return stats


__all__ = ["run_ingestion", "Document", "partition_file", "partition_corpus"]
