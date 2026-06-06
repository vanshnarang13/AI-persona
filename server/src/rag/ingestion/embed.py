from __future__ import annotations
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from src.services.llm_client import get_client
from src.config import settings

BATCH_SIZE = 100


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = get_client()
    resp = await client.embeddings.create(
        input=texts,
        model=settings.openai_embedding_model,
    )
    return [item.embedding for item in resp.data]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of BATCH_SIZE. Returns embeddings in same order."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        embeddings = await _embed_batch(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings
