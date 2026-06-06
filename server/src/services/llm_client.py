from __future__ import annotations
from openai import AsyncOpenAI
from src.config import settings
from typing import AsyncIterator

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=30.0)
    return _client


async def embed(text: str) -> list[float]:
    client = get_client()
    resp = await client.embeddings.create(
        input=text,
        model=settings.openai_embedding_model,
    )
    return resp.data[0].embedding


async def chat_complete(
    messages: list[dict],
    mode: str = "chat",
    stream: bool = False,
) -> str | AsyncIterator[str]:
    client = get_client()
    model = (
        settings.openai_chat_model_fast
        if mode == "voice"
        else settings.openai_chat_model_smart
    )
    if stream:
        return _stream(client, model, messages)
    resp = await client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content or ""


async def _stream(client: AsyncOpenAI, model: str, messages: list[dict]) -> AsyncIterator[str]:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
