from __future__ import annotations
from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    top_k: int | None = None
    mode: str = "chat"


class RetrievedChunk(BaseModel):
    id: str
    content: str
    source: str
    score: float
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    chunks: list[RetrievedChunk]
    query: str
    latency_ms: float
