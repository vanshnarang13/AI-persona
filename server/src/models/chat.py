from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None
    history: list[Message] = Field(default_factory=list)
    mode: Literal["chat", "voice"] = "chat"


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
