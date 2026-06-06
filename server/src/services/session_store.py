from __future__ import annotations
import json
import uuid
from src.services.redis_client import get_redis
from src.models.chat import Message

_SESSION_TTL = 60 * 60 * 24  # 24 hours
_MAX_HISTORY = 20             # keep last 20 turns (10 exchanges)
_KEY_PREFIX = "session:"


def new_session_id() -> str:
    return str(uuid.uuid4())


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


async def load_history(session_id: str) -> list[Message]:
    r = get_redis()
    raw = await r.get(_key(session_id))
    if not raw:
        return []
    return [Message(**m) for m in json.loads(raw)]


async def save_history(session_id: str, history: list[Message]) -> None:
    r = get_redis()
    trimmed = history[-_MAX_HISTORY:]
    await r.setex(
        _key(session_id),
        _SESSION_TTL,
        json.dumps([m.model_dump() for m in trimmed]),
    )


async def append_turn(session_id: str, user_msg: str, assistant_reply: str) -> None:
    history = await load_history(session_id)
    history.append(Message(role="user", content=user_msg))
    history.append(Message(role="assistant", content=assistant_reply))
    await save_history(session_id, history)


async def delete_session(session_id: str) -> None:
    r = get_redis()
    await r.delete(_key(session_id))
