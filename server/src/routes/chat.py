from __future__ import annotations
import json
import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.models.chat import ChatRequest, ChatResponse
from src.agents.persona_agent.graph import run_persona_turn, stream_persona_turn
from src.agents.booking_agent.graph import run_booking_turn
from src.agents.booking_agent.graph import is_booking_intent
from src.services.redis_client import get_redis
from src.services.session_store import new_session_id, load_history, append_turn

log = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])

_BOOKING_ACTIVE = "booking_active:"


async def _booking_active(session_id: str) -> bool:
    return bool(await get_redis().get(f"{_BOOKING_ACTIVE}{session_id}"))


async def _set_booking(session_id: str, active: bool) -> None:
    r = get_redis()
    key = f"{_BOOKING_ACTIVE}{session_id}"
    if active:
        await r.setex(key, 3600, "1")
    else:
        await r.delete(key)


async def _handle_booking(request: Request, body: ChatRequest, session_id: str) -> str:
    is_first = not await _booking_active(session_id)
    if is_first:
        await _set_booking(session_id, True)

    reply = await run_booking_turn(
        graph=request.app.state.booking_graph,
        user_message=body.message,
        session_id=session_id,
        is_first_turn=is_first,
    )

    if any(kw in reply.lower() for kw in ("confirmed for", "calendar invite", "book directly")):
        await _set_booking(session_id, False)

    await append_turn(session_id, body.message, reply)
    return reply


async def _handle_persona(request: Request, body: ChatRequest, session_id: str) -> str:
    history = await load_history(session_id)
    history_dicts = [{"role": m.role, "content": m.content} for m in history]

    reply = await run_persona_turn(
        graph=request.app.state.persona_graph,
        query=body.message,
        mode=body.mode,
        history=history_dicts,
    )
    await append_turn(session_id, body.message, reply)
    return reply


async def _sse_stream(request: Request, body: ChatRequest, session_id: str):
    in_booking = await _booking_active(session_id)
    use_booking = in_booking or is_booking_intent(body.message)

    if use_booking:
        try:
            reply = await _handle_booking(request, body, session_id)
            yield f"data: {json.dumps({'token': reply})}\n\n"
        except Exception as e:
            log.exception("chat.booking_error", session_id=session_id, error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            yield "data: [DONE]\n\n"
        return

    # Persona — stream tokens via LangGraph
    history = await load_history(session_id)
    history_dicts = [{"role": m.role, "content": m.content} for m in history]
    accumulated = []
    try:
        async for token in stream_persona_turn(
            graph=request.app.state.persona_graph,
            query=body.message,
            mode=body.mode,
            history=history_dicts,
        ):
            accumulated.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as e:
        log.exception("chat.persona_error", session_id=session_id, error=str(e))
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        yield "data: [DONE]\n\n"
        if accumulated:
            await append_turn(session_id, body.message, "".join(accumulated))


@router.post("/chat")
async def chat_stream(request: Request, body: ChatRequest):
    session_id = body.session_id or new_session_id()
    return StreamingResponse(
        _sse_stream(request, body, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: Request, body: ChatRequest):
    session_id = body.session_id or new_session_id()

    in_booking = await _booking_active(session_id)
    if in_booking or is_booking_intent(body.message):
        reply = await _handle_booking(request, body, session_id)
    else:
        reply = await _handle_persona(request, body, session_id)

    return ChatResponse(reply=reply, session_id=session_id)


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    from src.services.session_store import delete_session
    await delete_session(session_id)
    await _set_booking(session_id, False)
    return {"cleared": session_id}
