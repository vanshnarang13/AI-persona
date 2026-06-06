from __future__ import annotations

import json
import re
import structlog
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Request
from pydantic import BaseModel

from src.rag.retrieval import retrieve
from src.services.redis_client import get_redis_binary
from src.services.calcom_client import get_availability, create_booking
from src.agents.booking_agent.tools import parse_preference_filter

router = APIRouter(tags=["vapi"])
log = structlog.get_logger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class ToolCallResult(BaseModel):
    toolCallId: str
    result: str


class ToolCallsResponse(BaseModel):
    results: list[ToolCallResult]


@router.post("/vapi/tools", response_model=ToolCallsResponse)
async def vapi_tools(request: Request):
    body = await request.json()
    log.info("vapi_tools.raw_body", body_keys=list(body.keys()),
             body_preview=json.dumps(body)[:500])

    message = body.get("message", {})
    # Vapi sends toolCalls; toolCallList is also present but not always
    tool_calls = message.get("toolCallList") or message.get("toolCalls") or []
    if not tool_calls:
        tool_calls = body.get("toolCallList") or body.get("toolCalls") or []

    log.info("vapi_tools.parsed", tool_count=len(tool_calls))

    results = []
    for call in tool_calls:
        call_id = call.get("id", "")
        fn = call.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            args = {}

        log.info("vapi_tools.dispatch", tool=name, call_id=call_id,
                 args_keys=list(args.keys()) if isinstance(args, dict) else str(args)[:50])

        result = await _dispatch(request, name, args)

        log.info("vapi_tools.result", tool=name, result_len=len(result),
                 result_preview=result[:100])

        results.append(ToolCallResult(toolCallId=call_id, result=result))

    return ToolCallsResponse(results=results)


async def _dispatch(request: Request, name: str, args: dict) -> str:
    if name == "retrieve_context":
        return await _retrieve_context(request, args)
    if name == "check_availability":
        return await _check_availability(args)
    if name == "create_booking":
        return await _create_booking(args)
    return f"Unknown tool: {name}"


async def _retrieve_context(request: Request, args: dict) -> str:
    query = args.get("query", "")
    if not query:
        return "No query provided."
    chunks, _ = await retrieve(
        query=query,
        mode="voice",
        pool=request.app.state.pool,
        bm25_index=request.app.state.bm25,
        redis=get_redis_binary(),
    )
    if not chunks:
        return "No relevant information found."
    return "\n\n".join(f"[{i}] {c.content}" for i, c in enumerate(chunks, 1))


async def _check_availability(args: dict) -> str:
    preference = str(args.get("preference", "")).strip()
    slots = await get_availability(days_ahead=7)
    if not slots:
        return "No available slots in the next 7 days."

    by_day: dict[str, list] = {}
    for s in slots:
        day = s.start.astimezone(IST).strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(s)

    time_filter = parse_preference_filter(preference) if preference else None

    lines = []
    counter = 1
    for day in sorted(by_day):
        day_slots = by_day[day]
        if time_filter:
            day_slots = [s for s in day_slots if time_filter(s.start.astimezone(IST))]
        if not day_slots:
            continue
        day_label = datetime.fromisoformat(day).strftime("%A %d %b")
        lines.append(f"\n{day_label}:")
        for s in day_slots:
            dt = s.start.astimezone(IST)
            lines.append(f"  {counter}. {dt.strftime('%I:%M %p IST')} [iso: {s.start.isoformat()}]")
            counter += 1

    if not lines:
        return f"No slots match '{preference}'. Try a different time or day."

    header = f"Available slots ({preference}):" if preference else "Available slots:"
    return header + "\n".join(lines)


def _clean_email(raw: str) -> str:
    """Normalize speech-to-text email transcription artifacts."""
    e = raw.lower().strip()
    # Fix common provider name STT splits before anything else
    e = e.replace("g mail", "gmail").replace("g-mail", "gmail")
    e = e.replace("hot mail", "hotmail").replace("hot-mail", "hotmail")
    e = e.replace("you mail", "youmail").replace("out look", "outlook")
    # Spoken separators
    e = re.sub(r"\bat the rate\b|\bat rate\b", "@", e)
    e = re.sub(r"\bdot\b", ".", e)
    e = re.sub(r"\bunderscore\b", "_", e)
    e = re.sub(r"\bhyphen\b|\bdash\b", "-", e)
    # "word at domain" → "word@domain" (only when no "@" yet)
    if "@" not in e:
        e = re.sub(r"(\w)\s+at\s+(\w)", r"\1@\2", e)
    # Strip spaces separately in local and domain parts
    parts = e.split("@", 1)
    if len(parts) == 2:
        local  = re.sub(r"\s+", "", parts[0])
        domain = re.sub(r"\s+", "", parts[1])
        e = f"{local}@{domain}"
    return e


async def _create_booking(args: dict) -> str:
    name  = args.get("name", "")
    email = _clean_email(args.get("email", ""))
    start = args.get("start", "")
    notes = args.get("notes", "Booked via Vansh's AI persona (voice)")
    if not all([name, email, start]):
        return "Missing required fields: name, email, and start time are all required."
    if "@" not in email or "." not in email.split("@")[-1]:
        return f"Email '{email}' looks invalid — please repeat your email address clearly."
    try:
        booking = await create_booking(
            name=name,
            email=email,
            start=datetime.fromisoformat(start),
            notes=notes,
        )
        dt = booking.start.astimezone(IST)
        result = (
            f"Booking confirmed for {dt.strftime('%A %d %b at %I:%M %p IST')}. "
            f"Calendar invite sent to {email}."
        )
        if booking.meeting_url:
            result += f" Meeting link: {booking.meeting_url}"
        return result
    except Exception as e:
        log.exception("vapi_tools.booking_failed", name=name, email=email, error=str(e))
        return f"Booking failed: {e}. Please try cal.com/vansh-narang directly."
