from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.services.calcom_client import (
    get_availability,
    create_booking,
)
from src.config.prompts import BOOKING_NOTE

IST = timezone(timedelta(hours=5, minutes=30))


def _fmt_slots(slots: dict) -> str:
    lines = []
    for slot_id, s in slots.items():
        dt = datetime.fromisoformat(s["start"]).astimezone(IST)
        lines.append(f"  {slot_id}: {dt.strftime('%A %d %b %Y, %I:%M %p IST')}")
    return "\n".join(lines)


def parse_preference_filter(preference: str):
    """
    Returns a callable (datetime -> bool) that matches the user's time preference.
    Understands: day names, 'after X pm', 'before X am', 'morning/afternoon/evening'.
    Returns None if no time filter can be parsed.
    """
    pref = preference.lower()
    filters = []

    # Day-of-week filter
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    for name, weekday in day_map.items():
        if name in pref:
            filters.append(lambda dt, wd=weekday: dt.weekday() == wd)
            break

    # Weekday / weekend
    if "weekday" in pref:
        filters.append(lambda dt: dt.weekday() < 5)
    elif "weekend" in pref:
        filters.append(lambda dt: dt.weekday() >= 5)

    # Named periods
    if "morning" in pref:
        filters.append(lambda dt: dt.hour < 12)
    elif "afternoon" in pref:
        filters.append(lambda dt: 12 <= dt.hour < 17)
    elif "evening" in pref:
        filters.append(lambda dt: dt.hour >= 17)

    # "at X pm" — exact hour, show slots at that hour only (±30 min window)
    m = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", pref)
    if m:
        h, mins, period = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if period == "pm" and h != 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        filters.append(lambda dt, _h=h, _m=mins: (_h, _m) <= (dt.hour, dt.minute) < (_h + 1, _m))

    # "after X am/pm" — e.g. "after 2 pm", "after 14:00"
    m = re.search(r"after\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", pref)
    if m:
        h, mins, period = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if period == "pm" and h != 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        filters.append(lambda dt, _h=h, _m=mins: (dt.hour, dt.minute) >= (_h, _m))

    # "before X am/pm"
    m = re.search(r"before\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", pref)
    if m:
        h, mins, period = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if period == "pm" and h != 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        filters.append(lambda dt, _h=h, _m=mins: (dt.hour, dt.minute) < (_h, _m))

    # "between X and Y"
    m = re.search(r"between\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+and\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", pref)
    if m:
        h1, m1, p1 = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        h2, m2, p2 = int(m.group(4)), int(m.group(5) or 0), m.group(6)
        if p1 == "pm" and h1 != 12: h1 += 12
        if p2 == "pm" and h2 != 12: h2 += 12
        filters.append(lambda dt, _h1=h1, _m1=m1, _h2=h2, _m2=m2:
                        (_h1, _m1) <= (dt.hour, dt.minute) < (_h2, _m2))

    if not filters:
        return None

    def combined(dt: datetime) -> bool:
        return all(f(dt) for f in filters)

    return combined


# ── Tool 1: fetch availability ─────────────────────────────────────────────────

@tool
async def get_availability_tool(preference: str) -> str:
    """
    Fetch Vansh's available 30-min interview slots from Cal.com.

    Call whenever the user wants to schedule a meeting, interview, or call.
    Pass the user's stated time preference (e.g. 'tomorrow morning',
    'Friday 10-12', 'weekday afternoons') so the response can be filtered.

    Returns a numbered slot list with slot IDs for use with select_slot_tool.
    """
    all_slots = await get_availability(days_ahead=7)
    if not all_slots:
        return json.dumps({
            "status": "no_slots",
            "message": "No slots available in the next 7 days.",
            "slots": {},
        })

    # Group by day (IST) so every day gets fair representation
    by_day: dict[str, list] = {}
    for s in all_slots:
        day = s.start.astimezone(IST).strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(s)

    selected: list = []
    for day in sorted(by_day):
        selected.extend(by_day[day])

    # All slots go into state (needed for select_slot_tool to validate IDs)
    all_slot_dict: dict[str, dict] = {
        f"slot_{i}": {"start": s.start.isoformat(), "end": s.end.isoformat()}
        for i, s in enumerate(selected, start=1)
    }

    # Apply preference filter only for the formatted display shown to the user
    time_filter = parse_preference_filter(preference)
    if time_filter:
        display_slots = {
            k: v for k, v in all_slot_dict.items()
            if time_filter(datetime.fromisoformat(v["start"]).astimezone(IST))
        }
        filter_note = f" (filtered to: {preference})"
    else:
        display_slots = all_slot_dict
        filter_note = ""

    return json.dumps({
        "status": "ok",
        "preference": preference,
        "slots": all_slot_dict,          # full set — LLM must use these IDs
        "formatted_slots": _fmt_slots(display_slots) + filter_note,
        "displayed_count": len(display_slots),
        "total_count": len(all_slot_dict),
    })


# ── Tool 2: lock in a slot ─────────────────────────────────────────────────────

@tool
async def select_slot_tool(
    slot_id: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Lock in a slot the user has chosen.

    Pass the slot_id (e.g. 'slot_3') from the list returned by
    get_availability_tool.  The slot must exist in the current conversation —
    do NOT invent slot IDs.
    """
    available: dict = state.get("available_slots") or {}
    if slot_id not in available:
        known = ", ".join(available.keys()) or "none yet — call get_availability_tool first"
        return json.dumps({
            "status": "error",
            "message": f"Unknown slot_id '{slot_id}'. Known slots: {known}",
        })

    slot = available[slot_id]
    dt = datetime.fromisoformat(slot["start"]).astimezone(IST)
    return json.dumps({
        "status": "ok",
        "slot_id": slot_id,
        "confirmed_time": dt.strftime("%A %d %b %Y, %I:%M %p IST"),
    })


# ── Tool 3: create booking ─────────────────────────────────────────────────────

@tool
async def create_booking_tool(
    name: str,
    email: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Book the selected slot with Vansh Narang on Cal.com.

    Call ONLY when ALL of the following are confirmed:
    - name  : recruiter's full name
    - email : recruiter's email address
    - a slot has already been selected via select_slot_tool

    A calendar invite is sent to the provided email automatically.
    """
    slot_id: str | None = state.get("selected_slot_id")
    available: dict = state.get("available_slots") or {}

    if not slot_id:
        return json.dumps({
            "status": "error",
            "message": "No slot selected yet. Call select_slot_tool first.",
        })
    slot = available.get(slot_id)
    if not slot:
        return json.dumps({
            "status": "error",
            "message": f"Slot '{slot_id}' not found in available slots. Call get_availability_tool.",
        })

    booking = await create_booking(
        name=name,
        email=email,
        start=datetime.fromisoformat(slot["start"]),
        notes=BOOKING_NOTE,
    )

    dt = datetime.fromisoformat(slot["start"]).astimezone(IST)
    return json.dumps({
        "status": "ok",
        "booking_id": booking.booking_id,
        "meeting_url": booking.meeting_url,
        "confirmed_time": dt.strftime("%A %d %b %Y, %I:%M %p IST"),
        "calendar_invite_sent_to": email,
    })
