import structlog
import httpx
from datetime import datetime, timedelta, timezone
from src.config import settings, app as app_config
from src.models.booking import TimeSlot, BookingResponse

log = structlog.get_logger(__name__)

CALCOM_BASE = "https://api.cal.com/v2"

def _headers(api_version: str) -> dict:
    return {
        "Authorization": f"Bearer {settings.calcom_api_key}",
        "cal-api-version": api_version,
        "Content-Type": "application/json",
    }


async def get_availability(days_ahead: int = 7) -> list[TimeSlot]:
    now = datetime.now(timezone.utc)
    start_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{CALCOM_BASE}/slots",
            headers=_headers("2024-09-04"),
            params={
                "eventTypeId": settings.calcom_event_type_id,
                "start": start_time,   # v2 uses "start"/"end", not "startTime"/"endTime"
                "end": end_time,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            log.warning("calcom.availability_unexpected", status=data.get("status"))

    slot_duration = app_config.CALCOM_SLOT_DURATION_MINUTES
    slots: list[TimeSlot] = []
    # v2 response: {"data": {"2026-06-08": [{"start": "..."}, ...], ...}}
    for day_slots in data.get("data", {}).values():
        for s in day_slots:
            slot_start = datetime.fromisoformat(s["start"].replace("Z", "+00:00"))
            slots.append(TimeSlot(start=slot_start, end=slot_start + timedelta(minutes=slot_duration)))
    return slots


async def create_booking(
    name: str,
    email: str,
    start: datetime,
    notes: str = "",
) -> BookingResponse:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{CALCOM_BASE}/bookings",
            headers=_headers("2026-02-25"),
            json={
                "eventTypeId": settings.calcom_event_type_id,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "attendee": {
                    "name": name,
                    "email": email,
                    "timeZone": "Asia/Kolkata",
                    "language": "en",
                },
                "bookingFieldsResponses": {"notes": notes} if notes else {},
            },
        )
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data")
        if data is None:
            log.error("calcom.booking_missing_data", body=str(body)[:200])
            raise ValueError(f"Unexpected Cal.com booking response: {body}")

    return BookingResponse(
        booking_id=str(data["id"]),
        meeting_url=data.get("meetingUrl") or data.get("videoCallData", {}).get("url"),
        start=datetime.fromisoformat(data["start"].replace("Z", "+00:00")),
        end=datetime.fromisoformat(data["end"].replace("Z", "+00:00")),
        confirmation_sent=True,
    )
