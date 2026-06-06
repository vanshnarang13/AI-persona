from __future__ import annotations
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime


class AvailabilityRequest(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=30)


class TimeSlot(BaseModel):
    start: datetime
    end: datetime


class BookingRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    email: EmailStr
    start: datetime
    notes: str = ""


class BookingResponse(BaseModel):
    booking_id: str
    meeting_url: str | None = None
    start: datetime
    end: datetime
    confirmation_sent: bool = False


class AvailabilityResponse(BaseModel):
    slots: list[TimeSlot]
