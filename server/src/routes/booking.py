from __future__ import annotations
from fastapi import APIRouter, Query
from src.models.booking import BookingRequest, BookingResponse, AvailabilityResponse
from src.services.calcom_client import get_availability as calcom_availability
from src.services.calcom_client import create_booking as calcom_create

router = APIRouter(tags=["booking"])


@router.get("/booking/availability", response_model=AvailabilityResponse)
async def get_availability(days_ahead: int = Query(default=7, ge=1, le=30)):
    slots = await calcom_availability(days_ahead=days_ahead)
    return AvailabilityResponse(slots=slots)


@router.post("/booking/book", response_model=BookingResponse)
async def create_booking(body: BookingRequest):
    return await calcom_create(
        name=body.name,
        email=str(body.email),
        start=body.start,
        notes=body.notes,
    )
