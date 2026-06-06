"""
Voice evaluation script — pulls real metrics from Vapi call logs.

Measures:
1. First-response latency  — seconds from call start to first assistant word
2. Tool call latency       — time between tool dispatch and result returned
3. Transcription accuracy  — key proper noun recognition rate via LLM judge
4. Task completion rate    — booking success across all calls with booking attempts
5. Turn-handover failures  — calls where tool silence caused mic to pass back

Run:
    cd server && PYTHONPATH=. python evaluation/eval_voice.py
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "client" / ".env.local")

VAPI_KEY     = os.getenv("VAPI_API_KEY", "")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# Key proper nouns that Deepgram should recognise correctly
KEY_TERMS = [
    "Vansh", "Narang", "IIT Roorkee", "AtlasRAG", "TradeSmith",
    "LangGraph", "pgvector", "Scaler", "Noos Technologies",
    "steganography", "Celery", "FastAPI",
]


def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg: str) -> None: print(f"  {RED}✗{RESET} {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}!{RESET} {msg}")
def section(t: str)-> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {t}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


async def fetch_calls(client: httpx.AsyncClient, limit: int = 30) -> list[dict]:
    r = await client.get(
        "https://api.vapi.ai/call",
        headers={"Authorization": f"Bearer {VAPI_KEY}"},
        params={"limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    return r.json() if isinstance(r.json(), list) else []


def extract_latency(call: dict) -> dict:
    """
    First-response latency = delta between when the user finished speaking
    and when the bot started its first real reply.
    Measured as: first_bot_sfs - last_user_sfs_before_that_bot_message.

    Tool call latency = tool_result.secondsFromStart - tool_call.secondsFromStart.
    """
    msgs = call.get("messages", [])
    first_response_latency = None
    tool_latencies = []

    last_user_sfs: float | None = None
    pending_tool: dict | None = None

    for m in msgs:
        role = m.get("role", "")
        sfs  = float(m.get("secondsFromStart", 0))

        if role == "user" and m.get("message"):
            last_user_sfs = sfs

        if (role == "bot" and m.get("message")
                and last_user_sfs is not None
                and first_response_latency is None):
            first_response_latency = sfs - last_user_sfs

        if role == "tool_calls":
            pending_tool = m

        if role == "tool_call_result" and pending_tool is not None:
            latency = sfs - float(pending_tool.get("secondsFromStart", sfs))
            if 0 <= latency < 60:
                tool_latencies.append(latency)
            pending_tool = None

    return {
        "first_response_s": first_response_latency,
        "tool_latencies_s": tool_latencies,
    }


def check_booking(call: dict) -> dict:
    """Returns booking attempted/succeeded/failed for a call."""
    msgs = call.get("messages", [])
    attempted = False
    succeeded = False
    failed    = False
    failure_reason = ""

    for m in msgs:
        role = m.get("role", "")
        if role == "tool_calls":
            for tc in m.get("toolCalls", []):
                if tc.get("function", {}).get("name") == "create_booking":
                    attempted = True
        if role == "tool_call_result" and attempted:
            result = str(m.get("result", ""))
            if "Booking confirmed" in result:
                succeeded = True
            elif "Booking failed" in result or "400" in result or "Missing required" in result:
                failed = True
                failure_reason = result[:120]

    return {"attempted": attempted, "succeeded": succeeded,
            "failed": failed, "failure_reason": failure_reason}


def detect_turn_handover_failure(call: dict) -> bool:
    """
    Detects if a tool call was made but no bot reply followed —
    a sign that Vapi handed the turn back before the tool result was used.
    """
    msgs = call.get("messages", [])
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "tool_call_result":
            # Look ahead for a bot message
            found_bot = any(
                msgs[j].get("role") == "bot" and msgs[j].get("message")
                for j in range(i + 1, min(i + 4, len(msgs)))
            )
            if not found_bot and i < len(msgs) - 1:
                return True
        i += 1
    return False


async def score_transcription_accuracy(
    call: dict,
    judge: AsyncOpenAI,
) -> dict:
    """
    Use GPT-4o-mini as a judge to score whether key proper nouns were
    transcribed correctly in the user messages.
    """
    msgs = call.get("messages", [])
    user_text = " ".join(
        m.get("message", "") for m in msgs if m.get("role") == "user"
    ).strip()

    if not user_text:
        return {"scored": False, "accuracy": None, "corrupted_terms": []}

    prompt = f"""You are evaluating speech-to-text transcription quality for a recruiter call.

User transcript:
{user_text}

Key proper nouns that should be recognised correctly if mentioned:
{', '.join(KEY_TERMS)}

Tasks:
1. List any key proper nouns from the list above that appear corrupted or misspelled in the transcript (e.g. "Vansh" → "Bunch", "AtlasRAG" → "Atlas Rag").
2. Estimate transcription accuracy for proper nouns as a percentage (100% = all correctly captured, lower = corruptions found).

Reply in JSON only:
{{"corrupted": ["list of corrupted terms with what they became"], "accuracy_pct": 0-100}}"""

    resp = await judge.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
        return {
            "scored": True,
            "accuracy": data.get("accuracy_pct"),
            "corrupted_terms": data.get("corrupted", []),
        }
    except Exception:
        return {"scored": False, "accuracy": None, "corrupted_terms": []}


async def run() -> None:
    section("Voice Quality Evaluation")
    print(f"  Fetching Vapi call logs...")

    async with httpx.AsyncClient() as http:
        calls = await fetch_calls(http, limit=30)

    if not calls:
        fail("No calls found — check VAPI_API_KEY")
        return

    print(f"  Analysing {len(calls)} calls\n")

    judge = AsyncOpenAI(api_key=OPENAI_KEY, timeout=20.0)

    first_response_latencies = []
    all_tool_latencies       = []
    booking_attempted        = 0
    booking_succeeded        = 0
    booking_failures         = []
    turn_handover_failures   = 0
    transcription_scores     = []
    corrupted_terms_seen     = []
    call_records             = []

    section("Per-call breakdown")

    for c in calls:
        call_id = c["id"][:12]
        started = c.get("startedAt", "")[:10]
        ended   = c.get("endedReason", "?")
        msgs    = c.get("messages", [])

        lat     = extract_latency(c)
        booking = check_booking(c)
        thf     = detect_turn_handover_failure(c)

        if lat["first_response_s"] is not None:
            first_response_latencies.append(lat["first_response_s"])
        all_tool_latencies.extend(lat["tool_latencies_s"])

        if booking["attempted"]:
            booking_attempted += 1
            if booking["succeeded"]:
                booking_succeeded += 1
            elif booking["failed"]:
                booking_failures.append(booking["failure_reason"])

        if thf:
            turn_handover_failures += 1

        # Transcription scoring (only for calls with substantive user content)
        txn = await score_transcription_accuracy(c, judge)
        if txn["scored"]:
            transcription_scores.append(txn["accuracy"])
            corrupted_terms_seen.extend(txn["corrupted_terms"])

        status = "OK" if not thf else "TURN-HANDOVER"
        bk_str = ""
        if booking["attempted"]:
            bk_str = f"  booking={'✓' if booking['succeeded'] else '✗'}"
        frp    = f"{lat['first_response_s']:.1f}s" if lat["first_response_s"] is not None else "n/a"
        print(f"  {call_id} | {started} | end={ended:25s} | 1st-resp={frp:6s} | {status}{bk_str}")

        call_records.append({
            "call_id":                c["id"],
            "started_at":             c.get("startedAt"),
            "ended_reason":           ended,
            "messages":               len(msgs),
            "first_response_s":       lat["first_response_s"],
            "tool_latencies_s":       lat["tool_latencies_s"],
            "booking_attempted":      booking["attempted"],
            "booking_succeeded":      booking["succeeded"],
            "booking_failed":         booking["failed"],
            "turn_handover_failure":  thf,
            "transcription_accuracy": txn.get("accuracy"),
            "corrupted_terms":        txn.get("corrupted_terms", []),
        })

    section("Results")

    # First-response latency
    if first_response_latencies:
        frl = sorted(first_response_latencies)
        p50 = frl[len(frl)//2]
        p95 = frl[int(len(frl)*0.95)]
        avg = mean(frl)
        icon = GREEN if p50 < 2.0 else YELLOW
        print(f"  {icon}First-response latency{RESET}")
        print(f"    p50={p50:.2f}s  p95={p95:.2f}s  avg={avg:.2f}s  n={len(frl)}")
        if p50 < 2.0:
            ok(f"p50 {p50:.2f}s under 2s target")
        else:
            warn(f"p50 {p50:.2f}s over 2s target")
    else:
        warn("No first-response latency data (calls may have no user turns)")

    # Tool latency
    if all_tool_latencies:
        tl = sorted(all_tool_latencies)
        print(f"\n  Tool call round-trip latency (server + network)")
        print(f"    p50={median(tl):.2f}s  p95={tl[int(len(tl)*0.95)]:.2f}s  n={len(tl)}")

    # Booking success
    print(f"\n  Booking task completion")
    if booking_attempted:
        rate = booking_succeeded / booking_attempted
        icon = GREEN if rate >= 0.8 else YELLOW
        print(f"    {icon}{booking_succeeded}/{booking_attempted} = {rate:.0%}{RESET}")
        if rate >= 0.8:
            ok(f"{rate:.0%} booking success rate")
        else:
            warn(f"{rate:.0%} booking success rate — below 80% target")
        for f in booking_failures:
            fail(f"  Failure: {f[:80]}")
    else:
        warn("No booking attempts found in these calls")

    # Turn-handover failures
    print(f"\n  Turn-handover failures (tool call with no bot follow-up)")
    if turn_handover_failures == 0:
        ok("No turn-handover failures detected")
    else:
        fail(f"{turn_handover_failures} turn-handover failures detected")

    # Transcription accuracy
    print(f"\n  Transcription accuracy (proper noun recognition, LLM-judged)")
    if transcription_scores:
        avg_txn = mean(transcription_scores)
        icon = GREEN if avg_txn >= 85 else YELLOW
        print(f"    {icon}avg={avg_txn:.1f}%{RESET}  n={len(transcription_scores)}")
        if avg_txn >= 85:
            ok(f"{avg_txn:.1f}% proper noun accuracy")
        else:
            warn(f"{avg_txn:.1f}% — some proper nouns being corrupted by STT")
        if corrupted_terms_seen:
            print(f"    Corruptions observed:")
            for t in set(corrupted_terms_seen):
                print(f"      {RED}✗{RESET} {t}")
    else:
        warn("No transcription scores (calls had no user speech)")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"eval_voice_{ts}.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_calls_analysed": len(calls),
        "first_response_latency": {
            "p50_s": round(sorted(first_response_latencies)[len(first_response_latencies)//2], 3) if first_response_latencies else None,
            "p95_s": round(sorted(first_response_latencies)[int(len(first_response_latencies)*0.95)], 3) if first_response_latencies else None,
            "avg_s": round(mean(first_response_latencies), 3) if first_response_latencies else None,
            "n": len(first_response_latencies),
        },
        "tool_call_latency": {
            "p50_s": round(median(all_tool_latencies), 3) if all_tool_latencies else None,
            "p95_s": round(sorted(all_tool_latencies)[int(len(all_tool_latencies)*0.95)], 3) if all_tool_latencies else None,
            "n": len(all_tool_latencies),
        },
        "booking": {
            "attempted": booking_attempted,
            "succeeded": booking_succeeded,
            "success_rate": round(booking_succeeded / booking_attempted, 4) if booking_attempted else None,
        },
        "transcription": {
            "avg_accuracy_pct": round(mean(transcription_scores), 1) if transcription_scores else None,
            "n_calls_scored": len(transcription_scores),
            "corrupted_terms": list(set(corrupted_terms_seen)),
        },
        "turn_handover_failures": turn_handover_failures,
        "calls": call_records,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved: {out}")


if __name__ == "__main__":
    asyncio.run(run())
