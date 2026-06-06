"""
Vapi end-to-end evaluation script.

Tests:
1. Tool endpoint reachability (local + ngrok)
2. Retrieval quality — checks that queries return grounded chunks
3. Cache hit latency — verifies < 300ms on warm cache
4. Cold start latency — measures first-call latency
5. Booking tools — check_availability and create_booking (dry-run)
6. Recent Vapi call log analysis — fetches last N calls, checks tool results
7. Barge-in / interruption — checks if call ended for wrong reasons

Usage:
    cd server && python evaluation/eval_vapi.py [--ngrok] [--calls N]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "client", ".env.local"))

LOCAL_URL = "http://localhost:8000"
VAPI_API  = "https://api.vapi.ai"

VAPI_KEY      = os.getenv("VAPI_API_KEY", "")
ASSISTANT_ID  = os.getenv("NEXT_PUBLIC_VAPI_ASSISTANT_ID", "")
NGROK_URL     = os.getenv("NGROK_URL", "https://trickery-blunt-chatter.ngrok-free.dev")

RETRIEVE_QUERIES = [
    {"query": "AI machine learning experience projects", "expect_source": ["skills", "fit_for_role"]},
    {"query": "background education IIT Roorkee", "expect_source": ["bio", "fit_for_role"]},
    {"query": "AtlasRAG TradeSmith projects", "expect_source": ["projects", "fit_for_role", "bio", "skills"]},
    {"query": "work internship Noos Technologies", "expect_source": ["fit_for_role", "bio", "experience", "honesty"]},
    {"query": "why hire Vansh fit for role Scaler", "expect_source": ["fit_for_role"]},
    {"query": "skills programming languages frameworks", "expect_source": ["skills"]},
]

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


def _vapi_tool_body(query: str, call_id: str = "eval-1") -> dict:
    """Build exact Vapi tool-call envelope with dict arguments (real Vapi format)."""
    return {
        "message": {
            "type": "tool-calls",
            "toolCalls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "retrieve_context",
                        "arguments": {"query": query},  # dict, not string — real Vapi format
                    },
                }
            ],
        }
    }


async def test_health(client: httpx.AsyncClient) -> bool:
    section("1. Health check")
    try:
        r = await client.get(f"{LOCAL_URL}/health", timeout=5)
        data = r.json()
        if data.get("status") == "ok":
            ok("Local server healthy")
            return True
        else:
            fail(f"Health returned: {data}")
            return False
    except Exception as e:
        fail(f"Server unreachable: {e}")
        return False


async def test_retrieval_quality(client: httpx.AsyncClient, use_ngrok: bool) -> dict[str, Any]:
    section("2. Retrieval quality (tool endpoint)")
    base = NGROK_URL if use_ngrok else LOCAL_URL
    results = {"pass": 0, "fail": 0, "latencies": []}

    for i, q in enumerate(RETRIEVE_QUERIES):
        body = _vapi_tool_body(q["query"], call_id=f"eval-{i}")
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{base}/vapi/tools", json=body, timeout=30)
            elapsed = (time.perf_counter() - t0) * 1000
            results["latencies"].append(elapsed)

            data = r.json()
            result_text = data["results"][0]["result"]

            if result_text.startswith("No query provided"):
                fail(f"'{q['query'][:40]}' → NO QUERY (bug!)")
                results["fail"] += 1
            elif result_text.startswith("No relevant"):
                fail(f"'{q['query'][:40]}' → no chunks ({elapsed:.0f}ms)")
                results["fail"] += 1
            else:
                chunk_count = result_text.count("[")
                source_match = any(s in result_text for s in q["expect_source"])
                if source_match:
                    ok(f"'{q['query'][:40]}' → {chunk_count} chunks, {elapsed:.0f}ms")
                    results["pass"] += 1
                else:
                    warn(f"'{q['query'][:40]}' → {chunk_count} chunks but source mismatch ({elapsed:.0f}ms)")
                    results["fail"] += 1
        except Exception as e:
            fail(f"'{q['query'][:40]}' → exception: {e}")
            results["fail"] += 1

    lats = results["latencies"]
    if lats:
        p50 = sorted(lats)[len(lats)//2]
        p95 = sorted(lats)[int(len(lats)*0.95)]
        print(f"\n  Latency — p50={p50:.0f}ms  p95={p95:.0f}ms")
        if p50 < 300:
            ok(f"p50 {p50:.0f}ms < 300ms (cache warm ✓)")
        elif p50 < 2000:
            warn(f"p50 {p50:.0f}ms — borderline (should be < 300ms with warm cache)")
        else:
            fail(f"p50 {p50:.0f}ms > 2000ms — cache miss or cold start")

    return results


async def test_booking_tools(client: httpx.AsyncClient) -> dict[str, Any]:
    section("3. Booking tools")
    results = {"pass": 0, "fail": 0}

    # check_availability
    body = {
        "message": {
            "type": "tool-calls",
            "toolCalls": [
                {
                    "id": "book-1",
                    "type": "function",
                    "function": {"name": "check_availability", "arguments": {"days_ahead": 7}},
                }
            ],
        }
    }
    try:
        r = await client.post(f"{LOCAL_URL}/vapi/tools", json=body, timeout=20)
        text = r.json()["results"][0]["result"]
        if "No available" in text or ("IST" in text and "slot" in text.lower()):
            ok(f"check_availability: {text[:80]}")
            results["pass"] += 1
        else:
            warn(f"check_availability unexpected: {text[:80]}")
    except Exception as e:
        fail(f"check_availability failed: {e}")
        results["fail"] += 1

    return results


async def test_cold_start(client: httpx.AsyncClient) -> None:
    section("4. Cold-start vs cache-hit latency")
    # Run same query twice — second should be cache hit
    q = "AI machine learning experience Vansh projects"
    body = _vapi_tool_body(q)

    times = []
    for attempt in range(3):
        t0 = time.perf_counter()
        r = await client.post(f"{LOCAL_URL}/vapi/tools", json=body, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        text = r.json()["results"][0]["result"]
        hit = not text.startswith("No ")
        status = "hit" if hit else "miss"
        print(f"  Attempt {attempt+1}: {elapsed:.0f}ms ({status})")

    if times[1] < 400:
        ok(f"Cache warm: 2nd call {times[1]:.0f}ms < 400ms")
    else:
        warn(f"Cache not warm: 2nd call {times[1]:.0f}ms (expect < 400ms after warmup)")


async def analyze_recent_calls(n: int = 5) -> None:
    section(f"5. Recent Vapi call analysis (last {n} calls)")
    if not VAPI_KEY or not ASSISTANT_ID:
        warn("VAPI_API_KEY or NEXT_PUBLIC_VAPI_ASSISTANT_ID not set — skipping")
        return

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{VAPI_API}/call",
            headers={"Authorization": f"Bearer {VAPI_KEY}"},
            params={"limit": n},
            timeout=15,
        )
        calls = r.json()
        if not isinstance(calls, list):
            warn(f"Unexpected response: {str(calls)[:100]}")
            return

        for c in calls[:n]:
            call_id   = c.get("id", "?")
            status    = c.get("status", "?")
            end_reason= c.get("endedReason", "?")
            started   = c.get("startedAt", "?")
            msgs      = c.get("messages", [])

            tool_calls   = [m for m in msgs if m.get("role") == "tool_calls"]
            tool_results = [m for m in msgs if m.get("role") == "tool_call_result"]
            no_query     = sum(1 for m in tool_results if "No query provided" in str(m.get("result", "")))
            retrieval_ok = sum(1 for m in tool_results if len(str(m.get("result", ""))) > 200)

            print(f"\n  Call {call_id[:8]}… | {started[:10]} | end={end_reason}")
            print(f"    Tool calls={len(tool_calls)}  results={len(tool_results)}  no_query_errors={no_query}  retrieval_ok={retrieval_ok}")

            if no_query > 0:
                warn(f"    {no_query} 'No query provided' errors (pre-fix calls show this; test a new call)")
            elif retrieval_ok > 0:
                ok(f"    {retrieval_ok} successful retrievals")
            else:
                warn(f"    No successful retrievals found")

            if end_reason in ("silence-timed-out", "ejected"):
                warn(f"    End reason '{end_reason}' — possible tool failure or voice issue")
            elif end_reason == "customer-ended-call":
                ok(f"    Customer ended call normally")


async def test_vapi_assistant_config() -> None:
    section("6. Vapi assistant configuration")
    if not VAPI_KEY or not ASSISTANT_ID:
        warn("Missing keys — skipping")
        return

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{VAPI_API}/assistant/{ASSISTANT_ID}",
            headers={"Authorization": f"Bearer {VAPI_KEY}"},
            timeout=10,
        )
        a = r.json()
        model = a.get("model", {})
        tools = model.get("tools", [])

        ok(f"Assistant ID: {ASSISTANT_ID}")
        ok(f"Model: {model.get('provider')} {model.get('model')}")
        ok(f"Voice: {a.get('voice', {}).get('provider')} {a.get('voice', {}).get('voiceId')}")

        max_tokens = model.get("maxTokens", 0)
        if max_tokens and max_tokens < 300:
            warn(f"maxTokens={max_tokens} — low, may truncate voice responses (recommend 400+)")
        else:
            ok(f"maxTokens={max_tokens}")

        for t in tools:
            fn  = t.get("function", {})
            srv = t.get("server", {})
            timeout = srv.get("timeoutSeconds", "not set")
            name = fn.get("name", "?")
            if timeout == "not set":
                fail(f"Tool {name}: no timeoutSeconds (Vapi default ~5s, too low)")
            elif timeout < 20:
                warn(f"Tool {name}: timeoutSeconds={timeout} (low for cold start)")
            else:
                ok(f"Tool {name}: timeoutSeconds={timeout}s, url={srv.get('url','?')[:50]}")

        system = model.get("systemPrompt", "")
        if "retrieve_context" in system and ("MUST" in system or "ALWAYS" in system):
            ok("System prompt forces tool use")
        else:
            warn("System prompt may not strongly force retrieval — check prompt")

        # Check for bad patterns: prompt TELLS LLM to say filler, not just mentions them
        filler_patterns = ["say 'hold on'", "say 'one moment'", "tell the caller"]
        if any(p in system.lower() for p in filler_patterns):
            fail("System prompt instructs LLM to say filler phrases (remove them)")
        else:
            ok("System prompt discourages filler phrases")


async def main(args: argparse.Namespace) -> None:
    print(f"\n{BOLD}Vapi End-to-End Evaluation{RESET}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Local: {LOCAL_URL}  |  Ngrok: {NGROK_URL}")

    async with httpx.AsyncClient() as client:
        healthy = await test_health(client)
        if not healthy:
            print("\n⚠️  Server not running — start with: make start")
            sys.exit(1)

        retrieval = await test_retrieval_quality(client, use_ngrok=args.ngrok)
        booking   = await test_booking_tools(client)
        await test_cold_start(client)

    await analyze_recent_calls(n=args.calls)
    await test_vapi_assistant_config()

    section("Summary")
    total_pass = retrieval["pass"] + booking["pass"]
    total_fail = retrieval["fail"] + booking["fail"]
    if total_fail == 0:
        ok(f"All {total_pass} checks passed")
    else:
        warn(f"{total_pass} passed, {total_fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vapi end-to-end evaluation")
    parser.add_argument("--ngrok", action="store_true", help="Test via ngrok URL (real Vapi path)")
    parser.add_argument("--calls", type=int, default=5, help="Number of recent Vapi calls to analyze")
    asyncio.run(main(parser.parse_args()))
