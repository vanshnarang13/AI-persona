"""
Credential health check — run from server/ directory:
    python scripts/check_credentials.py
"""
import os, sys, asyncio
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

GREEN = "\033[92m✓\033[0m"
RED   = "\033[91m✗\033[0m"
YELLOW = "\033[93m~\033[0m"

def ok(name, msg=""): print(f"  {GREEN} {name}" + (f" — {msg}" if msg else ""))
def fail(name, msg=""): print(f"  {RED} {name}" + (f" — {msg}" if msg else ""))
def warn(name, msg=""): print(f"  {YELLOW} {name}" + (f" — {msg}" if msg else ""))


# ── 1. OpenAI ──────────────────────────────────────────────────────────────
def check_openai():
    print("\n[1] OpenAI")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        fail("OPENAI_API_KEY", "not set"); return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Embeddings
        resp = client.embeddings.create(input="health check", model="text-embedding-3-small")
        dims = len(resp.data[0].embedding)
        ok("Embeddings", f"text-embedding-3-small → {dims}d vector")

        # Chat
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
        )
        reply = resp.choices[0].message.content.strip()
        ok("Chat completion", f"gpt-4o-mini → '{reply}'")

    except Exception as e:
        fail("OpenAI", str(e))


# ── 2. Supabase client ────────────────────────────────────────────────────
def check_supabase():
    print("\n[2] Supabase")
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")

    if not url: fail("SUPABASE_URL", "not set"); return
    if not key: fail("SUPABASE_SERVICE_KEY", "not set"); return

    try:
        from supabase import create_client
        client = create_client(url, key)
        # Simple health — list tables via pg catalog
        result = client.table("pg_tables").select("tablename").limit(1).execute()
        ok("Supabase client", f"connected to {url}")
    except Exception as e:
        # Supabase free tier may restrict pg_tables — try a raw health ping instead
        try:
            import httpx
            resp = httpx.get(f"{url}/rest/v1/", headers={"apikey": key}, timeout=8)
            if resp.status_code in (200, 400):  # 400 = no table specified, still authenticated
                ok("Supabase client", f"REST API reachable (HTTP {resp.status_code})")
            else:
                fail("Supabase client", f"HTTP {resp.status_code}")
        except Exception as e2:
            fail("Supabase client", str(e2))


# ── 3. Postgres / asyncpg ────────────────────────────────────────────────
async def check_database():
    print("\n[3] Database (asyncpg)")
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        fail("DATABASE_URL", "not set"); return

    try:
        import asyncpg
        # Strip SQLAlchemy driver prefix — asyncpg needs plain postgresql://
        dsn = db_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn, statement_cache_size=0, timeout=10)
        version = await conn.fetchval("SELECT version()")
        await conn.close()
        ok("asyncpg connection", version.split(",")[0])
    except Exception as e:
        fail("asyncpg connection", str(e))


# ── 4. Redis / Upstash ────────────────────────────────────────────────────
async def check_redis():
    print("\n[4] Redis (Upstash)")
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        fail("REDIS_URL", "not set"); return

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(redis_url, decode_responses=True, socket_timeout=8)
        await r.set("persona:healthcheck", "ok", ex=30)
        val = await r.get("persona:healthcheck")
        await r.aclose()
        ok("Redis", f"SET/GET round-trip → '{val}'")
    except Exception as e:
        fail("Redis", str(e))


# ── 5. Cal.com ────────────────────────────────────────────────────────────
async def check_calcom():
    print("\n[5] Cal.com")
    api_key  = os.getenv("CALCOM_API_KEY", "")
    username = os.getenv("CALCOM_USERNAME", "")
    event_id = os.getenv("CALCOM_EVENT_TYPE_ID", "")

    if not api_key: fail("CALCOM_API_KEY", "not set"); return

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.cal.com/v2/event-types",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "cal-api-version": "2024-06-14",
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            raw = data.get("data", [])
            # v2 returns either a list of event types or a list of groups
            if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "eventTypes" in raw[0]:
                event_types = raw[0].get("eventTypes", [])
            elif isinstance(raw, list):
                event_types = raw
            else:
                event_types = raw.get("eventTypeGroups", [{}])[0].get("eventTypes", [])
            names = [e["title"] for e in event_types[:3]]
            ok("Cal.com API key", f"found {len(event_types)} event type(s): {names}")
            if event_id and not any(str(e["id"]) == str(event_id) for e in event_types):
                warn("CALCOM_EVENT_TYPE_ID", f"{event_id} not found in your event types — double-check it")
            elif event_id:
                ok("CALCOM_EVENT_TYPE_ID", f"{event_id} confirmed")
        else:
            fail("Cal.com", f"HTTP {resp.status_code} — {resp.text[:120]}")
    except Exception as e:
        fail("Cal.com", str(e))


# ── 6. Vapi ───────────────────────────────────────────────────────────────
async def check_vapi():
    print("\n[6] Vapi")
    api_key       = os.getenv("VAPI_API_KEY", "")
    phone_number_id = os.getenv("VAPI_PHONE_NUMBER_ID", "")

    if not api_key: fail("VAPI_API_KEY", "not set"); return

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://api.vapi.ai/phone-number",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code == 200:
            numbers = resp.json()
            count = len(numbers) if isinstance(numbers, list) else "?"
            ok("Vapi API key", f"{count} phone number(s) on account")
            if isinstance(numbers, list) and phone_number_id:
                ids = [n.get("id") for n in numbers]
                if phone_number_id in ids:
                    num = next(n for n in numbers if n.get("id") == phone_number_id)
                    ok("VAPI_PHONE_NUMBER_ID", num.get("number", phone_number_id))
                else:
                    fail("VAPI_PHONE_NUMBER_ID", f"{phone_number_id} not found on this account")
        else:
            fail("Vapi", f"HTTP {resp.status_code} — {resp.text[:120]}")
    except Exception as e:
        fail("Vapi", str(e))


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    print("=" * 50)
    print("  Persona Agent — Credential Health Check")
    print("=" * 50)

    check_openai()
    check_supabase()
    await check_database()
    await check_redis()
    await check_calcom()
    await check_vapi()

    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
