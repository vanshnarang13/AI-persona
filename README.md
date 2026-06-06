# AI Persona — Vansh Narang

An autonomous AI persona that recruiters can **call** (voice) or **chat with** (web), ask about background, projects, and GitHub repos, and **book a real interview** — no human in the loop.

Built as a screening assignment for the Scaler AI Engineer Intern role.

---

## What it does

- Answers questions about Vansh's background, skills, projects, and GitHub work — every answer is grounded in a retrieved corpus, never invented
- Books a 30-minute interview call via Cal.com directly in the conversation
- Works over voice (phone call via Vapi) and web chat (Next.js SSE stream) from the same backend
- Resists prompt injection and falls back gracefully when confidence is low

---

## Architecture

```
Phone (Vapi)  ─┐
               ├─► FastAPI server (async)
Web chat  ─────┘     ├── /chat          SSE streaming
                     ├── /vapi/tools    single dispatch endpoint (all 3 tools)
                     ├── /booking       LangGraph StateGraph
                     └── /health

                     Retrieval pipeline
                     ├── Vector search   pgvector HNSW + text-embedding-3-small
                     ├── BM25            rank_bm25, Redis-cached (TTL 2h)
                     ├── RRF fusion      weights 0.55/0.45, K=60
                     └── Rerank          cross-encoder (chat only, skipped on voice)

                     Agents
                     ├── PersonaAgent   create_react_agent, gpt-4o
                     └── BookingAgent   LangGraph StateGraph, AsyncPostgresSaver

                     Services
                     ├── Supabase pgvector   vector store + booking checkpoints
                     ├── Upstash Redis        BM25 index cache + retrieval cache
                     └── Cal.com v2 API       availability + bookings
```

**Voice path:** Vapi → Deepgram nova-2 STT → gpt-4o → `/vapi/tools` dispatch → ElevenLabs George TTS

**Chat path:** Next.js → SSE stream → PersonaAgent → hybrid retrieval → gpt-4o

Both paths hit the same retrieval pipeline, so answers are consistent across channels.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn, SSE |
| LLM | gpt-4o (tool-calling), gpt-4o-mini (evals) |
| Embeddings | text-embedding-3-small (1536d) |
| Vector DB | Supabase pgvector, HNSW (m=16, ef=64) |
| Lexical search | rank_bm25, Redis-cached |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Cache | Upstash Redis (binary client) |
| Booking state | LangGraph AsyncPostgresSaver on Supabase Postgres |
| Voice | Vapi · Deepgram nova-2 · ElevenLabs George |
| Booking | Cal.com v2 REST API |
| Frontend | Next.js 15, Tailwind CSS |
| Deploy | Render (backend) · Vercel (frontend) |

---

## Repo layout

```
persona-agent/
├── server/
│   ├── src/
│   │   ├── server.py                 # FastAPI entrypoint + lifespan
│   │   ├── routes/                   # chat, retrieve, booking, health, vapi_tools
│   │   ├── rag/
│   │   │   ├── ingestion/            # partition → chunk → embed → vectorize
│   │   │   └── retrieval/            # vector, lexical_bm25, hybrid_rrf, rerank, pipeline
│   │   ├── agents/
│   │   │   ├── persona_agent/        # grounded Q&A with guardrails
│   │   │   └── booking_agent/        # LangGraph StateGraph + InjectedState tools
│   │   ├── services/                 # supabase, redis, llm, calcom clients
│   │   ├── config/                   # settings, prompts, tuning
│   │   └── models/                   # Pydantic schemas
│   ├── data/corpus/                  # knowledge base (bio, resume, projects, GitHub)
│   ├── ingest/                       # ingestion scripts
│   ├── evaluation/                   # eval scripts + golden set
│   ├── Dockerfile
│   ├── render.yaml
│   └── requirements.txt
├── client/                           # Next.js 15 frontend
└── voice/
    └── vapi_assistant_config.json    # Vapi assistant reference config
```

---

## Running locally

**Prerequisites:** Python 3.9+, Node 18+, a Supabase project with pgvector, Upstash Redis, Cal.com account, OpenAI API key, Vapi account.

```bash
# 1. Copy and fill env
cp server/.env.example server/.env
cp client/.env.local.example client/.env.local

# 2. Run DB migration
psql $DATABASE_URL -f server/supabase/migrations/0001_init.sql

# 3. Ingest corpus
cd server && PYTHONPATH=. python ingest/ingest_corpus.py

# 4. Start backend (port 8000)
cd server && python -m uvicorn src.server:app --port 8000 --reload

# 5. Start frontend (port 3000)
cd client && npm install && npm run dev

# 6. Voice tunnel (ngrok)
ngrok http --domain=<your-static-domain> 8000
```

---

## Retrieval configuration

Tuned via grid search on a 67-question golden set. Locked in `server/src/config/tuning.py`.

| Parameter | Value |
|---|---|
| TOP_K chat / voice | 8 / 3 |
| Vector / BM25 weights | 0.55 / 0.45 |
| HNSW ef_search | 100 |
| Confidence threshold | 0.35 |

---

## Evaluation

```bash
# Chat groundedness (67 questions, LLM-as-judge)
cd server && PYTHONPATH=. python evaluation/run_evals.py

# Voice quality (pulls real Vapi call logs)
cd server && PYTHONPATH=. python evaluation/eval_voice.py
```

**Results:**

| Metric | Result |
|---|---|
| Hallucination rate | 1.5% (1/67, false positive) |
| Retrieval precision@3 | 92.1% |
| Injection defense | 100% (5/5) |
| Voice first-response latency p50 | 4.1s end-to-end |
| Server-side retrieval (warm cache) | ~33ms |
| Booking success rate | 83% (5/6) |
| Transcription accuracy | 97.5% |

---

## Key implementation notes

**Vapi tool arguments:** Vapi sends `arguments` as a dict, not a JSON string. Always handle both: `json.loads(x) if isinstance(x, str) else x`.

**Slot availability:** The `check_availability` tool has no `days_ahead` parameter — the server always fetches 7 days. The LLM passes a natural language `preference` string; the server does all date math via `parse_preference_filter()`.

**Turn-handover fix:** Each Vapi tool definition has `messages[type: "request-start"]` to keep the audio turn active during tool execution and prevent Vapi from handing the mic back to the user.

**Redis client:** Must be binary (`decode_responses=False`). BM25 index and retrieval cache use pickle serialization.

**Booking state:** Uses LangGraph `InjectedState` — the LLM never touches raw ISO datetimes. Users pick slots by number; the graph resolves the slot ID from state.
