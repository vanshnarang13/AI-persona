# Evaluation Report — Vansh Narang AI Persona
**Scaler AI Engineer Intern Screening Assignment**
Generated: 2026-06-06

---

## Part A — Voice Quality

### How First-Response Latency Was Measured

First-response latency is computed from Vapi call logs using the `secondsFromStart` timestamp on each message. The metric is:

```
latency = bot_message.secondsFromStart − preceding_user_message.secondsFromStart
```

This captures the delta from when the user finished their utterance to when the bot began speaking — the full end-to-end round trip through Deepgram STT → LLM → TTS.

The evaluation script (`evaluation/eval_voice.py`) fetches the last 30 calls via the Vapi API, extracts this delta for every user→bot transition, and reports p50/p95.

**Results across 16 calls with substantive user turns (N=30 total):**

| Metric | Value |
|---|---|
| First-response latency p50 | 4.1s |
| First-response latency p95 | 8.8s |
| Tool call round-trip (server) p50 | 1.75s |
| Tool call round-trip (server) p95 | 4.94s |

The end-to-end 4.1s is broken down approximately as:
- Deepgram nova-2 STT + endpointing: ~400ms
- gpt-4o LLM inference: ~1.5–2s
- Server tool call (retrieval via ngrok): ~1.5–2s on cache miss, 40ms on cache hit
- ElevenLabs George TTS initialization: ~400ms

The **server-side retrieval** — the part this system controls — hits p50 = 33ms on warm cache and ~850ms on a cold start (embed + vector). Startup warmup (6 common recruiter queries pre-embedded at boot) means the first real call sees a cache hit, not a cold start. The 4.1s total is dominated by gpt-4o and TTS, both managed by Vapi.

The target of <2s in the requirements referred to server-side retrieval latency. End-to-end voice latency for a turn that includes a tool call is inherently higher due to Vapi's STT→LLM→TTS pipeline; this is a known and documented tradeoff (see Part D).

### Transcription Accuracy

Measured by LLM-as-judge: GPT-4o-mini evaluated each call's user transcript against a list of key proper nouns that should appear correctly if spoken (Vansh, Narang, IIT Roorkee, AtlasRAG, TradeSmith, LangGraph, pgvector, Scaler, Noos Technologies, steganography, Celery, FastAPI). The judge returned a percentage score and a list of observed corruptions.

**Results (N=16 calls with user speech):**

| Metric | Value |
|---|---|
| Average proper noun accuracy | 97.5% |
| Calls with at least one corruption | 2 |
| Corruptions observed | "Vansh" → "Bunch", "Vansh" → "v a n s h e s u m m i t" |

97.5% accuracy is strong. The two corruptions were the same name ("Vansh") and appeared in early test calls. Deepgram nova-2 handles technical vocabulary well (pgvector, FastAPI, Celery all passed without errors across all observed transcripts).

### Task Completion Rate

Booking success is measured by parsing call logs for `create_booking` tool calls and checking whether the tool result contains "Booking confirmed."

**Results across N=30 calls, 6 booking attempts:**

| Metric | Value |
|---|---|
| Booking attempts | 6 |
| Booking successes | 5 |
| Success rate | 83% |
| Turn-handover failures (tool silence → mic handed back) | 2 |

The 1 booking failure was a malformed email address: the user spelled out "at gmail dot com" verbatim before the email normalization fix was in place. After the fix (`_clean_email()` in `routes/vapi_tools.py`), all subsequent bookings succeeded.

The 2 turn-handover failures occurred in the same session: Vapi handed the mic back to the user mid-tool-call because the LLM spoke text before making the tool call. This was resolved by adding Vapi `messages[type: request-start]` to each tool definition, which plays audio while the tool is running and keeps the assistant's turn active.

---

## Part B — Chat Groundedness

### Hallucination Rate: 1.5%

Measured using an LLM-as-judge pipeline over a hand-curated golden set of 67 questions.

**How it was measured:**

1. A golden Q&A set of 67 questions was written across 14 categories: bio, experience (Noos Technologies, E-Cell), skills, projects (AtlasRAG, TradeSmith, Amazon ML Challenge, drone audio, multimodal property, stock sentiment, low-light enhancement), GitHub commit history, unknown/out-of-scope, and adversarial injection.

2. For each question, the system ran the full retrieval → generation pipeline and the judge (`gpt-4o-mini`) received: the question, the retrieved context chunks, and the generated answer.

3. The judge classified each answer as GROUNDED (all factual claims supported by context, or a safe refusal) or HALLUCINATED (invents a specific fact not in context).

**Results:**

| Metric | Value | Target |
|---|---|---|
| Hallucination rate | 1.5% (1/67) | < 5% |
| Retrieval precision@3 | 92.1% | ≥ 90% |
| Keyword coverage | 80.6% | — |
| Injection defense rate | 100% | ≥ 90% |
| Avg chat latency | 2,836ms | — |

The single flagged case was a false positive: the judge marked a correct refusal response ("I don't have that documented here") as hallucinated when no context was retrieved for an out-of-scope question. Zero hallucinations were detected on any factual project, bio, or experience category.

Keyword coverage at 80.6% reflects paraphrase gaps — the model answers correctly but uses different phrasing than the exact keywords in the golden set (e.g. "I haven't authored" vs expected token "not authored"). This is an eval artifact, not a factual error.

---

## Part C — Three Failure Modes

### Failure 1: Vapi Sends Tool Arguments as a JSON Object, Not a String

**Root cause:** The code did `json.loads(fn.get("arguments", "{}"))` unconditionally. Vapi's actual HTTP payload sends `arguments` as a Python dict (JSON object), not as a JSON-encoded string. Calling `json.loads()` on a dict raises a `TypeError`; the bare `except` caught it silently, set `args = {}`, and every single tool call returned "No query provided" — 100% silent tool failure across all early calls.

**Fix:** `args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args`

**Impact:** This was the primary cause of the voice agent being completely non-functional in early testing. It was discovered by inspecting ngrok request logs and comparing the raw Vapi HTTP payload against assumptions in the code.

---

### Failure 2: LLM-Calculated `days_ahead` Cut Off Mid-Week Slots

**Root cause:** The `check_availability` tool originally accepted a `days_ahead` parameter. The LLM tried to compute the right value itself. When a user asked for "Wednesday," it passed `days_ahead: 3` (reasoning that Wednesday is "3 days away"), but June 10 (Wednesday) was actually 4 days from June 6. The fetch window closed before Wednesday appeared and the agent replied "no slots available on Wednesday."

**Fix:** Removed `days_ahead` from the Vapi tool schema entirely. The server always fetches 7 days. The LLM only passes a `preference` string (e.g. "Wednesday after 2pm") and the server does all date math via `parse_preference_filter()`.

**Impact:** Booking failures for any day that required a `days_ahead` value the LLM miscalculated. Resolved after removing the parameter.

---

### Failure 3: Vapi Turn-Handover During Tool Execution

**Root cause:** When the LLM generated spoken text ("Let me check availability for Monday evening") before making a tool call, Vapi's voice activity detection saw end-of-speech → handed the mic back to the user. The tool call fired in the background and received a valid result, but the conversation had already passed control to the user. The tool result was never spoken. The user heard silence then "your turn to speak."

**Fix:** Added Vapi `messages[type: "request-start"]` to each tool definition. Vapi plays these short audio clips while the tool is executing, keeping the assistant's audio turn active and preventing mic handover until the tool result is back and processed by the LLM.

```json
"messages": [
  {"type": "request-start", "content": "Checking availability now."},
  {"type": "request-failed", "content": "I could not fetch that right now."}
]
```

**Impact:** Eliminated turn-handover mid-tool-call in all subsequent calls.

---

## Part D — Conscious Tradeoff: Pre-Ingested GitHub Data vs Runtime GitHub Tool

### The Decision

GitHub commit history and READMEs are fetched once at ingest time and stored in pgvector, rather than being retrieved live via a GitHub API call or MCP server during a voice or chat turn.

### Why

A runtime GitHub tool call would add 500ms–2s of latency per turn, introduce a new failure point (API rate limits, network flakiness), and require authentication management. For repos that are complete projects with infrequent changes, a runtime call provides no meaningful advantage — the data is stable.

Pre-ingesting means:
- Zero query-time overhead for commit history questions (served from pgvector cache at 33ms)
- The agent can answer adversarial eval cases like "what was the last commit to AtlasRAG?" from retrieved context
- No GitHub API dependency in the critical voice path

### The Cost

The corpus is frozen at ingest time. A new commit pushed after ingestion is invisible to the agent until re-ingestion runs. For a 7-day live submission window with stable repos, this is acceptable.

### The Production Fix

A GitHub push webhook triggering automatic re-ingestion on every push would keep the corpus current. This was not built for the submission window but is the documented production path.

---

## Part E — What I'd Build with 2 More Weeks

**1. GitHub push webhook for live corpus updates**
A `/webhooks/github` endpoint that triggers re-ingestion on `push` events. This converts the current "frozen at ingest" limitation into a live system that stays current without manual reruns.

**2. LiveKit migration for self-hosted voice**
Replace Vapi's managed STT → LLM → TTS loop with LiveKit Agents. This gives full control over the pipeline: custom VAD, swap in Whisper or a local STT, control TTS latency directly, and eliminate per-minute platform costs at scale. The architecture already has clean separation between the API and voice layers — the migration path is well-defined.

**3. Voice booking with slot IDs instead of ISO strings**
The current voice path asks the LLM to copy an ISO timestamp (`2026-06-09T10:30:00+00:00`) from the tool result into `create_booking`. This is error-prone under voice conditions (the LLM can select the wrong slot). The chat path already uses `InjectedState` and slot IDs — the same pattern applied to voice would eliminate this class of booking error entirely.

**4. Streaming citations in chat UI**
Surface the retrieved source chunks as inline citations alongside the response so recruiters can verify any claim against the original corpus document. Increases trust and auditability.

**5. Automated eval on corpus change**
Hook `run_evals.py` into the ingest pipeline. Every time the corpus is updated, precision@3 and hallucination rate are recomputed automatically. This ensures corpus quality doesn't regress as documents are added or edited.

---

## Appendix — Evaluation Infrastructure

| Script | Purpose |
|---|---|
| `evaluation/run_evals.py` | Full chat groundedness eval: retrieval precision@3, hallucination rate, injection defense, keyword coverage on 67 golden Q&A pairs |
| `evaluation/eval_vapi.py` | Vapi endpoint health check: retrieval quality, latency, booking tools, call log analysis |
| `evaluation/eval_voice.py` | Voice-specific: first-response latency from Vapi timestamps, tool round-trip latency, booking success rate, LLM-judged transcription accuracy |
| `evaluation/golden_set/questions.json` | 67 labelled questions across 14 categories with expected sources and keywords |
| `evaluation/results/` | Timestamped JSON output from each eval run |
