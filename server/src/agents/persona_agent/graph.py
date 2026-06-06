from __future__ import annotations
from typing import TypedDict, AsyncIterator

import asyncpg
import redis.asyncio as aioredis
import structlog
from langgraph.graph import StateGraph, END, START

from src.config import PERSONA_SYSTEM_PROMPT, GROUNDING_FALLBACK, INJECTION_DEFLECT
from src.models.retrieval import RetrievedChunk
from src.rag.retrieval import retrieve as _retrieve, BM25Index
from src.services import llm_client
from src.agents.persona_agent.guardrails import detect_injection, check_grounding, sanitize_input

log = structlog.get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class PersonaState(TypedDict):
    raw_query: str
    mode:      str
    history:   list
    # internal
    query:     str
    injected:  bool
    chunks:    list
    grounded:  bool
    context:   str
    # output
    reply:     str


# ── Shared helper ─────────────────────────────────────────────────────────────

def _build_messages(query: str, context: str, history: list) -> list[dict]:
    """Build the messages list for the LLM from state fields."""
    history_lines = []
    for m in (history or [])[-10:]:
        role = "Recruiter" if m.get("role") == "user" else "Vansh"
        history_lines.append(f"{role}: {m.get('content', '')}")

    system = PERSONA_SYSTEM_PROMPT.format(
        context=context or "",
        history="\n".join(history_lines) or "None",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": query},
    ]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def guardrail_node(state: PersonaState) -> dict:
    query = sanitize_input(state["raw_query"])
    injected = detect_injection(query)
    if injected:
        log.warning("persona.injection_detected", query=query[:80])
    return {"query": query, "injected": injected}


def inject_deflect_node(state: PersonaState) -> dict:
    return {"reply": INJECTION_DEFLECT}


def grounding_node(state: PersonaState) -> dict:
    raw_chunks = state.get("chunks") or []
    has_history = bool(state.get("history"))

    # LangGraph serialises TypedDict to plain dicts — restore to RetrievedChunk
    if raw_chunks and isinstance(raw_chunks[0], dict):
        chunks = [RetrievedChunk(**c) for c in raw_chunks]
    else:
        chunks = raw_chunks

    grounded = check_grounding(chunks)

    context = ""
    if grounded:
        parts = [f"[{i}] Source: {c.source}\n{c.content}" for i, c in enumerate(chunks, 1)]
        context = "\n\n---\n\n".join(parts)

    # Allow follow-up questions to reach generate_node even if retrieval score
    # is below threshold — history provides enough grounding for conversational turns.
    return {"grounded": grounded or has_history, "context": context}


def fallback_node(state: PersonaState) -> dict:
    log.info("persona.grounding_fallback", query=state.get("query", "")[:80])
    return {"reply": GROUNDING_FALLBACK}


async def generate_node(state: PersonaState) -> dict:
    messages = _build_messages(state["query"], state.get("context", ""), state.get("history"))
    reply = await llm_client.chat_complete(messages, mode=state["mode"], stream=False)
    log.info("persona.generate", mode=state["mode"], chunks=len(state.get("chunks") or []))
    return {"reply": reply}


# ── Routing ───────────────────────────────────────────────────────────────────

def _after_guardrail(state: PersonaState) -> str:
    return "inject_deflect" if state["injected"] else "retrieve"


def _after_grounding(state: PersonaState) -> str:
    return "generate" if state["grounded"] else "fallback"


# ── Build ─────────────────────────────────────────────────────────────────────

def build_persona_graph(pool: asyncpg.Pool, bm25_index: BM25Index, redis: aioredis.Redis):
    """Services captured in closure; compiled graph stored on app.state."""

    async def retrieve_node(state: PersonaState) -> dict:
        chunks, latency = await _retrieve(
            query=state["query"],
            mode=state["mode"],
            pool=pool,
            bm25_index=bm25_index,
            redis=redis,
        )
        return {"chunks": [
            {"id": c.id, "content": c.content, "source": c.source,
             "score": c.score, "metadata": c.metadata}
            for c in chunks
        ]}

    g = StateGraph(PersonaState)
    g.add_node("guardrail",      guardrail_node)
    g.add_node("retrieve",       retrieve_node)
    g.add_node("grounding",      grounding_node)
    g.add_node("generate",       generate_node)
    g.add_node("inject_deflect", inject_deflect_node)
    g.add_node("fallback",       fallback_node)

    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", _after_guardrail,
        {"inject_deflect": "inject_deflect", "retrieve": "retrieve"})
    g.add_edge("retrieve", "grounding")
    g.add_conditional_edges("grounding", _after_grounding,
        {"generate": "generate", "fallback": "fallback"})
    for node in ("inject_deflect", "fallback", "generate"):
        g.add_edge(node, END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_EMPTY_STATE = {
    "raw_query": "", "mode": "chat", "history": [],
    "query": "", "injected": False, "chunks": [],
    "grounded": False, "context": "", "reply": "",
}


async def run_persona_turn(graph, query: str, mode: str, history: list) -> str:
    result = await graph.ainvoke({**_EMPTY_STATE, "raw_query": query, "mode": mode, "history": history})
    return result.get("reply", "")


async def stream_persona_turn(graph, query: str, mode: str, history: list) -> AsyncIterator[str]:
    """
    Runs the graph (non-streaming) for guardrail + retrieval + grounding,
    then streams the LLM response token-by-token.
    Injection deflections and grounding fallbacks are yielded as a single chunk.
    """
    result = await graph.ainvoke({**_EMPTY_STATE, "raw_query": query, "mode": mode, "history": history})

    # Deflection or fallback — send whole reply immediately
    if result.get("injected") or not result.get("grounded"):
        yield result.get("reply", "")
        return

    # Grounded — stream tokens
    messages = _build_messages(
        result.get("query", query),
        result.get("context", ""),
        history,
    )
    stream = await llm_client.chat_complete(messages, mode=mode, stream=True)
    async for token in stream:
        yield token
