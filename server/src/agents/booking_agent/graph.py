from __future__ import annotations

import json
import re
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import structlog

from src.config.prompts import BOOKING_SYSTEM_PROMPT, AGENT_ERROR_FALLBACK
from src.config.settings import settings
from src.agents.booking_agent.tools import (
    get_availability_tool,
    select_slot_tool,
    create_booking_tool,
)

log = structlog.get_logger(__name__)

_BOOKING_INTENT_RE = re.compile(
    r"\b(schedule|book|meeting|call|interview|available|slot|appointment|set up)\b",
    re.IGNORECASE,
)


def is_booking_intent(text: str) -> bool:
    return bool(_BOOKING_INTENT_RE.search(text))


# ── State ──────────────────────────────────────────────────────────────────────

class BookingState(TypedDict):
    messages:         Annotated[list, add_messages]
    available_slots:  dict              # {slot_id: {start, end}} — populated by get_availability_tool
    selected_slot_id: Optional[str]     # set by select_slot_tool
    customer_name:    Optional[str]
    customer_email:   Optional[str]


# ── LLM + tools ───────────────────────────────────────────────────────────────

_TOOLS = [get_availability_tool, select_slot_tool, create_booking_tool]

_llm = ChatOpenAI(
    model=settings.openai_chat_model_smart,
    temperature=0,
    api_key=settings.openai_api_key,
)
_llm_with_tools = _llm.bind_tools(_TOOLS)


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def agent_node(state: BookingState) -> dict:
    response = await _llm_with_tools.ainvoke(
        [SystemMessage(content=BOOKING_SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}


_raw_tool_node = ToolNode(_TOOLS)


async def tools_node(state: BookingState) -> dict:
    """
    Wraps ToolNode to extract structured state from tool results.

    InjectedState tools (select_slot_tool, create_booking_tool) read
    available_slots / selected_slot_id directly from state.
    This node promotes their outputs back into top-level state fields.
    """
    result = await _raw_tool_node.ainvoke(state)
    updates: dict = {"messages": result["messages"]}

    for msg in result["messages"]:
        tool_name = getattr(msg, "name", None)
        if not tool_name:
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if tool_name == "get_availability_tool" and data.get("status") == "ok":
            updates["available_slots"] = data["slots"]
            log.info("booking.slots_fetched", count=len(data["slots"]))

        elif tool_name == "select_slot_tool" and data.get("status") == "ok":
            updates["selected_slot_id"] = data["slot_id"]
            log.info("booking.slot_selected", slot_id=data["slot_id"])

        elif tool_name == "create_booking_tool" and data.get("status") == "ok":
            log.info("booking.created",
                     booking_id=data.get("booking_id"),
                     time=data.get("confirmed_time"))

    return updates


# ── Router ─────────────────────────────────────────────────────────────────────

def should_continue(state: BookingState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_booking_graph(checkpointer) -> StateGraph:
    g = StateGraph(BookingState)

    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")

    return g.compile(checkpointer=checkpointer)


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_booking_turn(
    graph,
    user_message: str,
    session_id: str,
    is_first_turn: bool = False,   # kept for API compat; agent handles state via checkpointer
) -> str:
    config = {"configurable": {"thread_id": f"booking:{session_id}"}}
    result = await graph.ainvoke(
        {"messages": [("user", user_message)]},
        config=config,
    )
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and msg.content:
            return msg.content
    return AGENT_ERROR_FALLBACK
