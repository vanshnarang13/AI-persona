from typing import Any


def parse_vapi_tool_call(body: dict) -> tuple[str, dict]:
    """Extract tool name and args from a Vapi tool-call webhook body."""
    tool_call = body.get("message", {}).get("toolCalls", [{}])[0]
    name: str = tool_call.get("function", {}).get("name", "")
    args: dict = tool_call.get("function", {}).get("arguments", {})
    return name, args


def build_tool_result(result: Any) -> dict:
    """Wrap a result in Vapi's expected tool-result envelope."""
    return {"results": [{"result": str(result)}]}
