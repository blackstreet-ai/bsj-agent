"""
bsj_agent_v2.agents.researcher

Factory for the BSJ researcher agent using Google ADK LlmAgent.
- Uses Tavily MCP tools (HTTP streamable) built from env via tools.mcp_utils
- Avoids response_mime_type with tools to prevent INVALID_ARGUMENT
- Heavily commented for ADK learning
"""
from __future__ import annotations

from typing import Any, List

try:
    from google.adk.agents import LlmAgent
    from google.genai import types
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore
    types = None  # type: ignore

from ...tools.mcp_utils import build_tavily_toolset


def create_agent(*, debug: bool = False) -> Any:
    """Create the researcher LlmAgent.

    Returns:
      An ADK `LlmAgent` configured to research a topic using MCP tools.
    """
    if LlmAgent is None:
        raise RuntimeError("ADK not available. Install google-adk and retry.")

    tools: List[Any] = build_tavily_toolset(debug=debug)

    def _before(tool: Any = None, args: dict | None = None, **kwargs):
        if not debug:
            return
        try:
            name = getattr(tool, "name", getattr(tool, "__class__", type("_", (), {}))).__class__.__name__ if tool else "<unknown>"
            print(f"[v2 TOOL> before] {name} args={args}")
        except Exception:
            pass

    def _after(tool: Any = None, args: dict | None = None, _ctx: Any = None, tool_response: dict | None = None, **kwargs):
        if not debug:
            return
        try:
            name = getattr(tool, "name", getattr(tool, "__class__", type("_", (), {}))).__class__.__name__ if tool else "<unknown>"
            preview = str(tool_response)
            if preview and len(preview) > 400:
                preview = preview[:400] + "...<trunc>"
            print(f"[v2 TOOL< after] {name} resp={preview}")
        except Exception:
            pass

    instruction = (
        "You are the BSJ researcher. Read session.state.topic and stay STRICTLY on that topic.\n"
        "Use tools FIRST: perform SEARCH using a tool whose name contains 'search' or 'web'; then FETCH full content using a tool whose name contains 'crawl' or 'fetch'.\n"
        "Do not answer until you have used at least one search tool and one fetch/crawl tool. If tools are unavailable, output minimal empty schema.\n"
        "Output JSON only with keys: topics (array of strings), key_stats (array of {label, value, source}), citations (array of {title,url})."
    )

    return LlmAgent(
        name="bsj_researcher",
        model="gemini-2.5-pro",
        description="Research subtopics, key stats, and citations.",
        instruction=instruction,
        tools=tools,
        output_key="research",
        before_tool_callback=_before,
        after_tool_callback=_after,
        # Do NOT set response_mime_type here due to tool-calling constraints.
        generate_content_config=None if types is not None else None,
    )
