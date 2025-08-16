"""
bsj_agent.agents.researcher

Factory to create the BSJ Researcher LlmAgent using ADK. This agent is responsible for
SEARCH then FETCH via MCP tools (Tavily + Firecrawl) and returns strict JSON to session.state.

ADK concepts are explained in-line for learning: LlmAgent with tools, output_key, and
GenerateContentConfig to nudge raw JSON outputs.
"""
from __future__ import annotations

from typing import Any, List

try:
    from google.adk.agents import LlmAgent
    from google.genai import types
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore
    types = None  # type: ignore

from ...tools.mcp_utils import build_researcher_toolsets


def create_agent(*, debug: bool = False) -> Any:
    """
    Build and return the BSJ Researcher as an ADK LlmAgent.

    - Uses MCP toolsets built from environment (Tavily/Firecrawl)
    - Enforces JSON output via output_key + response_mime_type
    - Includes before/after tool callbacks when debug=True
    """
    if LlmAgent is None:
        raise RuntimeError("ADK not available. Install google-adk and retry.")

    tools: List[Any] = build_researcher_toolsets(debug=debug)

    def _debug_before_tool(tool: Any = None, args: dict | None = None, **kwargs):  # minimal, local to agent
        if not debug:
            return
        try:
            name = getattr(tool, "name", getattr(tool, "__class__", type("_", (), {}))).__class__.__name__ if tool else "<unknown>"
            print(f"[TOOL> before] {name} args={args}")
        except Exception:
            pass

    def _debug_after_tool(tool: Any = None, args: dict | None = None, _ctx: Any = None, tool_response: dict | None = None, **kwargs):
        if not debug:
            return
        try:
            name = getattr(tool, "name", getattr(tool, "__class__", type("_", (), {}))).__class__.__name__ if tool else "<unknown>"
            preview = str(tool_response)
            if len(preview) > 500:
                preview = preview[:500] + "...<truncated>"
            print(f"[TOOL< after] {name} resp={preview}")
        except Exception:
            pass

    instruction = (
        "You are the BSJ researcher. Read session.state.topic and stay STRICTLY on that topic.\n"
        "Use tools FIRST: perform SEARCH using a tool whose name contains 'search' or 'web'; then FETCH full content using a tool whose name contains 'crawl' or 'fetch'. (Examples: Tavily via MCP, Firecrawl via MCP.)\n"
        "Do not answer until you have used at least one search tool and one fetch/crawl tool. If tools are unavailable, return {\"research\": {\"topics\": [], \"key_stats\": [], \"citations\": []}, \"error\": \"TOOLS_UNAVAILABLE\"}.\n"
        "Style: Plan -> Tool calls -> Synthesis -> JSON only.\n"
        "Synthesize facts and key stats ONLY from fetched content.\n"
        "Respond with ONLY valid JSON as {\"research\": {\"topics\": [], \"key_stats\": [], \"citations\": [{\"title\": \"...\", \"url\": \"...\"}]}}.\n"
        "Provide at least 3 citations with accurate titles and URLs. Avoid unrelated domains; remain on session.state.topic."
    )

    return LlmAgent(
        name="bsj_researcher",
        model="gemini-2.5-pro",
        description="Research subtopics, key stats, and citations for the BSJ topic.",
        instruction=instruction,
        tools=tools,
        output_key="research",
        before_tool_callback=_debug_before_tool if debug else None,
        after_tool_callback=_debug_after_tool if debug else None,
        # Do NOT set response_mime_type here: Gemini function calling with tools
        # rejects application/json mime type. We keep strict JSON by instruction
        # and output_key; downstream agents expect structured state.
    )
