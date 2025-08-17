"""
bsj_agent_v2.agents.researcher

Factory for the BSJ researcher agent using Google ADK LlmAgent.
- Uses Tavily MCP tools (HTTP streamable) built from env via tools.mcp_utils
- Avoids response_mime_type with tools to prevent INVALID_ARGUMENT
- Heavily commented for ADK learning
"""
from __future__ import annotations

from typing import Any, List
import json

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
        "Do not answer until you have used at least one search tool and one fetch/crawl tool. If tools are unavailable, return minimal findings.\n"
        "Produce: topics (bulleted subtopics), key_stats (label – value – source), and citations (title with URL)."
    )

    def _after_model(callback_context, llm_response):
        """Parse JSON when present, update state, and keep ADK-compatible return.

        Important: ADK postprocessors expect an LLM response object with `.content`.
        Returning a raw string will break downstream processors (e.g., NL planning).

        Behavior:
        - Parse the model text as JSON (if possible) and update state['research'].
        - Build a Markdown summary and store it in state['research_markdown'] for UI.
        - Return the original `llm_response` object (not a string).
        """
        try:
            text = ""
            if getattr(llm_response, "content", None) and getattr(llm_response.content, "parts", None):
                text = "".join([(p.text or "") for p in llm_response.content.parts])
            # Try parse JSON if the model produced it
            parsed = None
            if text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None

            if parsed is not None:
                try:
                    callback_context.state["research"] = parsed
                except Exception:
                    pass

                topics = parsed.get("topics") if isinstance(parsed, dict) else None
                key_stats = parsed.get("key_stats") if isinstance(parsed, dict) else None
                citations = parsed.get("citations") if isinstance(parsed, dict) else None
                lines = ["# Research Findings"]
                if topics:
                    lines.append("\n**Topics**:")
                    for t in (topics if isinstance(topics, list) else [topics]):
                        lines.append(f"- {t}")
                if key_stats:
                    lines.append("\n**Key Stats**:")
                    for s in (key_stats if isinstance(key_stats, list) else [key_stats]):
                        if isinstance(s, dict):
                            label = s.get("label", "")
                            value = s.get("value", "")
                            source = s.get("source", "")
                            lines.append(f"- {label}: {value} — {source}")
                        else:
                            lines.append(f"- {s}")
                if citations:
                    lines.append("\n**Citations**:")
                    for c in (citations if isinstance(citations, list) else [citations]):
                        if isinstance(c, dict):
                            title = c.get("title", "")
                            url = c.get("url", "")
                            lines.append(f"- {title} — {url}")
                        else:
                            lines.append(f"- {c}")
                try:
                    callback_context.state["research_markdown"] = "\n".join(lines)
                except Exception:
                    pass
                return llm_response

            # Fallback: no JSON, render raw text as Markdown
            if text:
                try:
                    callback_context.state["research_markdown"] = "# Research Findings\n\n" + text
                except Exception:
                    pass
                return llm_response
            return llm_response
        except Exception:
            return llm_response

    return LlmAgent(
        name="bsj_researcher",
        model="gemini-2.5-pro",
        description="Research subtopics, key stats, and citations.",
        instruction=instruction,
        tools=tools,
        output_key="research",
        before_tool_callback=_before,
        after_tool_callback=_after,
        after_model_callback=_after_model,
        # Do NOT set response_mime_type here due to tool-calling constraints.
        generate_content_config=None if types is not None else None,
    )
