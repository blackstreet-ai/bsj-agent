"""
MCP utilities for building ADK MCPToolset instances from environment variables.

- Normalizes URLs (handles trailing slashes and /sse specifics for SSE endpoints)
- Returns a list of toolsets to attach to LlmAgent.tools
- Extensively commented for ADK learning purposes
"""
from __future__ import annotations

import os
from typing import Any, List
import os.path

try:
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        SseConnectionParams,
        StreamableHTTPConnectionParams,
    )
except Exception:  # pragma: no cover
    MCPToolset = None  # type: ignore
    SseConnectionParams = None  # type: ignore
    StreamableHTTPConnectionParams = None  # type: ignore


def _normalize_base(url: str) -> str:
    url = url.strip()
    return url.rstrip('/')


def _ensure_sse(url: str) -> str:
    base = _normalize_base(url)
    return base if base.endswith('/sse') else f"{base}/sse"


def build_researcher_toolsets(debug: bool = False) -> List[Any]:
    """
    Create MCP toolsets for the researcher agent.

    Env vars:
      - TAVILY_MCP_URL (+ TAVILY_API_KEY)
      - FIRECRAWL_MCP_URL (+ FIRECRAWL_API_KEY) [SSE]
    """
    toolsets: List[Any] = []

    if MCPToolset is None:
        if debug:
            print("[MCP] ADK MCPToolset not available; skipping tool setup")
        return toolsets

    tavily_url = _normalize_base(os.path.expandvars(os.getenv("TAVILY_MCP_URL", "")))
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    firecrawl_url = os.path.expandvars(os.getenv("FIRECRAWL_MCP_URL", "").strip())
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "").strip()

    # Log detection
    if debug:
        print(
            f"[MCP] Env detected: tavily_url={'set' if tavily_url else 'unset'}, "
            f"firecrawl_url={'set' if firecrawl_url else 'unset'}"
        )

    # Tavily (HTTP streamable)
    if tavily_url and StreamableHTTPConnectionParams is not None:
        # If Tavily expects key via query parameter and it's missing/empty, append or fill it.
        if tavily_key and 'tavily' in tavily_url:
            if 'tavilyApiKey=' not in tavily_url:
                sep = '&' if '?' in tavily_url else '?'
                tavily_url = f"{tavily_url}{sep}tavilyApiKey={tavily_key}"
            elif tavily_url.endswith('tavilyApiKey='):
                tavily_url = f"{tavily_url}{tavily_key}"
        conn = StreamableHTTPConnectionParams(
            url=tavily_url,
            headers=({"Authorization": f"Bearer {tavily_key}"} if tavily_key else None),
        )
        toolsets.append(MCPToolset(connection_params=conn))
        if debug:
            print(f"[MCP] Tavily toolset configured url={tavily_url}")

    # Firecrawl (SSE)
    if firecrawl_url and SseConnectionParams is not None:
        url = _ensure_sse(firecrawl_url)
        conn = SseConnectionParams(
            url=url,
            headers=({"Authorization": f"Bearer {firecrawl_key}"} if firecrawl_key else None),
        )
        toolsets.append(MCPToolset(connection_params=conn))
        if debug:
            print(f"[MCP] Firecrawl toolset configured url={url}")

    if debug:
        print(f"[MCP] researcher_tools count: {len(toolsets)}")
        for i, t in enumerate(toolsets):
            try:
                conn = getattr(t, "connection_params", None)
                url = getattr(conn, "url", None) if conn else None
                print(f"[MCP] toolset[{i}] class={t.__class__.__name__} url={url}")
            except Exception:
                pass

    return toolsets
