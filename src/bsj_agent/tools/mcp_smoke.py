"""
ADK MCP smoke test for Tavily (search) and Firecrawl (fetch).

- Lists available tools from each MCP server
- Executes a sample search and fetch to verify connectivity and basic responses

Usage:
  PYTHONPATH=src python3 -m bsj_agent.tools.mcp_smoke "AI in African fintech"

Env vars used:
  TAVILY_MCP_URL, TAVILY_API_KEY (optional)
  FIRECRAWL_MCP_URL, FIRECRAWL_API_KEY (optional)

This is a developer utility; not part of the production pipeline.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

try:
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        SseConnectionParams,
        StreamableHTTPConnectionParams,
    )
except Exception as e:  # pragma: no cover - smoke tool guard
    raise SystemExit(f"ADK MCP imports failed: {e}. Ensure google-adk is installed.")


async def _list_and_sample(toolset: MCPToolset, label: str, query: str) -> None:
    print(f"\n== {label} ==")
    # List tools
    tools = await toolset.get_tools()
    names = [getattr(t, "name", t.__class__.__name__) for t in tools]
    print(f"Tools ({len(names)}): {names}")

    # Heuristic: pick a search-like tool, then a fetch/crawl-like tool
    search_tool = next((t for t in tools if any(k in t.name.lower() for k in ("search", "web"))), None)
    fetch_tool = next((t for t in tools if any(k in t.name.lower() for k in ("crawl", "fetch", "scrape"))), None)

    if search_tool:
        try:
            print(f"[CALL] {search_tool.name} -> query='{query}'")
            resp: Any = await search_tool.run_async(args={"query": query})  # type: ignore[attr-defined]
            preview = json.dumps(resp, ensure_ascii=False) if not isinstance(resp, str) else resp
            print(f"[RESP] {search_tool.name}: {preview[:600]}{'...<truncated>' if len(preview) > 600 else ''}")
        except Exception as e:
            print(f"[ERROR] calling {search_tool.name}: {e}")
    else:
        print("No obvious search tool found.")

    if fetch_tool:
        # Try to find a URL candidate in previous response (very light heuristic)
        url = None
        try:
            if 'preview' in locals():
                # try crude extraction
                import re
                m = re.search(r"https?://[\w\-\./%?#=&:]+", preview)
                if m:
                    url = m.group(0)
        except Exception:
            pass
        if not url:
            # fallback to a neutral URL
            url = "https://example.com"
        try:
            print(f"[CALL] {fetch_tool.name} -> url='{url}'")
            args = {k: v for k, v in (('url', url), ('q', url))}  # some servers accept 'q'
            resp2: Any = await fetch_tool.run_async(args=args)  # type: ignore[attr-defined]
            text = json.dumps(resp2, ensure_ascii=False) if not isinstance(resp2, str) else resp2
            print(f"[RESP] {fetch_tool.name}: {text[:600]}{'...<truncated>' if len(text) > 600 else ''}")
        except Exception as e:
            print(f"[ERROR] calling {fetch_tool.name}: {e}")
    else:
        print("No obvious fetch/crawl tool found.")


async def main(topic: str) -> None:
    # Tavily
    tavily_url = os.getenv("TAVILY_MCP_URL", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    # Firecrawl
    firecrawl_url = os.getenv("FIRECRAWL_MCP_URL", "").strip()
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "").strip()

    tasks = []

    if tavily_url:
        conn = StreamableHTTPConnectionParams(
            url=tavily_url,
            headers=(
                {"Authorization": f"Bearer {tavily_key}"} if tavily_key else None
            ),
        )
        tasks.append(_list_and_sample(MCPToolset(connection_params=conn), "TAVILY", topic))
    else:
        print("[SKIP] Tavily MCP not configured (TAVILY_MCP_URL unset)")

    if firecrawl_url:
        conn = SseConnectionParams(
            url=firecrawl_url,
            headers=(
                {"Authorization": f"Bearer {firecrawl_key}"} if firecrawl_key else None
            ),
        )
        tasks.append(_list_and_sample(MCPToolset(connection_params=conn), "FIRECRAWL", topic))
    else:
        print("[SKIP] Firecrawl MCP not configured (FIRECRAWL_MCP_URL unset)")

    if not tasks:
        print("No MCP servers configured. Set env vars and retry.")
        return

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: PYTHONPATH=src python3 -m bsj_agent.tools.mcp_smoke \"<topic/query>\"")
        raise SystemExit(2)
    asyncio.run(main(sys.argv[1]))
