from typing import Any, Dict, List

try:
    # Placeholder: adapt to ADK tool interface when wiring real search
    from google.adk.tools import Tool  # type: ignore
except Exception:  # pragma: no cover
    Tool = object  # type: ignore


def tool_web_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Stub web search tool.
    Returns a list of {title, url, snippet} dicts.
    """
    print("[WARN] tool_web_search is a stub. Provide real search integration.")
    return [
        {"title": f"Result {i+1} for {query}", "url": f"https://example.com/{i+1}", "snippet": "..."}
        for i in range(k)
    ]
