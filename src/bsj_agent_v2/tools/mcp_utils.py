"""
bsj_agent_v2.tools.mcp_utils

Tavily-only MCP Toolset builder for the Researcher agent (Google ADK).
- Expands env vars embedded in the URL (e.g., ${TAVILY_API_KEY}).
- Ensures the tavilyApiKey query parameter is present (fills if empty).
- Returns a list with a single MCPToolset (or empty if misconfigured).

This keeps v2 minimal and focused while remaining extensible.
"""
from __future__ import annotations

import os
import os.path
from typing import Any, List
from typing import Dict, Mapping, MutableMapping
import json

try:
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        StreamableHTTPConnectionParams,
    )
    # Auth types from ADK for tool calling auth config
    from google.adk.auth.auth_credential import (
        AuthCredential,
        AuthCredentialTypes,
        HttpAuth,
        HttpCredentials,
    )
    from fastapi.openapi.models import HTTPBase
except Exception:  # pragma: no cover
    MCPToolset = None  # type: ignore
    StreamableHTTPConnectionParams = None  # type: ignore
    AuthCredential = None  # type: ignore
    AuthCredentialTypes = None  # type: ignore
    HttpAuth = None  # type: ignore
    HttpCredentials = None  # type: ignore
    HTTPBase = None  # type: ignore


def _normalize_base(url: str) -> str:
    url = url.strip()
    return url.rstrip('/')


def build_tavily_toolset(debug: bool = False) -> List[Any]:
    """Create a Tavily MCP toolset from env vars.

    Env:
      - TAVILY_MCP_URL (base URL; may include "?tavilyApiKey=${TAVILY_API_KEY}")
      - TAVILY_API_KEY
    """
    toolsets: List[Any] = []

    if MCPToolset is None or StreamableHTTPConnectionParams is None:
        if debug:
            print("[MCP v2] ADK MCPToolset not available; skipping Tavily setup")
        return toolsets

    tavily_url = _normalize_base(os.path.expandvars(os.getenv("TAVILY_MCP_URL", "")))
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if debug:
        print(f"[MCP v2] Env detected: tavily_url={'set' if tavily_url else 'unset'}")

    if not tavily_url:
        return toolsets

    # Fill the query parameter if it exists empty or not present.
    if tavily_key and "tavily" in tavily_url:
        if "tavilyApiKey=" not in tavily_url:
            sep = "&" if "?" in tavily_url else "?"
            tavily_url = f"{tavily_url}{sep}tavilyApiKey={tavily_key}"
        elif tavily_url.endswith("tavilyApiKey="):
            tavily_url = f"{tavily_url}{tavily_key}"

    conn = StreamableHTTPConnectionParams(
        url=tavily_url,
        headers=({"Authorization": f"Bearer {tavily_key}"} if tavily_key else None),
    )

    # Provide explicit auth config to ADK to remove auth warnings in Web UI.
    # We expose an HTTP Bearer scheme so the UI knows the tool is authenticated.
    auth_scheme = None
    auth_credential = None
    if HTTPBase is not None and AuthCredential is not None and tavily_key:
        # FastAPI OpenAPI models expose concrete security scheme models like HTTPBase
        # instead of the Union SecurityScheme, so instantiate HTTPBase directly.
        auth_scheme = HTTPBase(scheme="bearer")
        auth_credential = AuthCredential(
            auth_type=AuthCredentialTypes.HTTP,
            http=HttpAuth(
                scheme="bearer",
                credentials=HttpCredentials(token=tavily_key),
            ),
        )

    toolsets.append(
        MCPToolset(
            connection_params=conn,
            auth_scheme=auth_scheme,
            auth_credential=auth_credential,
        )
    )

    if debug:
        print(f"[MCP v2] Tavily toolset configured url={tavily_url}")

    return toolsets


# ------------------------------
# State helpers for Web UI
# ------------------------------
def flatten_state(
    state: Mapping[str, Any] | None,
    *,
    sep: str = ".",
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Flattens a nested mapping for cleaner, compact Web UI display.

    - Only flattens dict-like mappings up to max_depth.
    - Lists/tuples are JSON-serialized to a short string preview.
    - Non-JSON-serializable values are stringified.
    """
    if not state:
        return {}

    def _shorten(value: Any) -> Any:
        # Ensure JSON-safe values; shorten long strings
        try:
            json.dumps(value)
            if isinstance(value, str) and len(value) > 4000:
                return value[:4000] + "…"
            return value
        except Exception:
            return str(value)

    out: Dict[str, Any] = {}

    def _walk(prefix: str, obj: Any, depth: int) -> None:
        if depth > max_depth:
            out[prefix] = _shorten(obj)
            return

        if isinstance(obj, Mapping):
            for k, v in obj.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                _walk(key, v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            # Compact preview for sequences
            preview = obj[:5] if isinstance(obj, list) else list(obj)[:5]
            out[prefix] = _shorten(preview)
        else:
            out[prefix] = _shorten(obj)

    _walk("", state, 0)
    return out


def validate_state_for_ui(state: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Validates and sanitizes a state mapping for ADK Web UI consumption.

    - Ensures keys are strings and values are JSON-serializable.
    - Truncates overly long strings to keep UI responsive.
    - Drops keys that fail basic validation.
    """
    flat = flatten_state(state)
    safe: Dict[str, Any] = {}

    for k, v in flat.items():
        if not isinstance(k, str) or not k:
            continue
        try:
            json.dumps(v)
            safe[k] = v
        except Exception:
            safe[k] = str(v)

    return safe
