"""
bsj_agent_v2.agent

Single-file entrypoint module required by ADK tools (web/eval/cli) that exposes
`root_agent` for the BSJ v2 pipeline.

This module is intentionally tiny and delegates graph construction to
`bsj_agent_v2.workflow.build_root`.
"""
from __future__ import annotations

try:
    from .workflow import build_root
except Exception as e:  # pragma: no cover
    raise

# Construct the root agent once for the app. Toggle debug here if desired.
root_agent = build_root(debug=False)
