"""
bsj_agent_v2

Clean, ADK-native BSJ pipeline scaffold (non-destructive to existing bsj_agent code).
"""
# =============================================================================
# ADK App Export (root_agent)
# -----------------------------------------------------------------------------
# ADK's Web UI (started via `adk web <agents_dir>`) auto-discovers "apps" by
# importing packages/modules inside <agents_dir> and looking for a symbol named
# `root_agent`.
#
# By exposing `root_agent` here, the ADK AgentLoader can treat `bsj_agent_v2`
# as an app. This enables running the full BSJ v2 pipeline (with Human-in-the-
# Loop long-running review gates) directly from the ADK Web UI.
#
# Usage (from repository root):
#   1) Start the ADK Web UI pointing to our `src/` as agents_dir:
#        adk web src
#   2) Open the UI (default http://localhost:8000/dev-ui), select app
#      "bsj_agent_v2", create a session, and set initial state with a `topic`.
#   3) Click Run. When the pipeline reaches review gates, the UI will surface
#      Approve/Reject controls that resolve the long-running tool.
#
from __future__ import annotations

from .workflow import build_root

# root_agent: the ADK entry point loaded by the Web UI / AgentLoader
# - We build once at import time for simplicity. If you prefer a fresh build per
#   session, you can wrap this in a callable or use an `agent.py` submodule.
root_agent = build_root(debug=False)

