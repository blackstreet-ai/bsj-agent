"""
bsj_agent_v2.workflow

ADK-native orchestration for BSJ pipeline (v2):
- researcher -> scriptwriter -> (thumbnail_promptor || captioner) -> voiceover
- Modular agent factories under bsj_agent_v2.agents.*
- Designed to work with InMemoryRunner and ADK Web UI.
"""
from __future__ import annotations

from typing import Any, List

try:
    from google.adk.agents import SequentialAgent, ParallelAgent
except Exception:  # pragma: no cover
    SequentialAgent = None  # type: ignore
    ParallelAgent = None  # type: ignore

from .agents.researcher import create_agent as create_researcher
from .agents.scriptwriter import create_agent as create_scriptwriter
from .agents.thumbnail_promptor import create_agent as create_thumbnail_promptor
from .agents.captioner import create_agent as create_captioner
from .agents.voiceover import create_agent as create_voiceover
 


def build_root(*, debug: bool = False) -> Any:
    """Build the root agent graph for BSJ v2.

    Returns:
      A composed ADK agent (SequentialAgent) that can be run by InMemoryRunner
      or served via ADK Web UI.
    """
    if SequentialAgent is None or ParallelAgent is None:
        raise RuntimeError("ADK not available. Install google-adk and retry.")

    researcher = create_researcher(debug=debug)
    scriptwriter = create_scriptwriter()
    assets_parallel = ParallelAgent(
        name="bsj_assets_parallel",
        sub_agents=[create_thumbnail_promptor(), create_captioner()],
    )
    voiceover = create_voiceover()

    root = SequentialAgent(
        name="bsj_pipeline_v2",
        sub_agents=[
            researcher,
            scriptwriter,
            assets_parallel,
            voiceover,
        ],
    )
    return root
