"""
bsj_agent.agents.voiceover

Factory for BSJ Voiceover LlmAgent. Produces voiceover-ready text from the script
and research context in session.state.
"""
from __future__ import annotations

from typing import Any

try:
    from google.adk.agents import LlmAgent
    from google.genai import types
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore
    types = None  # type: ignore


def create_agent() -> Any:
    if LlmAgent is None:
        raise RuntimeError("ADK not available. Install google-adk and retry.")

    instruction = (
        "Read session.state.script.draft and session.state.topic. Output ONLY JSON under 'voiceover' with key 'text' containing finalized, on-topic narration.\n"
        "Explicitly reflect at least 2 ideas from session.state.research.topics."
    )

    return LlmAgent(
        name="bsj_voiceover",
        model="gemini-2.5-flash",
        description="Transform script into voiceover-ready text.",
        instruction=instruction,
        output_key="voiceover",
        tools=[],
        generate_content_config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ) if types is not None else None,
    )
