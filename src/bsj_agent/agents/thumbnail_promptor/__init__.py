"""
bsj_agent.agents.thumbnail_promptor

Factory for Thumbnail Promptor LlmAgent. Produces 3 Afrofuturist-style prompts
from session.state.script.summary.
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
        "Read session.state.script.summary. Output JSON under 'thumbnail_prompts' as an array of 3 strings. No prose outside JSON."
    )

    return LlmAgent(
        name="bsj_thumbnail_promptor",
        model="gemini-2.5-flash",
        description="Generate 3 Afrofuturist-style thumbnail prompts.",
        instruction=instruction,
        output_key="thumbnail_prompts",
        tools=[],
        generate_content_config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ) if types is not None else None,
    )
