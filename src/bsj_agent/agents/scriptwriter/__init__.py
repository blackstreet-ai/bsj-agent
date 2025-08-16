"""
bsj_agent.agents.scriptwriter

Factory for BSJ Scriptwriter LlmAgent (ADK). Consumes research in session.state and
emits a structured script object to session.state['script'].
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
        "You are the BSJ scriptwriter. Use session.state.research.topics and session.state.research.key_stats to craft a BSJ-tone narrative.\n"
        "Respond with ONLY valid JSON (no markdown, no code fences, no prose).\n"
        "- beats: array of 5-8 short strings capturing the narrative beats\n"
        "- draft: a single string, 400-700 words, culturally grounded in BSJ voice\n"
        "- summary: a single string <= 60 words\n"
        "Output the object as {\"script\": { ... }}. Stay STRICTLY on session.state.topic."
    )

    return LlmAgent(
        name="bsj_scriptwriter",
        model="gemini-2.5-flash",
        description="Transform research into a narrative script.",
        instruction=instruction,
        output_key="script",
        tools=[],
        generate_content_config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ) if types is not None else None,
    )
