"""
bsj_agent.agents.captioner

Factory for BSJ Captioner LlmAgent. Generates captions + hashtags for platforms
from session.state.script.summary and session.state.topic.
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
        "Read session.state.script.summary and session.state.topic. Output ONLY JSON under 'captions' with keys youtube[], tiktok[], instagram[], hashtags[]. "
        "All content must stay on the given topic. No prose outside JSON. Provide at least 2 items for youtube, tiktok, instagram, and at least 8 hashtags."
    )

    return LlmAgent(
        name="bsj_captioner",
        model="gemini-2.5-flash",
        description="Generate captions and hashtags for YouTube, TikTok, and Instagram.",
        instruction=instruction,
        output_key="captions",
        tools=[],
        generate_content_config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ) if types is not None else None,
    )
