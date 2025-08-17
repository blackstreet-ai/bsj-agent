"""
bsj_agent_v2.agents.captioner

Generates captions + hashtags for YouTube, TikTok, Instagram from script summary.
"""
from __future__ import annotations

from typing import Any
import json

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
        "You are the BSJ captioner. Using session.state.script.summary and session.state.topic, "
        "produce captions for YouTube, TikTok, Instagram plus a common hashtag set."
    )

    def _after(callback_context, llm_response):
        """Parse JSON, store Markdown in state, and return original response.

        - Keeps ADK postprocessing intact by returning `llm_response`.
        - Stores structured content in state['captions'] and a Markdown view in
          state['captions_markdown'] for UI consumption.
        """
        try:
            text = ""
            if getattr(llm_response, "content", None) and llm_response.content.parts:
                text = "".join([p.text or "" for p in llm_response.content.parts])
            parsed = json.loads(text)
            # Save structured value
            try:
                state = callback_context.state
                state["captions"] = parsed.get("captions") if isinstance(parsed, dict) else parsed
            except Exception:
                pass

            caps = parsed.get("captions") if isinstance(parsed, dict) else parsed
            lines = ["# Captions"]
            if isinstance(caps, dict):
                for platform in ["youtube", "tiktok", "instagram"]:
                    val = caps.get(platform)
                    if val:
                        lines.append(f"- **{platform}**: {val}")
                tags = caps.get("hashtags")
                if tags:
                    lines.append(f"- **hashtags**: {tags}")
            elif caps:
                lines.append(str(caps))
            md = "\n".join(lines)
            try:
                callback_context.state["captions_markdown"] = md
            except Exception:
                pass
            return llm_response
        except Exception:
            return llm_response

    return LlmAgent(
        name="bsj_captioner",
        model="gemini-2.5-flash",
        description="Generate platform captions and hashtags.",
        instruction=instruction,
        output_key="captions",
        after_model_callback=_after,
        # Let the model respond naturally; our callback renders Markdown and updates state.
    )
