"""
bsj_agent_v2.agents.thumbnail_promptor

Generates 3 Afrofuturist-style thumbnail prompts from script summary.
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
        "You are the BSJ thumbnail promptor. Using session.state.script.summary, "
        "produce exactly 3 Afrofuturist-style thumbnail prompts that would look great as YouTube thumbnails."
    )

    def _after(callback_context, llm_response):
        try:
            text = ""
            if getattr(llm_response, "content", None) and llm_response.content.parts:
                text = "".join([p.text or "" for p in llm_response.content.parts])
            parsed = json.loads(text)
            # Save to state
            try:
                state = callback_context.state
                # Accept either {'thumbnail_prompts': [...]} or direct list
                val = parsed.get("thumbnail_prompts") if isinstance(parsed, dict) else parsed
                state["thumbnail_prompts"] = val
            except Exception:
                pass

            prompts = (
                parsed.get("thumbnail_prompts") if isinstance(parsed, dict) else parsed
            )
            lines = ["# Thumbnail Prompts"]
            if prompts:
                for p in (prompts if isinstance(prompts, list) else [prompts]):
                    lines.append(f"- {p}")
            md = "\n".join(lines)
            return md
        except Exception:
            return None

    return LlmAgent(
        name="bsj_thumbnail_promptor",
        model="gemini-2.5-flash",
        description="Generate 3 Afrofuturist thumbnail prompts.",
        instruction=instruction,
        output_key="thumbnail_prompts",
        after_model_callback=_after,
        # Let the model respond naturally; our callback renders Markdown and updates state.
    )
