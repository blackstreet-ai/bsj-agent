"""
bsj_agent_v2.agents.voiceover

Produces voiceover-ready text from script draft and research.
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
        "You are the BSJ voiceover generator. Using session.state.script.draft and session.state.topic, "
        "produce voiceover-ready text in BSJ voice that is clear and engaging."
    )

    def _after(callback_context, llm_response):
        try:
            text = ""
            if getattr(llm_response, "content", None) and llm_response.content.parts:
                text = "".join([p.text or "" for p in llm_response.content.parts])
            parsed = json.loads(text)
            # Save structured value
            try:
                state = callback_context.state
                state["voiceover"] = parsed.get("voiceover") if isinstance(parsed, dict) else parsed
            except Exception:
                pass

            vo = parsed.get("voiceover") if isinstance(parsed, dict) else parsed
            lines = ["# Voiceover"]
            if isinstance(vo, dict):
                txt = vo.get("text") or vo.get("script") or vo
                lines.append(str(txt))
            elif vo:
                lines.append(str(vo))
            md = "\n".join(lines)
            return md
        except Exception:
            return None

    return LlmAgent(
        name="bsj_voiceover",
        model="gemini-2.5-flash",
        description="Produce voiceover-ready text from script.",
        instruction=instruction,
        output_key="voiceover",
        after_model_callback=_after,
        # Let the model respond naturally; our callback renders Markdown and updates state.
    )
