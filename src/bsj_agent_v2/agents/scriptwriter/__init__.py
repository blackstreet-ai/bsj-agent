"""
bsj_agent_v2.agents.scriptwriter

Factory for the BSJ scriptwriter agent using Google ADK LlmAgent.
- Consumes session.state.research
- Produces script JSON: beats[], draft, summary
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
        "You are the BSJ scriptwriter. Use session.state.research.topics and "
        "session.state.research.key_stats to craft a BSJ-tone narrative.\n"
        "Produce: beats (5-8 bullet points), a 400-700 word draft, and a summary (<=60 words)."
    )

    def _after(callback_context, llm_response):
        """Parse JSON into state and return Markdown summary instead of raw JSON.

        ADK detail: Returning non-None content from after_model_callback replaces
        the model's event content in the chat UI (see `BaseAgent.__handle_after_agent_callback`).
        We also update `session.state['script']` with parsed JSON for downstream agents.
        """
        try:
            text = ""
            if getattr(llm_response, "content", None) and llm_response.content.parts:
                text = "".join([p.text or "" for p in llm_response.content.parts])
            # Attempt to parse JSON (model is instructed to output JSON only)
            parsed = json.loads(text)
            # Save structured value for the pipeline and UI state panel
            try:
                state = callback_context.state
                state["script"] = parsed
            except Exception:
                pass

            # Build compact Markdown for the Chat/Events panel
            beats = parsed.get("beats") if isinstance(parsed, dict) else None
            draft = parsed.get("draft") if isinstance(parsed, dict) else None
            summary = parsed.get("summary") if isinstance(parsed, dict) else None

            lines = ["# Script"]
            if beats:
                lines.append("\n**Beats**:")
                for b in (beats if isinstance(beats, list) else [beats]):
                    lines.append(f"- {b}")
            if summary:
                lines.append("\n**Summary**:\n")
                lines.append(str(summary))
            if draft:
                lines.append("\n**Draft**:\n")
                lines.append(str(draft))

            md = "\n".join(lines)
            return md
        except Exception:
            # If parsing fails, fall back to default rendering
            return None

    return LlmAgent(
        name="bsj_scriptwriter",
        model="gemini-2.5-flash",
        description="Transform research into a narrative script.",
        instruction=instruction,
        output_key="script",
        after_model_callback=_after,
        # Tools are not used here; enforcing JSON mime type is supported.
        # Let the model respond naturally; our callback renders Markdown and updates state.
    )
