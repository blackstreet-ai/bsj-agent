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
        """Parse JSON, store Markdown in state, and return the original response.

        ADK expects an LLM response object flowing through postprocessors.
        Returning a raw string can break downstream processors. We therefore:
        - Parse and store structured JSON at state['script'].
        - Build Markdown and store at state['script_markdown'] for UI.
        - Return `llm_response` to keep the flow ADK-compatible.
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
            try:
                callback_context.state["script_markdown"] = md
            except Exception:
                pass
            return llm_response
        except Exception:
            # Fall back to letting ADK render the original response
            return llm_response

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
