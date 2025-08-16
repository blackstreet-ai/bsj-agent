"""
bsj_agent_v2.agents.normalizer

A tiny utility agent that runs as a normal ADK agent and normalizes
JSON-string outputs written by upstream LLM agents into real Python objects.

Why: Some LLM responses are captured as strings in `session.state` even when
we request application/json. This agent ensures ADK Web shows structured
objects instead of code-fenced or stringified JSON.

Keys normalized if they exist and are strings:
- script
- thumbnail_prompts
- captions
- voiceover

Implementation notes (ADK):
- We mutate state via `before_agent_callback` using `CallbackContext.state`,
  which tracks deltas correctly for the runner and UI.
- The agent itself does not call any LLMs or tools; its _run_async_impl is a
  no-op.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from ...tools.mcp_utils import validate_state_for_ui
try:
    from google.genai import types
except Exception:  # pragma: no cover
    types = None  # type: ignore


_CODEFENCE_RE = re.compile(r"^```(?:json)?\n|\n```$", re.IGNORECASE)


def _strip_code_fences(s: str) -> str:
    # remove leading/trailing triple backticks optionally labeled json
    return _CODEFENCE_RE.sub("", s.strip())


class JSONNormalizerAgent(BaseAgent):
    """ADK agent that normalizes JSON-string fields in state.

    It does all work in before_agent_callback to leverage ADK's State delta.
    """

    def __init__(self, *, name: str = "bsj_json_normalizer") -> None:
        super().__init__(name=name, description="Normalize JSON strings in state")
        self.before_agent_callback = self._before

    async def _before(self, *, callback_context: CallbackContext) -> Optional[Any]:
        state = callback_context.state
        keys = ("script", "thumbnail_prompts", "captions", "voiceover")
        for k in keys:
            try:
                v = state.get(k)
            except Exception:
                v = None
            if isinstance(v, str):
                raw = _strip_code_fences(v)
                try:
                    parsed = json.loads(raw)
                    state[k] = parsed
                except Exception:
                    # If it's not valid JSON, leave as-is.
                    pass
        # Provide a UI-friendly, flattened & validated snapshot without
        # mutating originals, so ADK Web renders a compact, readable view.
        try:
            state["_ui_state"] = validate_state_for_ui(state)
        except Exception:
            # Best-effort only; never block pipeline.
            pass
        # Emit a single Markdown Event that summarizes major outputs for the UI.
        # This renders nicely in the Chat/Events pane while preserving JSON in state.
        try:
            md_lines: list[str] = []
            # Research
            research = state.get("research")
            if research:
                md_lines.append("# Research Findings")
                topics = None
                key_stats = None
                citations = None
                if isinstance(research, dict):
                    topics = research.get("topics")
                    key_stats = research.get("key_stats")
                    citations = research.get("citations")
                if topics:
                    md_lines.append("\n**Topics**:")
                    for t in (topics if isinstance(topics, list) else [topics]):
                        md_lines.append(f"- {t}")
                if key_stats:
                    md_lines.append("\n**Key Stats**:")
                    for ks in (key_stats if isinstance(key_stats, list) else [key_stats]):
                        try:
                            label = ks.get("label", "stat")
                            value = ks.get("value", "-")
                            src = ks.get("source")
                            if src:
                                md_lines.append(f"- {label}: {value} (source: {src})")
                            else:
                                md_lines.append(f"- {label}: {value}")
                        except Exception:
                            md_lines.append(f"- {ks}")
                if citations:
                    md_lines.append("\n**Citations**:")
                    for c in (citations if isinstance(citations, list) else [citations]):
                        try:
                            title = c.get("title", "link")
                            url = c.get("url", "-")
                            md_lines.append(f"- [{title}]({url})")
                        except Exception:
                            md_lines.append(f"- {c}")

            # Script
            script = state.get("script")
            if script:
                md_lines.append("\n# Script")
                if isinstance(script, dict):
                    beats = script.get("beats")
                    draft = script.get("draft")
                    summary = script.get("summary")
                    if beats:
                        md_lines.append("\n**Beats**:")
                        for b in (beats if isinstance(beats, list) else [beats]):
                            md_lines.append(f"- {b}")
                    if summary:
                        md_lines.append("\n**Summary**:\n")
                        md_lines.append(str(summary))
                    if draft:
                        md_lines.append("\n**Draft**:\n")
                        md_lines.append(str(draft))
                else:
                    md_lines.append(str(script))

            # Thumbnail prompts
            thumbs = state.get("thumbnail_prompts")
            if thumbs:
                md_lines.append("\n# Thumbnail Prompts")
                for p in (thumbs if isinstance(thumbs, list) else [thumbs]):
                    md_lines.append(f"- {p}")

            # Captions
            captions = state.get("captions")
            if captions:
                md_lines.append("\n# Captions")
                for cap in (captions if isinstance(captions, list) else [captions]):
                    if isinstance(cap, dict):
                        platform = cap.get("platform", "post")
                        text = cap.get("text") or cap.get("caption") or cap
                        tags = cap.get("hashtags")
                        line = f"- **{platform}**: {text}"
                        if tags:
                            line += f"  {tags}"
                        md_lines.append(line)
                    else:
                        md_lines.append(f"- {cap}")

            # Voiceover
            voice = state.get("voiceover")
            if voice:
                md_lines.append("\n# Voiceover")
                if isinstance(voice, dict):
                    vo_sum = voice.get("summary") or voice.get("script") or voice
                    md_lines.append(str(vo_sum))
                else:
                    md_lines.append(str(voice))

            md_text = "\n".join(md_lines).strip()
            if md_text:
                if types is not None:
                    return types.Content(parts=[types.Part(text=md_text)])
                return md_text
        except Exception:
            # Never block the pipeline on formatting errors
            pass
        return None

    async def _run_async_impl(self, ctx) -> Any:  # type: ignore[override]
        # No LLM or tools. We rely on before callback to mutate state.
        if False:  # pragma: no cover - satisfy async generator type
            yield Event(invocation_id=ctx.invocation_id, author=self.name, branch=ctx.branch)
        return


def create_agent() -> BaseAgent:
    return JSONNormalizerAgent()
