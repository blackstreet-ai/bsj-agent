"""
bsj_agent_v2.agents.review_gate

Generic Human-In-The-Loop (HITL) review agent using Google ADK's
LongRunningFunctionTool. This creates a true pause/resume gate in the
pipeline, following the official ADK samples under:
- `adk-python/contributing/samples/human_in_loop/`
- `adk-python/contributing/samples/tool_human_in_the_loop/`

We expose a generic factory that can be reused for different checkpoints
(e.g., research review, script review) by pointing to the correct
state keys that contain the Markdown for display.

Key ideas:
- The model is instructed to call a long-running tool to request approval.
- The tool function returns a "pending" result so ADK pauses and waits
  for human resolution via the Web UI or programmatic completion.
- We also write a small status record into the session state for auditing.

This file is extensively commented for ADK learners, per project standards.
"""
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime

try:
    # ADK core Agent type and LongRunningFunctionTool for HITL
    from google.adk import Agent
    from google.adk.tools.long_running_tool import LongRunningFunctionTool
    from google.adk.tools.tool_context import ToolContext
    from google.genai import types
except Exception:  # pragma: no cover
    Agent = None  # type: ignore
    LongRunningFunctionTool = None  # type: ignore
    ToolContext = None  # type: ignore
    types = None  # type: ignore


def _now_iso() -> str:
    """Utility to format current time as ISO8601 string."""
    return datetime.now().isoformat()


def _make_review_tool(approval_state_key: str, markdown_state_key: str):
    """Factory that creates a long-running approval tool function.

    The returned function signature follows the ADK pattern for long-running
    tools: it must accept a `ToolContext` parameter. The model will invoke this
    tool to request human approval. We return a "pending" status and write a
    small status object into `tool_context.state` for observability.

    Args:
      approval_state_key: State key where we store review status metadata,
        e.g., "research_review" or "script_review".
      markdown_state_key: State key where the upstream agent stored
        Markdown for the human to read, e.g., "research_markdown".

    Returns:
      A callable suitable to wrap with `LongRunningFunctionTool`.
    """

    def ask_for_approval(reason: str, tool_context: ToolContext) -> Dict[str, Any] | None:
        """Request approval for the current artifact in state.

        Parameters:
          reason: Short string provided by the model explaining what is being
            reviewed (e.g., "Research findings" or "Script draft").
          tool_context: ADK ToolContext providing access to session state and
            logging; required by LongRunningFunctionTool to enable HITL pause.

        Returns:
          A dict with at least 'status': 'pending' to trigger a HITL pause.
          Can include any additional metadata (ticket/message IDs, etc.).
        """
        try:
            state = getattr(tool_context, "state", {}) or {}
            # Pull the corresponding markdown for display context (optional)
            markdown = state.get(markdown_state_key)

            # Record a status object in state for visibility/auditing
            state[approval_state_key] = {
                "status": "pending",
                "reason": reason,
                "markdown_key": markdown_state_key,
                "has_markdown": bool(markdown),
                "created_at": _now_iso(),
            }
            # Hint the UI to not replace the pending card with a summary
            # (mirrors sample get_user_choice_tool behavior)
            try:
                tool_context.actions.skip_summarization = True
            except Exception:
                pass
        except Exception:
            # Swallow state errors to avoid breaking the pause behavior
            pass

        # For LongRunningFunctionTool, returning None marks the call as pending
        # and surfaces a resolvable card in the Web UI. The human will submit the
        # final tool response (e.g., {"status": "approved" | "rejected", ...})
        # via the UI, which resumes the pipeline.
        return None

    return ask_for_approval


def create_review_agent(
    *,
    name: str,
    markdown_state_key: str,
    approval_state_key: str,
    model: str = "gemini-2.5-flash",
) -> Any:
    """Create a generic HITL review agent.

    This agent is minimal: it doesn't generate new content. Instead, it guides
    the model to call a long-running approval tool which triggers a human
    approval pause. The human resolves the tool call in the ADK Web UI, after
    which the pipeline resumes.

    Parameters:
      name: Agent name, e.g., "bsj_research_review".
      markdown_state_key: State key containing the Markdown to review.
      approval_state_key: State key where review gate status should be recorded.
      model: Model to use; defaults to a cost-effective fast model.

    Returns:
      An ADK `Agent` instance configured with a LongRunningFunctionTool.
    """
    if Agent is None or LongRunningFunctionTool is None:
        raise RuntimeError("ADK not available. Install google-adk and retry.")

    # Instruction: be explicit that the agent should call the HITL tool.
    instruction = (
        "You are a STRICT review checkpoint."
        "\n- Do NOT generate normal text output."
        "\n- Read session.state.%s silently."
        "\n- Your ONLY action is to CALL the long-running tool"
        " ask_for_approval(reason=...) to request human approval."
        "\n- After calling the tool, STOP and WAIT. Do not produce any other content."
        % markdown_state_key
    )

    # Build the long-running tool with state keys bound
    ask_for_approval = _make_review_tool(
        approval_state_key=approval_state_key,
        markdown_state_key=markdown_state_key,
    )

    return Agent(
        name=name,
        model=model,
        description=(
            "HITL review gate that pauses the pipeline for human approval."
        ),
        instruction=instruction,
        tools=[LongRunningFunctionTool(func=ask_for_approval)],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=16,
            candidate_count=1,
        )
        if types is not None
        else None,
    )
