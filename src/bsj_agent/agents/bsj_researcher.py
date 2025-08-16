from typing import Any, Dict

try:
    from google.adk.agents import LlmAgent
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore


def get_agent() -> Any:
    """
    Return the bsj_researcher agent instance or a stub if ADK is unavailable.
    """
    # Always return a simple sync stub for now so the pipeline runs without
    # requiring ADK's async Runner integration.
    class _Stub:
        name = "bsj_researcher"

        def run(self, session: Dict[str, Any]) -> Dict[str, Any]:
            if LlmAgent is None:
                print("[WARN] google-adk not installed. Running bsj_researcher stub.")
            else:
                print("[INFO] Using bsj_researcher stub adapter (sync). TODO: wire ADK Runner.")
            # Pass-through stub that echoes inputs
            session.setdefault("state", {})
            session["state"].setdefault("research", {"notes": [], "citations": []})
            return session

    return _Stub()
