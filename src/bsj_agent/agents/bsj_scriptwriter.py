from typing import Any, Dict

try:
    from google.adk.agents import LlmAgent
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore


def get_agent() -> Any:
    # Always return a simple sync stub for now so the pipeline runs without
    # requiring ADK's async Runner integration.
    class _Stub:
        name = "bsj_scriptwriter"

        def run(self, session: Dict[str, Any]) -> Dict[str, Any]:
            if LlmAgent is None:
                print("[WARN] google-adk not installed. Running bsj_scriptwriter stub.")
            else:
                print("[INFO] Using bsj_scriptwriter stub adapter (sync). TODO: wire ADK Runner.")
            session.setdefault("state", {})
            research = session["state"].get("research", {})
            session["state"]["script"] = {
                "beats": ["Intro", "Body", "Conclusion"],
                "draft": "Placeholder script based on research." if research else "Placeholder script.",
                "summary": "One-liner summary",
            }
            return session

    return _Stub()
