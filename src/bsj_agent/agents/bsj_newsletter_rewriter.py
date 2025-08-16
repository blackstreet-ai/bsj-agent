from typing import Any, Dict

try:
    from google.adk.agents import LlmAgent
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore


def get_agent() -> Any:
    # Always return a simple sync stub for now so the pipeline runs without
    # requiring ADK's async Runner integration.
    class _Stub:
        name = "bsj_newsletter_rewriter"

        def run(self, session: Dict[str, Any]) -> Dict[str, Any]:
            if LlmAgent is None:
                print("[WARN] google-adk not installed. Running bsj_newsletter_rewriter stub.")
            else:
                print("[INFO] Using bsj_newsletter_rewriter stub adapter (sync). TODO: wire ADK Runner.")
            script = session.get("state", {}).get("script", {})
            session.setdefault("state", {})
            session["state"]["newsletter"] = {
                "subject_lines": [
                    "This week in tech & culture",
                    "BSJ Brief: Key insights",
                    "Afrofuturist lens on today's news",
                ],
                "body": script.get("summary", "Newsletter-ready summary placeholder."),
            }
            return session

    return _Stub()
