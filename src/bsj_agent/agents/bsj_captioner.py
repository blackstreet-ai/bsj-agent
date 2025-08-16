from typing import Any, Dict

try:
    from google.adk.agents import LlmAgent
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore


def get_agent() -> Any:
    # Always return a simple sync stub for now so the pipeline runs without
    # requiring ADK's async Runner integration.
    class _Stub:
        name = "bsj_captioner"

        def run(self, session: Dict[str, Any]) -> Dict[str, Any]:
            if LlmAgent is None:
                print("[WARN] google-adk not installed. Running bsj_captioner stub.")
            else:
                print("[INFO] Using bsj_captioner stub adapter (sync). TODO: wire ADK Runner.")
            session.setdefault("state", {})
            session["state"]["captions"] = {
                "youtube": ["YT caption 1", "YT caption 2", "YT caption 3"],
                "tiktok": ["TT caption 1", "TT caption 2", "TT caption 3"],
                "instagram": ["IG caption 1", "IG caption 2", "IG caption 3"],
                "hashtags": ["#BSJ", "#TechCulture", "#Afrofuturism"],
            }
            return session

    return _Stub()
