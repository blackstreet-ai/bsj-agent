from typing import Any, Dict

try:
    from google.adk.agents import LlmAgent
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore


def get_agent() -> Any:
    # Always return a simple sync stub for now so the pipeline runs without
    # requiring ADK's async Runner integration.
    class _Stub:
        name = "bsj_thumbnail_promptor"

        def run(self, session: Dict[str, Any]) -> Dict[str, Any]:
            if LlmAgent is None:
                print("[WARN] google-adk not installed. Running bsj_thumbnail_promptor stub.")
            else:
                print("[INFO] Using bsj_thumbnail_promptor stub adapter (sync). TODO: wire ADK Runner.")
            session.setdefault("state", {})
            session["state"]["thumbnail_prompts"] = [
                "Afrofuturist collage, bold typography, high contrast, BSJ colors",
                "Editorial portrait with neon accents, tech-meets-culture vibe",
                "Minimalist geometric shapes with Afrocentric palette"
            ]
            return session

    return _Stub()
