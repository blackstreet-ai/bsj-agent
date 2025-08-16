from typing import Any, Dict

try:
    from google.adk.agents import LlmAgent
except Exception:  # pragma: no cover
    LlmAgent = None  # type: ignore


def get_agent() -> Any:
    # Always return a simple sync stub for now so the pipeline runs without
    # requiring ADK's async Runner integration.
    class _Stub:
        name = "bsj_voiceover"

        def run(self, session: Dict[str, Any]) -> Dict[str, Any]:
            if LlmAgent is None:
                print("[WARN] google-adk not installed. Running bsj_voiceover stub.")
            else:
                print("[INFO] Using bsj_voiceover stub adapter (sync). TODO: wire ADK Runner.")
            session.setdefault("state", {})
            script = session["state"].get("script", {})
            # Simulate producing TTS-ready text
            session["state"]["voiceover"] = {
                "text": script.get("draft", "No script available."),
                "voice_id": "elevenlabs_bsj_voice_placeholder",
                "status": "stub_generated"
            }
            return session

    return _Stub()
