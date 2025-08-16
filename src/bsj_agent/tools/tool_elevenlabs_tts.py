from typing import Optional


def tool_elevenlabs_tts(text: str, voice_id: Optional[str] = None) -> bytes:
    """
    Stub ElevenLabs TTS tool. Returns fake audio bytes.
    Replace with real ElevenLabs API integration and auth handling.
    """
    print("[WARN] tool_elevenlabs_tts is a stub. No real audio generated.")
    payload = f"VOICE:{voice_id or 'default'}\n{text}".encode("utf-8")
    return payload
