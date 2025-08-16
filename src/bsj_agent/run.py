import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Optional

from .workflow import BsjPipeline, run_adk_pipeline


# Load environment variables.
# When installed as a console script, __file__ lives in site-packages, so we
# prefer CWD. Allow override via BSJ_DOTENV env var.
_explicit_env = os.environ.get("BSJ_DOTENV")
if _explicit_env and Path(_explicit_env).exists():
    load_dotenv(dotenv_path=_explicit_env, override=False)
else:
    # Load from current working directory if available; doesn't error if missing.
    load_dotenv(override=False)


def _run_stub(topic: str, include_newsletter: bool) -> dict[str, Any]:
    pipeline = BsjPipeline(include_newsletter=include_newsletter)
    return pipeline.run(topic)


def _extract_text(content) -> str:
    # content is google.genai.types.Content
    if not content or not getattr(content, "parts", None):
        return ""
    texts = []
    for p in content.parts:
        if getattr(p, "text", None):
            texts.append(p.text)
    return "\n".join(texts).strip()


def _run_adk(topic: str, *, debug: bool = False) -> tuple[Optional[dict], Optional[str]]:
    """Return (session_state, error_message)."""
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
        from google.genai import types
    except Exception as e:
        return None, f"ADK import failed: {e}"

    # Auth hint
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return None, (
            "Missing Google auth. Set GOOGLE_API_KEY for Gemini API or GOOGLE_APPLICATION_CREDENTIALS for Vertex."
        )

    # Root agent: single-pass generation of BSJ outputs into a JSON blob
    root = LlmAgent(
        name="bsj_root",
        model="gemini-2.5-flash",
        description="Produce BSJ research, script, prompts, captions, and voiceover fields as JSON.",
        instruction=(
            "You are the Black Street Journal (BSJ) content generator. Given a topic, produce a concise,"
            " factual, culturally grounded output in this strict JSON schema under key bsj_output: "
            "{research:{topics:[], key_stats:[], citations:[{title, url}]}, script:{beats:[], draft, summary},"
            " thumbnail_prompts:[3 strings], captions:{youtube:[], tiktok:[], instagram:[], hashtags:[]},"
            " voiceover:{text, voice_id?}}. Only output JSON (no prose)."
        ),
        output_key="bsj_output",
        tools=[],
    )

    runner = InMemoryRunner(agent=root)

    # Ensure a session exists before calling runner.run
    try:
        # Uses the in-memory session service (sync helper is fine for local runs)
        runner._in_memory_session_service.create_session_sync(  # type: ignore[attr-defined]
            app_name=runner.app_name,
            user_id="bsj_user",
            session_id="bsj_session",
            state={"topic": topic},
        )
    except Exception:
        # If it already exists or service unavailable, ignore and proceed
        pass

    # Initialize a session and run with the user's topic as text content
    user_msg = types.Content(role="user", parts=[types.Part(text=f"Topic: {topic}")])

    try:
        events = list(
            runner.run(
                user_id="bsj_user",
                session_id="bsj_session",
                new_message=user_msg,
            )
        )
    except Exception as e:
        return None, f"ADK run failed: {e}"

    # Debug: inspect last few events for content and state changes
    if debug:
        try:
            print(f"[DEBUG] ADK events received: {len(events)}")
            for i, ev in enumerate(events[-10:]):  # limit to last 10 to avoid noise
                etype = type(ev).__name__
                author = getattr(ev, "author", None)
                content = getattr(ev, "content", None)
                has_text = False
                preview = ""
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        if getattr(p, "text", None):
                            has_text = True
                            preview = (p.text or "")[:120]
                            break
                state_delta = getattr(ev, "state_delta", None)
                state_keys = list(state_delta.keys()) if isinstance(state_delta, dict) else None
                print(
                    f"[DEBUG] Event[{i}] {etype} author={author} has_text={has_text} "
                    f"state_delta_keys={state_keys} preview={preview!r}"
                )
        except Exception:
            pass

    # Try to read the final session state from the in-memory session service first
    try:
        sess = runner._in_memory_session_service.get_session_sync(  # type: ignore[attr-defined]
            app_name=runner.app_name,
            user_id="bsj_user",
            session_id="bsj_session",
        )
        state = getattr(sess, "state", {}) or {}
        if debug:
            try:
                print(f"[DEBUG] Session state top-level keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")
                if isinstance(state, dict):
                    for k in ("bsj_output", "output", "state"):
                        if k in state:
                            v = state[k]
                            print(f"[DEBUG] Found key '{k}' in state: type={type(v).__name__}")
            except Exception:
                pass
        if isinstance(state, dict) and state:
            bsj = state.get("bsj_output") or state.get("output") or None
            # If ADK stored JSON as string, parse it
            if isinstance(bsj, str):
                raw = bsj.strip()
                if raw.startswith("```"):
                    # strip fenced code block
                    first_nl = raw.find("\n")
                    if first_nl != -1:
                        raw = raw[first_nl + 1 :]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()
                try:
                    bsj = json.loads(raw)
                except Exception:
                    # Try to recover braces
                    s = raw.find("{")
                    e = raw.rfind("}")
                    if s >= 0 and e > s:
                        try:
                            bsj = json.loads(raw[s : e + 1])
                        except Exception:
                            bsj = None
                    else:
                        bsj = None
            # Unwrap if the parsed dict still has a top-level 'bsj_output'
            if isinstance(bsj, dict) and "bsj_output" in bsj and isinstance(bsj["bsj_output"], dict):
                bsj = bsj["bsj_output"]
            if isinstance(bsj, dict):
                session: dict[str, Any] = {"state": {"topic": topic}}
                session["state"]["research"] = bsj.get("research", {})
                session["state"]["script"] = bsj.get("script", {})
                session["state"]["thumbnail_prompts"] = bsj.get("thumbnail_prompts", [])
                session["state"]["captions"] = bsj.get("captions", {})
                session["state"]["voiceover"] = bsj.get("voiceover", {})
                return session, None
    except Exception:
        # Ignore and fall back to parsing streamed content
        pass

    # Find last event with textual content (author may be agent name rather than 'model')
    model_text = ""
    for ev in events:
        content = getattr(ev, "content", None)
        if content:
            extracted = _extract_text(content)
            if extracted:
                model_text = extracted

    if not model_text:
        return None, "No model output captured."

    # Try to parse as JSON
    try:
        data = json.loads(model_text)
    except Exception:
        # Try to recover from fenced blocks
        start = model_text.find("{")
        end = model_text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(model_text[start : end + 1])
            except Exception as e:
                return None, f"Model output not valid JSON: {e}\nRaw: {model_text[:4000]}"
        else:
            return None, f"Model output not JSON. Raw: {model_text[:4000]}"

    # Map into our session.state shape
    session: dict[str, Any] = {"state": {"topic": topic}}
    bsj = data.get("bsj_output", data)
    session["state"]["research"] = bsj.get("research", {})
    session["state"]["script"] = bsj.get("script", {})
    session["state"]["thumbnail_prompts"] = bsj.get("thumbnail_prompts", [])
    session["state"]["captions"] = bsj.get("captions", {})
    session["state"]["voiceover"] = bsj.get("voiceover", {})
    return session, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BSJ agent pipeline")
    parser.add_argument("topic", help="Topic to research and script")
    parser.add_argument(
        "--include-newsletter",
        action="store_true",
        help="Include optional newsletter rewriting stage (stub engine only)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print final session JSON to stdout",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for ADK events and session state",
    )
    parser.add_argument(
        "--adk-multistage",
        action="store_true",
        help="Use ADK multi-stage pipeline (research → script → thumbs/captions → voiceover)",
    )
    parser.add_argument(
        "--engine",
        choices=["adk", "stub"],
        default="adk",
        help="Execution engine: 'adk' (real Gemini via ADK Runner) or 'stub' (local stubs)",
    )
    args = parser.parse_args()

    if args.engine == "adk":
        if args.adk_multistage:
            # Auth check for multi-stage path
            if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                print("[ERROR] Missing Google auth. Set GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS.")
                print("Falling back to stub engine. Set GOOGLE_API_KEY or Vertex creds for real run.")
                session = _run_stub(args.topic, args.include_newsletter)
            else:
                session, err = run_adk_pipeline(
                    args.topic,
                    include_newsletter=args.include_newsletter,
                    debug=args.debug,
                )
                if err:
                    print(f"[ERROR] {err}")
                    print("Falling back to stub engine. Set GOOGLE_API_KEY or Vertex creds for real run.")
                    session = _run_stub(args.topic, args.include_newsletter)
        else:
            session, err = _run_adk(args.topic, debug=args.debug)
            if err:
                print(f"[ERROR] {err}")
                print("Falling back to stub engine. Set GOOGLE_API_KEY or Vertex creds for real run.")
                session = _run_stub(args.topic, args.include_newsletter)
    else:
        session = _run_stub(args.topic, args.include_newsletter)

    if args.print_json:
        print(json.dumps(session, indent=2, ensure_ascii=False))
    else:
        state = session.get("state", {})
        print("=== BSJ Pipeline Complete ===")
        print(f"Topic: {state.get('topic')}")
        print(f"Research keys: {list(state.get('research', {}).keys())}")
        print(f"Script beats: {len(state.get('script', {}).get('beats', []))}")
        print(f"Thumbnail prompts: {len(state.get('thumbnail_prompts', []))}")
        caps = state.get('captions', {})
        print(
            "Captions (yt/tt/ig): "
            f"{len(caps.get('youtube', []))}/"
            f"{len(caps.get('tiktok', []))}/"
            f"{len(caps.get('instagram', []))}"
        )
        print(f"Voiceover present: {'text' in state.get('voiceover', {})}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
