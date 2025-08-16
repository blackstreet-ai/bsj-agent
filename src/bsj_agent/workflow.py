from __future__ import annotations

from typing import Any, Dict

# ADK imports for the async-first agent runner path. We orchestrate them
# synchronously by using InMemoryRunner which exposes a sync .run() wrapper.
try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except Exception:
    # Keep local stubs usable even if ADK is not installed; the ADK path will
    # be guarded when invoked.
    LlmAgent = None  # type: ignore
    InMemoryRunner = None  # type: ignore
    types = None  # type: ignore

from .agents import (
    bsj_researcher,
    bsj_scriptwriter,
    bsj_thumbnail_promptor,
    bsj_captioner,
    bsj_voiceover,
    bsj_newsletter_rewriter,
)


class BsjPipeline:
    """
    Minimal synchronous orchestration matching the charter:
    researcher -> review -> scriptwriter -> review -> (thumbnail + captioner) -> voiceover -> (optional) newsletter

    session structure is a dict with a "state" key used by each stage.
    """

    # Initialize the BSJ pipeline orchestrator
    def __init__(self, include_newsletter: bool = False) -> None:
        self.include_newsletter = include_newsletter
        # Instantiate agents
        self.researcher = bsj_researcher()
        self.scriptwriter = bsj_scriptwriter()
        self.thumbnail_promptor = bsj_thumbnail_promptor()
        self.captioner = bsj_captioner()
        self.voiceover = bsj_voiceover()
        self.newsletter_rewriter = bsj_newsletter_rewriter()

    # Run the BSJ pipeline
    def run(self, topic: str) -> Dict[str, Any]:
        session: Dict[str, Any] = {"state": {"topic": topic}}

        # Research
        session = self.researcher.run(session)
        self._human_review(session, stage="research")

        # Scriptwriting
        session = self.scriptwriter.run(session)
        self._human_review(session, stage="script")

        # Parallel branch (serialized here)
        session = self.thumbnail_promptor.run(session)
        session = self.captioner.run(session)

        # Voiceover
        session = self.voiceover.run(session)

        # Optional newsletter
        if self.include_newsletter:
            session = self.newsletter_rewriter.run(session)

        return session

    def _human_review(self, session: Dict[str, Any], stage: str) -> None:
        """
        Placeholder human review gate. In production, wire to UI/CLI confirmation.
        """
        session.setdefault("meta", {}).setdefault("reviews", []).append({
            "stage": stage,
            "status": "auto-approved (stub)",
        })


# === ADK Orchestration (Pattern A: Python sequencing) ===
def run_adk_pipeline(topic: str, include_newsletter: bool = False, *, debug: bool = False) -> tuple[Dict[str, Any] | None, str | None]:
    """
    Execute the BSJ pipeline using ADK `LlmAgent`s for each stage and route
    outputs to `session.state` using `output_key`.

    Returns (session_state_dict, error_message).

    Design:
    - We create a fresh InMemoryRunner per stage but carry forward the entire
      session.state by pre-creating the session with the previous state's dict.
      This avoids needing a custom shared session service instance.
    - Each agent reads from expected keys in state, and writes exactly one key
      via `output_key`.
    - After each stage we read back the session state and pass it forward.

    Note: The ADK agents are async-first; InMemoryRunner provides a sync facade.
    """
    if LlmAgent is None or InMemoryRunner is None or types is None:
        return None, "ADK not available (imports failed). Install google-adk."

    # Short run identifier derived from topic to make logs easy to correlate
    run_id = hex(abs(hash(("bsj", topic))) % (1 << 32))[2:]
    if debug:
        print(f"[RUN {run_id}] topic={topic}")

    # 1) Researcher — reads topic, writes research
    researcher = LlmAgent(
        name="bsj_researcher",
        model="gemini-2.5-flash",
        description="Research subtopics, key stats, and citations for the BSJ topic.",
        instruction=(
            "You are the BSJ researcher. Read session.state.topic and stay STRICTLY on that topic.\n"
            "Output ONLY JSON under 'research' with keys: topics[], key_stats[], citations[{title,url}].\n"
            "Do not reinterpret or change the topic. If unsure, acknowledge uncertainty but remain on-topic."
        ),
        output_key="research",
        tools=[],
    )

    state: Dict[str, Any] = {"topic": topic}
    if debug:
        print(f"[START:bsj_researcher] run={run_id}")
    state = _adk_run_single_stage(agent=researcher, state=state, expected_key="research", debug=debug, run_id=run_id)
    _validate_stage_output(state, key="research", expected_type=dict)

    # 2) Scriptwriter — reads research, writes script
    scriptwriter = LlmAgent(
        name="bsj_scriptwriter",
        model="gemini-2.5-flash",
        description="Transform research into a narrative script.",
        instruction=(
            "You are the BSJ scriptwriter. Use session.state.research.topics and session.state.research.key_stats to craft a BSJ-tone narrative.\n"
            "Respond with ONLY valid JSON (no markdown, no code fences, no prose).\n"
            "- beats: array of 5-8 short strings capturing the narrative beats\n"
            "- draft: a single string, 400-700 words, culturally grounded in BSJ voice\n"
            "- summary: a single string <= 60 words\n"
            "Output the object as {\"script\": { ... }}. Stay STRICTLY on session.state.topic. Do NOT mention unrelated domains (e.g., sports, specific teams)."
        ),
        output_key="script",
        tools=[],
    )
    if debug:
        print(f"[START:bsj_scriptwriter] run={run_id}")
    state = _adk_run_single_stage(agent=scriptwriter, state=state, expected_key="script", debug=debug, run_id=run_id)
    _validate_stage_output(state, key="script", expected_type=dict)
    # One-time retry if empty/invalid content
    if _script_is_empty(state.get("script", {})):
        if debug:
            print("[RETRY:bsj_scriptwriter] Detected empty script fields. Retrying with stricter format reminder.")
        scriptwriter_repair = LlmAgent(
            name="bsj_scriptwriter_repair",
            model="gemini-2.5-flash",
            description="Repair and complete script to required schema.",
            instruction=(
                "Your previous output did not satisfy the schema. Using session.state.research.topics and key_stats, "
                "respond with ONLY JSON (no markdown). Output exactly: {\"script\": {\"beats\": [5-8 strings], "
                "\"draft\": string 400-700 words, \"summary\": string <= 60 words}}."
            ),
            output_key="script",
            tools=[],
        )
        if debug:
            print(f"[START:bsj_scriptwriter_repair] run={run_id}")
        state = _adk_run_single_stage(agent=scriptwriter_repair, state=state, expected_key="script", debug=debug, run_id=run_id)
        _validate_stage_output(state, key="script", expected_type=dict)

    # Topic-grounding: if draft exists but appears off-topic relative to research topics, do a repair pass
    try:
        if _script_off_topic(state.get("script", {}), state.get("research", {}), state.get("topic", "")):
            if debug:
                print("[RETRY:bsj_scriptwriter] Off-topic detected vs research/topic. Retrying with strict on-topic constraint.")
            scriptwriter_on_topic = LlmAgent(
                name="bsj_scriptwriter_on_topic",
                model="gemini-2.5-flash",
                description="Ensure script aligns to topic and research.",
                instruction=(
                    "You must write ONLY JSON as {\"script\": {\"beats\":[], \"draft\": \"...\", \"summary\": \"...\"}}.\n"
                    "Stay STRICTLY on session.state.topic. Use at least 3 phrases from session.state.research.topics and reflect key_stats where natural.\n"
                    "Do not mention unrelated domains (e.g., sports if topic is fashion)."
                ),
                output_key="script",
                tools=[],
            )
            if debug:
                print(f"[START:bsj_scriptwriter_on_topic] run={run_id}")
            state = _adk_run_single_stage(agent=scriptwriter_on_topic, state=state, expected_key="script", debug=debug, run_id=run_id)
            _validate_stage_output(state, key="script", expected_type=dict)
    except Exception as e:
        if debug:
            print(f"[RETRY:bsj_scriptwriter] off-topic check error: {e}")
    _append_review(state, stage="script")

    # 3) Thumbnail prompts — reads script.summary, writes thumbnail_prompts
    thumbnail_promptor = LlmAgent(
        name="bsj_thumbnail_promptor",
        model="gemini-2.5-flash",
        description="Generate 3 Afrofuturist-style thumbnail prompts.",
        instruction=(
            "Read session.state.script.summary. Output JSON under 'thumbnail_prompts' as an array of 3 strings. "
            "No prose outside JSON."
        ),
        output_key="thumbnail_prompts",
        tools=[],
    )
    if debug:
        print(f"[START:bsj_thumbnail_promptor] run={run_id}")
    state = _adk_run_single_stage(agent=thumbnail_promptor, state=state, expected_key="thumbnail_prompts", debug=debug, run_id=run_id)
    _validate_stage_output(state, key="thumbnail_prompts", expected_type=list)

    # 4) Captioner — reads script.summary, writes captions
    captioner = LlmAgent(
        name="bsj_captioner",
        model="gemini-2.5-flash",
        description="Generate captions and hashtags for YouTube, TikTok, and Instagram.",
        instruction=(
            "Read session.state.script.summary and session.state.topic. Output ONLY JSON under 'captions' with keys youtube[], tiktok[], instagram[], hashtags[]. "
            "All content must stay on the given topic. No prose outside JSON. Provide at least 2 items for youtube, tiktok, instagram, and at least 8 hashtags."
        ),
        output_key="captions",
        tools=[],
    )
    if debug:
        print(f"[START:bsj_captioner] run={run_id}")
    state = _adk_run_single_stage(agent=captioner, state=state, expected_key="captions", debug=debug, run_id=run_id)
    _validate_stage_output(state, key="captions", expected_type=dict)
    # Retry captioner if any list is empty
    caps = state.get("captions", {}) if isinstance(state.get("captions"), dict) else {}
    def _empty_list(v):
        return not isinstance(v, list) or len(v) == 0
    if _empty_list(caps.get("youtube")) or _empty_list(caps.get("tiktok")) or _empty_list(caps.get("instagram")) or _empty_list(caps.get("hashtags")):
        if debug:
            print("[RETRY:bsj_captioner] Detected empty captions lists. Retrying with stricter format reminder.")
        captioner_retry = LlmAgent(
            name="bsj_captioner_retry",
            model="gemini-2.5-flash",
            description="Fill captions with topic-grounded content.",
            instruction=(
                "Respond with ONLY JSON under 'captions' containing non-empty arrays with at least 2 youtube, 2 tiktok, 2 instagram items, and >=8 hashtags.\n"
                "Each item must clearly refer to session.state.topic and reflect session.state.script.summary. Avoid generic filler."
            ),
            output_key="captions",
            tools=[],
        )
        if debug:
            print(f"[START:bsj_captioner_retry] run={run_id}")
        state = _adk_run_single_stage(agent=captioner_retry, state=state, expected_key="captions", debug=debug, run_id=run_id)
        _validate_stage_output(state, key="captions", expected_type=dict)

    # 5) Voiceover — reads script.draft, writes voiceover
    voiceover = LlmAgent(
        name="bsj_voiceover",
        model="gemini-2.5-flash",
        description="Transform script into voiceover-ready text.",
        instruction=(
            "Read session.state.script.draft and session.state.topic. Output ONLY JSON under 'voiceover' with key 'text' containing finalized, on-topic narration.\n"
            "Explicitly reflect at least 2 ideas from session.state.research.topics."
        ),
        output_key="voiceover",
        tools=[],
    )
    if debug:
        print(f"[START:bsj_voiceover] run={run_id}")
    state = _adk_run_single_stage(agent=voiceover, state=state, expected_key="voiceover", debug=debug, run_id=run_id)
    _validate_stage_output(state, key="voiceover", expected_type=dict)
    # Retry voiceover if empty or off-topic
    try:
        v = state.get("voiceover", {}) if isinstance(state.get("voiceover"), dict) else {}
        vtext = v.get("text", "") if isinstance(v, dict) else ""
        if not isinstance(vtext, str) or not vtext.strip() or _script_off_topic({"draft": vtext}, state.get("research", {}), state.get("topic", "")):
            if debug:
                print("[RETRY:bsj_voiceover] Empty or off-topic. Retrying with strict on-topic constraint.")
            voiceover_retry = LlmAgent(
                name="bsj_voiceover_retry",
                model="gemini-2.5-flash",
                description="Rewrite voiceover on-topic.",
                instruction=(
                    "Respond with ONLY JSON as {\"voiceover\": {\"text\": \"...\"}}.\n"
                    "Stay strictly on session.state.topic, and reference at least 2 phrases from session.state.research.topics in natural language."
                ),
                output_key="voiceover",
                tools=[],
            )
            if debug:
                print(f"[START:bsj_voiceover_retry] run={run_id}")
            state = _adk_run_single_stage(agent=voiceover_retry, state=state, expected_key="voiceover", debug=debug, run_id=run_id)
            _validate_stage_output(state, key="voiceover", expected_type=dict)
    except Exception as e:
        if debug:
            print(f"[RETRY:bsj_voiceover] check error: {e}")

    # 6) Optional newsletter — reads script + research, writes newsletter
    if include_newsletter:
        newsletter = LlmAgent(
            name="bsj_newsletter_rewriter",
            model="gemini-2.5-flash",
            description="Rewrite for newsletter and subject lines.",
            instruction=(
                "Read session.state.script and session.state.research. Output JSON under 'newsletter' with keys 'body' and 'subjects'[3]."
            ),
            output_key="newsletter",
            tools=[],
        )
        if debug:
            print(f"[START:bsj_newsletter_rewriter] run={run_id}")
        state = _adk_run_single_stage(agent=newsletter, state=state, expected_key="newsletter", debug=debug, run_id=run_id)
        _validate_stage_output(state, key="newsletter", expected_type=dict)

    # Wrap into session-like dict consistent with the rest of the app
    session: Dict[str, Any] = {"state": state}
    return session, None


def _adk_run_single_stage(*, agent: Any, state: Dict[str, Any], expected_key: str, debug: bool, run_id: str) -> Dict[str, Any]:
    """
    Helper to run a single `LlmAgent` stage using a fresh InMemoryRunner while
    carrying forward cumulative session state.

    Steps:
    - Create runner with the provided agent
    - Pre-create session using current state
    - Run with a small user prompt (agent reads state internally)
    - Read back state from the session service and return it
    """
    assert InMemoryRunner is not None and types is not None
    runner = InMemoryRunner(agent=agent)

    # Re-create session for this stage with the cumulative state
    try:
        runner._in_memory_session_service.create_session_sync(  # type: ignore[attr-defined]
            app_name=runner.app_name,
            user_id=f"bsj_user_{run_id}",
            session_id=f"bsj_{run_id}_{agent.name}",
            state=state,
        )
    except Exception:
        pass

    # Ground the agent with topic to reduce drift
    topic = state.get("topic", "")
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text=f"Proceed with your stage using session.state. Topic: {topic}")],
    )
    events_text: list[str] = []
    for event in runner.run(user_id=f"bsj_user_{run_id}", session_id=f"bsj_{run_id}_{agent.name}", new_message=user_msg):
        # Best-effort extract text from events
        txt = _extract_event_text(event)
        if txt:
            events_text.append(txt)
            if debug:
                print(f"[ADK:{agent.name}] {txt}")

    # Read back the session state, normalize to a plain dict, and inject parsed content if needed
    try:
        sess = runner._in_memory_session_service.get_session_sync(  # type: ignore[attr-defined]
            app_name=runner.app_name,
            user_id="bsj_user",
            session_id="bsj_session",
        )
        raw_state = getattr(sess, "state", {}) or {}
    except Exception:
        raw_state = {}

    # Normalize to a plain dict without relying on the underlying ADK type
    if isinstance(raw_state, dict):
        work_state: Dict[str, Any] = dict(raw_state)
    else:
        # Fallback to previous state; attempt shallow copy
        work_state = dict(state)

    # If ADK didn't persist, parse events and inject into expected_key
    if expected_key and expected_key not in work_state:
        parsed = _parse_json_from_texts(events_text)
        if parsed is not None:
            if isinstance(parsed, dict) and expected_key in parsed:
                work_state[expected_key] = parsed[expected_key]
            else:
                work_state[expected_key] = parsed

    # If the expected key is present but is a string, attempt to coerce to JSON
    if expected_key and isinstance(work_state.get(expected_key), str):
        coerced = _parse_json_from_texts([work_state[expected_key]])
        if coerced is not None:
            if isinstance(coerced, dict) and expected_key in coerced:
                work_state[expected_key] = coerced[expected_key]
            else:
                work_state[expected_key] = coerced

    if debug:
        ek_present = expected_key in work_state
        ek_type = type(work_state.get(expected_key)).__name__ if ek_present else None
        print(f"[STATE:{agent.name}] keys={list(work_state.keys())} expected_key={expected_key} present={ek_present} type={ek_type}")

    return work_state


def _extract_event_text(event: Any) -> str:
    """
    Extract textual content from an ADK event in a tolerant way.
    """
    try:
        content = getattr(event, "content", None)
        if content and getattr(content, "parts", None):
            texts = []
            for part in content.parts:
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)
            return "\n".join(texts)
        # Fallback to string repr
        return ""
    except Exception:
        return ""


def _parse_json_from_texts(texts: list[str]) -> Any:
    """
    Combine text chunks, strip code fences, and parse JSON if present.
    Returns a Python object or None.
    """
    import json
    if not texts:
        return None
    blob = "\n".join(texts).strip()
    # Strip fenced blocks if any
    if blob.startswith("```"):
        # remove first line and trailing fence
        lines = [ln for ln in blob.splitlines() if not ln.strip().startswith("```")]
        blob = "\n".join(lines).strip()
    # Try direct JSON
    try:
        return json.loads(blob)
    except Exception:
        pass
    # Try to locate first {...} or [...] segment
    import re
    m = re.search(r"(\{.*\}|\[.*\])", blob, flags=re.S)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _validate_stage_output(state: Dict[str, Any], *, key: str, expected_type: type) -> None:
    """
    Light schema guard: ensure that `state[key]` exists and matches `expected_type`.
    If missing or wrong type, create a safe default and note it for later inspection.
    """
    if key not in state or not isinstance(state.get(key), expected_type):
        # Create minimal safe defaults
        defaults = {dict: {}, list: [], str: "", int: 0, float: 0.0}
        state[key] = defaults.get(expected_type, None)
        state.setdefault("meta", {}).setdefault("validation", []).append({
            "key": key,
            "expected": expected_type.__name__,
            "status": "corrected_to_default",
        })


def _append_review(state: Dict[str, Any], *, stage: str) -> None:
    """Append a human review placeholder after a critical stage (e.g., script)."""
    state.setdefault("meta", {}).setdefault("reviews", []).append({
        "stage": stage,
        "status": "auto-approved (adk)",
    })


def _script_is_empty(script_obj: Dict[str, Any]) -> bool:
    """Return True if beats/draft/summary missing or empty."""
    if not isinstance(script_obj, dict):
        return True
    beats = script_obj.get("beats")
    draft = script_obj.get("draft")
    summary = script_obj.get("summary")
    if not isinstance(beats, list) or len(beats) < 3:
        return True
    if not isinstance(draft, str) or len(draft.strip()) < 100:
        return True
    if not isinstance(summary, str) or len(summary.strip()) < 10:
        return True
    return False


def _extract_keywords(research_obj: Dict[str, Any]) -> set[str]:
    """Extract a small set of lowercase keywords from research.topics for coarse matching."""
    kws: set[str] = set()
    try:
        topics = research_obj.get("topics", []) if isinstance(research_obj, dict) else []
        for t in topics:
            if not isinstance(t, str):
                continue
            for w in t.lower().split():
                w = ''.join(ch for ch in w if ch.isalnum())
                if len(w) >= 5:
                    kws.add(w)
    except Exception:
        pass
    return kws


def _script_off_topic(script_obj: Dict[str, Any], research_obj: Dict[str, Any], topic: str = "") -> bool:
    """Heuristic off-topic detector.
    Returns True if:
    - draft is empty, OR
    - none of the (research keywords UNION topic keywords) appear in the draft, OR
    - a small blacklist keyword (e.g., sports terms) appears while topic doesn't include that domain.
    """
    try:
        draft = script_obj.get("draft", "") if isinstance(script_obj, dict) else ""
        if not isinstance(draft, str) or not draft.strip():
            return True
        kws = _extract_keywords(research_obj)
        # Add topic-derived keywords (>=4 chars)
        if isinstance(topic, str) and topic:
            for w in topic.lower().split():
                w = ''.join(ch for ch in w if ch.isalnum())
                if len(w) >= 4:
                    kws.add(w)
        text = draft.lower()
        # Quick blacklist for known drift domains
        sports_blacklist = {"celtics", "heat", "nba", "playoffs", "season", "fenway", "patriots", "bruins", "redsox", "boston"}
        if isinstance(topic, str) and topic:
            topic_l = topic.lower()
            if not any(s in topic_l for s in sports_blacklist):
                if any(s in text for s in sports_blacklist):
                    return True
        if not kws:
            return False  # cannot judge, don't block
        return not any(kw in text for kw in kws)
    except Exception:
        return False
