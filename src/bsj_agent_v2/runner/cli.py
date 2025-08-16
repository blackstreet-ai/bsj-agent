"""
bsj_agent_v2.runner.cli

Minimal CLI to run the BSJ v2 pipeline with ADK InMemoryRunner.
- Accepts --topic
- Prints final session.state as JSON
- Gemini API by default (GOOGLE_API_KEY required)

Usage:
  PYTHONPATH=src python3 -m bsj_agent_v2.runner.cli --topic "AI in African fintech"
"""
from __future__ import annotations

import argparse
import json
import os

try:
    from google.adk.runners import InMemoryRunner
    from google.genai import types
except Exception as e:  # pragma: no cover
    raise

from ..workflow import build_root


def main():
    parser = argparse.ArgumentParser(description="Run BSJ v2 pipeline")
    parser.add_argument("--topic", required=True, help="Topic for the pipeline")
    args = parser.parse_args()

    # Build the root agent graph. Set debug=True to see tool callbacks.
    agent = build_root(debug=True)
    runner = InMemoryRunner(agent=agent)

    user = "bsj_cli_user"
    sid = "bsj_cli_session"

    # Initialize session with initial state
    try:
        runner._in_memory_session_service.create_session_sync(
            app_name=runner.app_name,
            user_id=user,
            session_id=sid,
            state={"topic": args.topic},
        )
    except Exception:
        pass

    # Kick off a single turn (message content is not used by our agents; they read session.state)
    msg = types.Content(role="user", parts=[types.Part(text="Run using session.state")])
    for _ in runner.run(user_id=user, session_id=sid, new_message=msg):
        pass

    sess = runner._in_memory_session_service.get_session_sync(
        app_name=runner.app_name, user_id=user, session_id=sid
    )
    print(json.dumps(getattr(sess, "state", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
