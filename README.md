# BSJ Agent

Black Street Journal (BSJ) multi-agent pipeline built on Google ADK.

## Prerequisites
- Python 3.10+
- Recommended: virtual environment

## Setup
```bash
# From this folder
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip

# Install dependencies
pip install -e .
# Or if you want to co-develop with a local ADK checkout:
# pip install -e ../adk-python
```

## Development
Project layout:
```
src/
  bsj_agent/
    agents/                 # bsj_* agents
    tools/                  # tool_* stubs
    workflow.py             # Orchestration
    run.py                  # Entrypoint
```

## Running
```bash
# Ensure google-adk is installed by pip (installed via pyproject)
python -m bsj_agent.run
```

If google-adk isn't installed yet, the runner will warn and exit gracefully.

## Roadmap
- Implement bsj_researcher → bsj_scriptwriter → (thumbnail+captioner) → bsj_voiceover
- Add human review gates between major stages
- Wire stub tools: web_search, fetch_url, elevenlabs_tts
- Add evaluation harness and example dataset
