# BSJ Agent

Black Street Journal (BSJ) multi-agent pipeline built on Google ADK.

## Prerequisites
- Python 3.10+
- Recommended: virtual environment
- GOOGLE_API_KEY environment variable for Gemini models (used by ADK). See `.env.example`.

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

### MCP integration (Tavily search + Firecrawl fetch)
Researcher stage can leverage MCP servers for search and content fetching. Configure env vars in `.env` (see `.env.example`).

Required env vars:
- `TAVILY_MCP_URL` (e.g. `https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}`)
- `TAVILY_API_KEY`
- `FIRECRAWL_MCP_URL` (e.g. `https://mcp.firecrawl.dev/${FIRECRAWL_API_KEY}/sse`)
- `FIRECRAWL_API_KEY`

## Running
Preferred (v2 ADK-native pipeline):
```bash
# Ensure GOOGLE_API_KEY is set in environment or .env
PYTHONPATH=src python -m bsj_agent_v2.runner.cli --topic "AI in African fintech"

# Example output: final session.state as pretty JSON (research/script/voiceover and Markdown keys)
```

Legacy (v1) runner is still available:
```bash
# Using venv console script
.venv/bin/bsj-run "AI in African fintech" --adk-multistage --print-json --debug

# Or with python -m and PYTHONPATH
PYTHONPATH=src python -m bsj_agent.run "AI in African fintech" --adk-multistage --print-json --debug
```

Notes:
- The pipeline will auto-detect MCP variables and attach toolsets to the researcher stage when configured.
- If MCP servers are unreachable, the researcher falls back to model-only behavior (JSON-structured outputs are still expected), and validators/retries apply.

## Development
Project layout:
```
src/
  bsj_agent/
    agents/                 # bsj_* agents
    tools/                  # tool_* stubs
    workflow.py             # Orchestration
    run.py                  # Entrypoint
  bsj_agent_v2/
    agents/
      researcher/
      scriptwriter/
      thumbnail_promptor/
      captioner/
      voiceover/
      review_gate/          # Human review gate agent factories
    runner/
      cli.py                # Minimal CLI for InMemoryRunner
    tools/
      mcp_utils.py          # Shared MCP helpers (if applicable)
    workflow.py             # ADK-native orchestration (v2)
```

### v2 Orchestration overview
The v2 pipeline composes ADK agents in `bsj_agent_v2/workflow.py`:

- researcher → research_review → scriptwriter → script_review → (thumbnail_promptor || captioner) → voiceover
- Human review gates are implemented via `review_gate` with these state keys:
  - `research_markdown` + `research_review` (approval/notes)
  - `script_markdown` + `script_review` (approval/notes)

Outputs are written back to `session.state` and rendered as Markdown alongside structured data, e.g. `voiceover` and `voiceover_markdown`.

### Troubleshooting: module import in editable mode
If `python -m bsj_agent.run ...` fails with `ModuleNotFoundError: No module named 'bsj_agent'` even after `pip install -e .`, add the source path to `PYTHONPATH` or use the venv console script:

```bash
# Option A: prepend src to PYTHONPATH when running from project root
PYTHONPATH=src python -m bsj_agent.run "Your topic" --adk-multistage --print-json

# Option B: use the console script from the virtualenv
.venv/bin/bsj-run "Your topic" --adk-multistage --print-json

# Option C: activate the venv first, then run
source .venv/bin/activate
python -m bsj_agent.run "Your topic" --adk-multistage --print-json
```

## Roadmap
- v2 agent graph implemented: researcher → review → scriptwriter → review → (thumbnail+captioner) → voiceover
- Human review gates implemented for research and script stages
- Wire stub tools: web_search, fetch_url, elevenlabs_tts (continue integration and tests)
- Add evaluation harness and example dataset
