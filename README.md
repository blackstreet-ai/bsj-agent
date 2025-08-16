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

### MCP integration (Tavily search + Firecrawl fetch)
Researcher stage can leverage MCP servers for search and content fetching. Configure env vars in `.env` (see `.env.example`).

Required env vars:
- `TAVILY_MCP_URL` (e.g. `https://mcp.tavily.com/mcp/?tavilyApiKey=${TAVILY_API_KEY}`)
- `TAVILY_API_KEY`
- `FIRECRAWL_MCP_URL` (e.g. `https://mcp.firecrawl.dev/${FIRECRAWL_API_KEY}/sse`)
- `FIRECRAWL_API_KEY`

Example run:
```bash
# Using venv console script
.venv/bin/bsj-run "AI in African fintech" --adk-multistage --print-json --debug

# Or with python -m and PYTHONPATH
PYTHONPATH=src python -m bsj_agent.run "AI in African fintech" --adk-multistage --print-json --debug
```

Notes:
- The pipeline will auto-detect MCP variables and attach toolsets to the researcher stage.
- If MCP servers are unreachable, the researcher falls back to model-only behavior (still enforced to output JSON), and validators/retries apply.

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
- Implement bsj_researcher → bsj_scriptwriter → (thumbnail+captioner) → bsj_voiceover
- Add human review gates between major stages
- Wire stub tools: web_search, fetch_url, elevenlabs_tts
- Add evaluation harness and example dataset
