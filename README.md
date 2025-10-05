# Office-Cube AI Agent Workflow

Simulates a basic business office with AI agents featuring expanded prompts and real LLM integration via xAI Grok API for dynamic responses. Orchestration routes tasks to workers like CEO, Accountant, Architect.

## Run Commands
- Start: `python app.py`
- Test: `python test_app.py`

## Env Vars
- `LOG_LEVEL`: DEBUG or INFO (default: INFO)
- `XAI_API_KEY`: Required; get from https://x.ai/api (set via export XAI_API_KEY=your_key)

## Ports Used
- App: 0.0.0.0:8000 (access at http://localhost:8000)

## Test Instructions
Run `python test_app.py`—all tests should pass. Covers health, orchestration (with mocked API), validation, prompts, and LLM success/fallback.

For errors:
1. Check logs in terminal (JSON format).
2. "XAI_API_KEY required": Set env var with your xAI API key from https://x.ai/api.
3. Ensure Flask and requests installed (`pip install flask requests` assumed).
4. If port bind fails: Kill process on 8000 (`lsof -ti:8000 | xargs kill -9`), retry.
5. Invalid task: Keep 1-500 chars.
6. API timeout/errors: Check key validity; fallback used automatically.
