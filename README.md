# Office-Cube AI Agent Workflow

Simulates a basic business office with AI agents featuring expanded prompts, xAI LLM integration, and now LangChain-powered orchestration for dynamic routing and chaining.

## Run Commands
- Install deps: `pip install flask langchain langchain-xai` (new for LangChain)
- Start: `python app.py`
- Test: `python test_app.py`

## Env Vars
- `LOG_LEVEL`: DEBUG or INFO (default: INFO)
- `XAI_API_KEY`: Required; get from https://x.ai/api (set via export XAI_API_KEY=your_key)
- `LANGCHAIN_TEMP`: LLM temperature (default: 0.7)

## Ports Used
- App: 0.0.0.0:8000 (access at http://localhost:8000)

## Test Instructions
Run `python test_app.py`—all tests should pass. Covers health, LangChain orchestration (mocked), validation, prompts, success/fallback.

For errors:
1. Check logs in terminal (JSON format).
2. "XAI_API_KEY required": Set env var with your xAI API key from https://x.ai/api.
3. "No module named 'langchain'": `pip install langchain langchain-xai`.
4. If port bind fails: Kill process on 8000 (`lsof -ti:8000 | xargs kill -9`), retry.
5. Invalid task: Keep 1-500 chars.
6. LangChain/ API errors: Verify key; fallback used. Check temperature if outputs off.
