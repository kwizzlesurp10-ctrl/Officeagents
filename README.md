# Office-Cube AI Agent Workflow

Simulates a basic business office with AI agents featuring expanded prompts, xAI LLM integration, and now LangChain-powered orchestration for dynamic routing and chaining.

## Run Commands
- Install deps: `pip install -r Officeagents/requirements.txt`
- Start: `python Officeagents/app.py`
- Test: `python Officeagents/test_app.py`

## Env Vars
- `GOOGLE_API_KEY`: Use Google Gemini via `langchain-google-genai`
- `XAI_API_KEY`: Use xAI Grok via `langchain-xai`
- `LOG_LEVEL`: DEBUG or INFO (default: INFO)
- `LANGCHAIN_TEMP`: LLM temperature (default: 0.7)

Set one of `GOOGLE_API_KEY` or `XAI_API_KEY`. You can export them in your shell:

```bash
export GOOGLE_API_KEY="your_google_api_key"
# or
export XAI_API_KEY="your_xai_api_key"
```

Alternatively, copy `.env.example` to `.env` and fill in values. The app automatically loads `.env` via `python-dotenv`.

## Ports Used
- App: 0.0.0.0:8000 (access at http://localhost:8000)

## Test Instructions
Run `python Officeagents/test_app.py` — tests cover health, LangChain orchestration (mocked), validation, prompts, and success/fallback paths.

For errors:
1. Check logs in terminal (JSON format).
2. "GOOGLE_API_KEY or XAI_API_KEY required": Set one of these env vars.
3. "No module named 'langchain'": `pip install langchain langchain-xai`.
4. If port bind fails: Kill process on 8000 (`lsof -ti:8000 | xargs kill -9`), retry.
5. Invalid task: Keep 1-500 chars.
6. LangChain/ API errors: Verify key; fallback used. Check temperature if outputs off.


Keiths Chain of Thought (CoT): The query "LangChain orchestration" builds on the Office-Cube workflow, aiming to replace heuristic routing with LangChain-powered dynamic orchestration for more intelligent agent coordination. Success criteria: Integrate LangChain with xAI's ChatXAI for all agents; use a LangChain agent (ReAct-style supervisor) in orchestration to decide routing/chaining based on task analysis; preserve Flask UI/API; ensure responses are structured and chained (e.g., Architect -> Manager). Output dynamic, LLM-driven decisions. Tree of Thought (ToT): Branch 1: Simple chains per agent (prompt | llm) + LLM router chain. Branch 2: Full multi-agent with tools (each agent as tool)—too complex for minimalism. Branch 3: Supervisor chain that outputs JSON {agent: str, chain: bool, subtask: str}, then parse/invoke. Selected Branch 1 for simplicity, with JSON parsing for routing. Trade-offs resolved: LangChain adds deps (assume pip install langchain langchain-xai); sync chains vs. async (sync for Flask). Self-refine: Ensure JSON output mode for router to avoid parsing errors; fallback to heuristic if LangChain fails; validate chains in tests. Proactive: Update README for new deps; add env for LangChain temp. Meets checklist: Goal met (LangChain orchestration), minimal overwrites, security intact (prompts escaped), tests mock LangChain, no prohibited files.
