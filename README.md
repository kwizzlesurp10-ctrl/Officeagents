# Office-Cube AI Agent Workflow

Simulates a basic business office with AI agents. Orchestration routes tasks to workers like CEO, Accountant, Architect.

## Run Commands
- Start: `python app.py`
- Test: `python test_app.py`

## Env Vars
- `LOG_LEVEL`: DEBUG or INFO (default: INFO)

## Ports Used
- App: 0.0.0.0:8000 (access at http://localhost:8000)

## Test Instructions
Run `python test_app.py`—all tests should pass. Covers health, orchestration, validation.

For errors:
1. Check logs in terminal (JSON format).
2. Ensure Flask is installed (`pip install flask` assumed).
3. If port bind fails: Kill process on 8000, retry.
4. Invalid task: Keep 1-500 chars.
