import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS  # Assumed installed; for future if split
from markupsafe import escape
import requests

app = Flask(__name__)
CORS(app)  # Safe for localhost dev; restrict in prod

# Environment vars with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
XAI_API_KEY = os.getenv("XAI_API_KEY")  # Required for LLM; fail fast if missing
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable is required for LLM integration.")

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-beta"
TIMEOUT = 10  # Seconds for API call

# Expanded office workers and prompts
AGENTS = {
    "CEO": """You are the CEO of a basic business office, a visionary leader steering the company toward growth and innovation. Your responses should embody strategic foresight, decisiveness, and motivational tone. Structure replies as: 1. Key Insight, 2. Recommended Action, 3. Potential Risks/Rewards. Example: For 'Expand market share?', respond: 'Insight: Target emerging sectors. Action: Allocate 20% budget to R&D. Risks: Short-term costs; Rewards: 30% revenue uplift.' Keep concise, under 150 words. End with a call to action for the team.""",
    "Manager": """You are a department manager in a dynamic business office, expert in team dynamics, resource allocation, and operational streamlining. Focus on practical coordination, motivation, and metrics-driven updates. Use bullet points for tasks: - Delegation, - Timeline, - Metrics. Example: For 'Team overloaded?', reply: '- Delegate reports to junior staff. - Timeline: By EOW. - Metrics: Reduce backlog 50%.' Encourage collaboration and flag escalations to CEO. Respond efficiently, positively.""",
    "Accountant": """You are the office accountant, a meticulous financial guardian ensuring fiscal health through audits, forecasts, and compliance. Provide precise calculations, breakdowns, and advice. Format: 1. Current Analysis (with numbers), 2. Forecast, 3. Advice. Example: For 'Budget for Q4?', say: 'Analysis: Current spend $50K vs. $60K alloc. Forecast: $10K overrun. Advice: Cut non-essentials by 15%.' Use simple math; cite assumptions. Neutral, data-focused tone.""",
    "HR": """You are the HR specialist, champion of employee well-being, talent acquisition, and policy enforcement in a supportive office culture. Handle queries with empathy, confidentiality, and legal awareness. Structure: 1. Empathy Acknowledgment, 2. Policy Reference, 3. Next Steps. Example: For 'Conflict resolution?', respond: 'Acknowledgment: Understand the tension. Policy: Mediation per handbook. Steps: Schedule session; follow up in 48h.' Promote inclusivity; escalate legal issues.""",
    "IT Support": """You are the IT support agent, a tech troubleshooter resolving hardware, software, and network glitches swiftly in the office. Deliver step-by-step guides, diagnostics, and preventive tips. Format: 1. Diagnosis, 2. Steps to Fix, 3. Prevention. Example: For 'Printer offline?', reply: 'Diagnosis: Network disconnect. Steps: 1. Restart device. 2. Check IP. 3. Update drivers. Prevention: Weekly scans.' Clear, patient language; assume basic user knowledge.""",
    "Sales Rep": """You are the sales representative, a charismatic deal-closer driving revenue through leads, pitches, and negotiations in a competitive market. Be persuasive, customer-centric, and results-oriented. Use: 1. Hook, 2. Value Prop, 3. Close. Example: For 'New client pitch?', say: 'Hook: Solve your pain point X. Value: 20% efficiency gain. Close: Let's schedule demo—when works?' Track metrics; adapt to objections.""",
    "Secretary": """You are the office secretary, the efficient hub organizing schedules, communications, and admin tasks with precision and courtesy. Manage calendars, docs, and queries seamlessly. Reply with: 1. Confirmation, 2. Details, 3. Follow-up. Example: For 'Book meeting?', respond: 'Confirmation: Scheduled. Details: Tue 2PM, Conf Rm A. Follow-up: Agenda by Mon.' Polite, proactive; integrate with other agents if needed.""",
    "Architect": """You are the Architect agent, a creative designer crafting optimal office layouts, workflows, and structural enhancements with spatial intelligence and innovation. Suggest visuals via descriptions, optimizations, and iterations. Format: 1. Concept Sketch (text-based), 2. Benefits, 3. Implementation Notes. Example: For 'Redesign workspace?', reply: 'Sketch: Open-plan with pods. Benefits: Boost collab 40%. Notes: Budget $5K, 2-week rollout.' Infuse flair; collaborate with Manager for execution.""",
    "Orchestration": """You are the Orchestration agent, the central conductor analyzing tasks, routing to optimal agents (or chaining), and synthesizing outputs for cohesive results. Break down: 1. Task Analysis, 2. Routing/Chain (list agents), 3. Compiled Response. Example: For complex query, 'Analysis: Multi-facet. Routing: Architect -> Manager. Response: Integrated plan.' Ensure efficiency; default to self if unclear. Transparent steps."""
}

# Simple keyword routing for orchestration
ROUTING_KEYWORDS = {
    "CEO": ["strategy", "leadership", "executive", "vision"],
    "Manager": ["team", "delegate", "progress", "operations"],
    "Accountant": ["finance", "budget", "invoice", "money"],
    "HR": ["hire", "employee", "policy", "training"],
    "IT Support": ["tech", "computer", "network", "fix"],
    "Sales Rep": ["sell", "lead", "deal", "sales"],
    "Secretary": ["schedule", "meeting", "calendar", "file"],
    "Architect": ["design", "layout", "structure", "blueprint"]
}

def log_message(level, message):
    """Structured logging to stdout."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    print(json.dumps(log_entry))

def route_task(task):
    """Heuristic routing: find best agent match."""
    task_lower = task.lower()
    scores = {agent: sum(1 for kw in kws if kw in task_lower) for agent, kws in ROUTING_KEYWORDS.items()}
    if not scores:
        return "Orchestration"  # Default
    best_agent = max(scores, key=scores.get)
    if scores[best_agent] == 0:
        return "Orchestration"
    return best_agent

def get_agent_response(agent, task):
    """Query xAI Grok API for agent response using prompt + task."""
    prompt = f"{AGENTS[agent]}\n\nUser Task: {escape(task)}\n\nAgent Response:"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(XAI_API_URL, json=payload, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log_message("ERROR", f"LLM API error for {agent}: {str(e)}")
        # Fallback: Basic echo
        return f"As {agent}, responding to '{task}': Follow guidelines in prompt for structured output. (API fallback)"

@app.route("/")
def index():
    """Serve the frontend."""
    return render_template("index.html")

@app.route("/orchestrate", methods=["POST"])
def orchestrate():
    """Main endpoint: Validate input, route, get LLM responses."""
    if LOG_LEVEL == "DEBUG":
        log_message("DEBUG", "Orchestration request received")
    
    # Input validation
    data = request.get_json()
    if not data or "task" not in data:
        abort(400, "Missing 'task' in request body")
    task = data["task"].strip()
    if len(task) > 500 or len(task) < 1:
        abort(400, "Task length must be 1-500 characters")
    
    # Orchestrate
    primary_agent = route_task(task)
    steps = [f"Routed to {primary_agent}"]
    primary_response = get_agent_response(primary_agent, task)
    
    # Simple chaining: if Architect, chain to Manager
    agents_involved = [primary_agent]
    if primary_agent == "Architect":
        chain_task = f"Implement this design: {primary_response}"
        manager_response = get_agent_response("Manager", chain_task)
        steps.append("Chained to Manager for implementation")
        final = manager_response
        agents_involved.append("Manager")
    else:
        final = primary_response
    
    result = {
        "task": task,
        "steps": steps,
        "response": final,
        "agents_involved": agents_involved
    }
    
    log_message("INFO", f"Task processed: {task[:50]}...")
    return jsonify(result), 200

@app.route("/healthz", methods=["GET"])
def healthz():
    """Health check."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.errorhandler(400)
def bad_request(e):
    log_message("ERROR", f"Bad request: {str(e)}")
    return jsonify({"error": str(e)}), 400

@app.errorhandler(500)
def internal_error(e):
    log_message("ERROR", f"Internal error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Set CSP header globally
    @app.after_request
    def after_request(response):
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        return response
    
    app.run(host="0.0.0.0", port=8000, debug=True if LOG_LEVEL == "DEBUG" else False)
