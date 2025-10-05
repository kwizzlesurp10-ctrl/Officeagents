import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS  # Assumed installed; for future if split
from markupsafe import escape

app = Flask(__name__)
CORS(app)  # Safe for localhost dev; restrict in prod

# Environment vars with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Essential office workers and prompts
AGENTS = {
    "CEO": "You are the CEO of a basic business office. Provide high-level strategic advice, decisions, and leadership insights. Respond concisely to executive queries.",
    "Manager": "You are a department manager. Handle team coordination, task delegation, and progress updates. Focus on operational efficiency.",
    "Accountant": "You are the office accountant. Manage finances, budgets, invoices, and reports. Give accurate financial calculations and advice.",
    "HR": "You are the HR specialist. Deal with recruitment, employee relations, policies, and training. Ensure compliance and positivity.",
    "IT Support": "You are the IT support agent. Troubleshoot tech issues, software setups, and network problems. Provide step-by-step fixes.",
    "Sales Rep": "You are the sales representative. Pitch products, handle leads, negotiate deals, and track sales metrics. Be persuasive and data-driven.",
    "Secretary": "You are the office secretary. Schedule meetings, manage calendars, handle correspondence, and organize files. Be efficient and polite.",
    "Architect": "You are the Architect agent. Design office layouts, workflows, and structures. Suggest blueprints, improvements, and spatial optimizations with creative flair.",
    "Orchestration": "You are the Orchestration agent. Analyze the task, route it to the best-suited agent (or chain them), and compile a final response. List steps taken."
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

def simulate_agent_response(agent, task):
    """Simulate agent response using prompt + task. In real, integrate LLM."""
    # Placeholder simulation: echo with role flavor
    prompt = AGENTS[agent]
    response = f"{prompt} Task: {escape(task)}\nSimulated Response: As {agent}, I would handle this by [brief action]. Outcome: Resolved efficiently."
    return response

@app.route("/")
def index():
    """Serve the frontend."""
    return render_template("index.html")

@app.route("/orchestrate", methods=["POST"])
def orchestrate():
    """Main endpoint: Validate input, route, simulate chain."""
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
    primary_response = simulate_agent_response(primary_agent, task)
    
    # Simple chaining: if Architect, suggest to Manager
    if primary_agent == "Architect":
        manager_response = simulate_agent_response("Manager", f"Implement: {primary_response}")
        steps.append("Chained to Manager for implementation")
        final = manager_response
    else:
        final = primary_response
    
    result = {
        "task": task,
        "steps": steps,
        "response": final,
        "agents_involved": [primary_agent]
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
