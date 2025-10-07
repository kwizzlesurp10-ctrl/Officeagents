import os
from dotenv import load_dotenv
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from markupsafe import escape
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_orchestrator import run_graph

app = Flask(__name__)
CORS(app)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///office_cube.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load environment variables from a local .env if present
load_dotenv()

# Environment vars with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_TEMP = float(os.getenv("LANGCHAIN_TEMP", "0.7"))

# LangChain setup
if XAI_API_KEY:
    llm = ChatXAI(model="grok-beta", xai_api_key=XAI_API_KEY, temperature=LANGCHAIN_TEMP)
elif GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=LANGCHAIN_TEMP)
else:
    # Defer error until first usage to allow importing app without keys (for tests/health)
    llm = None

# Orchestration logic is now in Officeagents/graph_orchestrator.py

def log_message(level, message):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    print(json.dumps(log_entry))

def handle_secret_service(task):
    # Sanitize task by removing sensitive data patterns
    sanitized_task = re.sub(r'(password|api_key|credit_card)\s*[:=]?\s*\S+', '[REDACTED]', task, flags=re.IGNORECASE)
    return {
        "steps": ["Handled by Secret Service"],
        "response": f"Sensitive task '{sanitized_task}' has been processed securely.",
        "agents_involved": ["SecretService"]
    }

def orchestrate_with_langchain(task, max_chains=5):
    # Quick check for sensitive keywords to route to SecretService without LLM
    if re.search(r'(password|api_key|credit_card)', task, re.IGNORECASE):
        return handle_secret_service(task)

    try:
        final_response = run_graph(task)
        return {
            "steps": ["Graph orchestration complete"],
            "response": final_response,
            "agents_involved": ["GraphOrchestrator"]
        }
    except Exception as e:
        log_message("ERROR", f"LangChain orchestration error: {str(e)}")
        return {
            "steps": ["Fallback: Direct to Orchestration"],
            "response": f"Task '{task}' processed via fallback. (Error: {str(e)})",
            "agents_involved": ["Orchestration"]
        }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/orchestrate", methods=["POST"])
def orchestrate():
    if LOG_LEVEL == "DEBUG":
        log_message("DEBUG", "Orchestration request received")
    
    data = request.get_json()
    if not data or "task" not in data:
        abort(400, "Missing 'task' in request body")
    task_text = data["task"].strip()
    if len(task_text) > 500 or len(task_text) < 1:
        abort(400, "Task length must be 1-500 characters")
    
    max_chains = data.get("max_chains", 5)
    if not isinstance(max_chains, int) or max_chains < 1 or max_chains > 10:
        abort(400, "max_chains must be an integer between 1 and 10")

    result = orchestrate_with_langchain(escape(task_text), max_chains=max_chains)
    
    # Save to database
    new_task = Task(
        task=task_text,
        response=str(result['response']),
        steps=json.dumps(result['steps']),
        agents_involved=json.dumps(result['agents_involved'])
    )
    try:
        db.session.add(new_task)
        db.session.commit()
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            pass
        log_message("ERROR", f"DB write failed: {str(e)}")
    
    full_result = {
        "task": task_text,
        **result
    }
    
    log_message("INFO", f"Task processed and saved: {task_text[:50]}...")
    return jsonify(full_result), 200

@app.route("/healthz", methods=["GET"])
def healthz():
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
    with app.app_context():
        db.create_all()
    
    @app.after_request
    def after_request(response):
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
        return response
    
    app.run(host="0.0.0.0", port=8000, debug=True if LOG_LEVEL == "DEBUG" else False)
