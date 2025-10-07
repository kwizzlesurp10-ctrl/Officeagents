import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from markupsafe import escape
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)
CORS(app)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///office_cube.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Environment vars with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_TEMP = float(os.getenv("LANGCHAIN_TEMP", "0.7"))

# LangChain setup
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=LANGCHAIN_TEMP)
    print("Using Google Gemini Pro LLM")
elif XAI_API_KEY:
    llm = ChatXAI(model="grok-beta", xai_api_key=XAI_API_KEY, temperature=LANGCHAIN_TEMP)
    print("Using xAI Grok Beta LLM")
else:
    raise ValueError("Either GOOGLE_API_KEY or XAI_API_KEY environment variable is required for LLM integration.")

# Expanded office workers prompts
AGENT_PROMPTS = {
    "CEO": """You are the CEO of a basic business office...""",
    "Manager": """You are a department manager...""",
    "Accountant": """You are the office accountant...""",
    "HR": """You are the HR specialist...""",
    "IT Support": """You are the IT support agent...""",
    "Sales Rep": """You are the sales representative...""",
    "Secretary": """You are the office secretary...""",
    "Architect": """You are the Architect agent...""",
    "Orchestration": """You are the Orchestration agent..."""
}

# Task model
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task = db.Column(db.String(500), nullable=False)
    response = db.Column(db.Text, nullable=False)
    steps = db.Column(db.Text, nullable=False)
    agents_involved = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Task {self.id}: {self.task}>'

# Pre-build agent chains
agent_chains = {}
for agent_name, prompt_text in AGENT_PROMPTS.items():
    if agent_name == "Orchestration":
        router_prompt = ChatPromptTemplate.from_template(prompt_text + "\n\nTask: {task}\nOutput JSON:")
        agent_chains[agent_name] = router_prompt | llm | JsonOutputParser()
    else:
        agent_prompt = ChatPromptTemplate.from_template(prompt_text + "\n\nUser Task: {task}\n\nAgent Response:")
        agent_chains[agent_name] = agent_prompt | llm

def log_message(level, message):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    print(json.dumps(log_entry))

def orchestrate_with_langchain(task):
    try:
        router_output = agent_chains["Orchestration"].invoke({"task": task})
        primary_agent = router_output.get("agent", "Orchestration")
        subtask = router_output.get("subtask", task)
        chain_next = router_output.get("chain_next", False)
        next_agent = router_output.get("next_agent", "")

        steps = [f"LangChain routed to {primary_agent} for '{subtask}'"]
        agents_involved = [primary_agent]
        final_response = agent_chains[primary_agent].invoke({"task": subtask})

        if chain_next and next_agent in agent_chains:
            chain_subtask = f"Implement/follow up on: {final_response}"
            chain_response = agent_chains[next_agent].invoke({"task": chain_subtask})
            steps.append(f"Chained to {next_agent}")
            final_response = chain_response
            agents_involved.append(next_agent)

        return {
            "steps": steps,
            "response": final_response,
            "agents_involved": agents_involved
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
    
    result = orchestrate_with_langchain(escape(task_text))
    
    # Save to database
    new_task = Task(
        task=task_text,
        response=str(result['response']),
        steps=json.dumps(result['steps']),
        agents_involved=json.dumps(result['agents_involved'])
    )
    db.session.add(new_task)
    db.session.commit()
    
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
