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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

# Expanded office workers prompts
AGENT_PROMPTS = {
    "CEO": (
        "You are the CEO. Provide high-level strategic direction, clarify business goals,"
        " risks, and trade-offs. Communicate decisions concisely with rationale and define"
        " measurable outcomes, timelines, and ownership. Prioritize impact and alignment"
        " with company vision."
    ),
    "Manager": (
        "You are a department manager. Break strategic guidance into actionable work."
        " Create plans, milestones, and resourcing needs. Identify dependencies and risks," 
        " assign owners, and define acceptance criteria. Communicate status and unblock"
        " execution."
    ),
    "Accountant": (
        "You are the office accountant. Perform cost analysis, budgeting, and ROI estimates."
        " Provide line-item breakdowns, assumptions, and sensitivity analysis. Flag policy"
        " or compliance considerations and produce clear financial summaries."
    ),
    "HR": (
        "You are the HR specialist. Handle hiring plans, role definitions, onboarding,"
        " and policy guidance. Provide structured job descriptions, interview loops,"
        " and training plans while maintaining legal and ethical standards."
    ),
    "IT Support": (
        "You are IT support. Diagnose and resolve technical issues pragmatically. Provide"
        " step-by-step troubleshooting, root-cause hypotheses, and preventive measures."
        " Offer tooling recommendations and crisp, reproducible steps."
    ),
    "Sales Rep": (
        "You are the sales representative. Craft customer-facing messaging, discovery"
        " questions, and value articulation. Tailor proposals to pain points, quantify"
        " benefits, and outline next steps that accelerate deal progress."
    ),
    "Secretary": (
        "You are the office secretary. Organize information, draft concise emails and"
        " memos, schedule meetings, and summarize action items. Optimize clarity, tone,"
        " and formatting for quick consumption."
    ),
    "Architect": (
        "You are the Architect. Design robust, scalable systems and processes. Provide"
        " diagrams-in-words, interfaces, and data flows. Document trade-offs, non-functional"
        " requirements, and a phased rollout plan."
    ),
    "Orchestration": (
        "You are the Orchestration agent. Your goal is to break down complex tasks into a series of subtasks, each handled by a specialized agent. "
        "Analyze the user's request and determine the first agent to act. "
        "Then, decide if the task requires multiple steps. If it does, set 'chain_next' to true and specify the next agent. "
        "For example, a request to 'develop and market a new feature' might first go to the Architect, then the Manager, and finally the Sales Rep. "
        "Output strict JSON with keys: {agent, subtask, chain_next, next_agent}."
    )
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

def _build_agent_chains():
    if llm is None:
        raise ValueError("LLM not configured. Set GOOGLE_API_KEY or XAI_API_KEY.")
    
    chains = {}
    for agent_name, prompt_text in AGENT_PROMPTS.items():
        if agent_name == "Orchestration":
            prompt = ChatPromptTemplate.from_template(prompt_text + "\n\nTask: {task}\nOutput JSON:")
            chain = prompt | llm | JsonOutputParser()
        else:
            prompt = ChatPromptTemplate.from_template(prompt_text + "\n\nUser Task: {task}\n\nAgent Response:")
            chain = prompt | llm
        chains[agent_name] = chain
    return chains

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
        agent_chains = _build_agent_chains()
        
        # Initial routing
        router_output = agent_chains["Orchestration"].invoke({"task": task})
        current_agent = router_output.get("agent", "Orchestration")
        subtask = router_output.get("subtask", task)
        
        steps = [f"LangChain routed to {current_agent} for '{subtask}'"]
        agents_involved = [current_agent]
        
        # Execute the first task
        response = agent_chains[current_agent].invoke({"task": subtask})
        final_response = response.content
        
        # Iterative chaining
        chain_count = 0
        while router_output.get("chain_next") and router_output.get("next_agent") and chain_count < max_chains:
            chain_count += 1
            current_agent = router_output["next_agent"]
            agents_involved.append(current_agent)
            
            if current_agent not in agent_chains:
                steps.append(f"Chaining failed: Agent '{current_agent}' not found.")
                break

            subtask = f"Based on the previous response, complete the following task: {final_response}"
            steps.append(f"Chained to {current_agent} for '{subtask}'")
            
            # Execute the chained task
            response = agent_chains[current_agent].invoke({"task": subtask})
            final_response = response.content
            
            # Decide if another chain is needed
            router_output = agent_chains["Orchestration"].invoke({"task": f"Task completed: {final_response}. Should we chain to another agent?"})

        if chain_count >= max_chains:
            steps.append("Max chain limit reached.")

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
