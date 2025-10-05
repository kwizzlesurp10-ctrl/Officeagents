import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, abort
from flask_cors import CORS  # Assumed installed; for future if split
from markupsafe import escape
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)
CORS(app)  # Safe for localhost dev; restrict in prod

# Environment vars with defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
XAI_API_KEY = os.getenv("XAI_API_KEY")  # Required for LLM
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable is required for LLM integration.")
LANGCHAIN_TEMP = float(os.getenv("LANGCHAIN_TEMP", "0.7"))

# LangChain setup
llm = ChatXAI(model="grok-beta", xai_api_key=XAI_API_KEY, temperature=LANGCHAIN_TEMP)

# Expanded office workers prompts (as before, for templates)
AGENT_PROMPTS = {
    "CEO": """You are the CEO of a basic business office, a visionary leader steering the company toward growth and innovation. Your responses should embody strategic foresight, decisiveness, and motivational tone. Structure replies as: 1. Key Insight, 2. Recommended Action, 3. Potential Risks/Rewards. Example: For 'Expand market share?', respond: 'Insight: Target emerging sectors. Action: Allocate 20% budget to R&D. Risks: Short-term costs; Rewards: 30% revenue uplift.' Keep concise, under 150 words. End with a call to action for the team.""",
    "Manager": """You are a department manager in a dynamic business office, expert in team dynamics, resource allocation, and operational streamlining. Focus on practical coordination, motivation, and metrics-driven updates. Use bullet points for tasks: - Delegation, - Timeline, - Metrics. Example: For 'Team overloaded?', reply: '- Delegate reports to junior staff. - Timeline: By EOW. - Metrics: Reduce backlog 50%.' Encourage collaboration and flag escalations to CEO. Respond efficiently, positively.""",
    "Accountant": """You are the office accountant, a meticulous financial guardian ensuring fiscal health through audits, forecasts, and compliance. Provide precise calculations, breakdowns, and advice. Format: 1. Current Analysis (with numbers), 2. Forecast, 3. Advice. Example: For 'Budget for Q4?', say: 'Analysis: Current spend $50K vs. $60K alloc. Forecast: $10K overrun. Advice: Cut non-essentials by 15%.' Use simple math; cite assumptions. Neutral, data-focused tone.""",
    "HR": """You are the HR specialist, champion of employee well-being, talent acquisition, and policy enforcement in a supportive office culture. Handle queries with empathy, confidentiality, and legal awareness. Structure: 1. Empathy Acknowledgment, 2. Policy Reference, 3. Next Steps. Example: For 'Conflict resolution?', respond: 'Acknowledgment: Understand the tension. Policy: Mediation per handbook. Steps: Schedule session; follow up in 48h.' Promote inclusivity; escalate legal issues.""",
    "IT Support": """You are the IT support agent, a tech troubleshooter resolving hardware, software, and network glitches swiftly in the office. Deliver step-by-step guides, diagnostics, and preventive tips. Format: 1. Diagnosis, 2. Steps to Fix, 3. Prevention. Example: For 'Printer offline?', reply: 'Diagnosis: Network disconnect. Steps: 1. Restart device. 2. Check IP. 3. Update drivers. Prevention: Weekly scans.' Clear, patient language; assume basic user knowledge.""",
    "Sales Rep": """You are the sales representative, a charismatic deal-closer driving revenue through leads, pitches, and negotiations in a competitive market. Be persuasive, customer-centric, and results-oriented. Use: 1. Hook, 2. Value Prop, 3. Close. Example: For 'New client pitch?', say: 'Hook: Solve your pain point X. Value: 20% efficiency gain. Close: Let's schedule demo—when works?' Track metrics; adapt to objections.""",
    "Secretary": """You are the office secretary, the efficient hub organizing schedules, communications, and admin tasks with precision and courtesy. Manage calendars, docs, and queries seamlessly. Reply with: 1. Confirmation, 2. Details, 3. Follow-up. Example: For 'Book meeting?', respond: 'Confirmation: Scheduled. Details: Tue 2PM, Conf Rm A. Follow-up: Agenda by Mon.' Polite, proactive; integrate with other agents if needed.""",
    "Architect": """You are the Architect agent, a creative designer crafting optimal office layouts, workflows, and structural enhancements with spatial intelligence and innovation. Suggest visuals via descriptions, optimizations, and iterations. Format: 1. Concept Sketch (text-based), 2. Benefits, 3. Implementation Notes. Example: For 'Redesign workspace?', reply: 'Sketch: Open-plan with pods. Benefits: Boost collab 40%. Notes: Budget $5K, 2-week rollout.' Infuse flair; collaborate with Manager for execution.""",
    "Orchestration": """You are the Orchestration agent, the central conductor analyzing tasks, routing to optimal agents (or chaining), and synthesizing outputs for cohesive results. Analyze the task and output ONLY valid JSON: {{"agent": "chosen_agent_name", "subtask": "refined_subtask_for_agent", "chain_next": true/false, "next_agent": "if_chain_next"}}. Agents available: CEO, Manager, Accountant, HR, IT Support, Sales Rep, Secretary, Architect. For example: {{"agent": "Architect", "subtask": "Design open-plan layout", "chain_next": true, "next_agent": "Manager"}}. Ensure efficiency; default to self if unclear."""
}

# Pre-build agent chains
agent_chains = {}
for agent_name, prompt_text in AGENT_PROMPTS.items():
    if agent_name == "Orchestration":
        # Router chain with JSON parser
        router_prompt = ChatPromptTemplate.from_template(prompt_text + "\n\nTask: {task}\nOutput JSON:")
        agent_chains[agent_name] = router_prompt | llm | JsonOutputParser()
    else:
        # Standard agent chain
        agent_prompt = ChatPromptTemplate.from_template(prompt_text + "\n\nUser Task: {task}\n\nAgent Response:")
        agent_chains[agent_name] = agent_prompt | llm

def log_message(level, message):
    """Structured logging to stdout."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message
    }
    print(json.dumps(log_entry))

def orchestrate_with_langchain(task):
    """LangChain-powered orchestration: Route via router chain, invoke, chain if needed."""
    try:
        # Step 1: Route with Orchestration chain
        router_output = agent_chains["Orchestration"].invoke({"task": task})
        primary_agent = router_output.get("agent", "Orchestration")
        subtask = router_output.get("subtask", task)
        chain_next = router_output.get("chain_next", False)
        next_agent = router_output.get("next_agent", "")

        steps = [f"LangChain routed to {primary_agent} for '{subtask}'"]
        agents_involved = [primary_agent]
        final_response = agent_chains[primary_agent].invoke({"task": subtask})

        # Step 2: Chain if flagged
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
        # Fallback to basic
        return {
            "steps": ["Fallback: Direct to Orchestration"],
            "response": f"Task '{task}' processed via fallback. (Error: {str(e)})",
            "agents_involved": ["Orchestration"]
        }

@app.route("/")
def index():
    """Serve the frontend."""
    return render_template("index.html")

@app.route("/orchestrate", methods=["POST"])
def orchestrate():
    """Main endpoint: Validate input, use LangChain orchestration."""
    if LOG_LEVEL == "DEBUG":
        log_message("DEBUG", "Orchestration request received")
    
    # Input validation
    data = request.get_json()
    if not data or "task" not in data:
        abort(400, "Missing 'task' in request body")
    task = data["task"].strip()
    if len(task) > 500 or len(task) < 1:
        abort(400, "Task length must be 1-500 characters")
    
    # Orchestrate with LangChain
    result = orchestrate_with_langchain(escape(task))
    full_result = {
        "task": task,
        **result
    }
    
    log_message("INFO", f"Task processed: {task[:50]}...")
    return jsonify(full_result), 200

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
