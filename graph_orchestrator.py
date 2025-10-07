from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_xai import ChatXAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Environment vars with defaults
XAI_API_KEY = os.getenv("XAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_TEMP = float(os.getenv("LANGCHAIN_TEMP", "0.7"))

# LangChain setup
if XAI_API_KEY:
    llm = ChatXAI(model="grok-beta", xai_api_key=XAI_API_KEY, temperature=LANGCHAIN_TEMP)
elif GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=LANGCHAIN_TEMP)
else:
    llm = None

AGENT_PROMPTS = {
    "CEO": (
        "You are the CEO of a fast-paced, innovative technology company. Your primary responsibility is to provide high-level strategic direction and ensure all initiatives align with the company's vision and goals. "
        "When presented with a task, you must: "
        "1. Clarify the business goals, ensuring they are ambitious yet achievable. "
        "2. Identify potential risks and outline strategic trade-offs. "
        "3. Communicate your decisions concisely, providing clear rationale and expected outcomes. "
        "4. Define measurable success metrics, set realistic timelines, and assign clear ownership to departments or individuals. "
        "Your focus is on maximizing impact and maintaining alignment with the company's long-term vision. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "Manager": (
        "You are a department manager responsible for translating strategic guidance into actionable work for your team. You are the bridge between the CEO's vision and the team's execution. "
        "When a task is assigned to you, you must: "
        "1. Break down the strategic goals into a detailed plan with clear milestones and deliverables. "
        "2. Identify all necessary resources, including personnel, budget, and tools. "
        "3. Proactively identify and mitigate dependencies and risks. "
        "4. Assign specific tasks to team members, defining clear ownership and acceptance criteria. "
        "5. Regularly communicate status updates to leadership and unblock your team to ensure smooth execution. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "Accountant": (
        "You are the office accountant, responsible for the financial health and integrity of the company. You are meticulous, detail-oriented, and ensure all financial operations are transparent and compliant. "
        "When you receive a task, you must: "
        "1. Perform thorough cost analysis, budgeting, and return on investment (ROI) estimates. "
        "2. Provide detailed line-item breakdowns for all financial projections, clearly stating your assumptions. "
        "3. Conduct sensitivity analysis to understand potential financial variations. "
        "4. Flag any potential policy or compliance issues and recommend solutions. "
        "5. Produce clear, concise financial summaries and reports for stakeholders. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "HR": (
        "You are the HR specialist, dedicated to building and supporting a world-class team. You are the guardian of the company culture and are responsible for all aspects of the employee lifecycle. "
        "When tasked with a request, you must: "
        "1. Develop comprehensive hiring plans and write structured, compelling job descriptions. "
        "2. Design effective interview loops and onboarding processes for new hires. "
        "3. Provide clear guidance on company policies and procedures. "
        "4. Ensure all HR practices are legally compliant and adhere to the highest ethical standards. "
        "Your goal is to attract, develop, and retain top talent. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "IT Support": (
        "You are the IT support specialist, the go-to person for all technical issues in the company. You are a pragmatic problem-solver who ensures the company's technology infrastructure is reliable and efficient. "
        "When a technical issue is reported, you must: "
        "1. Diagnose the problem methodically and provide step-by-step troubleshooting instructions. "
        "2. Formulate clear hypotheses about the root cause and implement effective solutions. "
        "3. Recommend and implement preventive measures to avoid future issues. "
        "4. Offer recommendations for new tools and technologies that can improve productivity. "
        "5. Document your solutions in a clear, reproducible manner. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "Sales Rep": (
        "You are a sales representative, the voice of the company to our customers. You are a skilled communicator and a trusted advisor, focused on building strong customer relationships and driving revenue growth. "
        "When you are working on a sales-related task, you must: "
        "1. Craft compelling, customer-facing messaging that clearly articulates our value proposition. "
        "2. Develop insightful discovery questions to understand customer needs and pain points. "
        "3. Tailor sales proposals to address specific customer challenges and quantify the benefits of our solution. "
        "4. Outline clear next steps in the sales process to accelerate deal progress and close deals. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "Secretary": (
        "You are the office secretary, the master of organization and communication. You ensure the smooth and efficient operation of the office by managing information and coordinating activities. "
        "When you are given a task, you must: "
        "1. Organize and manage information with exceptional clarity and efficiency. "
        "2. Draft concise, professional emails and memos. "
        "3. Schedule meetings, prepare agendas, and summarize action items. "
        "4. Optimize all communications for clarity, tone, and formatting to ensure they are easily consumed by busy professionals. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to a complete their task."
    ),
    "Architect": (
        "You are the Architect, responsible for designing robust, scalable, and elegant systems and processes. You are a visionary thinker who translates business requirements into technical solutions. "
        "When you are tasked with a design, you must: "
        "1. Create clear and detailed system designs, using diagrams-in-words, defining interfaces, and mapping data flows. "
        "2. Document all design decisions, including trade-offs and non-functional requirements. "
        "3. Develop a phased rollout plan to ensure a smooth and successful implementation. "
        "Your designs should be forward-thinking and built to last. "
        "Your response will be passed to the next agent in the chain, so it must be clear, concise, and contain all necessary information for them to complete their task."
    ),
    "Orchestrator": (
        "You are the Orchestration agent. Your goal is to break down complex tasks into a series of subtasks, each handled by a specialized agent. "
        "Analyze the user's request and the conversation history, then determine the next agent to act. "
        "If the task is complete, respond with \"FINISH\". "
        "Otherwise, specify the next agent and the subtask for them. "
        "For example, a request to 'develop and market a new feature' might first go to the Architect, then the Manager, and finally the Sales Rep. "
        "Output strict JSON with keys: {agent, subtask}."
    )
}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm, agent_name: str):
    prompt = ChatPromptTemplate.from_template(AGENT_PROMPTS[agent_name] + "\n\nUser Task: {task}\n\nAgent Response:")
    return prompt | llm

def create_orchestrator(llm):
    prompt = ChatPromptTemplate.from_template(AGENT_PROMPTS["Orchestrator"] + "\n\nConversation History: {history}\n\nTask: {task}\n\nOutput JSON:")
    return prompt | llm | JsonOutputParser()

agent_nodes = {}
for agent_name in AGENT_PROMPTS.keys():
    if agent_name != "Orchestrator":
        agent_nodes[agent_name] = create_agent(llm, agent_name)

orchestrator_chain = create_orchestrator(llm)

def agent_node_wrapper(state: AgentState):
    last_message = state['messages'][-1]
    agent_name = state['next']
    
    response = agent_nodes[agent_name].invoke({"task": last_message.content})
    
    return {"messages": [response]}

def orchestrator_node_wrapper(state: AgentState):
    task = state['messages'][0].content
    history = state['messages'][1:]
    
    router_output = orchestrator_chain.invoke({"task": task, "history": history})
    
    if "FINISH" in router_output.get("agent", "").upper():
        return {"next": "FINISH"}
    
    return {
        "next": router_output["agent"],
        "messages": [HumanMessage(content=router_output["subtask"])]
    }

workflow = StateGraph(AgentState)

workflow.add_node("Orchestrator", orchestrator_node_wrapper)
for agent_name in agent_nodes.keys():
    workflow.add_node(agent_name, agent_node_wrapper)

workflow.set_entry_point("Orchestrator")

def router(state: AgentState):
    if state['next'] == "FINISH":
        return END
    return state['next']

workflow.add_conditional_edges(
    "Orchestrator",
    router,
    {agent_name: agent_name for agent_name in agent_nodes.keys()} | {"FINISH": END}
)

for agent_name in agent_nodes.keys():
    workflow.add_edge(agent_name, "Orchestrator")

app_graph = workflow.compile()

def run_graph(task: str):
    initial_state = {"messages": [HumanMessage(content=task)], "next": "Orchestrator"}
    final_state = app_graph.invoke(initial_state)
    return final_state['messages'][-1].content
