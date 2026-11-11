import sqlite3
from typing import Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from .state import AgentState
from .config import SQLITE_DB_PATH
from .agents import (
    query_router_node,
    rag_agent_node,
    order_agent_node,
    support_agent_node,
    off_topic_agent_node,
    greeting_agent_node,
    validator_node
)


def route_by_intent(state: AgentState) -> str:
    """
    Router function that directs execution based on classified intent.
    """
    intent = state.get("intent", "faq_docs")
    if intent == "order_query":
        return "order_agent"
    elif intent == "complaint":
        return "support_agent"
    elif intent == "greeting":
        return "greeting_agent"
    elif intent == "off_topic":
        return "off_topic_agent"
    return "rag_agent"


def build_multiagent_graph():
    """
    Constructs and compiles the StateGraph workflow according to idea.txt:
    Query Router -> (RAG Agent | Order Tool | Support Agent) -> Validator -> Final Response
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("query_router", query_router_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("support_agent", support_agent_node)
    workflow.add_node("order_agent", order_agent_node)
    workflow.add_node("off_topic_agent", off_topic_agent_node)
    workflow.add_node("greeting_agent", greeting_agent_node)
    workflow.add_node("validator", validator_node)

    # 2. Add Start Edge -> Query Router
    workflow.add_edge(START, "query_router")

    # 3. Add Conditional Edge from Query Router to Branch Agents
    workflow.add_conditional_edges(
        "query_router",
        route_by_intent,
        {
            "rag_agent": "rag_agent",
            "order_agent": "order_agent",
            "support_agent": "support_agent",
            "greeting_agent": "greeting_agent",
            "off_topic_agent": "off_topic_agent",
        }
    )

    # 4. Connect all Branch Agents to Validator Node
    workflow.add_edge("rag_agent", "validator")
    workflow.add_edge("order_agent", "validator")
    workflow.add_edge("support_agent", "validator")
    workflow.add_edge("greeting_agent", "validator")
    workflow.add_edge("off_topic_agent", "validator")

    # 5. Validator Node terminates to END
    workflow.add_edge("validator", END)

    # 6. SQLite Persistence Checkpointer
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    app = workflow.compile(checkpointer=checkpointer)
    return app


# Create compiled graph instance
multiagent_app = build_multiagent_graph()
