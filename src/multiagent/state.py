from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages ]

    intent: Optional[str]
    draft_response: Optional[str]
    validated_response: Optional[str]
    validation_status: Optional[str]
    validator_feedback: Optional[str]
    agent_logs: Annotated[List[Dict[str, Any]], operator.add]
    customer_id: Optional[str]