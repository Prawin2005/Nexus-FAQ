import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Ensure src/ is on sys.path so multiagent package resolves properly
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from langchain_core.messages import HumanMessage
from multiagent.graph import multiagent_app

app = FastAPI(
    title="Custom Multi-Agent Support System",
    description="Multi-Agent system based on idea.txt (Router -> RAG/Order/Support -> Validator -> Final Response).",
    version="0.2.0"
)

# Static files directory
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., example="Where is my order ORD12345?")
    thread_id: Optional[str] = Field(default="session_default", example="session_101")
    customer_id: Optional[str] = Field(default="CUST101", example="CUST101")


class ChatResponse(BaseModel):
    thread_id: str
    intent: str
    reply: str
    validation_status: str
    agent_logs: List[Dict[str, Any]]


@app.get("/")
async def serve_index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Multi-Agent System API is operational."}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    config = {"configurable": {"thread_id": req.thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=req.message.strip())],
        "draft_response": None,
        "validated_response": None,
        "validation_status": None,
        "validator_feedback": None,
        "intent": None,
        "customer_id": req.customer_id,
        "agent_logs": []
    }

    try:
        final_state = multiagent_app.invoke(initial_state, config=config)

        validated_reply = final_state.get("validated_response") or final_state.get("draft_response") or "Thank you for contacting customer support."

        return ChatResponse(
            thread_id=req.thread_id,
            intent=final_state.get("intent", "faq_docs"),
            reply=validated_reply,
            validation_status=final_state.get("validation_status", "APPROVED"),
            agent_logs=final_state.get("agent_logs", [])
        )
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "checkpointer": "sqlite", "flow": "idea.txt"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
