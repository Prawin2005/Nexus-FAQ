# 🤖 Nexus — Autonomous Multi-Agent Customer Support System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Guardrails](https://img.shields.io/badge/Guardrails-Deterministic%20Validation-emerald.svg)]()

> A production-ready, autonomous multi-agent customer support architecture powered by **LangGraph**, **FastAPI**, **ChromaDB**, and **SQLite checkpointing**. The system orchestrates intent routing, dense vector retrieval (RAG), live order tracking & refund actions, and strict anti-hallucination validation before delivering responses to the customer.

---

## 🏛️ System Architecture

The system coordinates specialized agents and deterministic guardrails through a stateful cyclical graph:

```text
                           [ USER QUERY ]
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │     Query Router     │
                      │  (Intent Classifier) │
                      └──────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
  │  RAG Agent   │        │  Order Tool  │        │Support Agent │
  │ (FAQ & Docs) │        │ (Order Track)│        │(Complaints & │
  │  [ChromaDB]  │        │ [Live API]   │        │   Refunds)   │
  └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │    Validator Node    │
                      │ (Anti-Hallucination  │
                      │  & Quality Checker)  │
                      └──────────┬───────────┘
                                 │
                                 ▼
                     [ VALIDATED RESPONSE ]
```

---

## ✨ Key Architectural Highlights

- **🎯 Intelligent Intent Routing:** Automatically categorizes customer inquiries into FAQ inquiries, Order Status queries, Complaint/Refund requests, Greetings, or Off-Topic inquiries.
- **📚 Dense Vector Retrieval (RAG):** Uses **ChromaDB** with OpenAI embeddings (`text-embedding-3-small`) to retrieve verified policies, return guidelines, payment rules, and shipping SLA documents.
- **🛡️ Strict Anti-Hallucination Guardrails:** Dedicated Validator Node cross-references draft responses against verified facts and order numbers to prevent hallucinations (e.g. inventing unverified order IDs).
- **💾 Fault-Tolerant Session Persistence:** Stateful conversation tracking via **SqliteSaver** checkpointing retains multi-turn context across user interactions.
- **⚡ Dual-Engine Support:** Configurable model provider supporting both **OpenAI** (`gpt-4o-mini`) and **Google Gemini** (`gemini-2.5-flash`).
- **💻 Interactive Support Portal:** Modern customer portal with real-time trace inspection, token streaming, and multi-agent execution logging.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | Asynchronous API endpoints & static asset serving |
| **Agent Orchestration** | **LangGraph** | Stateful cyclical graph state machine |
| **Vector Database** | **ChromaDB** | Semantic vector storage & similarity search |
| **LLM Provider** | **OpenAI / Google Gemini** | Context synthesis & domain-specific responses |
| **State Checkpointer** | **SQLite** | Multi-turn persistent thread retention |
| **Frontend UI** | **Vanilla HTML5 / CSS3 / JS** | Responsive customer support chat widget & portal |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Prawin2005/YOUR_REPO_NAME.git
cd MultiAgent
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows:
.\venv\Scripts\Activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API keys:
```ini
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini

# Optional Gemini configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```

### 5. Launch the Application
```bash
python main.py
```

- **Customer Support Portal:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check API:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 📂 Project Directory Structure

```text
MultiAgent/
├── src/
│   └── multiagent/
│       ├── __init__.py        # Package initializer
│       ├── config.py          # Environment settings & model loader
│       ├── state.py           # TypedDict schema for LangGraph state
│       ├── graph.py           # LangGraph StateGraph builder & router
│       ├── agents.py          # Agent nodes (Router, RAG, Order, Support, Validator)
│       ├── rag_store.py       # ChromaDB vector store & semantic search
│       └── tools.py           # Order lookup, refund, and escalation tools
├── static/
│   └── index.html             # Customer support web interface & trace drawer
├── main.py                    # FastAPI server entry point
├── requirements.txt           # Python dependency specifications
├── .env.example               # Template environment configuration
└── README.md                  # System documentation
```

---

## 📄 License
This project is licensed under the MIT License.

