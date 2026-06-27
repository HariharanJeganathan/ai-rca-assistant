# 🤖 AI Incident RCA Assistant

> Automated Root Cause Analysis powered by LangGraph AI agents, RAG, and FastAPI

[![CI Pipeline](https://github.com/YOUR_USERNAME/ai-rca-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/ai-rca-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 What This Does

Submit an incident report → AI agent reasons through it → Get a structured RCA document.

```
Incident Report (text/PDF)
        ↓
  [ChromaDB RAG]          ← Finds similar past incidents
        ↓
  [LangGraph Agent]       ← Multi-step reasoning
    → What happened?
    → Why did it happen?
    → What should be fixed?
        ↓
  Structured RCA Report   ← Saved to PostgreSQL
```

---

## 🏗️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI Orchestration** | LangGraph | Multi-step reasoning agent |
| **LLM** | Groq (LLaMA 3) / OpenAI / Azure OpenAI | Language model (switchable) |
| **RAG** | LangChain + ChromaDB | Similar incident retrieval |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Text → vectors |
| **API** | FastAPI | REST API framework |
| **Database** | PostgreSQL (Supabase) | RCA report storage |
| **Containers** | Docker + Docker Compose | Local dev environment |
| **CI/CD** | GitHub Actions | Automated testing |
| **Deploy** | Render | Cloud hosting (free tier) |

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- Docker Desktop
- Free Groq API key → [console.groq.com](https://console.groq.com)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-rca-assistant.git
cd ai-rca-assistant
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start with Docker
```bash
docker compose up --build
```

### 4. Open the app
- **API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **Frontend:** http://localhost:8000/ui

---

## 🔀 Switching LLM Provider

Edit `.env` — no code changes needed:

```env
# Use Groq (free, LLaMA 3)
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# Switch to OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here

# Switch to Azure OpenAI
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment
```

---

## 📁 Project Structure

```
ai-rca-assistant/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # LLM switcher & settings
│   ├── agents/
│   │   └── rca_agent.py     # LangGraph reasoning agent
│   ├── chains/
│   │   └── rca_chain.py     # LangChain prompts
│   ├── rag/
│   │   └── retriever.py     # ChromaDB vector search
│   ├── db/
│   │   └── postgres.py      # PostgreSQL integration
│   ├── models/
│   │   └── schemas.py       # Pydantic data models
│   └── requirements.txt
├── frontend/
│   └── index.html           # Simple web UI
├── docker-compose.yml       # Local dev stack
├── Dockerfile               # Container recipe
├── .github/workflows/ci.yml # GitHub Actions CI
└── .env.example             # Config template
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/rca/analyze` | Submit incident for RCA |
| `GET` | `/api/v1/rca/{incident_id}` | Get RCA report |
| `GET` | `/api/v1/rca/reports` | List all reports |
| `POST` | `/api/v1/incidents/ingest` | Add incident to knowledge base |

---

## 🧠 How the LangGraph Agent Works

```
                    ┌─────────────────┐
                    │  User Submits   │
                    │    Incident     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Parse & Embed  │ ← Extract key info
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  RAG Retrieval  │ ← Search ChromaDB
                    │ (Similar Incidents)   for past incidents
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Analyze Root    │ ← LLM reasons about
                    │     Cause       │   what went wrong
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Generate Action │ ← LLM produces
                    │     Items       │   corrective actions
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Save to DB     │ ← PostgreSQL
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Return Report  │
                    └─────────────────┘
```

---

## 🌐 Deploy to Render (Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set environment variables from `.env.example`
5. Deploy!

---

## 🧪 Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

---

## 📄 License

MIT License — free to use for personal and commercial projects.

---

## 👤 Author

Built as a portfolio project demonstrating:
- AI/LLM engineering (LangGraph, RAG, multi-provider)
- Modern Python backend (FastAPI, async, Pydantic)
- Production practices (Docker, CI/CD, environment config)
- ITSM domain knowledge (incident management, RCA)

## Final Link:
Landing:   https://ai-rca-assistant.onrender.com
App:       https://ai-rca-assistant.onrender.com/ui
API Docs:  https://ai-rca-assistant.onrender.com/docs
