"""
main.py — FastAPI Application Entry Point
==========================================
This is where the app STARTS.

When you run: uvicorn backend.main:app --reload
  → Python finds this file
  → Creates the FastAPI "app" object
  → Registers all the routes (URLs)
  → Starts listening for requests

We'll add more routes in Steps 2–5.
This file is intentionally kept thin — logic lives in other modules.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from config import settings, validate_config, get_llm, get_embeddings

# Set up logging — so we can see what's happening in the terminal
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Lifespan — runs code on startup and shutdown
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the app starts.
    Code after 'yield' runs when the app shuts down.

    Good place to: validate config, warm up models, connect to DB.
    """
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Validate all required environment variables
    try:
        validate_config()
    except EnvironmentError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise

    logger.info(f"✅ App started successfully. LLM: {settings.LLM_PROVIDER}")

    yield   # App runs while we're here

    # Shutdown
    logger.info("👋 Shutting down...")


# ============================================================
# Create FastAPI App
# ============================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## AI Incident RCA Assistant

Automated Root Cause Analysis powered by LangGraph AI agents.

### Features
- 🤖 **AI-powered RCA** using LangGraph reasoning agents
- 🔍 **RAG search** — finds similar past incidents from ChromaDB
- 🔀 **Multi-LLM** — switch between Groq, OpenAI, Azure via config
- 📊 **PostgreSQL** — stores all RCA reports
- 📄 **PDF/Text upload** — parse incident reports from files

### How to use
1. Submit an incident via `POST /api/v1/rca/analyze`
2. Check status via `GET /api/v1/rca/{incident_id}`
3. View all reports via `GET /api/v1/rca/reports`
    """,
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # Alternative docs at http://localhost:8000/redoc
    lifespan=lifespan,
)


# ============================================================
# CORS Middleware
# ============================================================
# CORS = Cross-Origin Resource Sharing
# This allows the frontend (HTML page) to call the API
# even if they're on different ports (e.g. :3000 vs :8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production, replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Routes — URL endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint — shows app is alive.
    Visit http://localhost:8000/ in browser.
    """
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Used by Docker, Railway, Render to know if app is alive.
    Returns 200 OK if everything is fine.
    """
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "debug_mode": settings.DEBUG,
        # We'll add real DB checks in Step 5
        "database_connected": False,    # placeholder
        "chromadb_connected": False,    # placeholder
    }


# ============================================================
# Include Routers — we'll add these in later steps
# ============================================================
# Step 2: from routers import rca_router
#         app.include_router(rca_router, prefix="/api/v1")
#
# (Commented out for now — files don't exist yet)


# ============================================================
# Run directly (for development only)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # Auto-restart when code changes
        log_level="debug" if settings.DEBUG else "info"
    )
