"""
main.py — FastAPI Application Entry Point
(Updated: Step 5 — PostgreSQL connected on startup)

WHAT CHANGED FROM STEP 4:
  - Startup now calls create_tables() to create DB schema
  - Startup checks DB connection and reports status
  - /health endpoint now shows real DB connection status
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings, validate_config
from routers.rca_router import router as rca_router
from routers.incidents_router import router as incidents_router

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Track connection status for /health endpoint
_db_connected = False
_chroma_connected = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_connected, _chroma_connected

    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Validate config
    try:
        validate_config()
    except EnvironmentError as e:
        logger.error(f"❌ Config error: {e}")
        raise

    # Connect PostgreSQL + create tables
    try:
        from db.database import create_tables, check_database_connection
        _db_connected = await check_database_connection()
        if _db_connected:
            await create_tables()
            logger.info("✅ PostgreSQL connected and tables ready")
        else:
            logger.warning("⚠️ PostgreSQL not connected — reports won't persist")
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL startup issue: {e}")
        _db_connected = False

    # Check ChromaDB
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        stats = await retriever.get_stats()
        _chroma_connected = stats.get("status") == "connected"
        if _chroma_connected:
            logger.info(f"✅ ChromaDB connected. Incidents in KB: {stats.get('total_incidents', 0)}")
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB startup issue: {e}")
        _chroma_connected = False

    logger.info(f"✅ App ready! LLM: {settings.LLM_PROVIDER.upper()}")
    logger.info("📖 API Docs: http://localhost:8000/docs")

    yield

    logger.info("👋 Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 🤖 AI Incident RCA Assistant

Automated Root Cause Analysis powered by LangGraph AI agents.

### Quick Start
1. **Seed knowledge base:** `POST /api/v1/incidents/seed`
2. **Submit incident:** `POST /api/v1/rca/analyze`
3. **View report:** `GET /api/v1/rca/reports/{incident_id}`
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rca_router, prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "quick_start": {
            "step_1": "POST /api/v1/incidents/seed  (load sample data)",
            "step_2": "POST /api/v1/rca/analyze     (submit an incident)",
            "step_3": "GET  /api/v1/rca/reports      (view all reports)"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "database_connected": _db_connected,
        "chromadb_connected": _chroma_connected,
    }


@app.get("/api/v1/stats", tags=["Stats"])
async def get_stats():
    """Combined stats from PostgreSQL + ChromaDB."""
    from services.rca_service import RCAService
    service = RCAService()
    db_stats = {}
    kb_stats = {}
    try:
        db_stats = await service.get_report_stats()
    except Exception as e:
        db_stats = {"error": str(e)}
    try:
        kb_stats = await service.get_knowledge_base_stats()
    except Exception as e:
        kb_stats = {"error": str(e)}
    return {
        "reports": db_stats,
        "knowledge_base": kb_stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ── Serve frontend HTML ────────────────────────────────────
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Mount static files if frontend folder exists
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/ui", tags=["Frontend"])
async def serve_ui():
    """Serve the frontend UI."""
    ui_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Frontend not found. Place index.html in /frontend folder."}
