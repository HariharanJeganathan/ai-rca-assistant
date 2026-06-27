"""
main.py — FastAPI Application Entry Point
Updated: Added chat router
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings, validate_config
from routers.rca_router import router as rca_router
from routers.incidents_router import router as incidents_router
from routers.chat_router import router as chat_router

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_db_connected = False
_chroma_connected = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_connected, _chroma_connected
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        validate_config()
    except EnvironmentError as e:
        logger.error(f"❌ Config error: {e}")
        raise

    try:
        from db.database import create_tables, check_database_connection
        _db_connected = await check_database_connection()
        if _db_connected:
            await create_tables()
            logger.info("✅ PostgreSQL connected")
        else:
            logger.warning("⚠️ PostgreSQL not connected — using memory fallback")
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL: {e}")
        _db_connected = False

    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        stats = await retriever.get_stats()
        _chroma_connected = stats.get("status") == "connected"
        if _chroma_connected:
            logger.info(f"✅ ChromaDB connected. KB: {stats.get('total_incidents', 0)} incidents")
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB: {e}")
        _chroma_connected = False

    logger.info(f"✅ Ready! LLM: {settings.LLM_PROVIDER.upper()}")
    logger.info("📖 Docs: http://localhost:8000/docs")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Incident RCA Assistant — FedEx MI",
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

app.include_router(rca_router,       prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")
app.include_router(chat_router,      prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Serve landing page at root URL."""
    landing_path = os.path.join(frontend_path, "landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path)
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "ui": "/ui",
        "docs": "/docs"
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
    return {"reports": db_stats, "knowledge_base": kb_stats}


# Serve frontend
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/ui", tags=["Frontend"])
async def serve_ui():
    """Serve the main app UI."""
    ui_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Frontend not found"}

@app.get("/home", tags=["Frontend"])
@app.get("/landing", tags=["Frontend"])
async def serve_landing():
    """Serve the landing page."""
    landing_path = os.path.join(frontend_path, "landing.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path)
    return {"message": "Landing page not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
