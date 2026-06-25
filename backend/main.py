"""
main.py — FastAPI Application Entry Point
==========================================
Updated in Step 2 to include all API routers.

WHAT CHANGED FROM STEP 1:
  - Added imports for rca_router and incidents_router
  - Registered both routers with app.include_router()
  - Added /api/v1 prefix to all routes

The app now has working API endpoints:
  GET  /                           → Welcome
  GET  /health                     → Health check
  GET  /docs                       → Swagger UI (auto-generated!)
  POST /api/v1/rca/analyze         → Submit incident for RCA
  GET  /api/v1/rca/reports         → List all reports
  GET  /api/v1/rca/reports/{id}    → Get one report
  POST /api/v1/rca/upload          → Upload PDF/TXT
  POST /api/v1/incidents/ingest    → Add to knowledge base
  GET  /api/v1/incidents/search    → Search knowledge base
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from config import settings, validate_config

# Import routers (NEW in Step 2)
from routers.rca_router import router as rca_router
from routers.incidents_router import router as incidents_router

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Lifespan — startup and shutdown
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        validate_config()
    except EnvironmentError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    logger.info(f"✅ App started. LLM: {settings.LLM_PROVIDER.upper()}")
    logger.info(f"📖 API Docs: http://localhost:8000/docs")
    yield
    logger.info("👋 Shutting down...")


# ============================================================
# Create FastAPI App
# ============================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 🤖 AI Incident RCA Assistant

Automated Root Cause Analysis powered by LangGraph AI agents.

### Available Endpoints
- **POST** `/api/v1/rca/analyze` — Submit an incident, get back full RCA
- **POST** `/api/v1/rca/upload` — Upload a PDF/TXT incident report
- **GET** `/api/v1/rca/reports` — List all RCA reports
- **GET** `/api/v1/rca/reports/{id}` — Get specific RCA report
- **POST** `/api/v1/incidents/ingest` — Add incident to knowledge base
- **GET** `/api/v1/incidents/search` — Search similar incidents
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ============================================================
# CORS Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Register Routers (NEW in Step 2)
# ============================================================
# All routes get prefixed with /api/v1
# So /rca/analyze becomes /api/v1/rca/analyze
app.include_router(rca_router, prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")

# ============================================================
# Root Routes
# ============================================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}!",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze": "POST /api/v1/rca/analyze",
            "upload":  "POST /api/v1/rca/upload",
            "reports": "GET  /api/v1/rca/reports",
            "search":  "GET  /api/v1/incidents/search"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "database_connected": False,   # Step 5
        "chromadb_connected": False,   # Step 3
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug" if settings.DEBUG else "info"
    )
