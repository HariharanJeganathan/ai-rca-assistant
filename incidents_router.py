"""
incidents_router.py — Knowledge Base Management Routes
=======================================================
This router manages the KNOWLEDGE BASE — the collection of
past incidents that ChromaDB uses for RAG search.

Think of it like a library:
  - rca_router.py = the librarian who reads and analyses books
  - incidents_router.py = the person who ADDS books to the library

When you add an incident to the knowledge base:
  1. Its text gets converted to a vector (numbers)
  2. That vector is stored in ChromaDB
  3. Future RCA requests search this library for similar incidents

URL Structure:
  POST /api/v1/incidents/ingest     ← Add one incident to knowledge base
  POST /api/v1/incidents/bulk       ← Add many incidents at once
  GET  /api/v1/incidents/search     ← Search for similar incidents
  GET  /api/v1/incidents/stats      ← How many incidents in the KB?
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from models.schemas import IncidentInput, IncidentSeverity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/incidents", tags=["Incidents - Knowledge Base"])


# ============================================================
# ROUTE 1: Add One Incident to Knowledge Base
# ============================================================
@router.post(
    "/ingest",
    summary="Add an incident to the RAG knowledge base",
    description="""
Add a resolved incident to ChromaDB so future RCA requests
can find it as a similar past incident.

**Best practice:** After every resolved incident, ingest it here
so the AI gets smarter over time.
    """
)
async def ingest_incident(
    incident: IncidentInput,
    resolution: Optional[str] = None,
    root_cause: Optional[str] = None
):
    """
    Ingest a single incident into the ChromaDB knowledge base.
    Optional: include the resolution and root cause for richer context.
    """
    logger.info(f"Ingesting incident: {incident.title}")

    try:
        # We'll wire this to ChromaDB in Step 3
        # For now, return a stub response
        return {
            "success": True,
            "message": f"Incident '{incident.title}' added to knowledge base",
            "note": "ChromaDB integration coming in Step 3"
        }

    except Exception as e:
        logger.error(f"Error ingesting incident: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ROUTE 2: Bulk Ingest Many Incidents
# ============================================================
@router.post(
    "/bulk",
    summary="Add multiple incidents to the knowledge base at once"
)
async def bulk_ingest_incidents(incidents: List[IncidentInput]):
    """
    Add many incidents at once — useful for importing historical data.

    Example: You have 100 past incidents in a JSON file.
    POST them all here in one call.
    """
    if len(incidents) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 incidents per bulk upload. Split into batches."
        )

    logger.info(f"Bulk ingesting {len(incidents)} incidents")

    # Stub — will connect to ChromaDB in Step 3
    return {
        "success": True,
        "message": f"{len(incidents)} incidents queued for ingestion",
        "count": len(incidents)
    }


# ============================================================
# ROUTE 3: Search Knowledge Base
# ============================================================
@router.get(
    "/search",
    summary="Search for similar incidents in the knowledge base"
)
async def search_incidents(
    query: str = Query(..., min_length=5, description="Search query text"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return")
):
    """
    Semantic search through the knowledge base.
    Returns incidents most similar to the query text.

    This uses ChromaDB vector search — not keyword search.
    That means it finds incidents that are CONCEPTUALLY similar,
    even if they use different words.
    """
    logger.info(f"Searching knowledge base: '{query}', top_k={top_k}")

    # Stub — will connect to ChromaDB in Step 3
    return {
        "success": True,
        "query": query,
        "results": [],
        "note": "ChromaDB search coming in Step 3"
    }


# ============================================================
# ROUTE 4: Knowledge Base Stats
# ============================================================
@router.get(
    "/stats",
    summary="Get knowledge base statistics"
)
async def get_knowledge_base_stats():
    """
    Returns statistics about what's in the knowledge base.
    Useful for monitoring how much data the AI has to work with.
    """
    # Stub — will connect to ChromaDB in Step 3
    return {
        "success": True,
        "total_incidents": 0,
        "by_severity": {
            "P1": 0, "P2": 0, "P3": 0, "P4": 0
        },
        "note": "ChromaDB stats coming in Step 3"
    }
