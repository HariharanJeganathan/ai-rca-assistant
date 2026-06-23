"""
incidents_router.py — Knowledge Base Routes (Updated: Step 3)
==============================================================
WHAT CHANGED FROM STEP 2:
  - /ingest now calls REAL ChromaDB via rca_service
  - /search now calls REAL ChromaDB vector search
  - /stats now returns REAL ChromaDB count
  - Added /seed endpoint to populate sample data for demos
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from models.schemas import IncidentInput
from services.rca_service import RCAService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/incidents", tags=["Incidents - Knowledge Base"])
rca_service = RCAService()


@router.post("/ingest", summary="Add incident to RAG knowledge base")
async def ingest_incident(
    incident: IncidentInput,
    incident_id: Optional[str] = Query(default=None),
    resolution: Optional[str] = None,
    root_cause: Optional[str] = None
):
    """Add a resolved incident to ChromaDB so future RCAs can reference it."""
    import uuid
    from datetime import datetime
    iid = incident_id or f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    try:
        success = await rca_service.ingest_incident(
            incident_id=iid,
            incident=incident,
            root_cause=root_cause,
            resolution=resolution
        )
        if success:
            return {"success": True, "message": f"Incident '{incident.title}' added to knowledge base", "incident_id": iid}
        else:
            return {"success": False, "message": f"Incident {iid} already exists in knowledge base"}
    except Exception as e:
        logger.error(f"Error ingesting incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk", summary="Bulk add incidents to knowledge base")
async def bulk_ingest(incidents: List[IncidentInput]):
    if len(incidents) > 100:
        raise HTTPException(status_code=400, detail="Max 100 per batch")
    results = []
    for i, incident in enumerate(incidents):
        import uuid
        from datetime import datetime
        iid = f"INC-BULK-{i:04d}-{str(uuid.uuid4())[:6].upper()}"
        try:
            success = await rca_service.ingest_incident(incident_id=iid, incident=incident)
            results.append({"incident_id": iid, "success": success})
        except Exception as e:
            results.append({"incident_id": iid, "success": False, "error": str(e)})
    return {"success": True, "total": len(incidents), "results": results}


@router.get("/search", summary="Semantic search in knowledge base")
async def search_incidents(
    query: str = Query(..., min_length=5, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20)
):
    """Real ChromaDB vector search — finds semantically similar incidents."""
    try:
        results = await rca_service.search_similar(query=query, top_k=top_k)
        return {"success": True, "query": query, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Knowledge base statistics")
async def get_stats():
    """Returns how many incidents are stored in ChromaDB."""
    try:
        stats = await rca_service.get_knowledge_base_stats()
        return {"success": True, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seed", summary="Seed sample incidents for demo/testing")
async def seed_knowledge_base():
    """
    Populates ChromaDB with 5 sample historical incidents.
    Run this once to give the AI something to search against.
    Great for demos and interviews!
    """
    try:
        count = await rca_service.seed_knowledge_base()
        return {
            "success": True,
            "message": f"Seeded {count} sample incidents into ChromaDB",
            "tip": "Now try POST /api/v1/rca/analyze — the AI will find similar incidents!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
