"""
rca_router.py — API Routes for RCA Operations
===============================================
A "router" in FastAPI is like a group of related URL endpoints.
Think of it like a department in a company:
  - HR department handles /employees, /payroll
  - This router handles /rca/analyze, /rca/reports, etc.

We keep routes in separate files (not all in main.py) because:
  ✅ Easier to find and edit
  ✅ Each file has one clear responsibility
  ✅ Teams can work on different routers simultaneously
  ✅ Interviewers love this — it shows you know "separation of concerns"

URL Structure this file handles:
  POST /api/v1/rca/analyze        ← Submit incident, get RCA back
  GET  /api/v1/rca/{incident_id}  ← Get one specific RCA report
  GET  /api/v1/rca/reports        ← List all RCA reports
  POST /api/v1/rca/upload         ← Upload a PDF/text file
  DELETE /api/v1/rca/{incident_id} ← Delete a report
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging
import uuid
from datetime import datetime

from models.schemas import (
    RCARequest,
    RCAResponse,
    RCAReport,
    IncidentInput,
    RCAStatus,
    ErrorResponse
)
from services.rca_service import RCAService

# Set up logging — so we can see what's happening
logger = logging.getLogger(__name__)

# ============================================================
# Create the Router
# ============================================================
# prefix="/rca" means all routes here start with /rca
# tags=["RCA"] groups them in the Swagger docs UI
router = APIRouter(prefix="/rca", tags=["RCA - Root Cause Analysis"])

# Create service instance (the service does the actual work)
rca_service = RCAService()


# ============================================================
# ROUTE 1: Submit Incident for RCA Analysis
# ============================================================
@router.post(
    "/analyze",
    response_model=RCAResponse,
    summary="Submit an incident for RCA analysis",
    description="""
Submit an incident report and get back a full Root Cause Analysis.

**What happens internally:**
1. Incident is saved with status = PENDING
2. ChromaDB is searched for similar past incidents
3. LangGraph agent reasons through the incident step by step
4. Structured RCA report is generated and saved to PostgreSQL
5. Full report is returned in the response

**Processing time:** ~10-30 seconds depending on LLM provider
    """
)
async def analyze_incident(
    request: RCARequest,
    background_tasks: BackgroundTasks
):
    """
    Main endpoint — Submit an incident, get back a full RCA.

    BackgroundTasks = FastAPI feature that lets us start work
    without making the user wait. We use it for heavy AI processing.
    """
    logger.info(f"Received RCA request: {request.incident.title}")

    try:
        # Generate a unique ID for this incident
        # uuid4() creates a random ID like: "3f7a1b2c-..."
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        logger.info(f"Processing incident: {incident_id}")

        # Call the service layer to do the actual AI work
        # (We'll build rca_service in Step 4 — for now it's a stub)
        report = await rca_service.analyze_incident(
            incident_id=incident_id,
            incident=request.incident,
            llm_provider_override=request.llm_provider.value if request.llm_provider else None
        )

        return RCAResponse(
            success=True,
            message="RCA analysis completed successfully",
            incident_id=incident_id,
            status=report.status,
            report=report
        )

    except Exception as e:
        logger.error(f"Error analyzing incident: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze incident: {str(e)}"
        )


# ============================================================
# ROUTE 2: Upload File (PDF or Text)
# ============================================================
@router.post(
    "/upload",
    response_model=RCAResponse,
    summary="Upload an incident report file (PDF or TXT)",
    description="Upload a PDF or text file containing an incident report. The AI will extract the content and perform RCA."
)
async def upload_incident_file(
    file: UploadFile = File(..., description="PDF or TXT file containing the incident report"),
    severity: Optional[str] = Query(default="P3", description="Incident severity: P1, P2, P3, P4")
):
    """
    Upload a PDF or text file — the AI reads and analyzes it.

    UploadFile = FastAPI's file upload type
    File(...) = required file parameter
    """
    logger.info(f"File upload received: {file.filename}, size: {file.size}")

    # Validate file type
    allowed_types = ["application/pdf", "text/plain"]
    allowed_extensions = [".pdf", ".txt", ".text"]

    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {allowed_extensions}. Got: {file_ext}"
        )

    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB in bytes
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: 10MB. Got: {len(content) / 1024 / 1024:.1f}MB"
        )

    try:
        # Extract text from the file
        extracted_text = await rca_service.extract_text_from_file(
            content=content,
            filename=file.filename,
            file_extension=file_ext
        )

        if not extracted_text or len(extracted_text.strip()) < 20:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful text from the file. Please check the file content."
            )

        # Create an incident from the extracted text
        incident = IncidentInput(
            title=f"Incident from file: {file.filename}",
            description=extracted_text,
            severity=severity,
            additional_context=f"Extracted from uploaded file: {file.filename}"
        )

        # Generate incident ID
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        # Run RCA analysis
        report = await rca_service.analyze_incident(
            incident_id=incident_id,
            incident=incident
        )

        return RCAResponse(
            success=True,
            message=f"File '{file.filename}' processed and RCA completed",
            incident_id=incident_id,
            status=report.status,
            report=report
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ROUTE 3: Get One RCA Report by ID
# ============================================================
@router.get(
    "/reports/{incident_id}",
    response_model=RCAResponse,
    summary="Get a specific RCA report by incident ID"
)
async def get_rca_report(incident_id: str):
    """
    Fetch one specific RCA report.

    Path parameter: {incident_id}
    Example: GET /api/v1/rca/reports/INC-20241201-A3B4C5D6
    """
    logger.info(f"Fetching report for incident: {incident_id}")

    try:
        report = await rca_service.get_report(incident_id=incident_id)

        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"No RCA report found for incident ID: {incident_id}"
            )

        return RCAResponse(
            success=True,
            message="Report retrieved successfully",
            incident_id=incident_id,
            status=report.status,
            report=report
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching report {incident_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ROUTE 4: List All RCA Reports
# ============================================================
@router.get(
    "/reports",
    summary="List all RCA reports",
    description="Returns a paginated list of all RCA reports. Use skip and limit for pagination."
)
async def list_rca_reports(
    skip: int = Query(default=0, ge=0, description="Number of records to skip (for pagination)"),
    limit: int = Query(default=10, ge=1, le=100, description="Max records to return (1-100)"),
    status: Optional[str] = Query(default=None, description="Filter by status: pending, analyzing, completed, failed"),
    severity: Optional[str] = Query(default=None, description="Filter by severity: P1, P2, P3, P4")
):
    """
    List reports with pagination and filtering.

    Query parameters come from the URL like:
    GET /api/v1/rca/reports?skip=0&limit=10&status=completed

    skip + limit = pagination pattern
    (skip=0, limit=10 → first 10 records)
    (skip=10, limit=10 → next 10 records, i.e. page 2)
    """
    logger.info(f"Listing reports: skip={skip}, limit={limit}, status={status}")

    try:
        reports, total = await rca_service.list_reports(
            skip=skip,
            limit=limit,
            status_filter=status,
            severity_filter=severity
        )

        return {
            "success": True,
            "total": total,
            "skip": skip,
            "limit": limit,
            "reports": reports
        }

    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ROUTE 5: Delete an RCA Report
# ============================================================
@router.delete(
    "/reports/{incident_id}",
    summary="Delete an RCA report"
)
async def delete_rca_report(incident_id: str):
    """
    Delete a specific RCA report from the database.
    """
    logger.info(f"Deleting report: {incident_id}")

    try:
        deleted = await rca_service.delete_report(incident_id=incident_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"No report found with incident ID: {incident_id}"
            )

        return {
            "success": True,
            "message": f"Report {incident_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Excel Parse Endpoint (Step 6b) ────────────────────────
@router.post("/parse-excel", summary="Parse FedEx MI Excel file")
async def parse_excel_file(
    file: UploadFile = File(..., description="FedEx MI Excel (.xlsx)")
):
    """
    Parse a FedEx Major Incident Excel sheet.
    Extracts all incident fields automatically based on the FedEx MI format.
    Returns structured data that can be sent directly to /analyze.
    """
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only .xlsx files supported")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    try:
        from services.fedex_excel_parser import FedExMIParser
        parser = FedExMIParser()
        parsed = parser.parse(content)
        return {
            "success": True,
            "filename": file.filename,
            "parsed": parsed,
            "incident_number": parsed.get("incident_number"),
            "message": f"Successfully parsed {parsed.get('incident_number','incident')}"
        }
    except Exception as e:
        logger.error(f"Excel parse error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Excel: {str(e)}")
