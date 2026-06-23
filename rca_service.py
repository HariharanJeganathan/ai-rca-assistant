"""
rca_service.py — Business Logic Layer
=======================================
The SERVICE LAYER sits between the API routes and the AI agent.

Think of it like a restaurant:
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │   Routes    │────▶│   Service   │────▶│  AI Agent   │
  │ (Waiter)    │     │  (Kitchen   │     │  (Chef)     │
  │             │     │   Manager)  │     │             │
  └─────────────┘     └─────────────┘     └─────────────┘

Why this pattern?
  ✅ Routes stay thin — just receive/send HTTP requests
  ✅ All business logic is in one place (easy to change)
  ✅ AI agent focuses only on reasoning (not HTTP stuff)
  ✅ Easy to test each layer independently
  ✅ Senior engineers always use this pattern — interviewers notice

This is called the "Service Layer Pattern" or "Repository Pattern".
"""

import logging
import io
from typing import Optional, List, Tuple
from datetime import datetime

from models.schemas import (
    IncidentInput,
    RCAReport,
    RCAAnalysis,
    RCAStatus,
)

logger = logging.getLogger(__name__)


class RCAService:
    """
    Service class that orchestrates:
      1. Text extraction (from PDF/TXT files)
      2. RAG search (find similar incidents in ChromaDB)
      3. AI analysis (run LangGraph agent)
      4. Database operations (save/fetch reports from PostgreSQL)

    Each method maps to one business operation.
    In Steps 3-5, we'll wire each method to its real implementation.
    Right now, they return stub data so the API actually works end-to-end.
    """

    def __init__(self):
        """
        Initialize service.
        In later steps we'll inject:
          - ChromaDB retriever
          - LangGraph agent
          - PostgreSQL connection
        """
        logger.info("[RCAService] Initialized (stub mode — AI coming in Steps 3-4)")

        # These will be real objects after Steps 3-5
        self.retriever = None    # ChromaDB (Step 3)
        self.agent = None        # LangGraph Agent (Step 4)
        self.db = None           # PostgreSQL (Step 5)

        # In-memory store for now (replaced by PostgreSQL in Step 5)
        self._reports_store: dict = {}

    # ============================================================
    # 1. ANALYZE INCIDENT — Main orchestration method
    # ============================================================
    async def analyze_incident(
        self,
        incident_id: str,
        incident: IncidentInput,
        llm_provider_override: Optional[str] = None
    ) -> RCAReport:
        """
        Main method — orchestrates the full RCA pipeline.

        Steps:
          1. Save incident as PENDING
          2. Search ChromaDB for similar incidents (RAG)
          3. Run LangGraph agent to generate RCA
          4. Save completed report to PostgreSQL
          5. Return the report

        Currently returns stub data.
        Will be fully wired in Steps 3 and 4.
        """
        logger.info(f"[RCAService] Starting analysis for {incident_id}")

        # Step 1: Create initial report record
        report = RCAReport(
            incident_id=incident_id,
            status=RCAStatus.ANALYZING,
            incident=incident,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            llm_provider_used=llm_provider_override or "groq"
        )

        # Save to in-memory store (PostgreSQL in Step 5)
        self._reports_store[incident_id] = report

        try:
            # Step 2: RAG — search for similar past incidents
            # (ChromaDB integration coming in Step 3)
            similar_incidents = await self._find_similar_incidents(incident)

            # Step 3: Run AI agent
            # (LangGraph agent coming in Step 4)
            analysis = await self._run_rca_agent(incident, similar_incidents)

            # Step 4: Update report with completed analysis
            report.analysis = analysis
            report.status = RCAStatus.COMPLETED
            report.completed_at = datetime.utcnow()
            report.updated_at = datetime.utcnow()

            # Save updated report
            self._reports_store[incident_id] = report

            logger.info(f"[RCAService] Analysis complete for {incident_id}")
            return report

        except Exception as e:
            # If anything fails, mark as FAILED
            report.status = RCAStatus.FAILED
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report
            logger.error(f"[RCAService] Analysis failed for {incident_id}: {e}")
            raise

    # ============================================================
    # 2. FIND SIMILAR INCIDENTS (RAG Stub)
    # ============================================================
    async def _find_similar_incidents(
        self,
        incident: IncidentInput
    ) -> List[str]:
        """
        Search ChromaDB for incidents similar to this one.
        Returns a list of similar incident descriptions.

        STUB: Returns empty list for now.
        Will be wired to ChromaDB in Step 3.
        """
        logger.info("[RCAService] Searching for similar incidents (stub)")

        # Step 3 will replace this with:
        # return await self.retriever.search(incident.description, top_k=3)
        return []

    # ============================================================
    # 3. RUN RCA AGENT (AI Stub)
    # ============================================================
    async def _run_rca_agent(
        self,
        incident: IncidentInput,
        similar_incidents: List[str]
    ) -> RCAAnalysis:
        """
        Run the LangGraph agent to generate the RCA.

        STUB: Returns a structured placeholder analysis.
        Will be replaced with real LangGraph agent in Step 4.
        """
        logger.info("[RCAService] Running RCA agent (stub mode)")

        # Build context string from similar incidents
        similar_context = ""
        if similar_incidents:
            similar_context = "\n".join([f"- {s}" for s in similar_incidents])

        # Stub analysis — real AI output in Step 4
        stub_analysis = RCAAnalysis(
            incident_summary=(
                f"Incident: {incident.title}. "
                f"Affected systems: {', '.join(incident.affected_systems) if incident.affected_systems else 'Not specified'}. "
                f"Severity: {incident.severity.value}."
            ),
            timeline_reconstruction=(
                incident.incident_timeline or
                "Timeline not provided. Please add incident timeline for better analysis."
            ),
            root_cause=(
                "⚠️ STUB: LangGraph AI agent will be connected in Step 4. "
                "Root cause analysis will appear here after the agent is wired up."
            ),
            contributing_factors=[
                "STUB: Factor 1 will be identified by AI agent",
                "STUB: Factor 2 will be identified by AI agent"
            ],
            impact_assessment=(
                f"Severity {incident.severity.value} incident affecting: "
                f"{', '.join(incident.affected_systems) if incident.affected_systems else 'systems not specified'}."
            ),
            immediate_actions_taken=[
                "STUB: Immediate actions will be extracted by AI agent"
            ],
            corrective_actions=[
                "STUB: Corrective actions will be generated by AI agent in Step 4"
            ],
            preventive_measures=[
                "STUB: Preventive measures will be generated by AI agent in Step 4"
            ],
            lessons_learned=(
                "STUB: Lessons learned will be generated by AI agent in Step 4. "
                f"Context: {incident.additional_context or 'No additional context provided.'}"
            ),
            similar_incidents=similar_incidents,
            confidence_score=0.0  # Will be real after Step 4
        )

        return stub_analysis

    # ============================================================
    # 4. GET ONE REPORT
    # ============================================================
    async def get_report(self, incident_id: str) -> Optional[RCAReport]:
        """
        Fetch one RCA report by incident ID.
        Looks in in-memory store (PostgreSQL in Step 5).
        """
        logger.info(f"[RCAService] Getting report: {incident_id}")
        return self._reports_store.get(incident_id)

    # ============================================================
    # 5. LIST REPORTS
    # ============================================================
    async def list_reports(
        self,
        skip: int = 0,
        limit: int = 10,
        status_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Tuple[List[RCAReport], int]:
        """
        List reports with optional filtering and pagination.
        Returns (list_of_reports, total_count).
        """
        logger.info(f"[RCAService] Listing reports: skip={skip}, limit={limit}")

        reports = list(self._reports_store.values())

        # Apply filters
        if status_filter:
            reports = [r for r in reports if r.status.value == status_filter]

        if severity_filter:
            reports = [r for r in reports if r.incident.severity.value == severity_filter]

        total = len(reports)

        # Apply pagination
        paginated = reports[skip: skip + limit]

        return paginated, total

    # ============================================================
    # 6. DELETE REPORT
    # ============================================================
    async def delete_report(self, incident_id: str) -> bool:
        """
        Delete a report by incident ID.
        Returns True if deleted, False if not found.
        """
        if incident_id in self._reports_store:
            del self._reports_store[incident_id]
            logger.info(f"[RCAService] Deleted report: {incident_id}")
            return True
        return False

    # ============================================================
    # 7. EXTRACT TEXT FROM FILE
    # ============================================================
    async def extract_text_from_file(
        self,
        content: bytes,
        filename: str,
        file_extension: str
    ) -> str:
        """
        Extract text from uploaded PDF or TXT files.

        For .txt files → decode bytes to string directly
        For .pdf files → use pypdf to extract text from pages
        """
        logger.info(f"[RCAService] Extracting text from: {filename}")

        if file_extension in [".txt", ".text"]:
            # Simple text file — just decode bytes
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1")

        elif file_extension == ".pdf":
            try:
                import pypdf

                # Read PDF from bytes using BytesIO
                # (BytesIO lets us treat bytes like a file object)
                pdf_reader = pypdf.PdfReader(io.BytesIO(content))

                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

                if not text_parts:
                    raise ValueError("No text could be extracted from PDF. It may be a scanned image PDF.")

                return "\n\n".join(text_parts)

            except ImportError:
                raise RuntimeError("pypdf not installed. Run: pip install pypdf")

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
