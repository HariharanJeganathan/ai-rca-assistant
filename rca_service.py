"""
rca_service.py — Business Logic Layer (Updated: Step 3 — ChromaDB wired in)
=============================================================================
WHAT CHANGED FROM STEP 2:
  - _find_similar_incidents() now calls REAL ChromaDB search
  - Added ingest_incident() to add incidents to the knowledge base
  - Added get_knowledge_base_stats() for the /stats endpoint
  - Retriever is injected via get_retriever()
"""

import logging
import io
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from models.schemas import IncidentInput, RCAReport, RCAAnalysis, RCAStatus
from rag.retriever import get_retriever, IncidentRetriever

logger = logging.getLogger(__name__)


class RCAService:
    """
    Orchestrates the full RCA pipeline:
      1. Text extraction (PDF/TXT files)
      2. RAG search (ChromaDB) ← NOW REAL in Step 3
      3. AI analysis (LangGraph) ← Coming in Step 4
      4. Database storage (PostgreSQL) ← Coming in Step 5
    """

    def __init__(self):
        self._reports_store: dict = {}
        self._retriever: Optional[IncidentRetriever] = None
        logger.info("[RCAService] Initialized")

    def _get_retriever(self) -> IncidentRetriever:
        """Get the ChromaDB retriever (lazy loaded)."""
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    # ============================================================
    # 1. ANALYZE INCIDENT
    # ============================================================
    async def analyze_incident(
        self,
        incident_id: str,
        incident: IncidentInput,
        llm_provider_override: Optional[str] = None
    ) -> RCAReport:
        """Full RCA pipeline orchestration."""
        logger.info(f"[RCAService] Starting analysis: {incident_id}")

        report = RCAReport(
            incident_id=incident_id,
            status=RCAStatus.ANALYZING,
            incident=incident,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            llm_provider_used=llm_provider_override or "groq"
        )
        self._reports_store[incident_id] = report

        try:
            # Step 3: REAL ChromaDB search
            similar_incidents = await self._find_similar_incidents(incident)

            # Step 4: AI agent (stub — becomes real in Step 4)
            analysis = await self._run_rca_agent(incident, similar_incidents)

            report.analysis = analysis
            report.status = RCAStatus.COMPLETED
            report.completed_at = datetime.utcnow()
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report

            logger.info(f"[RCAService] Analysis complete: {incident_id}")
            return report

        except Exception as e:
            report.status = RCAStatus.FAILED
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report
            logger.error(f"[RCAService] Analysis failed {incident_id}: {e}")
            raise

    # ============================================================
    # 2. FIND SIMILAR INCIDENTS — NOW REAL (Step 3)
    # ============================================================
    async def _find_similar_incidents(
        self,
        incident: IncidentInput
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB for similar past incidents.
        REAL implementation — was a stub in Step 2.
        """
        logger.info("[RCAService] Searching ChromaDB for similar incidents...")

        # Build search query from incident details
        search_query = f"{incident.title}. {incident.description}"
        if incident.affected_systems:
            search_query += f" Systems: {', '.join(incident.affected_systems)}"

        retriever = self._get_retriever()

        results = await retriever.search(
            query=search_query,
            top_k=3,
            min_relevance=0.3
        )

        if results:
            logger.info(f"[RCAService] Found {len(results)} similar incidents in ChromaDB")
            for r in results:
                logger.info(f"  → {r['incident_id']} (similarity: {r['similarity_score']})")
        else:
            logger.info("[RCAService] No similar incidents found in ChromaDB")

        return results

    # ============================================================
    # 3. RUN RCA AGENT (stub — real in Step 4)
    # ============================================================
    async def _run_rca_agent(
        self,
        incident: IncidentInput,
        similar_incidents: List[Dict[str, Any]]
    ) -> RCAAnalysis:
        """
        Run LangGraph agent to generate RCA.
        STUB — will be replaced with real agent in Step 4.
        Now passes similar_incidents context to the stub output.
        """
        logger.info("[RCAService] Running RCA agent (Step 4 will make this real)")

        # Format similar incidents for display
        similar_summaries = []
        for s in similar_incidents:
            similar_summaries.append(
                f"[{s['incident_id']}] {s['title']} "
                f"(similarity: {s['similarity_score']:.0%})"
            )

        return RCAAnalysis(
            incident_summary=(
                f"Incident: {incident.title}. "
                f"Severity: {incident.severity.value}. "
                f"Affected: {', '.join(incident.affected_systems) if incident.affected_systems else 'N/A'}."
            ),
            timeline_reconstruction=(
                incident.incident_timeline or
                "No timeline provided. Add timeline for better analysis."
            ),
            root_cause=(
                "⚠️ STUB: LangGraph AI agent will be wired in Step 4. "
                "Real root cause analysis will appear here."
            ),
            contributing_factors=[
                "STUB: Contributing factors will be identified by AI in Step 4"
            ],
            impact_assessment=(
                f"Severity {incident.severity.value} incident. "
                f"Affected systems: {', '.join(incident.affected_systems) if incident.affected_systems else 'N/A'}."
            ),
            immediate_actions_taken=["STUB: Actions will be extracted by AI in Step 4"],
            corrective_actions=["STUB: Will be generated by AI in Step 4"],
            preventive_measures=["STUB: Will be generated by AI in Step 4"],
            lessons_learned="STUB: Will be generated by AI in Step 4.",
            similar_incidents=similar_summaries,
            confidence_score=0.0
        )

    # ============================================================
    # 4. INGEST INCIDENT TO KNOWLEDGE BASE (NEW in Step 3)
    # ============================================================
    async def ingest_incident(
        self,
        incident_id: str,
        incident: IncidentInput,
        root_cause: Optional[str] = None,
        resolution: Optional[str] = None
    ) -> bool:
        """Add a resolved incident to ChromaDB knowledge base."""
        retriever = self._get_retriever()
        return await retriever.ingest(
            incident_id=incident_id,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            affected_systems=incident.affected_systems,
            root_cause=root_cause,
            resolution=resolution
        )

    # ============================================================
    # 5. KNOWLEDGE BASE STATS (NEW in Step 3)
    # ============================================================
    async def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Return ChromaDB knowledge base statistics."""
        retriever = self._get_retriever()
        return await retriever.get_stats()

    # ============================================================
    # 6. SEARCH KNOWLEDGE BASE (NEW in Step 3)
    # ============================================================
    async def search_similar(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Public method to search the knowledge base."""
        retriever = self._get_retriever()
        return await retriever.search(query=query, top_k=top_k)

    # ============================================================
    # 7. SEED SAMPLE DATA (NEW in Step 3)
    # ============================================================
    async def seed_knowledge_base(self) -> int:
        """Seed the knowledge base with sample incidents for demos."""
        retriever = self._get_retriever()
        return await retriever.seed_sample_incidents()

    # ============================================================
    # 8-10. GET / LIST / DELETE REPORTS
    # ============================================================
    async def get_report(self, incident_id: str) -> Optional[RCAReport]:
        return self._reports_store.get(incident_id)

    async def list_reports(
        self,
        skip: int = 0,
        limit: int = 10,
        status_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Tuple[List[RCAReport], int]:
        reports = list(self._reports_store.values())
        if status_filter:
            reports = [r for r in reports if r.status.value == status_filter]
        if severity_filter:
            reports = [r for r in reports if r.incident.severity.value == severity_filter]
        total = len(reports)
        return reports[skip: skip + limit], total

    async def delete_report(self, incident_id: str) -> bool:
        if incident_id in self._reports_store:
            del self._reports_store[incident_id]
            return True
        return False

    # ============================================================
    # 11. EXTRACT TEXT FROM FILE
    # ============================================================
    async def extract_text_from_file(
        self,
        content: bytes,
        filename: str,
        file_extension: str
    ) -> str:
        if file_extension in [".txt", ".text"]:
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1")
        elif file_extension == ".pdf":
            try:
                import pypdf
                pdf_reader = pypdf.PdfReader(io.BytesIO(content))
                text_parts = []
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"[Page {i+1}]\n{text}")
                if not text_parts:
                    raise ValueError("No text could be extracted from PDF.")
                return "\n\n".join(text_parts)
            except ImportError:
                raise RuntimeError("pypdf not installed. Run: pip install pypdf")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
