"""
rca_service.py — Business Logic Layer
(Updated: Step 4 — LangGraph agent wired in)

WHAT CHANGED FROM STEP 3:
  - _run_rca_agent() now calls the REAL LangGraph agent
  - Imports get_rca_agent from agents/rca_agent.py
  - Everything else stays the same
"""

import logging
import io
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from models.schemas import IncidentInput, RCAReport, RCAAnalysis, RCAStatus
from rag.retriever import get_retriever, IncidentRetriever

logger = logging.getLogger(__name__)


class RCAService:

    def __init__(self):
        self._reports_store: dict = {}
        self._retriever: Optional[IncidentRetriever] = None
        self._agent = None
        logger.info("[RCAService] Initialized")

    def _get_retriever(self) -> IncidentRetriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def _get_agent(self):
        """Get the LangGraph agent (lazy loaded)."""
        if self._agent is None:
            from agents.rca_agent import get_rca_agent
            self._agent = get_rca_agent()
        return self._agent

    # ============================================================
    # 1. ANALYZE INCIDENT
    # ============================================================
    async def analyze_incident(
        self,
        incident_id: str,
        incident: IncidentInput,
        llm_provider_override: Optional[str] = None
    ) -> RCAReport:
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
            # Step 3: ChromaDB RAG search
            similar_incidents = await self._find_similar_incidents(incident)

            # Step 4: REAL LangGraph agent
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
            logger.error(f"[RCAService] Failed {incident_id}: {e}")
            raise

    # ============================================================
    # 2. FIND SIMILAR INCIDENTS (ChromaDB)
    # ============================================================
    async def _find_similar_incidents(self, incident: IncidentInput) -> List[Dict]:
        logger.info("[RCAService] Searching ChromaDB...")
        query = f"{incident.title}. {incident.description}"
        if incident.affected_systems:
            query += f" Systems: {', '.join(incident.affected_systems)}"
        retriever = self._get_retriever()
        results = await retriever.search(query=query, top_k=3, min_relevance=0.3)
        logger.info(f"[RCAService] Found {len(results)} similar incidents")
        return results

    # ============================================================
    # 3. RUN RCA AGENT — NOW REAL (Step 4)
    # ============================================================
    async def _run_rca_agent(
        self,
        incident: IncidentInput,
        similar_incidents: List[Dict]
    ) -> RCAAnalysis:
        """
        Run the REAL LangGraph 7-step agent.
        Was a stub in Steps 2 and 3. Now fully wired.
        """
        logger.info("[RCAService] Running LangGraph agent (7 steps)...")
        agent = self._get_agent()
        analysis = await agent.run(
            incident=incident,
            incident_id=f"run-{datetime.utcnow().timestamp()}",
            similar_incidents=similar_incidents
        )
        logger.info(f"[RCAService] Agent complete. Confidence: {analysis.confidence_score:.0%}")
        return analysis

    # ============================================================
    # 4-10. Knowledge base + CRUD (unchanged from Step 3)
    # ============================================================
    async def ingest_incident(self, incident_id, incident, root_cause=None, resolution=None):
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

    async def get_knowledge_base_stats(self):
        return await self._get_retriever().get_stats()

    async def search_similar(self, query, top_k=5):
        return await self._get_retriever().search(query=query, top_k=top_k)

    async def seed_knowledge_base(self):
        return await self._get_retriever().seed_sample_incidents()

    async def get_report(self, incident_id):
        return self._reports_store.get(incident_id)

    async def list_reports(self, skip=0, limit=10, status_filter=None, severity_filter=None):
        reports = list(self._reports_store.values())
        if status_filter:
            reports = [r for r in reports if r.status.value == status_filter]
        if severity_filter:
            reports = [r for r in reports if r.incident.severity.value == severity_filter]
        total = len(reports)
        return reports[skip: skip + limit], total

    async def delete_report(self, incident_id):
        if incident_id in self._reports_store:
            del self._reports_store[incident_id]
            return True
        return False

    async def extract_text_from_file(self, content, filename, file_extension):
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
                raise RuntimeError("pypdf not installed.")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
