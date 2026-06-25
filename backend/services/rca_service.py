"""
rca_service.py — Business Logic Layer
(Updated: Step 5 — PostgreSQL wired in)

WHAT CHANGED FROM STEP 4:
  - _reports_store (in-memory dict) REPLACED by PostgreSQL
  - All CRUD methods now use RCARepository
  - DB session injected via get_db dependency
"""

import logging
import io
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from models.schemas import IncidentInput, RCAReport, RCAAnalysis, RCAStatus
from rag.retriever import get_retriever, IncidentRetriever
from db.postgres import RCARepository
from db.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


class RCAService:

    def __init__(self):
        self._retriever: Optional[IncidentRetriever] = None
        self._agent = None
        logger.info("[RCAService] Initialized with PostgreSQL backend")

    def _get_retriever(self) -> IncidentRetriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def _get_agent(self):
        if self._agent is None:
            from agents.rca_agent import get_rca_agent
            self._agent = get_rca_agent()
        return self._agent

    async def _get_repo(self) -> Tuple[RCARepository, AsyncSession]:
        """Get a repository + session pair."""
        session = AsyncSessionLocal()
        repo = RCARepository(session)
        return repo, session

    # ============================================================
    # 1. ANALYZE INCIDENT — Full pipeline
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

        # Save initial ANALYZING state to PostgreSQL
        async with AsyncSessionLocal() as session:
            repo = RCARepository(session)
            try:
                await repo.create_report(report)
                await session.commit()
            except ValueError:
                pass  # Already exists, continue

        try:
            # ChromaDB RAG search
            similar_incidents = await self._find_similar_incidents(incident)

            # LangGraph agent
            analysis = await self._run_rca_agent(incident, similar_incidents)

            # Update report with completed analysis
            report.analysis = analysis
            report.status = RCAStatus.COMPLETED
            report.completed_at = datetime.utcnow()
            report.updated_at = datetime.utcnow()

            # Save completed report to PostgreSQL
            async with AsyncSessionLocal() as session:
                repo = RCARepository(session)
                updated = await repo.update_report(report)
                await session.commit()
                if updated:
                    report = updated

            logger.info(f"[RCAService] Analysis complete: {incident_id}")
            return report

        except Exception as e:
            # Mark as FAILED in PostgreSQL
            report.status = RCAStatus.FAILED
            report.updated_at = datetime.utcnow()
            async with AsyncSessionLocal() as session:
                repo = RCARepository(session)
                await repo.update_report(report)
                await session.commit()
            logger.error(f"[RCAService] Failed: {incident_id}: {e}")
            raise

    # ============================================================
    # 2. FIND SIMILAR INCIDENTS (ChromaDB)
    # ============================================================
    async def _find_similar_incidents(self, incident: IncidentInput) -> List[Dict]:
        query = f"{incident.title}. {incident.description}"
        if incident.affected_systems:
            query += f" Systems: {', '.join(incident.affected_systems)}"
        retriever = self._get_retriever()
        results = await retriever.search(query=query, top_k=3, min_relevance=0.3)
        logger.info(f"[RCAService] Found {len(results)} similar incidents")
        return results

    # ============================================================
    # 3. RUN RCA AGENT (LangGraph)
    # ============================================================
    async def _run_rca_agent(self, incident, similar_incidents) -> RCAAnalysis:
        logger.info("[RCAService] Running LangGraph agent...")
        agent = self._get_agent()
        analysis = await agent.run(
            incident=incident,
            incident_id=f"run-{datetime.utcnow().timestamp()}",
            similar_incidents=similar_incidents
        )
        logger.info(f"[RCAService] Agent done. Confidence: {analysis.confidence_score:.0%}")
        return analysis

    # ============================================================
    # 4. GET ONE REPORT — from PostgreSQL
    # ============================================================
    async def get_report(self, incident_id: str) -> Optional[RCAReport]:
        async with AsyncSessionLocal() as session:
            repo = RCARepository(session)
            return await repo.get_report(incident_id)

    # ============================================================
    # 5. LIST REPORTS — from PostgreSQL
    # ============================================================
    async def list_reports(
        self,
        skip: int = 0,
        limit: int = 10,
        status_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Tuple[List[RCAReport], int]:
        async with AsyncSessionLocal() as session:
            repo = RCARepository(session)
            return await repo.list_reports(
                skip=skip,
                limit=limit,
                status_filter=status_filter,
                severity_filter=severity_filter
            )

    # ============================================================
    # 6. DELETE REPORT — soft delete in PostgreSQL
    # ============================================================
    async def delete_report(self, incident_id: str) -> bool:
        async with AsyncSessionLocal() as session:
            repo = RCARepository(session)
            result = await repo.delete_report(incident_id)
            await session.commit()
            return result

    # ============================================================
    # 7. GET DB STATS
    # ============================================================
    async def get_report_stats(self) -> Dict[str, Any]:
        async with AsyncSessionLocal() as session:
            repo = RCARepository(session)
            return await repo.get_stats()

    # ============================================================
    # 8-10. Knowledge base operations (ChromaDB — unchanged)
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

    # ============================================================
    # 11. EXTRACT TEXT FROM FILE
    # ============================================================
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
