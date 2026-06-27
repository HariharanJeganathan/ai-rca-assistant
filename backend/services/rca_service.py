"""
rca_service.py — Business Logic Layer
Updated: Graceful fallback when PostgreSQL is unavailable.
Analysis works even without DB — uses in-memory store as fallback.
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
        # Always keep in-memory store as fallback
        self._reports_store: dict = {}
        self._retriever: Optional[IncidentRetriever] = None
        self._agent = None
        self._db_available = None  # None = not checked yet
        logger.info("[RCAService] Initialized")

    def _get_retriever(self) -> IncidentRetriever:
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def _get_agent(self):
        if self._agent is None:
            from agents.rca_agent import get_rca_agent
            self._agent = get_rca_agent()
        return self._agent

    async def _check_db(self) -> bool:
        """Check if PostgreSQL is available. Cache result."""
        if self._db_available is not None:
            return self._db_available
        try:
            from db.database import check_database_connection
            self._db_available = await check_database_connection()
        except Exception:
            self._db_available = False
        logger.info(f"[RCAService] DB available: {self._db_available}")
        return self._db_available

    # ============================================================
    # 1. ANALYZE INCIDENT — Full pipeline with DB fallback
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

        # Save to in-memory store always
        self._reports_store[incident_id] = report

        # Try PostgreSQL too — but don't fail if unavailable
        db_ok = await self._check_db()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    await repo.create_report(report)
                    await session.commit()
            except Exception as e:
                logger.warning(f"[RCAService] DB save failed (continuing anyway): {e}")

        try:
            # ChromaDB RAG search
            similar_incidents = await self._find_similar_incidents(incident)

            # LangGraph AI agent
            analysis = await self._run_rca_agent(incident, similar_incidents)

            report.analysis = analysis
            report.status = RCAStatus.COMPLETED
            report.completed_at = datetime.utcnow()
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report

            # Try to update in PostgreSQL
            if db_ok:
                try:
                    from db.database import AsyncSessionLocal
                    from db.postgres import RCARepository
                    async with AsyncSessionLocal() as session:
                        repo = RCARepository(session)
                        await repo.update_report(report)
                        await session.commit()
                except Exception as e:
                    logger.warning(f"[RCAService] DB update failed (report still in memory): {e}")

            logger.info(f"[RCAService] Analysis complete: {incident_id}")
            return report

        except Exception as e:
            report.status = RCAStatus.FAILED
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report
            logger.error(f"[RCAService] Analysis failed {incident_id}: {e}")
            raise

    # ============================================================
    # 2. FIND SIMILAR INCIDENTS (ChromaDB)
    # ============================================================
    async def _find_similar_incidents(self, incident: IncidentInput) -> List[Dict]:
        try:
            query = f"{incident.title}. {incident.description}"
            if incident.affected_systems:
                query += f" Systems: {', '.join(incident.affected_systems)}"
            retriever = self._get_retriever()
            results = await retriever.search(query=query, top_k=3, min_relevance=0.3)
            logger.info(f"[RCAService] Found {len(results)} similar incidents")
            return results
        except Exception as e:
            logger.warning(f"[RCAService] ChromaDB search failed (continuing): {e}")
            return []

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
    # 4. GET REPORT — DB first, fallback to memory
    # ============================================================
    async def get_report(self, incident_id: str) -> Optional[RCAReport]:
        db_ok = await self._check_db()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    result = await repo.get_report(incident_id)
                    if result:
                        return result
            except Exception as e:
                logger.warning(f"[RCAService] DB get failed: {e}")
        # Fallback to memory
        return self._reports_store.get(incident_id)

    # ============================================================
    # 5. LIST REPORTS — DB first, fallback to memory
    # ============================================================
    async def list_reports(self, skip=0, limit=10, status_filter=None, severity_filter=None):
        db_ok = await self._check_db()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    return await repo.list_reports(skip, limit, status_filter, severity_filter)
            except Exception as e:
                logger.warning(f"[RCAService] DB list failed: {e}")

        # Fallback to memory
        reports = list(self._reports_store.values())
        if status_filter:
            reports = [r for r in reports if r.status.value == status_filter]
        if severity_filter:
            reports = [r for r in reports if r.incident.severity.value == severity_filter]
        return reports[skip:skip + limit], len(reports)

    # ============================================================
    # 6. DELETE REPORT
    # ============================================================
    async def delete_report(self, incident_id: str) -> bool:
        db_ok = await self._check_db()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    result = await repo.delete_report(incident_id)
                    await session.commit()
                    return result
            except Exception as e:
                logger.warning(f"[RCAService] DB delete failed: {e}")

        if incident_id in self._reports_store:
            del self._reports_store[incident_id]
            return True
        return False

    # ============================================================
    # 7. STATS
    # ============================================================
    async def get_report_stats(self) -> Dict[str, Any]:
        db_ok = await self._check_db()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    return await repo.get_stats()
            except Exception as e:
                logger.warning(f"[RCAService] DB stats failed: {e}")

        # Memory fallback stats
        reports = list(self._reports_store.values())
        return {
            "total_reports": len(reports),
            "by_status": {"completed": len([r for r in reports if r.status == RCAStatus.COMPLETED])},
            "by_severity": {},
            "average_confidence_score": 0.0,
            "note": "Database not connected — showing in-memory stats"
        }

    # ============================================================
    # 8-10. Knowledge base operations
    # ============================================================
    async def ingest_incident(self, incident_id, incident, root_cause=None, resolution=None):
        retriever = self._get_retriever()
        return await retriever.ingest(
            incident_id=incident_id, title=incident.title,
            description=incident.description, severity=incident.severity.value,
            affected_systems=incident.affected_systems,
            root_cause=root_cause, resolution=resolution
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
