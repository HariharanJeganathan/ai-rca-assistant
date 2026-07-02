"""
rca_service.py — Business Logic Layer
Updated: PostgreSQL as primary store, ChromaDB auto-ingestion after RCA,
         in-memory fallback when DB unavailable.
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
        # In-memory fallback (used when PostgreSQL is unavailable)
        self._reports_store: dict = {}
        self._retriever: Optional[IncidentRetriever] = None
        self._agent = None
        self._db_available: Optional[bool] = None
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

    async def _is_db_available(self) -> bool:
        """Check PostgreSQL availability. Re-check each time (Supabase can recover)."""
        try:
            from db.database import check_database_connection
            result = await check_database_connection()
            self._db_available = result
            return result
        except Exception:
            self._db_available = False
            return False

    # ============================================================
    # 1. ANALYZE INCIDENT
    # ============================================================
    async def analyze_incident(
        self,
        incident_id: str,
        incident: IncidentInput,
        llm_provider_override: Optional[str] = None
    ) -> RCAReport:
        logger.info(f"[RCAService] Starting: {incident_id}")

        report = RCAReport(
            incident_id=incident_id,
            status=RCAStatus.ANALYZING,
            incident=incident,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            llm_provider_used=llm_provider_override or "groq"
        )
        self._reports_store[incident_id] = report

        # Save ANALYZING state to DB
        db_ok = await self._is_db_available()
        if db_ok:
            await self._db_save(report, create=True)

        try:
            # Step 1: Search ChromaDB for similar incidents
            similar_incidents = await self._find_similar_incidents(incident)

            # Step 2: Run LangGraph agent
            analysis = await self._run_rca_agent(incident, similar_incidents)

            report.analysis = analysis
            report.status = RCAStatus.COMPLETED
            report.completed_at = datetime.utcnow()
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report

            # Step 3: Save completed report to DB
            if db_ok:
                await self._db_save(report, create=False)

            # Step 4: Auto-ingest this incident into ChromaDB knowledge base
            # So future similar incidents can find it!
            await self._auto_ingest_to_kb(incident_id, incident, analysis)

            logger.info(f"[RCAService] Complete: {incident_id}")
            return report

        except Exception as e:
            report.status = RCAStatus.FAILED
            report.updated_at = datetime.utcnow()
            self._reports_store[incident_id] = report
            if db_ok:
                await self._db_save(report, create=False)
            logger.error(f"[RCAService] Failed {incident_id}: {e}")
            raise

    # ============================================================
    # 2. AUTO-INGEST COMPLETED RCA INTO CHROMADB
    # ============================================================
    async def _auto_ingest_to_kb(
        self,
        incident_id: str,
        incident: IncidentInput,
        analysis: RCAAnalysis
    ):
        """
        After every successful RCA, automatically add the incident to ChromaDB.
        This means future similar incidents will find it in the knowledge base.
        The KB grows with every RCA you run — getting smarter over time.
        """
        try:
            retriever = self._get_retriever()
            root_cause = analysis.root_cause if analysis else None
            resolution = ". ".join(analysis.corrective_actions) if analysis and analysis.corrective_actions else None

            success = await retriever.ingest(
                incident_id=incident_id,
                title=incident.title,
                description=incident.description,
                severity=incident.severity.value,
                affected_systems=incident.affected_systems or [],
                root_cause=root_cause,
                resolution=resolution
            )
            if success:
                logger.info(f"[RCAService] Auto-ingested {incident_id} into ChromaDB KB")
        except Exception as e:
            logger.warning(f"[RCAService] Auto-ingest to KB failed (non-critical): {e}")

    # ============================================================
    # 3. FIND SIMILAR INCIDENTS
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
            logger.warning(f"[RCAService] ChromaDB search failed: {e}")
            return []

    # ============================================================
    # 4. RUN RCA AGENT
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
    # 5. DB SAVE HELPER
    # ============================================================
    async def _db_save(self, report: RCAReport, create: bool = True):
        """Save or update a report in PostgreSQL."""
        try:
            from db.database import AsyncSessionLocal
            from db.postgres import RCARepository
            async with AsyncSessionLocal() as session:
                repo = RCARepository(session)
                if create:
                    try:
                        await repo.create_report(report)
                    except ValueError:
                        await repo.update_report(report)  # Already exists
                else:
                    await repo.update_report(report)
                await session.commit()
        except Exception as e:
            logger.warning(f"[RCAService] DB save failed (using memory): {e}")

    # ============================================================
    # 6. GET REPORT — DB first, memory fallback
    # ============================================================
    async def get_report(self, incident_id: str) -> Optional[RCAReport]:
        db_ok = await self._is_db_available()
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
        return self._reports_store.get(incident_id)

    # ============================================================
    # 7. LIST REPORTS — DB first, memory fallback
    # ============================================================
    async def list_reports(self, skip=0, limit=10, status_filter=None, severity_filter=None):
        db_ok = await self._is_db_available()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    return await repo.list_reports(skip, limit, status_filter, severity_filter)
            except Exception as e:
                logger.warning(f"[RCAService] DB list failed: {e}")

        # Memory fallback
        reports = list(self._reports_store.values())
        if status_filter:
            reports = [r for r in reports if r.status.value == status_filter]
        if severity_filter:
            reports = [r for r in reports if r.incident.severity.value == severity_filter]
        return reports[skip:skip + limit], len(reports)

    # ============================================================
    # 8. DELETE REPORT
    # ============================================================
    async def delete_report(self, incident_id: str) -> bool:
        db_ok = await self._is_db_available()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    result = await repo.delete_report(incident_id)
                    await session.commit()
                    if result:
                        return True
            except Exception as e:
                logger.warning(f"[RCAService] DB delete failed: {e}")
        if incident_id in self._reports_store:
            del self._reports_store[incident_id]
            return True
        return False

    # ============================================================
    # 9. STATS
    # ============================================================
    async def get_report_stats(self) -> Dict[str, Any]:
        db_ok = await self._is_db_available()
        if db_ok:
            try:
                from db.database import AsyncSessionLocal
                from db.postgres import RCARepository
                async with AsyncSessionLocal() as session:
                    repo = RCARepository(session)
                    return await repo.get_stats()
            except Exception as e:
                logger.warning(f"[RCAService] DB stats failed: {e}")
        reports = list(self._reports_store.values())
        return {
            "total_reports": len(reports),
            "by_status": {"completed": sum(1 for r in reports if r.status == RCAStatus.COMPLETED)},
            "by_severity": {},
            "average_confidence_score": 0.0,
            "note": "Database not connected"
        }

    # ============================================================
    # 10. KNOWLEDGE BASE OPERATIONS
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
                    raise ValueError("No text extracted from PDF.")
                return "\n\n".join(text_parts)
            except ImportError:
                raise RuntimeError("pypdf not installed.")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
