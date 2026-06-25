"""
postgres.py — PostgreSQL Repository Layer
==========================================
The REPOSITORY PATTERN separates database operations from business logic.

Think of it like a library:
  - postgres.py = the LIBRARIAN (knows how to find/store books)
  - rca_service.py = the READER (asks librarian for books, doesn't care how stored)

WHY THIS PATTERN?
  If you later switch from PostgreSQL to MongoDB, you only change postgres.py.
  rca_service.py doesn't change at all. That's "loose coupling" — a senior
  engineering principle interviewers love to discuss.

EVERY DATABASE OPERATION is in this file:
  create_report()  → INSERT
  get_report()     → SELECT by ID
  list_reports()   → SELECT with filters + pagination
  update_report()  → UPDATE
  delete_report()  → soft DELETE (mark as deleted, don't actually remove)
  get_stats()      → aggregate queries (COUNT, GROUP BY)
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, and_, or_
from sqlalchemy.exc import IntegrityError

from db.models import RCAReportModel, RCAStatusEnum, SeverityEnum
from models.schemas import RCAReport, RCAAnalysis, IncidentInput, RCAStatus

logger = logging.getLogger(__name__)


class RCARepository:
    """
    Handles all PostgreSQL operations for RCA reports.

    Every method is async (non-blocking) — the app can handle
    other requests while waiting for the database.
    """

    def __init__(self, db: AsyncSession):
        """
        Takes a database session as input.
        The session is created by FastAPI's dependency injection (get_db).
        """
        self.db = db

    # ============================================================
    # CREATE — Save a new RCA report
    # ============================================================
    async def create_report(self, report: RCAReport) -> RCAReportModel:
        """
        INSERT a new RCA report into PostgreSQL.

        Converts Pydantic RCAReport → SQLAlchemy RCAReportModel
        then saves it to the database.
        """
        logger.info(f"[DB] Creating report for incident: {report.incident_id}")

        try:
            # Build the database model from the Pydantic schema
            db_report = RCAReportModel(
                incident_id=report.incident_id,
                status=RCAStatusEnum(report.status.value),
                llm_provider_used=report.llm_provider_used,

                # Incident details
                incident_title=report.incident.title,
                incident_description=report.incident.description,
                incident_severity=SeverityEnum(report.incident.severity.value),
                incident_affected_systems=report.incident.affected_systems,
                incident_timeline=report.incident.incident_timeline,
                incident_additional_context=report.incident.additional_context,

                # Analysis (may be None if still pending)
                **self._extract_analysis_fields(report.analysis),

                created_at=report.created_at or datetime.utcnow(),
                updated_at=report.updated_at or datetime.utcnow(),
                completed_at=report.completed_at,
            )

            self.db.add(db_report)
            await self.db.flush()   # flush = send to DB but don't commit yet
                                    # lets us get the auto-generated ID

            logger.info(f"[DB] Report created: id={db_report.id}, incident={report.incident_id}")
            return db_report

        except IntegrityError:
            await self.db.rollback()
            logger.warning(f"[DB] Report already exists for: {report.incident_id}")
            raise ValueError(f"Report already exists for incident: {report.incident_id}")

    # ============================================================
    # READ — Get one report by incident ID
    # ============================================================
    async def get_report(self, incident_id: str) -> Optional[RCAReport]:
        """
        SELECT one report by incident_id.
        Returns None if not found.
        """
        logger.info(f"[DB] Fetching report: {incident_id}")

        result = await self.db.execute(
            select(RCAReportModel).where(
                and_(
                    RCAReportModel.incident_id == incident_id,
                    RCAReportModel.is_deleted == False  # noqa: E712
                )
            )
        )
        db_report = result.scalar_one_or_none()

        if not db_report:
            logger.info(f"[DB] Report not found: {incident_id}")
            return None

        return self._to_pydantic(db_report)

    # ============================================================
    # READ — List reports with filters and pagination
    # ============================================================
    async def list_reports(
        self,
        skip: int = 0,
        limit: int = 10,
        status_filter: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Tuple[List[RCAReport], int]:
        """
        SELECT multiple reports with optional filtering and pagination.

        Returns: (list_of_reports, total_count)
        total_count is used for pagination UI (showing "Page 1 of 5")
        """
        logger.info(f"[DB] Listing reports: skip={skip}, limit={limit}")

        # Build WHERE conditions
        conditions = [RCAReportModel.is_deleted == False]  # noqa: E712

        if status_filter:
            try:
                conditions.append(
                    RCAReportModel.status == RCAStatusEnum(status_filter)
                )
            except ValueError:
                pass  # Invalid status filter — ignore it

        if severity_filter:
            try:
                conditions.append(
                    RCAReportModel.incident_severity == SeverityEnum(severity_filter)
                )
            except ValueError:
                pass  # Invalid severity filter — ignore it

        # COUNT query (for pagination total)
        count_result = await self.db.execute(
            select(func.count(RCAReportModel.id)).where(and_(*conditions))
        )
        total = count_result.scalar() or 0

        # SELECT query with pagination
        result = await self.db.execute(
            select(RCAReportModel)
            .where(and_(*conditions))
            .order_by(RCAReportModel.created_at.desc())  # Newest first
            .offset(skip)
            .limit(limit)
        )
        db_reports = result.scalars().all()

        reports = [self._to_pydantic(r) for r in db_reports]
        logger.info(f"[DB] Found {len(reports)} reports (total: {total})")

        return reports, total

    # ============================================================
    # UPDATE — Update report status and analysis
    # ============================================================
    async def update_report(self, report: RCAReport) -> Optional[RCAReport]:
        """
        UPDATE an existing report (e.g. add analysis when AI finishes).
        """
        logger.info(f"[DB] Updating report: {report.incident_id}")

        result = await self.db.execute(
            select(RCAReportModel).where(
                RCAReportModel.incident_id == report.incident_id
            )
        )
        db_report = result.scalar_one_or_none()

        if not db_report:
            logger.warning(f"[DB] Cannot update — report not found: {report.incident_id}")
            return None

        # Update fields
        db_report.status = RCAStatusEnum(report.status.value)
        db_report.updated_at = datetime.utcnow()

        if report.completed_at:
            db_report.completed_at = report.completed_at

        # Update analysis fields if analysis is available
        if report.analysis:
            analysis_fields = self._extract_analysis_fields(report.analysis)
            for field, value in analysis_fields.items():
                setattr(db_report, field, value)

        await self.db.flush()
        logger.info(f"[DB] Report updated: {report.incident_id}")
        return self._to_pydantic(db_report)

    # ============================================================
    # DELETE — Soft delete (mark as deleted, keep data)
    # ============================================================
    async def delete_report(self, incident_id: str) -> bool:
        """
        Soft delete — marks report as deleted but keeps data in DB.

        WHY SOFT DELETE?
          - Keeps audit trail (important for compliance)
          - Can be recovered if deleted by mistake
          - Common pattern in enterprise applications
        """
        logger.info(f"[DB] Soft deleting report: {incident_id}")

        result = await self.db.execute(
            update(RCAReportModel)
            .where(RCAReportModel.incident_id == incident_id)
            .values(is_deleted=True, updated_at=datetime.utcnow())
        )

        if result.rowcount == 0:
            logger.warning(f"[DB] Cannot delete — report not found: {incident_id}")
            return False

        logger.info(f"[DB] Report soft deleted: {incident_id}")
        return True

    # ============================================================
    # STATS — Aggregate queries
    # ============================================================
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics about all reports.
        Uses SQL COUNT and GROUP BY — efficient even with millions of rows.
        """
        logger.info("[DB] Getting report statistics")

        # Total count
        total_result = await self.db.execute(
            select(func.count(RCAReportModel.id))
            .where(RCAReportModel.is_deleted == False)  # noqa: E712
        )
        total = total_result.scalar() or 0

        # Count by status
        status_result = await self.db.execute(
            select(RCAReportModel.status, func.count(RCAReportModel.id))
            .where(RCAReportModel.is_deleted == False)  # noqa: E712
            .group_by(RCAReportModel.status)
        )
        by_status = {row[0].value: row[1] for row in status_result.all()}

        # Count by severity
        severity_result = await self.db.execute(
            select(RCAReportModel.incident_severity, func.count(RCAReportModel.id))
            .where(RCAReportModel.is_deleted == False)  # noqa: E712
            .group_by(RCAReportModel.incident_severity)
        )
        by_severity = {row[0].value: row[1] for row in severity_result.all()}

        # Average confidence score
        avg_result = await self.db.execute(
            select(func.avg(RCAReportModel.analysis_confidence_score))
            .where(
                and_(
                    RCAReportModel.is_deleted == False,  # noqa: E712
                    RCAReportModel.analysis_confidence_score.isnot(None)
                )
            )
        )
        avg_confidence = avg_result.scalar()

        return {
            "total_reports": total,
            "by_status": by_status,
            "by_severity": by_severity,
            "average_confidence_score": round(float(avg_confidence), 2) if avg_confidence else 0.0
        }

    # ============================================================
    # HELPERS — Convert between SQLAlchemy and Pydantic
    # ============================================================
    def _extract_analysis_fields(self, analysis: Optional[RCAAnalysis]) -> dict:
        """
        Extract analysis fields into a flat dict for the DB model.
        Returns empty/None values if analysis is None (pending state).
        """
        if not analysis:
            return {
                "analysis_summary": None,
                "analysis_root_cause": None,
                "analysis_impact": None,
                "analysis_timeline": None,
                "analysis_lessons_learned": None,
                "analysis_contributing_factors": [],
                "analysis_immediate_actions": [],
                "analysis_corrective_actions": [],
                "analysis_preventive_measures": [],
                "analysis_similar_incidents": [],
                "analysis_confidence_score": 0.0,
            }

        return {
            "analysis_summary": analysis.incident_summary,
            "analysis_root_cause": analysis.root_cause,
            "analysis_impact": analysis.impact_assessment,
            "analysis_timeline": analysis.timeline_reconstruction,
            "analysis_lessons_learned": analysis.lessons_learned,
            "analysis_contributing_factors": analysis.contributing_factors,
            "analysis_immediate_actions": analysis.immediate_actions_taken,
            "analysis_corrective_actions": analysis.corrective_actions,
            "analysis_preventive_measures": analysis.preventive_measures,
            "analysis_similar_incidents": analysis.similar_incidents,
            "analysis_confidence_score": analysis.confidence_score,
        }

    def _to_pydantic(self, db_report: RCAReportModel) -> RCAReport:
        """
        Convert SQLAlchemy model → Pydantic schema.
        This is what gets returned in API responses.
        """
        from models.schemas import IncidentSeverity

        incident = IncidentInput(
            title=db_report.incident_title,
            description=db_report.incident_description,
            severity=IncidentSeverity(db_report.incident_severity.value),
            affected_systems=db_report.incident_affected_systems or [],
            incident_timeline=db_report.incident_timeline,
            additional_context=db_report.incident_additional_context,
        )

        analysis = None
        if db_report.analysis_root_cause:
            analysis = RCAAnalysis(
                incident_summary=db_report.analysis_summary or "",
                timeline_reconstruction=db_report.analysis_timeline or "",
                root_cause=db_report.analysis_root_cause or "",
                contributing_factors=db_report.analysis_contributing_factors or [],
                impact_assessment=db_report.analysis_impact or "",
                immediate_actions_taken=db_report.analysis_immediate_actions or [],
                corrective_actions=db_report.analysis_corrective_actions or [],
                preventive_measures=db_report.analysis_preventive_measures or [],
                lessons_learned=db_report.analysis_lessons_learned or "",
                similar_incidents=db_report.analysis_similar_incidents or [],
                confidence_score=db_report.analysis_confidence_score or 0.0,
            )

        return RCAReport(
            id=db_report.id,
            incident_id=db_report.incident_id,
            status=RCAStatus(db_report.status.value),
            llm_provider_used=db_report.llm_provider_used,
            incident=incident,
            analysis=analysis,
            created_at=db_report.created_at,
            updated_at=db_report.updated_at,
            completed_at=db_report.completed_at,
        )
