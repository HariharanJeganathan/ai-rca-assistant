"""
db/models.py — Database Table Definitions (SQLAlchemy ORM Models)
==================================================================
These classes define what the PostgreSQL tables look like.

ORM = Object Relational Mapper
  Each Python class = one database table
  Each class attribute = one column in the table

WITHOUT ORM (raw SQL):
  CREATE TABLE rca_reports (
    id SERIAL PRIMARY KEY,
    incident_id VARCHAR(100) UNIQUE NOT NULL,
    ...
  );

WITH ORM (SQLAlchemy):
  class RCAReportModel(Base):
      __tablename__ = "rca_reports"
      id = Column(Integer, primary_key=True)
      incident_id = Column(String(100), unique=True, nullable=False)

Same result, but Python code — easier to version control and maintain.

WHY TWO SEPARATE MODEL FILES?
  - db/models.py   = DATABASE models (how data is stored in PostgreSQL)
  - models/schemas.py = API models (how data looks in JSON responses)

  They look similar but serve different purposes.
  The repository layer (postgres.py) converts between them.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Float,
    DateTime, Boolean, JSON, Enum as SQLEnum
)
from sqlalchemy.sql import func
from db.database import Base
import enum


# ============================================================
# Enums for database columns
# ============================================================
class RCAStatusEnum(str, enum.Enum):
    """Maps to the RCAStatus enum in schemas.py"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class SeverityEnum(str, enum.Enum):
    """Maps to the IncidentSeverity enum in schemas.py"""
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


# ============================================================
# Main RCA Report Table
# ============================================================
class RCAReportModel(Base):
    """
    Stores every RCA report in PostgreSQL.

    TABLE NAME: rca_reports

    This is the main table — everything goes here.
    The analysis JSON column stores the full AI-generated RCA
    as a JSON blob (flexible, no need for extra tables).
    """

    __tablename__ = "rca_reports"

    # ----------------------------------------------------------
    # Primary Key
    # ----------------------------------------------------------
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,     # PostgreSQL auto-generates: 1, 2, 3...
        index=True
    )

    # ----------------------------------------------------------
    # Incident Identification
    # ----------------------------------------------------------
    incident_id = Column(
        String(100),
        unique=True,            # No two reports for same incident
        nullable=False,
        index=True,             # Index = fast lookup by incident_id
        comment="Unique incident identifier e.g. INC-20241201-A3B4C5D6"
    )

    # ----------------------------------------------------------
    # Status & Provider
    # ----------------------------------------------------------
    status = Column(
        SQLEnum(RCAStatusEnum),
        default=RCAStatusEnum.PENDING,
        nullable=False,
        index=True              # Index = fast filtering by status
    )

    llm_provider_used = Column(
        String(50),
        nullable=True,
        comment="Which LLM was used: groq, openai, or azure"
    )

    # ----------------------------------------------------------
    # Incident Details (from user input)
    # ----------------------------------------------------------
    incident_title = Column(
        String(200),
        nullable=False
    )

    incident_description = Column(
        Text,                   # Text = unlimited length (vs String which has limit)
        nullable=False
    )

    incident_severity = Column(
        SQLEnum(SeverityEnum),
        default=SeverityEnum.P3,
        nullable=False,
        index=True
    )

    incident_affected_systems = Column(
        JSON,                   # Store list as JSON array: ["payment-svc", "checkout"]
        nullable=True,
        default=list
    )

    incident_timeline = Column(
        Text,
        nullable=True
    )

    incident_additional_context = Column(
        Text,
        nullable=True
    )

    # ----------------------------------------------------------
    # AI Analysis Output (stored as JSON)
    # ----------------------------------------------------------
    # We store the full RCAAnalysis as JSON instead of creating
    # separate tables for each field. This keeps things simple
    # and flexible — easy to add new fields later.

    analysis_summary = Column(Text, nullable=True)
    analysis_root_cause = Column(Text, nullable=True)
    analysis_impact = Column(Text, nullable=True)
    analysis_timeline = Column(Text, nullable=True)
    analysis_lessons_learned = Column(Text, nullable=True)

    # These are lists — stored as JSON arrays
    analysis_contributing_factors = Column(JSON, nullable=True, default=list)
    analysis_immediate_actions = Column(JSON, nullable=True, default=list)
    analysis_corrective_actions = Column(JSON, nullable=True, default=list)
    analysis_preventive_measures = Column(JSON, nullable=True, default=list)
    analysis_similar_incidents = Column(JSON, nullable=True, default=list)

    analysis_confidence_score = Column(
        Float,
        nullable=True,
        default=0.0,
        comment="AI confidence score 0.0 to 1.0"
    )

    # ----------------------------------------------------------
    # Timestamps
    # ----------------------------------------------------------
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),  # PostgreSQL sets this automatically
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),        # Auto-updates when record is modified
        nullable=False
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True               # Null until analysis is done
    )

    # ----------------------------------------------------------
    # Soft Delete (optional — keeps data for auditing)
    # ----------------------------------------------------------
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False
    )

    def __repr__(self):
        """String representation — useful for debugging"""
        return (
            f"<RCAReport id={self.id} "
            f"incident_id={self.incident_id} "
            f"status={self.status} "
            f"severity={self.incident_severity}>"
        )
