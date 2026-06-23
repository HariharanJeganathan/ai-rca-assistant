"""
schemas.py — Data Shapes (Pydantic Models)
==========================================
Pydantic models define the SHAPE of data flowing through the app.

Think of them like forms:
  - What fields are required?
  - What type is each field? (text, number, list...)
  - What are the default values?

FastAPI uses these automatically to:
  ✅ Validate incoming requests
  ✅ Show in API docs (Swagger UI)
  ✅ Serialize responses to JSON
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ============================================================
# Enums — fixed set of allowed values
# ============================================================

class IncidentSeverity(str, Enum):
    """Severity levels for incidents — P1 is most critical"""
    P1 = "P1"   # Critical — system down
    P2 = "P2"   # High — major feature broken
    P3 = "P3"   # Medium — partial impact
    P4 = "P4"   # Low — minor issue


class RCAStatus(str, Enum):
    """Where is the RCA in its lifecycle?"""
    PENDING = "pending"         # Just submitted, not analyzed yet
    ANALYZING = "analyzing"     # AI is currently working on it
    COMPLETED = "completed"     # RCA done
    FAILED = "failed"           # Something went wrong


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    OPENAI = "openai"
    AZURE = "azure"


# ============================================================
# Incident Input — what the user sends to us
# ============================================================

class IncidentInput(BaseModel):
    """
    Data the user provides when submitting an incident for RCA.

    Example request body:
    {
        "title": "Payment service down",
        "description": "Users cannot checkout. 502 errors in payment-svc.",
        "severity": "P1",
        "affected_systems": ["payment-service", "checkout-api"],
        "incident_timeline": "10:00 - Alert fired\n10:05 - On-call paged...",
        "additional_context": "Happened after deployment at 09:55"
    }
    """

    title: str = Field(
        ...,                        # "..." means REQUIRED (no default)
        min_length=5,
        max_length=200,
        description="Short title describing the incident",
        example="Payment service returning 502 errors"
    )

    description: str = Field(
        ...,
        min_length=20,
        description="Detailed description of what happened",
        example="Users are unable to complete checkout. Payment service is returning 502 Bad Gateway errors."
    )

    severity: IncidentSeverity = Field(
        default=IncidentSeverity.P3,
        description="Incident severity level (P1=Critical, P4=Low)"
    )

    affected_systems: List[str] = Field(
        default=[],
        description="List of systems/services affected",
        example=["payment-service", "checkout-api", "database"]
    )

    incident_timeline: Optional[str] = Field(
        default=None,
        description="Chronological timeline of events during the incident",
        example="09:55 - Deployment pushed\n10:00 - Alerts fired\n10:05 - On-call engineer paged"
    )

    additional_context: Optional[str] = Field(
        default=None,
        description="Any other relevant context (recent changes, logs, etc.)",
        example="This happened right after the v2.3.1 deployment of payment-service"
    )


# ============================================================
# RCA Output — what the AI produces
# ============================================================

class RCAAnalysis(BaseModel):
    """
    The structured Root Cause Analysis produced by the AI agent.
    Each field is one section of the RCA report.
    """

    incident_summary: str = Field(
        description="Brief summary of what happened"
    )

    timeline_reconstruction: str = Field(
        description="Reconstructed sequence of events leading to the incident"
    )

    root_cause: str = Field(
        description="The identified root cause of the incident"
    )

    contributing_factors: List[str] = Field(
        default=[],
        description="Other factors that contributed to the incident"
    )

    impact_assessment: str = Field(
        description="What was impacted and to what extent"
    )

    immediate_actions_taken: List[str] = Field(
        default=[],
        description="Actions taken to resolve the incident"
    )

    corrective_actions: List[str] = Field(
        default=[],
        description="Actions to fix the root cause permanently"
    )

    preventive_measures: List[str] = Field(
        default=[],
        description="Steps to prevent this from happening again"
    )

    lessons_learned: str = Field(
        description="Key takeaways from this incident"
    )

    similar_incidents: List[str] = Field(
        default=[],
        description="Similar past incidents found in the knowledge base (from ChromaDB RAG)"
    )

    confidence_score: float = Field(
        default=0.0,
        ge=0.0,    # greater than or equal to 0
        le=1.0,    # less than or equal to 1
        description="AI confidence in the analysis (0.0 to 1.0)"
    )


# ============================================================
# Full RCA Report — stored in PostgreSQL
# ============================================================

class RCAReport(BaseModel):
    """
    Complete RCA record as stored in the database.
    Combines the input incident + the AI analysis output.
    """

    id: Optional[int] = Field(default=None, description="Database auto-generated ID")
    incident_id: str = Field(description="Unique incident identifier (e.g. INC-2024-001)")
    status: RCAStatus = Field(default=RCAStatus.PENDING)
    llm_provider_used: Optional[str] = Field(default=None)

    # The original incident data
    incident: IncidentInput

    # The AI-generated analysis (None until analysis is complete)
    analysis: Optional[RCAAnalysis] = None

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        # Allow SQLAlchemy ORM models to be converted to Pydantic
        from_attributes = True


# ============================================================
# API Request/Response wrappers
# ============================================================

class RCARequest(BaseModel):
    """What the API receives when user submits an incident"""
    incident: IncidentInput
    llm_provider: Optional[LLMProvider] = Field(
        default=None,
        description="Override the default LLM provider for this request"
    )


class RCAResponse(BaseModel):
    """What the API sends back after starting an RCA"""
    success: bool
    message: str
    incident_id: str
    status: RCAStatus
    report: Optional[RCAReport] = None


class HealthResponse(BaseModel):
    """Response for the /health endpoint"""
    status: str
    app_name: str
    version: str
    llm_provider: str
    database_connected: bool
    chromadb_connected: bool


class ErrorResponse(BaseModel):
    """Standard error response shape"""
    success: bool = False
    error: str
    detail: Optional[str] = None
