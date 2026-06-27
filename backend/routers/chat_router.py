"""
chat_router.py — Incident Chat API
====================================
Provides a conversational endpoint that lets users chat
about a specific RCA report that has already been generated.

The AI has full context of:
  - The incident details
  - The generated RCA (root cause, actions, lessons)
  - The conversation history so far
  - Similar past incidents from ChromaDB

Supports:
  - 5 Why questions
  - Section rewrites
  - Follow-up questions
  - Similar incident queries
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat — Incident Assistant"])


# ── Request / Response schemas ─────────────────────────────
class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    incident_id: str
    message: str
    history: List[ChatMessage] = []
    report_context: Optional[dict] = None  # Full report passed from frontend

class ChatResponse(BaseModel):
    reply: str
    suggestions: List[str] = []


# ── Quick action prompts ───────────────────────────────────
QUICK_ACTIONS = {
    "5why": "Generate a detailed 5 Whys analysis for this incident in a numbered table format.",
    "mir_questions": "Generate 5 investigation questions I should ask in the MIR call for this incident.",
    "reword_root_cause": "Rewrite the root cause section in clearer, more concise language suitable for a management report.",
    "executive_summary": "Write a 3-sentence executive summary of this incident suitable for senior leadership.",
    "action_items_table": "Format the corrective and preventive actions as a table with columns: Action, Owner (blank), Due Date (blank), Status.",
    "similar_patterns": "Based on the similar past incidents found, what patterns do you see and what systemic fix would address all of them?",
}


@router.post("/message", response_model=ChatResponse)
async def chat_about_incident(request: ChatRequest):
    """
    Chat about a specific incident/RCA report.
    Maintains conversation history for context.
    """
    try:
        from config import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        llm = get_llm()

        # Build system prompt with full incident context
        report = request.report_context or {}
        incident = report.get("incident", {})
        analysis = report.get("analysis", {})

        system_prompt = f"""You are an expert Site Reliability Engineer and Incident Manager assistant.
You are helping analyze a specific incident that has already been through Root Cause Analysis.

=== INCIDENT CONTEXT ===
Incident ID: {report.get('incident_id', 'Unknown')}
Title: {incident.get('title', 'Unknown')}
Severity: {incident.get('severity', 'Unknown')}
Affected Systems: {', '.join(incident.get('affected_systems', []))}
Description: {incident.get('description', 'Not provided')}
Timeline: {incident.get('incident_timeline', 'Not provided')}
Additional Context: {incident.get('additional_context', 'Not provided')}

=== RCA ANALYSIS ALREADY GENERATED ===
Summary: {analysis.get('incident_summary', 'Not available')}
Root Cause: {analysis.get('root_cause', 'Not available')}
Impact: {analysis.get('impact_assessment', 'Not available')}
Contributing Factors: {', '.join(analysis.get('contributing_factors', []))}
Immediate Actions: {', '.join(analysis.get('immediate_actions_taken', []))}
Corrective Actions: {', '.join(analysis.get('corrective_actions', []))}
Preventive Measures: {', '.join(analysis.get('preventive_measures', []))}
Lessons Learned: {analysis.get('lessons_learned', 'Not available')}
Similar Past Incidents: {', '.join(analysis.get('similar_incidents', []))}
Confidence Score: {analysis.get('confidence_score', 0) * 100:.0f}%

=== YOUR ROLE ===
Help the user understand, refine, and expand this RCA.
You can:
- Generate 5 Why analysis
- Reword any section more clearly
- Answer questions about the incident
- Generate MIR meeting questions
- Create executive summaries
- Format action items as tables
- Spot patterns across similar incidents
- Suggest improvements to the analysis

Always be specific to THIS incident. Never give generic answers.
Format your responses clearly using markdown when helpful."""

        # Build message history for LangChain
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in request.history[-10:]:  # Keep last 10 messages for context
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # Add current message
        messages.append(HumanMessage(content=request.message))

        # Call LLM
        response = await llm.ainvoke(messages)
        reply = response.content.strip()

        # Generate contextual suggestions based on the conversation
        suggestions = _get_suggestions(request.message, analysis)

        return ChatResponse(reply=reply, suggestions=suggestions)

    except Exception as e:
        logger.error(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


def _get_suggestions(last_message: str, analysis: dict) -> List[str]:
    """Return contextual quick-action suggestions."""
    msg_lower = last_message.lower()

    # Context-aware suggestions
    if "5 why" in msg_lower or "root cause" in msg_lower:
        return [
            "Format this as a table for the MIR document",
            "What preventive measure addresses the deepest root cause?",
            "Generate MIR meeting questions based on this"
        ]
    elif "question" in msg_lower or "mir" in msg_lower:
        return [
            "Generate 5 Why analysis",
            "Write executive summary",
            "Format action items as a table"
        ]
    elif "reword" in msg_lower or "summary" in msg_lower:
        return [
            "Make it shorter — 2 sentences only",
            "Make it more technical",
            "Make it suitable for non-technical stakeholders"
        ]
    else:
        # Default suggestions
        return [
            "Generate 5 Why analysis",
            "Give me MIR meeting questions",
            "Write executive summary for leadership",
            "Format actions as a table with owner and due date"
        ]
