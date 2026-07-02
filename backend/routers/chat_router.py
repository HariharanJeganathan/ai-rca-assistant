"""
chat_router.py — Incident Chat API
Updated: Specific MIR questions (12 structured), incident-aware prompts
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat — Incident Assistant"])


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    incident_id: str
    message: str
    history: List[ChatMessage] = []
    report_context: Optional[dict] = None

class ChatResponse(BaseModel):
    reply: str
    suggestions: List[str] = []


def _build_mir_questions_prompt(incident: dict, analysis: dict) -> str:
    """
    Build a specific, incident-aware prompt for MIR questions.
    Produces exactly 12 questions across defined categories.
    """
    title = incident.get('title', 'the incident')
    severity = incident.get('severity', 'Unknown')
    systems = ', '.join(incident.get('affected_systems', [])) or 'affected systems'
    timeline = incident.get('incident_timeline', '')
    root_cause = analysis.get('root_cause', '')
    corrective = ', '.join(analysis.get('corrective_actions', []))
    preventive = ', '.join(analysis.get('preventive_measures', []))
    context = incident.get('additional_context', '')
    factors = ', '.join(analysis.get('contributing_factors', []))

    return f"""Generate exactly 12 MIR (Major Incident Review) meeting questions for this specific incident.
These questions will be used to guide the MIR call discussion.

=== INCIDENT DETAILS ===
Title: {title}
Severity: {severity}
Affected Systems: {systems}
Timeline: {timeline}
Additional Context: {context}

=== RCA FINDINGS ===
Root Cause: {root_cause}
Contributing Factors: {factors}
Corrective Actions: {corrective}
Preventive Measures: {preventive}

=== INSTRUCTIONS ===
Generate EXACTLY 12 questions structured as follows:

**5 Whys Analysis (5 questions)**
Questions that drill into WHY this specific incident happened — each "why" going one level deeper.
Reference the actual systems ({systems}), the actual root cause, and the actual timeline.
Do NOT ask generic "what was the root cause" questions — assume root cause is known, dig deeper.

**CAPA — Corrective & Preventive Actions (2 questions)**
Questions specifically about the corrective and preventive actions identified for THIS incident.
Reference the actual actions that were identified.

**Detailed Investigation (5 questions)**
Questions about detection gaps, response effectiveness, change management failures, monitoring, 
and process improvements — all specific to what happened in THIS incident.

FORMAT: Number each question 1-12. Group them with bold section headings.
Make every question specific to this incident. Zero generic questions.
If you mention systems, name them. If you mention the change, reference it. If you mention the timeline, use the actual times."""


@router.post("/message", response_model=ChatResponse)
async def chat_about_incident(request: ChatRequest):
    try:
        from config import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        from datetime import datetime

        llm = get_llm()

        report = request.report_context or {}
        incident = report.get("incident", {})
        analysis = report.get("analysis", {})

        # Extract date from incident context or timeline
        timeline = incident.get('incident_timeline', '')
        additional_context = incident.get('additional_context', '')

        # Try to find a date in the context
        incident_date = "the date specified in the incident"
        for line in (timeline + " " + additional_context).split('\n'):
            import re
            date_match = re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}', line)
            if date_match:
                incident_date = date_match.group(0)
                break

        system_prompt = f"""You are an expert Incident Manager and Problem Management specialist.
You are helping analyze a specific Major Incident that has already been through Root Cause Analysis.

=== INCIDENT CONTEXT ===
Incident ID: {report.get('incident_id', 'Unknown')}
Title: {incident.get('title', 'Unknown')}
Severity: {incident.get('severity', 'Unknown')}
Affected Systems: {', '.join(incident.get('affected_systems', []))}
Date/Time: {incident_date}
Description: {incident.get('description', 'Not provided')}
Timeline: {incident.get('incident_timeline', 'Not provided')}
Additional Context: {incident.get('additional_context', 'Not provided')}

=== RCA ANALYSIS ===
Summary: {analysis.get('incident_summary', 'Not available')}
Root Cause: {analysis.get('root_cause', 'Not available')}
Impact: {analysis.get('impact_assessment', 'Not available')}
Contributing Factors: {', '.join(analysis.get('contributing_factors', []))}
Immediate Actions: {', '.join(analysis.get('immediate_actions_taken', []))}
Corrective Actions: {', '.join(analysis.get('corrective_actions', []))}
Preventive Measures: {', '.join(analysis.get('preventive_measures', []))}
Lessons Learned: {analysis.get('lessons_learned', 'Not available')}
Similar Past Incidents: {', '.join(analysis.get('similar_incidents', []))}

=== RULES ===
1. ALWAYS be specific to THIS incident. Reference actual system names, actual times, actual changes.
2. NEVER give generic answers. If you catch yourself writing something that could apply to any incident, rewrite it.
3. For MIR questions: generate exactly 12, structured as 5 Whys (5) + CAPA (2) + Investigation (5).
4. For summaries: use the actual incident ID, actual systems, actual date/time from the context above.
5. Format cleanly — use numbered lists, bold section headers, no asterisk clutter."""

        # Check if this is a MIR questions request — use specialized prompt
        user_message = request.message
        is_mir_request = any(kw in user_message.lower() for kw in ['mir question', 'meeting question', '12 question', 'mir call', 'review question'])

        messages = [SystemMessage(content=system_prompt)]
        for msg in request.history[-10:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        if is_mir_request:
            # Use the specialized incident-specific MIR prompt
            specific_prompt = _build_mir_questions_prompt(incident, analysis)
            messages.append(HumanMessage(content=specific_prompt))
        else:
            messages.append(HumanMessage(content=user_message))

        response = await llm.ainvoke(messages)
        reply = response.content.strip()

        suggestions = _get_suggestions(user_message, analysis)
        return ChatResponse(reply=reply, suggestions=suggestions)

    except Exception as e:
        logger.error(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


def _get_suggestions(last_message: str, analysis: dict) -> List[str]:
    msg_lower = last_message.lower()
    if "5 why" in msg_lower or "root cause" in msg_lower:
        return ["Format as a table for the MIR document", "Generate 12 MIR questions based on this", "Write executive summary"]
    elif "question" in msg_lower or "mir" in msg_lower:
        return ["Format actions as a table with Owner and Due Date", "Write executive summary", "Reword root cause for management"]
    elif "reword" in msg_lower or "summary" in msg_lower:
        return ["Make it shorter — 2 sentences only", "Make it suitable for non-technical stakeholders", "Generate MIR questions"]
    else:
        return ["Generate 12 MIR questions", "5 Whys analysis", "Executive summary for leadership", "Format actions as table"]
