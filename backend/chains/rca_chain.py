"""
rca_chain.py — LangChain Prompts & Chains
==========================================
This file contains all the PROMPTS used by the LangGraph agent.

Prompts remain separate from agent logic so they can be optimized or
A/B tested without changing the 7-step RCA workflow.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)


# ============================================================
# PER-STEP COMPLETION TOKEN LIMITS
# ============================================================
# These limits only cap generated output. The LangGraph workflow,
# state, parsing, and 7-step process remain unchanged.
SUMMARY_MAX_TOKENS = 200
IMPACT_MAX_TOKENS = 250
ROOT_CAUSE_MAX_TOKENS = 600
FACTORS_MAX_TOKENS = 300
ACTIONS_MAX_TOKENS = 450
LESSONS_MAX_TOKENS = 400


# ============================================================
# PROMPT 1: Incident Summarizer
# ============================================================
INCIDENT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an SRE specializing in incident management and RCA.
Write a precise, factual 2-3 sentence incident summary.
Use only the provided evidence. Preserve actual incident ID, dates/times,
systems, severity, region, impact, and resolution details when available.
Never invent missing facts. Plain prose only; no headings or bullets."""
    ),
    (
        "human",
        """Summarize this incident in 2-3 sentences.

INCIDENT ID: {title}
SEVERITY: {severity}
AFFECTED SYSTEMS: {affected_systems}
DESCRIPTION: {description}
TIMELINE: {timeline}
ADDITIONAL CONTEXT: {additional_context}

Use actual dates/times from the evidence when available. If none are
provided, say "occurred recently" rather than inventing a date/time."""
    )
])


# ============================================================
# PROMPT 2: Root Cause Analyzer
# ============================================================
ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior IT Root Cause Analysis expert.
Use 5 Whys and fault-tree reasoning. Be systematic, technical, and
evidence-based. Never guess or invent facts."""
    ),
    (
        "human",
        """Identify the ROOT CAUSE of this incident using the 5 Whys.

INCIDENT DETAILS
Title: {title}
Severity: {severity}
Affected Systems: {affected_systems}
Description: {description}
Timeline: {timeline}
Additional Context: {additional_context}

SIMILAR PAST INCIDENTS
{similar_incidents}

Identify:
1. The primary/deepest underlying root cause
2. Why that root cause existed
3. The trigger that activated it

Be specific and technical. Reference the evidence and note relevant
patterns from similar incidents when present.

ROOT CAUSE ANALYSIS:"""
    )
])


# ============================================================
# PROMPT 3: Contributing Factors
# ============================================================
CONTRIBUTING_FACTORS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an incident analysis expert. Identify evidence-supported
people, process, technology, and environment factors that made the
incident occur, worsen, harder to detect, or slower to resolve.
Do not invent factors."""
    ),
    (
        "human",
        """List 3-5 distinct contributing factors.

INCIDENT: {title}
DESCRIPTION: {description}
ROOT CAUSE: {root_cause}
TIMELINE: {timeline}

Each factor must be one clear sentence. Output one factor per line,
starting with - ."""
    )
])


# ============================================================
# PROMPT 4: Action Items Generator
# ============================================================
ACTION_ITEMS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an SRE manager creating practical, implementable remediation
plans. Make each action specific enough for an engineer to execute.
Do not invent actions unsupported by the incident analysis."""
    ),
    (
        "human",
        """Generate remediation actions for this incident.

INCIDENT: {title}
ROOT CAUSE: {root_cause}
CONTRIBUTING FACTORS: {contributing_factors}
AFFECTED SYSTEMS: {affected_systems}

IMMEDIATE ACTIONS: 2-3 actions to stop or mitigate the incident.
CORRECTIVE ACTIONS: 3-5 tasks that permanently address the root cause.
PREVENTIVE MEASURES: 3-5 measures that reduce recurrence risk.

Make every item actionable and start each bullet with a verb.
Use exactly these section headers:
IMMEDIATE ACTIONS:
CORRECTIVE ACTIONS:
PREVENTIVE MEASURES:"""
    )
])


# ============================================================
# PROMPT 5: Lessons Learned
# ============================================================
LESSONS_LEARNED_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a blameless post-mortem facilitator. Extract constructive,
systemic learning and avoid individual blame."""
    ),
    (
        "human",
        """Write the Lessons Learned section.

INCIDENT: {title}
ROOT CAUSE: {root_cause}
CORRECTIVE ACTIONS: {corrective_actions}
SIMILAR PAST INCIDENTS: {similar_incidents}

Write 2-3 concise paragraphs covering:
1. What the incident taught us about systems/processes
2. What should change going forward
3. Broader organizational learning if similar incidents existed

LESSONS LEARNED:"""
    )
])


# ============================================================
# PROMPT 6: Impact Assessment
# ============================================================
IMPACT_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical incident manager assessing production impact.
Be precise about scope, duration, affected functionality, and business effect.
Use only the supplied evidence; do not invent impact."""
    ),
    (
        "human",
        """Assess the impact of this incident.

INCIDENT: {title}
SEVERITY: {severity}
AFFECTED SYSTEMS: {affected_systems}
DESCRIPTION: {description}
TIMELINE: {timeline}

Cover:
- affected users/customers
- unavailable or degraded functionality
- estimated duration
- business impact when determinable from the evidence

IMPACT ASSESSMENT:"""
    )
])


# ============================================================
# Chain Builder Functions
# ============================================================
def build_summary_chain(llm):
    return INCIDENT_SUMMARY_PROMPT | llm.bind(max_tokens=SUMMARY_MAX_TOKENS) | StrOutputParser()


def build_root_cause_chain(llm):
    return ROOT_CAUSE_PROMPT | llm.bind(max_tokens=ROOT_CAUSE_MAX_TOKENS) | StrOutputParser()


def build_contributing_factors_chain(llm):
    return CONTRIBUTING_FACTORS_PROMPT | llm.bind(max_tokens=FACTORS_MAX_TOKENS) | StrOutputParser()


def build_action_items_chain(llm):
    return ACTION_ITEMS_PROMPT | llm.bind(max_tokens=ACTIONS_MAX_TOKENS) | StrOutputParser()


def build_lessons_learned_chain(llm):
    return LESSONS_LEARNED_PROMPT | llm.bind(max_tokens=LESSONS_MAX_TOKENS) | StrOutputParser()


def build_impact_chain(llm):
    return IMPACT_ASSESSMENT_PROMPT | llm.bind(max_tokens=IMPACT_MAX_TOKENS) | StrOutputParser()


# ============================================================
# Helper: Parse bullet points from LLM output
# ============================================================
def parse_bullet_points(text: str) -> list:
    """Parse bullet/numbered LLM output into a Python list."""
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        for prefix in ["- ", "* ", "• ", "· "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        if line and len(line) > 3 and not line.endswith(":"):
            items.append(line)
    return items


def parse_action_items(text: str) -> dict:
    """Split action-item output into immediate, corrective, and preventive lists."""
    sections = {"immediate": [], "corrective": [], "preventive": []}
    current_section = None
    lines = text.strip().split("\n")

    for line in lines:
        line_lower = line.lower().strip()
        if "immediate" in line_lower:
            current_section = "immediate"
        elif "corrective" in line_lower:
            current_section = "corrective"
        elif "preventive" in line_lower or "prevent" in line_lower:
            current_section = "preventive"
        elif current_section and line.strip().startswith("-"):
            item = line.strip().lstrip("- ").strip()
            if item and len(item) > 3:
                sections[current_section].append(item)

    return sections
