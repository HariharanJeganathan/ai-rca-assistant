"""
rca_chain.py — LangChain Prompts & Chains
==========================================
This file contains all the PROMPTS used by the LangGraph agent.

WHAT IS A PROMPT TEMPLATE?
  Instead of hardcoding text, we use templates with placeholders.
  Like a mail-merge in Word — same template, different values each time.

  Example:
    Template: "Analyze this incident: {incident_title}"
    Filled:   "Analyze this incident: Payment service down"

WHY SEPARATE PROMPTS FROM AGENT LOGIC?
  - Easy to improve prompts without touching agent code
  - Easy to A/B test different prompts
  - Keeps the agent file clean and readable
  - Senior engineers always separate concerns like this

CHAIN = Prompt + LLM + Output Parser connected together
  Input → Prompt Template → LLM → Parse Output → Result
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

logger = logging.getLogger(__name__)


# ============================================================
# PROMPT 1: Incident Summarizer
# ============================================================
INCIDENT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Site Reliability Engineer (SRE) specializing in
incident management and root cause analysis. You have 15 years of experience
analyzing production incidents across large-scale distributed systems.

Your job is to read an incident report and produce a clear, concise summary
that captures the essential facts. Be factual, precise, and technical.
Do NOT speculate — only summarize what is stated."""
    ),
    (
        "human",
        """Please summarize this incident report in 3-4 sentences.
Focus on: what happened, when, and what was impacted.

INCIDENT TITLE: {title}
SEVERITY: {severity}
AFFECTED SYSTEMS: {affected_systems}
DESCRIPTION: {description}
TIMELINE: {timeline}

Write a clear, factual summary:"""
    )
])


# ============================================================
# PROMPT 2: Root Cause Analyzer
# ============================================================
ROOT_CAUSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior Root Cause Analysis expert with deep expertise in:
- Distributed systems and microservices
- Database performance and reliability
- Cloud infrastructure (AWS, Azure, GCP)
- CI/CD pipelines and deployments
- Network and security issues

You use the "5 Whys" methodology and fault tree analysis to identify
root causes. You are systematic, evidence-based, and thorough.
Never guess — base your analysis on the evidence provided."""
    ),
    (
        "human",
        """Analyze this incident and identify the ROOT CAUSE.

=== INCIDENT DETAILS ===
Title: {title}
Severity: {severity}
Affected Systems: {affected_systems}
Description: {description}
Timeline: {timeline}
Additional Context: {additional_context}

=== SIMILAR PAST INCIDENTS (from knowledge base) ===
{similar_incidents}

=== INSTRUCTIONS ===
Using the 5 Whys methodology, identify:
1. The PRIMARY root cause (the deepest underlying reason)
2. Why that root cause existed
3. What trigger activated it

Be specific and technical. Reference the timeline and context provided.
If similar past incidents are shown, note any patterns.

ROOT CAUSE ANALYSIS:"""
    )
])


# ============================================================
# PROMPT 3: Contributing Factors
# ============================================================
CONTRIBUTING_FACTORS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an incident analysis expert. Your role is to identify
ALL contributing factors to an incident — not just the root cause,
but everything that made the incident worse, harder to detect,
or slower to resolve. Think about: people, process, technology, and environment."""
    ),
    (
        "human",
        """Based on this incident analysis, list the contributing factors.

INCIDENT: {title}
DESCRIPTION: {description}
ROOT CAUSE IDENTIFIED: {root_cause}
TIMELINE: {timeline}

List 3-5 contributing factors. Each should be a distinct factor that
made the incident happen, worse, or harder to resolve.
Format each as a single clear sentence.

CONTRIBUTING FACTORS (one per line, starting with -):"""
    )
])


# ============================================================
# PROMPT 4: Action Items Generator
# ============================================================
ACTION_ITEMS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an SRE manager creating actionable remediation plans.
You focus on practical, implementable solutions with clear ownership.
Every action item should be specific enough that an engineer knows
exactly what to do without asking for clarification."""
    ),
    (
        "human",
        """Generate action items for this incident.

INCIDENT: {title}
ROOT CAUSE: {root_cause}
CONTRIBUTING FACTORS: {contributing_factors}
AFFECTED SYSTEMS: {affected_systems}

Generate TWO types of action items:

IMMEDIATE ACTIONS (what was/should be done to stop the bleeding):
List 2-3 immediate actions taken or needed.

CORRECTIVE ACTIONS (fix the root cause permanently):
List 3-5 specific engineering tasks to fix the underlying issue.

PREVENTIVE MEASURES (stop this from happening again):
List 3-5 measures to prevent recurrence (monitoring, tests, processes).

Format each as a clear, actionable bullet point starting with a verb.
Example: "Add circuit breaker to payment service API calls"

IMMEDIATE ACTIONS:
-

CORRECTIVE ACTIONS:
-

PREVENTIVE MEASURES:
-"""
    )
])


# ============================================================
# PROMPT 5: Lessons Learned
# ============================================================
LESSONS_LEARNED_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a blameless post-mortem facilitator. Your role is to
extract learning from incidents in a constructive, forward-looking way.
Focus on systemic improvements, not individual blame.
Use the blameless post-mortem culture pioneered by Google SRE."""
    ),
    (
        "human",
        """Write the "Lessons Learned" section for this incident post-mortem.

INCIDENT: {title}
ROOT CAUSE: {root_cause}
CORRECTIVE ACTIONS: {corrective_actions}
SIMILAR PAST INCIDENTS: {similar_incidents}

Write 2-3 paragraphs covering:
1. What this incident taught us about our systems/processes
2. What we will do differently going forward
3. Any broader organizational learning (if similar incidents occurred before)

Be constructive, specific, and forward-looking.

LESSONS LEARNED:"""
    )
])


# ============================================================
# PROMPT 6: Impact Assessment
# ============================================================
IMPACT_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical incident manager assessing business and technical
impact of production incidents. Be precise about scope, duration, and effect."""
    ),
    (
        "human",
        """Assess the impact of this incident.

INCIDENT: {title}
SEVERITY: {severity}
AFFECTED SYSTEMS: {affected_systems}
DESCRIPTION: {description}
TIMELINE: {timeline}

Describe the impact covering:
- Which users/customers were affected
- What functionality was unavailable or degraded
- Estimated duration of impact
- Business impact (if determinable from the information)

IMPACT ASSESSMENT:"""
    )
])


# ============================================================
# Chain Builder Functions
# ============================================================
def build_summary_chain(llm):
    """
    Chain = Prompt | LLM | OutputParser
    The | symbol is LangChain's "pipe" operator.
    Data flows left to right: prompt → llm → parse output
    """
    return INCIDENT_SUMMARY_PROMPT | llm | StrOutputParser()


def build_root_cause_chain(llm):
    return ROOT_CAUSE_PROMPT | llm | StrOutputParser()


def build_contributing_factors_chain(llm):
    return CONTRIBUTING_FACTORS_PROMPT | llm | StrOutputParser()


def build_action_items_chain(llm):
    return ACTION_ITEMS_PROMPT | llm | StrOutputParser()


def build_lessons_learned_chain(llm):
    return LESSONS_LEARNED_PROMPT | llm | StrOutputParser()


def build_impact_chain(llm):
    return IMPACT_ASSESSMENT_PROMPT | llm | StrOutputParser()


# ============================================================
# Helper: Parse bullet points from LLM output
# ============================================================
def parse_bullet_points(text: str) -> list:
    """
    Parse LLM output that contains bullet points into a Python list.

    Input:  "- Fix the database\n- Add monitoring\n- Update docs"
    Output: ["Fix the database", "Add monitoring", "Update docs"]
    """
    lines = text.strip().split("\n")
    items = []
    for line in lines:
        line = line.strip()
        # Remove common bullet point prefixes
        for prefix in ["- ", "* ", "• ", "· "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        # Remove numbered list prefixes like "1. ", "2. "
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        # Skip empty lines and section headers
        if line and len(line) > 3 and not line.endswith(":"):
            items.append(line)
    return items


def parse_action_items(text: str) -> dict:
    """
    Parse the action items prompt output into three separate lists.

    The prompt returns text with three sections:
      IMMEDIATE ACTIONS:
      - ...
      CORRECTIVE ACTIONS:
      - ...
      PREVENTIVE MEASURES:
      - ...

    This function splits them into three Python lists.
    """
    sections = {
        "immediate": [],
        "corrective": [],
        "preventive": []
    }

    current_section = None
    lines = text.strip().split("\n")

    for line in lines:
        line_lower = line.lower().strip()

        # Detect section headers
        if "immediate" in line_lower:
            current_section = "immediate"
        elif "corrective" in line_lower:
            current_section = "corrective"
        elif "preventive" in line_lower or "prevent" in line_lower:
            current_section = "preventive"
        elif current_section and line.strip().startswith("-"):
            # Add bullet point to current section
            item = line.strip().lstrip("- ").strip()
            if item and len(item) > 3:
                sections[current_section].append(item)

    return sections
