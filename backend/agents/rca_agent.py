"""
rca_agent.py — LangGraph Multi-Step Reasoning Agent
=====================================================
This is THE BRAIN of the entire application.

WHAT IS LANGGRAPH?
  LangGraph lets you build AI that reasons in STEPS — like a human expert
  who thinks through a problem methodically instead of guessing in one shot.

  Normal LLM call:  Question → One big answer (often wrong or shallow)
  LangGraph agent:  Question → Step 1 → Step 2 → Step 3 → Deep answer

OUR AGENT'S STEPS (the graph):
  ┌─────────────────────────────────────────────────────┐
  │                  RCA AGENT GRAPH                    │
  │                                                     │
  │  START                                              │
  │    ↓                                                │
  │  [1] summarize_incident    ← Read and summarize     │
  │    ↓                                                │
  │  [2] assess_impact         ← Who/what was affected  │
  │    ↓                                                │
  │  [3] analyze_root_cause    ← Why did it happen?     │
  │    ↓                                                │
  │  [4] identify_factors      ← What made it worse?   │
  │    ↓                                                │
  │  [5] generate_actions      ← What should we do?    │
  │    ↓                                                │
  │  [6] write_lessons         ← What did we learn?    │
  │    ↓                                                │
  │  [7] compile_report        ← Put it all together   │
  │    ↓                                                │
  │   END                                               │
  └─────────────────────────────────────────────────────┘

WHY THIS IS IMPRESSIVE IN INTERVIEWS:
  Most people just call llm.invoke("analyze this").
  LangGraph shows you understand:
  - Agentic AI (multi-step reasoning)
  - State management across steps
  - Graph-based workflow orchestration
  - How to break complex tasks into manageable steps
"""

import logging
from typing import TypedDict, Optional, List, Dict, Any, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from chains.rca_chain import (
    build_summary_chain,
    build_root_cause_chain,
    build_contributing_factors_chain,
    build_action_items_chain,
    build_lessons_learned_chain,
    build_impact_chain,
    parse_bullet_points,
    parse_action_items
)
from models.schemas import RCAAnalysis, IncidentInput
from config import get_llm

logger = logging.getLogger(__name__)


# ============================================================
# AGENT STATE — the memory that flows through all steps
# ============================================================
class RCAState(TypedDict):
    """
    The STATE is what gets passed between every node in the graph.
    Each step reads from state and writes its output back to state.

    Think of it like a shared whiteboard in a meeting room —
    each team member reads what others wrote and adds their own notes.

    TypedDict = Python dictionary with type hints (safer than plain dict)
    """

    # Input data
    incident_id: str
    title: str
    description: str
    severity: str
    affected_systems: List[str]
    timeline: str
    additional_context: str
    similar_incidents: List[Dict[str, Any]]

    # Outputs written by each step
    incident_summary: str           # Written by step 1
    impact_assessment: str          # Written by step 2
    root_cause: str                 # Written by step 3
    contributing_factors: List[str] # Written by step 4
    immediate_actions: List[str]    # Written by step 5
    corrective_actions: List[str]   # Written by step 5
    preventive_measures: List[str]  # Written by step 5
    lessons_learned: str            # Written by step 6

    # Metadata
    current_step: str
    errors: List[str]
    completed_at: Optional[str]


# ============================================================
# AGENT CLASS
# ============================================================
class RCAAgent:
    """
    The LangGraph-powered RCA Agent.

    This class:
    1. Builds the graph (connects nodes in order)
    2. Runs the graph with incident data as input
    3. Returns a structured RCAAnalysis object
    """

    def __init__(self, llm=None):
        """
        Initialize with an LLM.
        If none provided, uses whatever is configured in .env
        """
        self.llm = llm or get_llm()
        self.graph = self._build_graph()
        logger.info(f"[RCAAgent] Initialized with LLM: {type(self.llm).__name__}")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        StateGraph = a directed graph where:
          - Nodes = steps (functions that process the state)
          - Edges = connections between steps (what runs after what)
          - State = the data flowing through the graph
        """
        # Create a new graph that uses RCAState as its state type
        graph = StateGraph(RCAState)

        # Add all nodes (steps) to the graph
        # Each node is a method of this class
        graph.add_node("summarize_incident", self._summarize_incident)
        graph.add_node("assess_impact",      self._assess_impact)
        graph.add_node("analyze_root_cause", self._analyze_root_cause)
        graph.add_node("identify_factors",   self._identify_factors)
        graph.add_node("generate_actions",   self._generate_actions)
        graph.add_node("write_lessons",      self._write_lessons)
        graph.add_node("compile_report",     self._compile_report)

        # Define the flow — which node runs after which
        graph.set_entry_point("summarize_incident")
        graph.add_edge("summarize_incident", "assess_impact")
        graph.add_edge("assess_impact",      "analyze_root_cause")
        graph.add_edge("analyze_root_cause", "identify_factors")
        graph.add_edge("identify_factors",   "generate_actions")
        graph.add_edge("generate_actions",   "write_lessons")
        graph.add_edge("write_lessons",      "compile_report")
        graph.add_edge("compile_report",     END)

        # Compile the graph (validates connections and prepares for execution)
        return graph.compile()

    # ============================================================
    # NODE 1: Summarize Incident
    # ============================================================
    async def _summarize_incident(self, state: RCAState) -> RCAState:
        """
        Step 1: Read the incident and produce a clean summary.
        This gives all subsequent steps a clear, structured starting point.
        """
        logger.info(f"[RCAAgent] Step 1: Summarizing incident '{state['title']}'")
        state["current_step"] = "summarize_incident"

        try:
            chain = build_summary_chain(self.llm)
            summary = await chain.ainvoke({
                "title": state["title"],
                "severity": state["severity"],
                "affected_systems": ", ".join(state["affected_systems"]) or "Not specified",
                "description": state["description"],
                "timeline": state["timeline"] or "No timeline provided",
                "additional_context": state["additional_context"] or "Not provided"
            })
            state["incident_summary"] = summary.strip()
            logger.info(f"[RCAAgent] Summary complete ({len(summary)} chars)")

        except Exception as e:
            logger.error(f"[RCAAgent] Summary step failed: {e}")
            state["incident_summary"] = f"Incident: {state['title']}. Severity: {state['severity']}."
            state["errors"].append(f"summarize_incident: {str(e)}")

        return state

    # ============================================================
    # NODE 2: Assess Impact
    # ============================================================
    async def _assess_impact(self, state: RCAState) -> RCAState:
        """Step 2: Assess who and what was impacted."""
        logger.info("[RCAAgent] Step 2: Assessing impact")
        state["current_step"] = "assess_impact"

        try:
            chain = build_impact_chain(self.llm)
            impact = await chain.ainvoke({
                "title": state["title"],
                "severity": state["severity"],
                "affected_systems": ", ".join(state["affected_systems"]) or "Not specified",
                "description": state["description"],
                "timeline": state["timeline"] or "No timeline provided"
            })
            state["impact_assessment"] = impact.strip()
            logger.info("[RCAAgent] Impact assessment complete")

        except Exception as e:
            logger.error(f"[RCAAgent] Impact step failed: {e}")
            state["impact_assessment"] = (
                f"Severity {state['severity']} incident affecting: "
                f"{', '.join(state['affected_systems']) or 'systems not specified'}."
            )
            state["errors"].append(f"assess_impact: {str(e)}")

        return state

    # ============================================================
    # NODE 3: Analyze Root Cause
    # ============================================================
    async def _analyze_root_cause(self, state: RCAState) -> RCAState:
        """
        Step 3: The most important step — identify WHY it happened.
        Uses the 5 Whys methodology via the prompt.
        Also uses similar past incidents from ChromaDB as context.
        """
        logger.info("[RCAAgent] Step 3: Analyzing root cause")
        state["current_step"] = "analyze_root_cause"

        # Format similar incidents for the prompt
        similar_text = "No similar past incidents found in knowledge base."
        if state["similar_incidents"]:
            similar_parts = []
            for s in state["similar_incidents"]:
                similar_parts.append(
                    f"• [{s.get('incident_id', 'N/A')}] {s.get('title', 'N/A')}\n"
                    f"  Similarity: {s.get('similarity_score', 0):.0%}\n"
                    f"  Summary: {s.get('summary', 'N/A')[:200]}"
                )
            similar_text = "\n\n".join(similar_parts)

        try:
            chain = build_root_cause_chain(self.llm)
            root_cause = await chain.ainvoke({
                "title": state["title"],
                "severity": state["severity"],
                "affected_systems": ", ".join(state["affected_systems"]) or "Not specified",
                "description": state["description"],
                "timeline": state["timeline"] or "No timeline provided",
                "additional_context": state["additional_context"] or "None provided",
                "similar_incidents": similar_text
            })
            state["root_cause"] = root_cause.strip()
            logger.info("[RCAAgent] Root cause analysis complete")

        except Exception as e:
            logger.error(f"[RCAAgent] Root cause step failed: {e}")
            state["root_cause"] = f"Root cause analysis failed: {str(e)}"
            state["errors"].append(f"analyze_root_cause: {str(e)}")

        return state

    # ============================================================
    # NODE 4: Identify Contributing Factors
    # ============================================================
    async def _identify_factors(self, state: RCAState) -> RCAState:
        """Step 4: What other factors contributed to the incident?"""
        logger.info("[RCAAgent] Step 4: Identifying contributing factors")
        state["current_step"] = "identify_factors"

        try:
            chain = build_contributing_factors_chain(self.llm)
            factors_text = await chain.ainvoke({
                "title": state["title"],
                "description": state["description"],
                "root_cause": state["root_cause"],
                "timeline": state["timeline"] or "No timeline provided"
            })
            state["contributing_factors"] = parse_bullet_points(factors_text)
            logger.info(f"[RCAAgent] Found {len(state['contributing_factors'])} contributing factors")

        except Exception as e:
            logger.error(f"[RCAAgent] Factors step failed: {e}")
            state["contributing_factors"] = ["Contributing factor analysis unavailable"]
            state["errors"].append(f"identify_factors: {str(e)}")

        return state

    # ============================================================
    # NODE 5: Generate Action Items
    # ============================================================
    async def _generate_actions(self, state: RCAState) -> RCAState:
        """
        Step 5: Generate three types of actions:
          - Immediate: stop the bleeding NOW
          - Corrective: fix the root cause permanently
          - Preventive: stop it from ever happening again
        """
        logger.info("[RCAAgent] Step 5: Generating action items")
        state["current_step"] = "generate_actions"

        try:
            chain = build_action_items_chain(self.llm)
            actions_text = await chain.ainvoke({
                "title": state["title"],
                "root_cause": state["root_cause"],
                "contributing_factors": "\n".join(
                    [f"- {f}" for f in state["contributing_factors"]]
                ),
                "affected_systems": ", ".join(state["affected_systems"]) or "Not specified"
            })

            # Parse the three sections from the LLM output
            actions = parse_action_items(actions_text)
            state["immediate_actions"] = actions["immediate"] or ["Rollback or disable the affected component"]
            state["corrective_actions"] = actions["corrective"] or ["Fix identified root cause"]
            state["preventive_measures"] = actions["preventive"] or ["Add monitoring for early detection"]

            logger.info(
                f"[RCAAgent] Actions: {len(state['immediate_actions'])} immediate, "
                f"{len(state['corrective_actions'])} corrective, "
                f"{len(state['preventive_measures'])} preventive"
            )

        except Exception as e:
            logger.error(f"[RCAAgent] Actions step failed: {e}")
            state["immediate_actions"] = ["Immediately investigate and mitigate the issue"]
            state["corrective_actions"] = ["Fix the identified root cause"]
            state["preventive_measures"] = ["Add monitoring and alerting"]
            state["errors"].append(f"generate_actions: {str(e)}")

        return state

    # ============================================================
    # NODE 6: Write Lessons Learned
    # ============================================================
    async def _write_lessons(self, state: RCAState) -> RCAState:
        """Step 6: Write constructive, blameless lessons learned."""
        logger.info("[RCAAgent] Step 6: Writing lessons learned")
        state["current_step"] = "write_lessons"

        similar_text = "None found."
        if state["similar_incidents"]:
            similar_text = ", ".join([
                s.get("title", "N/A") for s in state["similar_incidents"]
            ])

        try:
            chain = build_lessons_learned_chain(self.llm)
            lessons = await chain.ainvoke({
                "title": state["title"],
                "root_cause": state["root_cause"],
                "corrective_actions": "\n".join(
                    [f"- {a}" for a in state["corrective_actions"]]
                ),
                "similar_incidents": similar_text
            })
            state["lessons_learned"] = lessons.strip()
            logger.info("[RCAAgent] Lessons learned complete")

        except Exception as e:
            logger.error(f"[RCAAgent] Lessons step failed: {e}")
            state["lessons_learned"] = "Lessons learned generation failed. Please review the root cause and actions manually."
            state["errors"].append(f"write_lessons: {str(e)}")

        return state

    # ============================================================
    # NODE 7: Compile Final Report
    # ============================================================
    async def _compile_report(self, state: RCAState) -> RCAState:
        """
        Step 7: Final step — just marks the state as complete.
        The actual RCAAnalysis object is built in run() using the state.
        """
        logger.info("[RCAAgent] Step 7: Compiling final report")
        state["current_step"] = "complete"
        state["completed_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"[RCAAgent] ✅ Analysis complete. "
            f"Errors: {len(state['errors'])}. "
            f"Steps completed: 7/7"
        )
        return state

    # ============================================================
    # PUBLIC METHOD: Run the full agent
    # ============================================================
    async def run(
        self,
        incident: IncidentInput,
        incident_id: str,
        similar_incidents: Optional[List[Dict[str, Any]]] = None
    ) -> RCAAnalysis:
        """
        Run the full 7-step LangGraph agent.

        Args:
            incident: The incident data from the user
            incident_id: Unique ID for this incident
            similar_incidents: Results from ChromaDB search (from Step 3)

        Returns:
            RCAAnalysis: Structured analysis with all sections filled
        """
        logger.info(f"[RCAAgent] Starting 7-step analysis for: {incident_id}")

        # Build the initial state
        initial_state: RCAState = {
            # Input
            "incident_id": incident_id,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity.value,
            "affected_systems": incident.affected_systems or [],
            "timeline": incident.incident_timeline or "",
            "additional_context": incident.additional_context or "",
            "similar_incidents": similar_incidents or [],

            # Outputs (empty — filled by each step)
            "incident_summary": "",
            "impact_assessment": "",
            "root_cause": "",
            "contributing_factors": [],
            "immediate_actions": [],
            "corrective_actions": [],
            "preventive_measures": [],
            "lessons_learned": "",

            # Metadata
            "current_step": "starting",
            "errors": [],
            "completed_at": None
        }

        # Run the graph
        # ainvoke = async invoke (non-blocking)
        final_state = await self.graph.ainvoke(initial_state)

        # Calculate confidence score
        # More similar incidents + fewer errors = higher confidence
        base_confidence = 0.75
        similar_bonus = min(len(similar_incidents or []) * 0.05, 0.15)
        error_penalty = len(final_state.get("errors", [])) * 0.1
        confidence = max(0.0, min(1.0, base_confidence + similar_bonus - error_penalty))

        # Format similar incident summaries for the report
        similar_summaries = []
        for s in (similar_incidents or []):
            similar_summaries.append(
                f"[{s.get('incident_id', 'N/A')}] {s.get('title', 'N/A')} "
                f"(similarity: {s.get('similarity_score', 0):.0%})"
            )

        # Build the structured RCAAnalysis from the final state
        analysis = RCAAnalysis(
            incident_summary=final_state.get("incident_summary", ""),
            timeline_reconstruction=final_state.get("timeline", incident.incident_timeline or "No timeline provided"),
            root_cause=final_state.get("root_cause", ""),
            contributing_factors=final_state.get("contributing_factors", []),
            impact_assessment=final_state.get("impact_assessment", ""),
            immediate_actions_taken=final_state.get("immediate_actions", []),
            corrective_actions=final_state.get("corrective_actions", []),
            preventive_measures=final_state.get("preventive_measures", []),
            lessons_learned=final_state.get("lessons_learned", ""),
            similar_incidents=similar_summaries,
            confidence_score=round(confidence, 2)
        )

        logger.info(
            f"[RCAAgent] Done. Confidence: {confidence:.0%}. "
            f"Errors during run: {final_state.get('errors', [])}"
        )

        return analysis


# ============================================================
# Singleton — one agent instance shared across the app
# ============================================================
from functools import lru_cache

@lru_cache(maxsize=1)
def get_rca_agent() -> RCAAgent:
    """
    Returns the shared RCAAgent instance.
    Only created once (lru_cache), reused for all requests.

    Usage in other files:
        from agents.rca_agent import get_rca_agent
        agent = get_rca_agent()
        analysis = await agent.run(incident, incident_id, similar_incidents)
    """
    return RCAAgent()
