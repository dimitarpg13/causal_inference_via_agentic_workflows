"""LangGraph state-machine wiring for the causal-inference workflow.

Architecture:
  START -> orchestrator -> [L1 | L2 | L3] -> validator
     -> (pass)     -> [sensitivity?] -> synthesizer -> END
     -> (re_route) -> orchestrator  (loop, bounded by max_iterations)
     -> (fail)     -> [sensitivity?] -> synthesizer -> END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from causal_inference.agents.l1_association import l1_association_node
from causal_inference.agents.l2_intervention import l2_intervention_node
from causal_inference.agents.l3_counterfactual import l3_counterfactual_node
from causal_inference.agents.orchestrator import orchestrator_node
from causal_inference.agents.sensitivity import sensitivity_node
from causal_inference.agents.synthesizer import synthesizer_node
from causal_inference.agents.validator import validator_node
from causal_inference.llm_client import LLMClient
from causal_inference.state import CausalState


def _route_by_rung(state: CausalState) -> str:
    rung = state.get("ladder_rung", "L1")
    return {
        "L1": "l1_association",
        "L2": "l2_intervention",
        "L3": "l3_counterfactual",
    }.get(rung, "l1_association")


def _validation_router(state: CausalState) -> str:
    result = state.get("validation_result", {})
    status = str(result.get("status", "pass")).lower()
    if status == "re_route":
        return "re_route"
    if status == "fail":
        return "post_validation"
    return "post_validation"


def _sensitivity_router(state: CausalState) -> str:
    """Route to sensitivity analysis if requested and confounders are a threat."""
    if state.get("run_sensitivity"):
        return "sensitivity"
    validation = state.get("validation_result", {})
    if validation.get("unmeasured_confounders_threat"):
        return "sensitivity"
    return "synthesizer"


def build_causal_graph(llm: LLMClient) -> Any:
    """Compile the LangGraph causal-inference state machine."""

    async def _orchestrator(state: CausalState) -> CausalState:
        return await orchestrator_node(state, llm)

    async def _l1(state: CausalState) -> CausalState:
        return await l1_association_node(state, llm)

    async def _l2(state: CausalState) -> CausalState:
        return await l2_intervention_node(state, llm)

    async def _l3(state: CausalState) -> CausalState:
        return await l3_counterfactual_node(state, llm)

    async def _validator(state: CausalState) -> CausalState:
        return await validator_node(state, llm)

    async def _sensitivity(state: CausalState) -> CausalState:
        return await sensitivity_node(state, llm)

    async def _synthesizer(state: CausalState) -> CausalState:
        return await synthesizer_node(state, llm)

    builder = StateGraph(CausalState)

    builder.add_node("orchestrator", _orchestrator)
    builder.add_node("l1_association", _l1)
    builder.add_node("l2_intervention", _l2)
    builder.add_node("l3_counterfactual", _l3)
    builder.add_node("validator", _validator)
    builder.add_node("sensitivity", _sensitivity)
    builder.add_node("synthesizer", _synthesizer)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        _route_by_rung,
        {
            "l1_association": "l1_association",
            "l2_intervention": "l2_intervention",
            "l3_counterfactual": "l3_counterfactual",
        },
    )

    for rung_node in ("l1_association", "l2_intervention", "l3_counterfactual"):
        builder.add_edge(rung_node, "validator")

    builder.add_conditional_edges(
        "validator",
        _validation_router,
        {
            "post_validation": "synthesizer",
            "re_route": "orchestrator",
        },
    )

    builder.add_edge("sensitivity", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def build_causal_graph_with_sensitivity(llm: LLMClient) -> Any:
    """Compile the LangGraph state machine with an optional sensitivity analysis step."""

    async def _orchestrator(state: CausalState) -> CausalState:
        return await orchestrator_node(state, llm)

    async def _l1(state: CausalState) -> CausalState:
        return await l1_association_node(state, llm)

    async def _l2(state: CausalState) -> CausalState:
        return await l2_intervention_node(state, llm)

    async def _l3(state: CausalState) -> CausalState:
        return await l3_counterfactual_node(state, llm)

    async def _validator(state: CausalState) -> CausalState:
        return await validator_node(state, llm)

    async def _sensitivity(state: CausalState) -> CausalState:
        return await sensitivity_node(state, llm)

    async def _synthesizer(state: CausalState) -> CausalState:
        return await synthesizer_node(state, llm)

    builder = StateGraph(CausalState)

    builder.add_node("orchestrator", _orchestrator)
    builder.add_node("l1_association", _l1)
    builder.add_node("l2_intervention", _l2)
    builder.add_node("l3_counterfactual", _l3)
    builder.add_node("validator", _validator)
    builder.add_node("post_validation", lambda s: s)
    builder.add_node("sensitivity", _sensitivity)
    builder.add_node("synthesizer", _synthesizer)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        _route_by_rung,
        {
            "l1_association": "l1_association",
            "l2_intervention": "l2_intervention",
            "l3_counterfactual": "l3_counterfactual",
        },
    )

    for rung_node in ("l1_association", "l2_intervention", "l3_counterfactual"):
        builder.add_edge(rung_node, "validator")

    builder.add_conditional_edges(
        "validator",
        _validation_router,
        {
            "post_validation": "post_validation",
            "re_route": "orchestrator",
        },
    )

    builder.add_conditional_edges(
        "post_validation",
        _sensitivity_router,
        {
            "sensitivity": "sensitivity",
            "synthesizer": "synthesizer",
        },
    )

    builder.add_edge("sensitivity", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


async def run_causal_workflow(
    question: str,
    llm: LLMClient,
    *,
    dag: dict | None = None,
    scm: dict | None = None,
    max_iterations: int = 3,
    run_sensitivity: bool = False,
) -> CausalState:
    """Run the full causal-inference agentic workflow on a question.

    Parameters
    ----------
    question : str
        The causal question to analyze.
    llm : LLMClient
        The LLM backend for agent reasoning.
    dag : dict, optional
        DAG specification. Falls back to a simple default if omitted.
    scm : dict, optional
        Structural Causal Model specification (required for L3).
    max_iterations : int
        Maximum re-routing iterations.
    run_sensitivity : bool
        If True, always run sensitivity analysis after validation.
    """
    if run_sensitivity:
        graph = build_causal_graph_with_sensitivity(llm)
    else:
        graph = build_causal_graph(llm)

    initial_state: CausalState = {
        "question": question,
        "causal_query": {},
        "ladder_rung": "L1",
        "dag": dag or _default_dag(),
        "scm": scm,
        "analysis_result": {},
        "validation_result": {},
        "sensitivity_result": {},
        "final_report": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "run_sensitivity": run_sensitivity,
    }

    return await graph.ainvoke(initial_state)


def _default_dag() -> dict:
    return {
        "nodes": ["X", "Y", "Z"],
        "edges": [["Z", "X"], ["Z", "Y"], ["X", "Y"]],
        "description": (
            "Default DAG: Z is a common cause of X and Y (confounder), "
            "and X directly causes Y."
        ),
    }
