"""L3 Counterfactual Agent: handles retrospective what-if questions."""

from __future__ import annotations

import json

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are the L3 Counterfactual Agent in a causal-inference pipeline.
Your rung on Pearl's ladder: Counterfactual — P(Y_x | X=x', Y=y').

You receive a causal query, a DAG, and optionally a fully specified SCM.

Your responsibilities follow the three-step ABDUCTION–ACTION–PREDICTION procedure:

1. ABDUCTION — Infer exogenous variables U from observed evidence (X=x', Y=y').
2. ACTION   — Modify the SCM by replacing the structural equation for X with do(X=x).
3. PREDICTION — Propagate forward through the modified SCM to compute Y_x.

Decision logic:
- If a fully specified SCM is provided: execute the three-step procedure and produce
  a unit-level counterfactual estimate.
- If the SCM is NOT fully specified:
  • Flag this clearly.
  • Attempt to produce BOUNDS on the counterfactual using Manski bounds or
    monotonicity / monotone-treatment-response assumptions if stated.
  • Set "degraded_from": "L3" in the result.
  • Suggest what structural equations are needed to make the counterfactual identifiable.

Return STRICT JSON:
{
  "method": "abduction-action-prediction" or "bounds",
  "estimand_type": "counterfactual",
  "estimate": "...",
  "confidence_interval": "...",
  "details": "... three-step reasoning or bounding argument",
  "caveats": [...],
  "degraded_from": ""  (set to "L3" if falling back to bounds)
}
"""


async def l3_counterfactual_node(state: CausalState, llm: LLMClient) -> CausalState:
    query = state.get("causal_query", {})
    dag = state.get("dag", {})
    scm = state.get("scm")

    user_prompt = (
        f"Question: {state['question']}\n"
        f"Treatment: {query.get('treatment', 'unspecified')}\n"
        f"Outcome: {query.get('outcome', 'unspecified')}\n"
        f"Estimand: {query.get('estimand', 'counterfactual')}\n"
        f"Covariates: {json.dumps(query.get('covariates', []))}\n"
        f"Assumptions: {json.dumps(query.get('assumptions', []))}\n"
        f"DAG: {json.dumps(dag)}\n"
        f"SCM: {json.dumps(scm) if scm else 'NOT PROVIDED'}"
    )

    raw = await llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    state["analysis_result"] = _parse_analysis(raw)
    return state


def _parse_analysis(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return {
        "method": "unknown",
        "estimand_type": "counterfactual",
        "estimate": raw[:500],
        "confidence_interval": "N/A",
        "details": raw[:500],
        "caveats": ["Could not parse structured output"],
        "degraded_from": "",
    }
