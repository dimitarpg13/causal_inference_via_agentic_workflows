"""Sensitivity Analysis Agent: evaluates robustness to unmeasured confounding.

This agent is invoked optionally after the Validator when unmeasured confounders
are flagged as a threat.  It reasons about E-values, Rosenbaum bounds, and
the general sensitivity of the causal conclusion to violations of the
no-unmeasured-confounding assumption.
"""

from __future__ import annotations

import json

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are the Sensitivity Analysis Agent in a causal-inference pipeline.

You are invoked when the Validator has flagged potential unmeasured confounders
or when the user explicitly requests a robustness assessment.

You receive the full analysis context: the causal query, DAG, analysis result,
and validation result.

Your responsibilities:

1. E-VALUE ANALYSIS — Compute or describe the E-value: the minimum strength of
   association that an unmeasured confounder would need with both the treatment
   and the outcome (on the risk-ratio scale) to explain away the observed effect.
   Interpret this: is the required confounding strength plausible?

2. ROSENBAUM BOUNDS (for L2) — Describe the sensitivity parameter Γ at which the
   causal conclusion would be overturned. Higher Γ = more robust.

3. MANSKI BOUNDS (for L3 without full SCM) — Report the width of the identification
   region. Narrower bounds = more informative even without point identification.

4. QUALITATIVE ASSESSMENT — Given the domain (from the DAG and question), are there
   plausible unmeasured confounders strong enough to overturn the result?

5. RECOMMENDATIONS — Suggest data collection, study designs, or additional analyses
   that would strengthen the causal claim.

Return STRICT JSON:
{
  "method": "e-value and/or rosenbaum-bounds and/or manski-bounds",
  "robustness_score": "high / moderate / low",
  "e_value": "... description or computed value",
  "rosenbaum_bounds": "... Γ value or description",
  "details": "... full reasoning",
  "interpretation": "... plain-language summary of robustness",
  "recommendations": ["..."]
}
"""


async def sensitivity_node(state: CausalState, llm: LLMClient) -> CausalState:
    query = state.get("causal_query", {})
    dag = state.get("dag", {})
    scm = state.get("scm")
    analysis = state.get("analysis_result", {})
    validation = state.get("validation_result", {})
    rung = state.get("ladder_rung", "L1")

    user_prompt = (
        f"Ladder rung: {rung}\n"
        f"Question: {state['question']}\n"
        f"Causal query: {json.dumps(query)}\n"
        f"DAG: {json.dumps(dag)}\n"
        f"SCM: {json.dumps(scm) if scm else 'NOT PROVIDED'}\n"
        f"Analysis result: {json.dumps(analysis)}\n"
        f"Validation result: {json.dumps(validation)}"
    )

    raw = await llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    state["sensitivity_result"] = _parse_sensitivity(raw)
    return state


def _parse_sensitivity(raw: str) -> dict:
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
        "method": "qualitative",
        "robustness_score": "unknown",
        "e_value": raw[:300],
        "rosenbaum_bounds": "N/A",
        "details": raw[:500],
        "interpretation": "Could not parse structured output",
        "recommendations": [],
    }
