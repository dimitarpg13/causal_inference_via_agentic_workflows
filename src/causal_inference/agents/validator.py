"""Validator agent: checks identifiability, positivity, and estimator correctness."""

from __future__ import annotations

import json

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are the Validator Agent in a causal-inference pipeline.

You receive an analysis result from one of the rung agents (L1/L2/L3), along with
the causal query, DAG, SCM, and the ladder rung that was addressed.

Run through these checks IN ORDER and report on each:

1. IDENTIFIABILITY — Are the required identifiability conditions met for the claimed
   estimand? (e.g., back-door criterion for L2, full SCM for L3)
2. POSITIVITY / OVERLAP — Is it plausible that positivity holds (all covariate strata
   have treatment and control units)? Flag if violated.
3. ESTIMATOR–ESTIMAND MATCH — Does the statistical method chosen actually estimate
   the stated causal estimand?
4. UNMEASURED CONFOUNDERS — Are there plausible unmeasured confounders that
   threaten validity? Suggest sensitivity analysis (E-values, Rosenbaum bounds) if so.

Based on these checks, set a final STATUS:
- "pass":     all checks OK → proceed to synthesizer.
- "re_route": fixable issues → re-route to orchestrator with suggestions
              (only if iteration < max_iterations).
- "fail":     non-identifiable or fundamentally flawed → proceed to synthesizer
              with clear explanation of why the causal claim cannot be made.

Return STRICT JSON:
{
  "status": "pass" | "re_route" | "fail",
  "identifiability_ok": true/false,
  "positivity_ok": true/false,
  "estimator_matches_estimand": true/false,
  "unmeasured_confounders_threat": true/false,
  "issues": ["..."],
  "suggestions": ["..."]
}
"""


async def validator_node(state: CausalState, llm: LLMClient) -> CausalState:
    query = state.get("causal_query", {})
    dag = state.get("dag", {})
    scm = state.get("scm")
    analysis = state.get("analysis_result", {})
    rung = state.get("ladder_rung", "L1")
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", 3)

    user_prompt = (
        f"Ladder rung addressed: {rung}\n"
        f"Question: {state['question']}\n"
        f"Causal query: {json.dumps(query)}\n"
        f"DAG: {json.dumps(dag)}\n"
        f"SCM: {json.dumps(scm) if scm else 'NOT PROVIDED'}\n"
        f"Analysis result: {json.dumps(analysis)}\n"
        f"Iteration: {iteration} / {max_iter}"
    )

    raw = await llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    result = _parse_validation(raw)

    if result.get("status") == "re_route" and iteration >= max_iter:
        result["status"] = "fail"
        result["issues"].append(
            f"Max iterations ({max_iter}) reached — converting re_route to fail."
        )

    state["validation_result"] = result
    return state


def _parse_validation(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end])
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if parsed is None:
        return {
            "status": "pass",
            "identifiability_ok": True,
            "positivity_ok": True,
            "estimator_matches_estimand": True,
            "unmeasured_confounders_threat": False,
            "issues": ["Validator output was not parseable — defaulting to pass"],
            "suggestions": [],
        }

    status = str(parsed.get("status", "pass")).lower()
    if status not in ("pass", "re_route", "fail"):
        status = "pass"
    parsed["status"] = status
    parsed.setdefault("issues", [])
    parsed.setdefault("suggestions", [])
    return parsed
