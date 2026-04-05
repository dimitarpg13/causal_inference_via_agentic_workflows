"""Orchestrator agent: classifies questions into Pearl's causal ladder rungs."""

from __future__ import annotations

import json
from typing import Any

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are a causal-inference orchestrator following Judea Pearl's causal hierarchy.

Given a user question you must:
1. Classify it into exactly ONE rung of the causal ladder:
   - L1 (Association):  questions about P(Y|X), correlation, prediction from observation.
   - L2 (Intervention): questions about P(Y|do(X)), what happens if we act / intervene.
   - L3 (Counterfactual): retrospective what-if questions P(Y_x | X=x', Y=y').

2. Extract structured variables:
   - treatment: the variable being manipulated or conditioned on
   - outcome:   the variable of interest
   - estimand:  the formal causal quantity (e.g. "ATE", "P(Y|X)", "counterfactual Y_{x=0}")
   - covariates: any mentioned confounders / mediators
   - assumptions: any causal assumptions stated in the question

Classification guidance:
- "Is X correlated with Y?" or "Does X predict Y?" → L1
- "What is the causal effect of X on Y?" or "What happens if we do X?" → L2
- "What would have happened if X had been different?" or "Had X not occurred..." → L3

Return STRICT JSON with exactly these keys:
{
  "ladder_rung": "L1" | "L2" | "L3",
  "treatment": "...",
  "outcome": "...",
  "estimand": "...",
  "covariates": ["..."],
  "assumptions": ["..."]
}
"""


async def orchestrator_node(state: CausalState, llm: LLMClient) -> CausalState:
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration

    user_prompt = state["question"]
    if iteration > 1:
        prev_validation = state.get("validation_result", {})
        user_prompt += (
            f"\n\n[Re-routed — iteration {iteration}]\n"
            f"Previous validation issues: {json.dumps(prev_validation.get('issues', []))}\n"
            f"Suggestions: {json.dumps(prev_validation.get('suggestions', []))}"
        )

    raw = await llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_orchestrator_output(raw)

    state["ladder_rung"] = parsed.pop("ladder_rung", "L1")  # type: ignore[assignment]
    state["causal_query"] = parsed
    return state


def _parse_orchestrator_output(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start:end])
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None

    if payload is None:
        return {
            "ladder_rung": "L1",
            "treatment": "unknown",
            "outcome": "unknown",
            "estimand": "P(Y|X)",
            "covariates": [],
            "assumptions": [],
        }

    rung = str(payload.get("ladder_rung", "L1")).upper()
    if rung not in ("L1", "L2", "L3"):
        rung = "L1"
    payload["ladder_rung"] = rung
    return payload
