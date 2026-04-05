"""L2 Intervention Agent: handles do-calculus / interventional questions."""

from __future__ import annotations

import json

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are the L2 Intervention Agent in a causal-inference pipeline.
Your rung on Pearl's ladder: Intervention — P(Y | do(X)).

You receive a causal query (treatment, outcome, covariates, assumptions) and a DAG.

Your responsibilities:
1. Check IDENTIFIABILITY against the DAG:
   - Back-door criterion: is there an admissible adjustment set?
   - Front-door criterion: applicable?
   - Instrumental variable available?
   If not identifiable, say so clearly and suggest what additional assumption
   or instrument would make it identifiable.
2. If identifiable, select an appropriate ESTIMATOR:
   - IPW (Inverse Probability Weighting)
   - Matching
   - AIPW (Augmented IPW / Doubly Robust)
   - Instrumental Variables (IV / 2SLS)
   - Front-door adjustment
3. Describe the estimation conceptually and provide:
   - ATE or ATT estimate (qualitative if no data)
   - Confidence interval reasoning
4. If NOT identifiable:
   - Set estimate to "Non-identifiable from given DAG"
   - List what is needed (assumptions, instruments, data)

Return STRICT JSON:
{
  "method": "...",
  "estimand_type": "interventional",
  "estimate": "...",
  "confidence_interval": "...",
  "details": "... identifiability reasoning, adjustment set, estimator choice",
  "caveats": [...]
}
"""


async def l2_intervention_node(state: CausalState, llm: LLMClient) -> CausalState:
    query = state.get("causal_query", {})
    dag = state.get("dag", {})

    user_prompt = (
        f"Question: {state['question']}\n"
        f"Treatment: {query.get('treatment', 'unspecified')}\n"
        f"Outcome: {query.get('outcome', 'unspecified')}\n"
        f"Estimand: {query.get('estimand', 'ATE')}\n"
        f"Covariates: {json.dumps(query.get('covariates', []))}\n"
        f"Assumptions: {json.dumps(query.get('assumptions', []))}\n"
        f"DAG: {json.dumps(dag)}"
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
        "estimand_type": "interventional",
        "estimate": raw[:500],
        "confidence_interval": "N/A",
        "details": raw[:500],
        "caveats": ["Could not parse structured output"],
    }
