"""L1 Association Agent: handles observational / correlational questions."""

from __future__ import annotations

import json

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are the L1 Association Agent in a causal-inference pipeline.
Your rung on Pearl's ladder: Association — P(Y | X).

You receive a causal query with treatment, outcome, covariates, and a DAG.

Your responsibilities:
1. Identify potential confounders from the DAG that could create spurious associations.
2. Select an appropriate statistical method:
   - Conditional-independence tests
   - Regression / partial correlation
   - Stratification
3. Describe (conceptually) the analysis you would perform and the expected result.
4. Explicitly flag whether the observed association is:
   - Likely spurious (confounded)
   - Potentially causal (but associational evidence only)
5. State all caveats: this is L1 only, no causal claim is justified.

Return STRICT JSON:
{
  "method": "...",
  "estimand_type": "associational",
  "estimate": "... (qualitative / conceptual description)",
  "confidence_interval": "N/A or description",
  "details": "... reasoning about confounders and method choice",
  "caveats": ["This is associational only — no causal claim", ...]
}
"""


async def l1_association_node(state: CausalState, llm: LLMClient) -> CausalState:
    query = state.get("causal_query", {})
    dag = state.get("dag", {})

    user_prompt = (
        f"Question: {state['question']}\n"
        f"Treatment: {query.get('treatment', 'unspecified')}\n"
        f"Outcome: {query.get('outcome', 'unspecified')}\n"
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
        "method": "regression",
        "estimand_type": "associational",
        "estimate": raw[:500],
        "confidence_interval": "N/A",
        "details": raw[:500],
        "caveats": ["Associational only — no causal claim"],
    }
