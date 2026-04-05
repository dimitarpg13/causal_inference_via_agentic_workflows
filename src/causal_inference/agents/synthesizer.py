"""Synthesizer agent: produces the final causal report."""

from __future__ import annotations

import json

from causal_inference.state import CausalState
from causal_inference.llm_client import LLMClient

SYSTEM_PROMPT = """\
You are the Synthesizer Agent in a causal-inference pipeline.

You receive:
- The original user question.
- The classified ladder rung (L1 / L2 / L3).
- The causal query (treatment, outcome, estimand, covariates, assumptions).
- The DAG and SCM (if available).
- The analysis result from the rung agent.
- The validation result from the validator.
- The sensitivity analysis result (if performed).

Produce a clear, structured CAUSAL REPORT with these sections:

## Causal Report

### Question
Restate the user's question.

### Ladder Rung
State which rung was addressed and why.

### Causal Query
Treatment, outcome, estimand.

### DAG / SCM Summary
Describe the assumed causal structure.

### Analysis
- Method used
- Result / estimate
- Confidence / uncertainty

### Assumptions
List every assumption required for the conclusion to hold.

### Validation
Summarize validator checks and any issues.

### Sensitivity Analysis
If performed, summarize robustness findings: E-values, Rosenbaum bounds, or Manski
bounds. If not performed, note this and state whether it would be recommended.

### Caveats & Limitations
Be explicit about what this analysis CANNOT claim.

### Conclusion
One-paragraph actionable summary.

Write the report in Markdown.
"""


async def synthesizer_node(state: CausalState, llm: LLMClient) -> CausalState:
    query = state.get("causal_query", {})
    dag = state.get("dag", {})
    scm = state.get("scm")
    analysis = state.get("analysis_result", {})
    validation = state.get("validation_result", {})
    sensitivity = state.get("sensitivity_result", {})
    rung = state.get("ladder_rung", "L1")

    user_prompt = (
        f"Original question: {state['question']}\n"
        f"Ladder rung: {rung}\n"
        f"Causal query: {json.dumps(query)}\n"
        f"DAG: {json.dumps(dag)}\n"
        f"SCM: {json.dumps(scm) if scm else 'NOT PROVIDED'}\n"
        f"Analysis result: {json.dumps(analysis)}\n"
        f"Validation result: {json.dumps(validation)}\n"
        f"Sensitivity analysis: {json.dumps(sensitivity) if sensitivity else 'NOT PERFORMED'}"
    )

    report = await llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    state["final_report"] = report
    return state
