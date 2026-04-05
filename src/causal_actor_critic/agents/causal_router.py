"""Causal-aware routing, diagnosis, correction, and finalization.

The causal diagnosis step determines whether a Critic rejection is due to
Actor strategy failure (fixable), environmental constraint (not fixable by
Actor), or a mix of both.  This prevents the deadlock cycle where the Actor
is sent to fix something that no strategy can fix.
"""

from __future__ import annotations

import json

from causal_inference.llm_client import LLMClient
from causal_actor_critic.state import SQLActorCriticState

DIAGNOSIS_SYSTEM_PROMPT = """\
You are a Causal Diagnosis Agent in an Actor-Critic SQL generation pipeline.

After the Critic rejects the Actor's SQL, you determine the ROOT CAUSE:

1. ACTOR_STRATEGY_FAILURE — The Actor chose a bad approach. The task is
   feasible; the Actor just needs better SQL. Examples: wrong table,
   missing GROUP BY, incorrect join condition.

2. ENVIRONMENTAL_CONSTRAINT — A physical or complexity constraint prevented
   success. No Actor strategy can fully satisfy the Critic's rubric under
   current conditions. Examples: query too complex for single response,
   token budget exceeded, ambiguous user request.

3. MIXED — Some issues are Actor failures (fixable), others are constraint-
   driven (not fixable). The Actor should fix what it can; the rest should
   be accepted with caveats.

You receive:
- The Critic's issues and feedback
- Constraint metadata (complexity score, constraint class, attempt count)
- The user's original question

Analyze each Critic issue and classify its root cause.

Return STRICT JSON:
{
    "root_cause": "ACTOR_STRATEGY_FAILURE" | "ENVIRONMENTAL_CONSTRAINT" | "MIXED",
    "actor_issues": ["list of issues the Actor can fix"],
    "environmental_issues": ["list of constraint-driven issues"],
    "targeted_feedback": "Specific, actionable feedback for the Actor focusing ONLY on fixable issues",
    "should_reroute": true/false,
    "should_relax_rubric": true/false
}
"""


async def causal_diagnose(
    state: SQLActorCriticState, llm: LLMClient
) -> dict:
    """Perform causal diagnosis on a Critic rejection.

    Determines whether failure is due to Actor strategy (fixable) or
    environmental constraints (not fixable by Actor alone).
    """
    verdict = state.get("critic_verdict", "")

    if verdict == "pass":
        return {
            "causal_diagnosis": {
                "root_cause": "ACTOR_STRATEGY_FAILURE",
                "actor_issues": [],
                "environmental_issues": [],
                "targeted_feedback": "",
                "should_reroute": False,
                "should_relax_rubric": False,
            }
        }

    constraint_meta = state.get("constraint_metadata", {})
    issues = state.get("critic_issues", [])
    feedback = state.get("critic_feedback", "")

    user_prompt = (
        f"## User Question\n{state['user_query']}\n\n"
        f"## Critic Verdict: {verdict}\n\n"
        f"## Critic Issues\n{json.dumps(issues, indent=2)}\n\n"
        f"## Critic Feedback\n{feedback}\n\n"
        f"## Constraint Metadata\n{json.dumps(constraint_meta, indent=2)}"
    )

    raw = await llm.generate(
        system_prompt=DIAGNOSIS_SYSTEM_PROMPT, user_prompt=user_prompt
    )

    diagnosis = _parse_diagnosis(raw)

    if constraint_meta.get("constraint_class") == "CONTEXT_EXHAUSTED":
        diagnosis["should_reroute"] = False
        diagnosis["should_relax_rubric"] = True

    return {"causal_diagnosis": diagnosis}


def _parse_diagnosis(raw: str) -> dict:
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
        "root_cause": "MIXED",
        "actor_issues": ["Could not parse diagnosis"],
        "environmental_issues": [],
        "targeted_feedback": raw[:500],
        "should_reroute": True,
        "should_relax_rubric": False,
    }


def causal_route(state: SQLActorCriticState) -> str:
    """Route based on causal diagnosis rather than raw Critic verdict.

    Returns one of: "pass", "reroute_actor", "accept_with_caveat", "finalize"
    """
    verdict = state.get("critic_verdict", "non_salvageable")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    diagnosis = state.get("causal_diagnosis", {})

    if verdict == "pass":
        return "finalize"

    if attempt >= max_attempts:
        return "finalize"

    root_cause = diagnosis.get("root_cause", "MIXED")

    if root_cause == "ENVIRONMENTAL_CONSTRAINT":
        return "accept_with_caveat"

    if root_cause == "ACTOR_STRATEGY_FAILURE":
        if verdict == "salvageable":
            return "apply_correction"
        return "reroute_actor"

    # MIXED: fix what's fixable
    if diagnosis.get("should_reroute", True):
        if verdict == "salvageable":
            return "apply_correction"
        return "reroute_actor"

    return "accept_with_caveat"


def apply_correction(state: SQLActorCriticState) -> dict:
    """Apply Critic's correction, using causal diagnosis for targeted feedback."""
    corrected = state.get("corrected_sql", "")
    attempt = state.get("attempt", 0) + 1
    diagnosis = state.get("causal_diagnosis", {})

    targeted = diagnosis.get("targeted_feedback", "")
    if targeted:
        feedback = targeted
    else:
        feedback = state.get("critic_feedback", "")

    return {
        "generated_sql": corrected,
        "sql_explanation": f"Critic-corrected SQL (attempt {attempt})",
        "critic_feedback": feedback,
        "attempt": attempt,
        "correction_history": [
            {"attempt": attempt, "source": "critic_correction", "sql": corrected}
        ],
    }


def accept_with_caveat(state: SQLActorCriticState) -> dict:
    """Accept the current SQL with a caveat about environmental constraints."""
    diagnosis = state.get("causal_diagnosis", {})
    env_issues = diagnosis.get("environmental_issues", [])
    sql = state.get("generated_sql", "")
    explanation = state.get("sql_explanation", "")

    caveat = (
        "Note: This query was accepted under environmental constraints. "
        f"The following limitations apply: {'; '.join(env_issues) if env_issues else 'complexity constraints detected'}. "
        "The Actor's strategy was appropriate given these constraints."
    )

    return {
        "final_sql": sql,
        "final_explanation": f"{explanation}\n\n{caveat}",
        "status": "accepted_with_caveat",
    }


def finalize(state: SQLActorCriticState) -> dict:
    """Produce the terminal output."""
    verdict = state.get("critic_verdict", "")
    sql = state.get("generated_sql", "")
    explanation = state.get("sql_explanation", "")
    attempt = state.get("attempt", 0)

    if verdict == "pass":
        status = "accepted"
    else:
        status = "best_effort"

    return {
        "final_sql": sql,
        "final_explanation": explanation,
        "status": status,
    }
