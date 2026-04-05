"""Constraint Collector — the deconfounding layer.

Sits between the Actor's output and the Critic's evaluation, gathering active
constraint signals and packaging them as structured metadata.  This implements
Pearl's backdoor adjustment in practice: by making hidden constraints observable
to the Critic, we block the backdoor path and enable correct causal attribution.
"""

from __future__ import annotations

from causal_actor_critic.state import SQLActorCriticState


# Heuristic complexity signals extracted from the user query
_COMPLEXITY_SIGNALS = {
    "window function": 0.15,
    "running total": 0.12,
    "year-over-year": 0.12,
    "yoy": 0.12,
    "growth rate": 0.10,
    "rank": 0.08,
    "percentile": 0.10,
    "pivot": 0.12,
    "recursive": 0.15,
    "cumulative": 0.10,
    "moving average": 0.12,
    "lag": 0.08,
    "lead": 0.08,
    "partition by": 0.08,
    "self-join": 0.10,
    "subquery": 0.06,
    "cte": 0.04,
    "having": 0.04,
    "case when": 0.04,
    "union": 0.06,
    "except": 0.06,
    "intersect": 0.06,
    "exists": 0.06,
    "correlated": 0.12,
}


def compute_query_complexity(user_query: str) -> tuple[float, list[str]]:
    """Score query complexity based on analytical pattern detection.

    Returns (score in [0, 1], list of detected complexity factors).
    """
    query_lower = user_query.lower()
    score = 0.0
    factors: list[str] = []

    for signal, weight in _COMPLEXITY_SIGNALS.items():
        if signal in query_lower:
            score += weight
            factors.append(signal)

    join_keywords = ["join", "multiple tables", "across tables", "combine"]
    if any(kw in query_lower for kw in join_keywords):
        score += 0.08
        factors.append("multi-table join")

    temporal_keywords = ["monthly", "quarterly", "yearly", "weekly", "daily",
                         "time series", "trend", "over time", "by month"]
    if any(kw in query_lower for kw in temporal_keywords):
        score += 0.06
        factors.append("temporal aggregation")

    return min(score, 1.0), factors


def classify_constraint(
    complexity: float,
    attempt: int,
    max_attempts: int,
    sql_length: int,
    token_budget: int = 4096,
) -> str:
    """Classify the active constraint regime."""
    context_utilization = attempt / max(max_attempts, 1)

    if context_utilization >= 0.9:
        return "CONTEXT_EXHAUSTED"
    if sql_length > token_budget * 0.8:
        return "SIZE_CONSTRAINED"
    if complexity > 0.5:
        return "COMPLEXITY_CONSTRAINED"
    return "UNCONSTRAINED"


def collect_constraints(state: SQLActorCriticState) -> dict:
    """Collect constraint metadata and attach it to the state.

    This is the core deconfounding operation: making hidden environmental
    variables observable to the Critic.
    """
    user_query = state.get("user_query", "")
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    generated_sql = state.get("generated_sql", "")

    complexity, factors = compute_query_complexity(user_query)
    constraint_class = classify_constraint(
        complexity, attempt, max_attempts, len(generated_sql)
    )

    metadata = {
        "query_complexity": round(complexity, 3),
        "token_budget_remaining": round(1.0 - len(generated_sql) / 4096, 3),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "context_utilization": round(attempt / max(max_attempts, 1), 3),
        "constraint_class": constraint_class,
        "complexity_factors": factors,
    }

    return {"constraint_metadata": metadata}
