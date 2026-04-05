"""LangGraph state schema for the causal-enhanced Actor-Critic SQL workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class ConstraintMetadata(TypedDict, total=False):
    """Structured metadata about active constraints — the deconfounding signal."""
    query_complexity: float
    token_budget_remaining: float
    attempt: int
    max_attempts: int
    context_utilization: float
    constraint_class: Literal[
        "UNCONSTRAINED",
        "COMPLEXITY_CONSTRAINED",
        "SIZE_CONSTRAINED",
        "CONTEXT_EXHAUSTED",
    ]
    complexity_factors: list[str]


class CausalDiagnosis(TypedDict, total=False):
    """Output of the causal diagnosis step after a Critic rejection."""
    root_cause: Literal[
        "ACTOR_STRATEGY_FAILURE",
        "ENVIRONMENTAL_CONSTRAINT",
        "MIXED",
    ]
    actor_issues: list[str]
    environmental_issues: list[str]
    targeted_feedback: str
    should_reroute: bool
    should_relax_rubric: bool


class SQLActorCriticState(TypedDict, total=False):
    """Full state flowing through the causal Actor-Critic LangGraph."""

    # Input
    user_query: str
    data_dictionary: str
    domain_rules: str

    # Actor context
    actor_system_prompt: str
    critic_system_prompt: str

    # Actor output
    generated_sql: str
    sql_explanation: str

    # Constraint metadata (deconfounding signal)
    constraint_metadata: dict

    # Critic output
    critic_verdict: str
    critic_issues: list[dict]
    critic_feedback: str
    corrected_sql: str

    # Causal diagnosis
    causal_diagnosis: dict

    # Loop control
    attempt: int
    max_attempts: int
    correction_history: Annotated[list[dict], operator.add]

    # Final output
    final_sql: str
    final_explanation: str
    status: str
