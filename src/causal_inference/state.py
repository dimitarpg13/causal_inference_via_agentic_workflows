"""Shared state schema for the causal inference agentic workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class CausalQuery(TypedDict, total=False):
    treatment: str
    outcome: str
    estimand: str
    covariates: list[str]
    assumptions: list[str]


class DAGSpec(TypedDict, total=False):
    nodes: list[str]
    edges: list[tuple[str, str]]
    latent_edges: list[tuple[str, str]]


class SCMSpec(TypedDict, total=False):
    structural_equations: dict[str, str]
    exogenous: dict[str, str]


class AnalysisResult(TypedDict, total=False):
    method: str
    estimand_type: str
    estimate: str
    confidence_interval: str
    details: str
    caveats: list[str]
    degraded_from: str


class ValidationResult(TypedDict, total=False):
    status: Literal["pass", "re_route", "fail"]
    identifiability_ok: bool
    positivity_ok: bool
    estimator_matches_estimand: bool
    unmeasured_confounders_threat: bool
    issues: list[str]
    suggestions: list[str]


class SensitivityResult(TypedDict, total=False):
    """Output of the sensitivity analysis agent."""
    method: str
    robustness_score: str
    e_value: str
    rosenbaum_bounds: str
    details: str
    interpretation: str
    recommendations: list[str]


class CausalState(TypedDict, total=False):
    question: str
    causal_query: dict[str, Any]
    ladder_rung: Literal["L1", "L2", "L3"]
    dag: dict[str, Any]
    scm: dict[str, Any] | None
    analysis_result: dict[str, Any]
    validation_result: dict[str, Any]
    sensitivity_result: dict[str, Any]
    final_report: str
    iteration: int
    max_iterations: int
    run_sensitivity: bool
