"""Synthetic data generators for classic causal inference scenarios.

Each generator returns a pandas DataFrame together with metadata describing
the ground-truth causal structure, true effects, and the DAG specification
that can be fed directly into the agentic workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataset:
    """Container for a synthetic causal dataset plus its ground-truth metadata."""

    data: pd.DataFrame
    dag: dict[str, Any]
    scm: dict[str, Any] | None
    true_ate: float | None
    description: str
    scenario: str
    confounders: list[str] = field(default_factory=list)
    mediators: list[str] = field(default_factory=list)


def generate_confounded_data(
    n: int = 2000,
    seed: int = 42,
    *,
    confounder_effect_on_treatment: float = 0.6,
    confounder_effect_on_outcome: float = 0.4,
    treatment_effect: float = 0.3,
) -> SyntheticDataset:
    """Smoking -> Cancer with Genetics as a confounder.

    DAG:  Genetics -> Smoking, Genetics -> Cancer, Smoking -> Cancer
    True ATE of Smoking on Cancer = treatment_effect
    Naive correlation is inflated by confounding through Genetics.
    """
    rng = np.random.default_rng(seed)

    genetics = rng.normal(0, 1, n)
    smoking = (
        confounder_effect_on_treatment * genetics + rng.normal(0, 0.5, n)
    )
    cancer_score = (
        treatment_effect * smoking
        + confounder_effect_on_outcome * genetics
        + rng.normal(0, 0.3, n)
    )

    df = pd.DataFrame({
        "genetics": genetics,
        "smoking": smoking,
        "cancer_score": cancer_score,
    })

    dag = {
        "nodes": ["Genetics", "Smoking", "Cancer"],
        "edges": [
            ["Genetics", "Smoking"],
            ["Genetics", "Cancer"],
            ["Smoking", "Cancer"],
        ],
        "description": (
            "Genetics confounds the Smoking-Cancer relationship. "
            "The naive correlation between Smoking and Cancer overstates "
            "the true causal effect."
        ),
    }

    scm = {
        "structural_equations": {
            "Genetics": "U_G",
            "Smoking": f"{confounder_effect_on_treatment} * Genetics + U_S",
            "Cancer": (
                f"{treatment_effect} * Smoking "
                f"+ {confounder_effect_on_outcome} * Genetics + U_C"
            ),
        },
        "exogenous": {
            "U_G": "Normal(0, 1)",
            "U_S": "Normal(0, 0.5)",
            "U_C": "Normal(0, 0.3)",
        },
    }

    return SyntheticDataset(
        data=df,
        dag=dag,
        scm=scm,
        true_ate=treatment_effect,
        description=(
            f"Confounded scenario: Genetics -> Smoking -> Cancer with "
            f"Genetics -> Cancer. True ATE = {treatment_effect}. "
            f"Naive correlation is inflated by confounding."
        ),
        scenario="confounded",
        confounders=["genetics"],
    )


def generate_frontdoor_data(
    n: int = 2000,
    seed: int = 42,
    *,
    smoking_to_tar: float = 0.7,
    tar_to_cancer: float = 0.5,
    genetics_to_smoking: float = 0.5,
    genetics_to_cancer: float = 0.3,
) -> SyntheticDataset:
    """Front-door criterion scenario: Smoking -> Tar -> Cancer, Genetics confounds.

    Genetics is an unmeasured common cause of Smoking and Cancer, but the
    front-door path through Tar allows identification of the causal effect.
    """
    rng = np.random.default_rng(seed)

    genetics = rng.normal(0, 1, n)
    smoking = genetics_to_smoking * genetics + rng.normal(0, 0.5, n)
    tar = smoking_to_tar * smoking + rng.normal(0, 0.3, n)
    cancer_score = (
        tar_to_cancer * tar
        + genetics_to_cancer * genetics
        + rng.normal(0, 0.3, n)
    )

    df = pd.DataFrame({
        "smoking": smoking,
        "tar": tar,
        "cancer_score": cancer_score,
    })

    true_ate = smoking_to_tar * tar_to_cancer

    dag = {
        "nodes": ["Smoking", "Tar", "Cancer", "Genetics"],
        "edges": [
            ["Smoking", "Tar"],
            ["Tar", "Cancer"],
            ["Genetics", "Smoking"],
            ["Genetics", "Cancer"],
        ],
        "latent": ["Genetics"],
        "description": (
            "Front-door scenario: Smoking -> Tar -> Cancer. "
            "Genetics is an unmeasured confounder of Smoking and Cancer. "
            "The front-door criterion through Tar identifies the causal effect."
        ),
    }

    return SyntheticDataset(
        data=df,
        dag=dag,
        scm=None,
        true_ate=true_ate,
        description=(
            f"Front-door scenario. True indirect effect via Tar = {true_ate:.3f}. "
            f"Back-door adjustment is blocked by unmeasured Genetics, but "
            f"front-door criterion through Tar gives identification."
        ),
        scenario="frontdoor",
        confounders=["genetics (unmeasured)"],
        mediators=["tar"],
    )


def generate_collider_data(
    n: int = 2000,
    seed: int = 42,
    *,
    talent_to_success: float = 0.5,
    beauty_to_success: float = 0.5,
) -> SyntheticDataset:
    """Collider bias scenario: Talent -> Success <- Beauty.

    Talent and Beauty are independent, but conditioning on Success
    (a collider) induces a spurious negative correlation between them.
    """
    rng = np.random.default_rng(seed)

    talent = rng.normal(0, 1, n)
    beauty = rng.normal(0, 1, n)
    success = (
        talent_to_success * talent
        + beauty_to_success * beauty
        + rng.normal(0, 0.5, n)
    )
    is_successful = (success > np.median(success)).astype(int)

    df = pd.DataFrame({
        "talent": talent,
        "beauty": beauty,
        "success": success,
        "is_successful": is_successful,
    })

    dag = {
        "nodes": ["Talent", "Beauty", "Success"],
        "edges": [
            ["Talent", "Success"],
            ["Beauty", "Success"],
        ],
        "description": (
            "Collider scenario: Talent and Beauty are independent causes "
            "of Success. Conditioning on Success induces spurious negative "
            "correlation between Talent and Beauty (Berkson's paradox)."
        ),
    }

    return SyntheticDataset(
        data=df,
        dag=dag,
        scm=None,
        true_ate=None,
        description=(
            "Collider bias demo. Talent and Beauty are marginally independent "
            "but become negatively correlated when conditioned on Success."
        ),
        scenario="collider",
    )


def generate_rct_data(
    n: int = 1000,
    seed: int = 42,
    *,
    true_effect: float = 0.6,
) -> SyntheticDataset:
    """Simple randomized controlled trial: Drug (X) -> Recovery (Y).

    No confounders by design. The naive difference in means is an unbiased
    estimate of the ATE.
    """
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, 0.5, n)
    noise = rng.normal(0, 0.3, n)
    recovery = true_effect * treatment + noise

    df = pd.DataFrame({
        "treatment": treatment,
        "recovery": recovery,
    })

    dag = {
        "nodes": ["Treatment", "Recovery"],
        "edges": [["Treatment", "Recovery"]],
        "description": "Simple RCT: Treatment -> Recovery with no confounders.",
    }

    scm = {
        "structural_equations": {
            "Treatment": "U_T",
            "Recovery": f"{true_effect} * Treatment + U_R",
        },
        "exogenous": {
            "U_T": "Bernoulli(0.5)",
            "U_R": "Normal(0, 0.3)",
        },
    }

    return SyntheticDataset(
        data=df,
        dag=dag,
        scm=scm,
        true_ate=true_effect,
        description=(
            f"RCT scenario: no confounding. True ATE = {true_effect}. "
            f"The naive difference-in-means estimator is unbiased."
        ),
        scenario="rct",
    )


def generate_iv_data(
    n: int = 2000,
    seed: int = 42,
    *,
    instrument_strength: float = 0.6,
    treatment_effect: float = 0.4,
    confounder_on_treatment: float = 0.5,
    confounder_on_outcome: float = 0.3,
) -> SyntheticDataset:
    """Instrumental variable scenario.

    DAG:  Z -> X -> Y, U -> X, U -> Y
    Z is a valid instrument: correlated with X, affects Y only through X,
    and independent of U.
    """
    rng = np.random.default_rng(seed)

    z = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    x = (
        instrument_strength * z
        + confounder_on_treatment * u
        + rng.normal(0, 0.3, n)
    )
    y = (
        treatment_effect * x
        + confounder_on_outcome * u
        + rng.normal(0, 0.3, n)
    )

    df = pd.DataFrame({"instrument": z, "treatment": x, "outcome": y})

    dag = {
        "nodes": ["Instrument (Z)", "Treatment (X)", "Outcome (Y)", "Confounder (U)"],
        "edges": [
            ["Instrument (Z)", "Treatment (X)"],
            ["Treatment (X)", "Outcome (Y)"],
            ["Confounder (U)", "Treatment (X)"],
            ["Confounder (U)", "Outcome (Y)"],
        ],
        "latent": ["Confounder (U)"],
        "description": (
            "IV scenario: Z is a valid instrument for the effect of X on Y. "
            "U is an unmeasured confounder. OLS is biased; 2SLS with Z is consistent."
        ),
    }

    return SyntheticDataset(
        data=df,
        dag=dag,
        scm=None,
        true_ate=treatment_effect,
        description=(
            f"IV scenario. True ATE = {treatment_effect}. OLS biased upward "
            f"due to confounding through U. 2SLS with instrument Z is consistent."
        ),
        scenario="iv",
    )
