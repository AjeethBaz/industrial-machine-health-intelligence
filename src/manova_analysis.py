"""
MANOVA analysis of the five sensor variables by observed Machine
failure status.

Uses the RAW (non-standardized) sensor values so group means remain
interpretable in original physical units. Machine failure is the
actual observed binary outcome recorded in the AI4I 2020 dataset; no
synthetic operational states (Normal/Degraded/Near-Failure) are
created. This module is independent of the Hotelling's T^2 pipeline
and does not use T^2 or PCA scores in any way.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

from src.data_loader import load_dataset

SENSOR_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

REQUIRED_COLUMNS: List[str] = SENSOR_COLUMNS + ["Machine failure"]


@dataclass
class ManovaResult:
    """Container for MANOVA results and supporting diagnostics."""

    group_sizes: Dict[str, int]
    descriptive_stats: pd.DataFrame
    mean_differences: pd.Series
    multivariate_tests: pd.DataFrame
    covariance_matrices: Dict[str, pd.DataFrame]
    covariance_diagnostics: Dict[str, bool]
    limitations: List[str]


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure all required sensor and grouping columns exist.

    Args:
        df: Raw dataset loaded via load_dataset().

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Required columns are: {REQUIRED_COLUMNS}."
        )


def _compute_group_sizes(df: pd.DataFrame) -> Dict[str, int]:
    """Return observation counts for each Machine failure group."""
    non_failure = int((df["Machine failure"] == 0).sum())
    failure = int((df["Machine failure"] == 1).sum())
    return {"Non-failure": non_failure, "Failure": failure}


def _compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute count, mean, standard deviation, and median for each
    sensor variable, grouped by Machine failure.

    Args:
        df: Raw dataset with sensor and grouping columns.

    Returns:
        A DataFrame indexed by (sensor, Machine failure group) with
        descriptive statistics as columns.
    """
    grouped = df.groupby("Machine failure")[SENSOR_COLUMNS].agg(
        ["count", "mean", "std", "median"]
    )
    grouped = grouped.rename(index={0: "Non-failure", 1: "Failure"})
    stacked = grouped.stack(level=0, future_stack=True)
    stacked.index.names = ["Machine failure", "Sensor"]
    stacked = stacked.reorder_levels(["Sensor", "Machine failure"]).sort_index(level=0)
    return stacked


def _compute_mean_differences(df: pd.DataFrame) -> pd.Series:
    """
    Compute failure-minus-non-failure mean difference for every
    sensor variable.

    Args:
        df: Raw dataset with sensor and grouping columns.

    Returns:
        A Series indexed by sensor name with mean differences.
    """
    means = df.groupby("Machine failure")[SENSOR_COLUMNS].mean()
    diff = means.loc[1] - means.loc[0]
    diff.name = "failure_minus_non_failure"
    return diff


def _compute_covariance_diagnostics(df: pd.DataFrame):
    """
    Compute per-group covariance matrices for the sensor variables
    and check basic numerical validity (finite values, expected
    dimensions, non-singularity).

    Args:
        df: Raw dataset with sensor and grouping columns.

    Returns:
        Tuple of (covariance_matrices, diagnostics) where
        covariance_matrices maps group label to a covariance
        DataFrame, and diagnostics maps group label to a boolean
        indicating whether the matrix appears numerically valid.
    """
    covariance_matrices: Dict[str, pd.DataFrame] = {}
    diagnostics: Dict[str, bool] = {}

    group_labels = {0: "Non-failure", 1: "Failure"}
    p = len(SENSOR_COLUMNS)

    for code, label in group_labels.items():
        subset = df.loc[df["Machine failure"] == code, SENSOR_COLUMNS]
        cov = subset.cov()
        covariance_matrices[label] = cov

        is_finite = np.isfinite(cov.values).all()
        correct_shape = cov.shape == (p, p)
        determinant = np.linalg.det(cov.values) if correct_shape else 0.0
        is_nonsingular = np.isfinite(determinant) and abs(determinant) > 1e-12

        diagnostics[label] = bool(is_finite and correct_shape and is_nonsingular)

    return covariance_matrices, diagnostics


def _run_manova_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the MANOVA model and extract all standard multivariate test
    statistics from statsmodels.

    Model: Y1 + Y2 + Y3 + Y4 + Y5 ~ Machine failure

    Column names containing spaces and brackets are referenced using
    Patsy's Q("...") quoting syntax so the formula parses correctly
    without renaming any columns in the underlying data.

    Args:
        df: Raw dataset with sensor and grouping columns.

    Returns:
        A DataFrame indexed by test name (Wilks' lambda, Pillai's
        trace, Hotelling-Lawley trace, Roy's greatest root) with
        columns: Value, F Value, Num DF, Den DF, Pr > F.
    """
    manova_df = df[SENSOR_COLUMNS + ["Machine failure"]].copy()

    dependent_terms = " + ".join([f'Q("{col}")' for col in SENSOR_COLUMNS])
    formula = f'{dependent_terms} ~ Q("Machine failure")'

    fit = MANOVA.from_formula(formula, data=manova_df)
    test_results = fit.mv_test()

    stats_table = test_results.results['Q("Machine failure")']["stat"].copy()
    stats_table.index = [
        "Wilks' lambda" if "Wilks" in str(i) else
        "Pillai's trace" if "Pillai" in str(i) else
        "Hotelling-Lawley trace" if "Hotelling" in str(i) else
        "Roy's greatest root" if "Roy" in str(i) else str(i)
        for i in stats_table.index
    ]
    return stats_table


def run_manova_analysis(df: pd.DataFrame) -> ManovaResult:
    """
    Run the full MANOVA analysis pipeline: validation, descriptive
    statistics, mean differences, covariance diagnostics, and the
    multivariate test itself.

    Args:
        df: Raw dataset as returned by load_dataset().

    Returns:
        A ``ManovaResult`` containing all outputs and diagnostics.
    """
    _validate_columns(df)

    group_sizes = _compute_group_sizes(df)
    descriptive_stats = _compute_descriptive_stats(df)
    mean_differences = _compute_mean_differences(df)
    covariance_matrices, covariance_diagnostics = _compute_covariance_diagnostics(df)
    multivariate_tests = _run_manova_test(df)

    limitations: List[str] = []
    if group_sizes["Failure"] < len(SENSOR_COLUMNS) + 1:
        limitations.append(
            "The failure group sample size is very small relative to the "
            "number of dependent variables; covariance estimates for this "
            "group may be unstable."
        )
    if not all(covariance_diagnostics.values()):
        limitations.append(
            "One or more group covariance matrices failed numerical "
            "validity checks (non-finite values, unexpected shape, or "
            "near-singularity)."
        )
    if group_sizes["Failure"] < 30:
        limitations.append(
            "Multivariate normality and homogeneity of covariance matrices "
            "(assumptions underlying MANOVA) were not formally tested here; "
            "results should be interpreted with caution given the small "
            "failure group size."
        )
    limitations.append(
        "MANOVA significance indicates the multivariate sensor means differ "
        "between observed failure and non-failure groups; it does not imply "
        "that the sensor variables cause machine failure."
    )

    return ManovaResult(
        group_sizes=group_sizes,
        descriptive_stats=descriptive_stats,
        mean_differences=mean_differences,
        multivariate_tests=multivariate_tests,
        covariance_matrices=covariance_matrices,
        covariance_diagnostics=covariance_diagnostics,
        limitations=limitations,
    )


if __name__ == "__main__":
    raw_df = load_dataset()
    result = run_manova_analysis(raw_df)

    print("=" * 60)
    print("MANOVA RESULTS")
    print("=" * 60)

    print("\nGroup sizes:")
    print(f"Non-failure: {result.group_sizes['Non-failure']}")
    print(f"Failure: {result.group_sizes['Failure']}")

    print("\nDescriptive statistics by Machine failure:")
    print(result.descriptive_stats)

    print("\nGroup mean differences (failure - non-failure):")
    print(result.mean_differences)

    print("\nMultivariate tests:")
    for test_name in [
        "Wilks' lambda",
        "Pillai's trace",
        "Hotelling-Lawley trace",
        "Roy's greatest root",
    ]:
        if test_name in result.multivariate_tests.index:
            row = result.multivariate_tests.loc[test_name]
            print(f"\n{test_name}:")
            print(f"  Value: {row['Value']:.6f}")
            print(f"  F statistic: {row['F Value']:.4f}")
            print(f"  df numerator: {row['Num DF']}")
            print(f"  df denominator: {row['Den DF']}")
            print(f"  p-value: {row['Pr > F']:.6f}")

    print("\nCovariance matrix diagnostics:")
    for group, is_valid in result.covariance_diagnostics.items():
        print(f"  {group}: {'valid' if is_valid else 'INVALID / UNSTABLE'}")

    print("\nLimitations:")
    for note in result.limitations:
        print(f"  - {note}")