"""
Phase-II Hotelling's T-squared multivariate statistical process
monitoring.

The mean vector and covariance matrix are estimated exclusively from
the standardized Phase-I healthy baseline (produced by
preprocessing.prepare_phase_data). Phase-II data and the
Machine failure label are never used to estimate these baseline
parameters or the control limit; Machine failure is preserved in the
output purely as an outcome variable for later comparison against
T^2 alerts.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import f as f_dist

from src.data_loader import load_dataset
from src.preprocessing import PhaseData, prepare_phase_data

ALPHA: float = 0.05


@dataclass
class HotellingsT2Result:
    """Container for Hotelling's T-squared monitoring results."""

    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    covariance_determinant: float
    ucl: float
    alpha: float
    p: int
    n: int
    f_critical: float
    phase1_t2: pd.DataFrame
    phase2_results: pd.DataFrame
    sensor_columns: List[str]


def estimate_baseline_parameters(
    phase1_healthy: pd.DataFrame, sensor_columns: List[str]
):
    """
    Estimate the mean vector and sample covariance matrix from the
    standardized Phase-I healthy baseline.

    Args:
        phase1_healthy: Standardized healthy Phase-I DataFrame.
        sensor_columns: The five sensor variable names.

    Returns:
        Tuple of (mean_vector, covariance_matrix) as numpy arrays.

    Raises:
        ValueError: If the covariance matrix does not have the
            expected (p, p) dimensions.
    """
    x = phase1_healthy[sensor_columns].values
    mean_vector = np.mean(x, axis=0)
    covariance_matrix = np.cov(x, rowvar=False)

    p = len(sensor_columns)
    if covariance_matrix.shape != (p, p):
        raise ValueError(
            f"Covariance matrix has shape {covariance_matrix.shape}, "
            f"expected ({p}, {p})."
        )
    return mean_vector, covariance_matrix


def _check_covariance_solvable(covariance_matrix: np.ndarray) -> float:
    """
    Verify the covariance matrix is numerically suitable for solving
    the linear system used in the T^2 calculation.

    Args:
        covariance_matrix: The (p, p) sample covariance matrix.

    Returns:
        The determinant of the covariance matrix, as a diagnostic.

    Raises:
        ValueError: If the matrix is singular or near-singular.
    """
    determinant = float(np.linalg.det(covariance_matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError(
            "Covariance matrix is singular or near-singular "
            f"(determinant={determinant}). Hotelling's T2 cannot be "
            "reliably calculated. Check for constant or collinear "
            "sensor variables in the Phase-I healthy baseline."
        )
    return determinant


def calculate_hotellings_t2(
    data: pd.DataFrame,
    sensor_columns: List[str],
    mean_vector: np.ndarray,
    covariance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate Hotelling's T^2 statistic for each observation in
    ``data`` relative to the given baseline mean and covariance.

    T^2 = (x - mu)' S^-1 (x - mu), solved via scipy.linalg.solve
    instead of explicit matrix inversion.

    Args:
        data: DataFrame containing the sensor columns to score.
        sensor_columns: The five sensor variable names.
        mean_vector: Baseline mean vector (from Phase-I healthy data).
        covariance_matrix: Baseline covariance matrix (from Phase-I
            healthy data).

    Returns:
        A numpy array of T^2 values, one per row in ``data``.
    """
    x = data[sensor_columns].values
    deviations = x - mean_vector

    solved = solve(covariance_matrix, deviations.T, assume_a="pos")
    t2_values = np.sum(deviations.T * solved, axis=0)
    return t2_values


def calculate_control_limit(p: int, n: int, alpha: float = ALPHA):
    """
    Calculate the classical Phase-II Hotelling's T^2 upper control
    limit based on the F distribution.

    UCL = [p(n + 1)(n - 1) / (n(n - p))] * F_(p, n-p; 1-alpha)

    Args:
        p: Number of monitored variables.
        n: Number of Phase-I healthy baseline observations.
        alpha: Significance level (default 0.05).

    Returns:
        Tuple of (ucl, f_critical).

    Raises:
        ValueError: If n <= p, which makes the control limit formula
            undefined.
    """
    if n <= p:
        raise ValueError(
            f"Phase-I baseline size (n={n}) must exceed the number of "
            f"variables (p={p}) to calculate the control limit."
        )

    f_critical = f_dist.ppf(1 - alpha, p, n - p)
    ucl = (p * (n + 1) * (n - 1) / (n * (n - p))) * f_critical
    return float(ucl), float(f_critical)


def run_hotellings_t2(phase_data: PhaseData, alpha: float = ALPHA) -> HotellingsT2Result:
    """
    Run the full Phase-I / Phase-II Hotelling's T^2 monitoring
    procedure using the Phase-I healthy baseline.

    Args:
        phase_data: A ``PhaseData`` object from
            ``preprocessing.prepare_phase_data()``.
        alpha: Significance level for the control limit (default 0.05).

    Returns:
        A ``HotellingsT2Result`` containing baseline parameters, the
        control limit, and Phase-I/Phase-II T^2 results.
    """
    sensor_columns = phase_data.sensor_columns
    phase1_healthy = phase_data.phase1_healthy
    phase2 = phase_data.phase2

    mean_vector, covariance_matrix = estimate_baseline_parameters(
        phase1_healthy, sensor_columns
    )
    covariance_determinant = _check_covariance_solvable(covariance_matrix)

    p = len(sensor_columns)
    n = len(phase1_healthy)
    ucl, f_critical = calculate_control_limit(p=p, n=n, alpha=alpha)

    phase1_t2_values = calculate_hotellings_t2(
        phase1_healthy, sensor_columns, mean_vector, covariance_matrix
    )
    phase2_t2_values = calculate_hotellings_t2(
        phase2, sensor_columns, mean_vector, covariance_matrix
    )

    phase1_t2 = pd.DataFrame(
        {
            "UDI": phase1_healthy["UDI"].values,
            "Hotelling_T2": phase1_t2_values,
        }
    )
    phase1_t2["T2_Alert"] = (phase1_t2["Hotelling_T2"] > ucl).astype(int)

    phase2_results = phase2[["UDI"] + sensor_columns + ["Machine failure"]].copy()
    phase2_results["Hotelling_T2"] = phase2_t2_values
    phase2_results["T2_Alert"] = (phase2_results["Hotelling_T2"] > ucl).astype(int)

    return HotellingsT2Result(
        mean_vector=mean_vector,
        covariance_matrix=covariance_matrix,
        covariance_determinant=covariance_determinant,
        ucl=ucl,
        alpha=alpha,
        p=p,
        n=n,
        f_critical=f_critical,
        phase1_t2=phase1_t2,
        phase2_results=phase2_results,
        sensor_columns=sensor_columns,
    )


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    result = run_hotellings_t2(prepared)

    print("=" * 60)
    print("HOTELLING'S T-SQUARED MONITORING SUMMARY")
    print("=" * 60)

    print(f"\nNumber of variables (p): {result.p}")
    print(f"Phase-I healthy baseline size (n): {result.n}")
    print(f"Alpha: {result.alpha}")
    print(f"F critical value: {result.f_critical:.4f}")
    print(f"Hotelling's T2 UCL: {result.ucl:.4f}")

    p1 = result.phase1_t2["Hotelling_T2"]
    print("\nPhase-I T2 summary:")
    print(f"  min:    {p1.min():.4f}")
    print(f"  mean:   {p1.mean():.4f}")
    print(f"  median: {p1.median():.4f}")
    print(f"  max:    {p1.max():.4f}")

    p2 = result.phase2_results["Hotelling_T2"]
    print("\nPhase-II T2 summary:")
    print(f"  min:    {p2.min():.4f}")
    print(f"  mean:   {p2.mean():.4f}")
    print(f"  median: {p2.median():.4f}")
    print(f"  max:    {p2.max():.4f}")

    n_alerts = int(result.phase2_results["T2_Alert"].sum())
    alert_pct = n_alerts / len(result.phase2_results) * 100
    print(f"\nPhase-II number of T2 alerts: {n_alerts}")
    print(f"Phase-II alert percentage: {alert_pct:.2f}%")

    n_failures = int(result.phase2_results["Machine failure"].sum())
    print(f"\nPhase-II actual machine failures: {n_failures}")

    both = int(
        (
            (result.phase2_results["T2_Alert"] == 1)
            & (result.phase2_results["Machine failure"] == 1)
        ).sum()
    )
    print(f"Observations that are both T2 alerts and actual failures: {both}")