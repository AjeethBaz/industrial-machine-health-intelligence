"""
Statistical diagnostics and assumption checks for the existing
multivariate SPC analysis (PCA, Hotelling's T^2, MANOVA).

This module performs read-only diagnostics on outputs already
produced by preprocessing.py, hotellings_t2.py, and manova_analysis.py.
It does not modify data, refit models, change the UCL, remove outliers,
repair any covariance matrix, or alter the underlying statistical
pipeline.

The additional Mardia, Box's M, and Levene/Brown-Forsythe outputs are
assumption diagnostics for academic reporting only. They do not change
PCA, Hotelling's T^2, MANOVA, health classification, or maintenance
recommendation results.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_dataset
from src.preprocessing import PhaseData, prepare_phase_data
from src.hotellings_t2 import HotellingsT2Result, run_hotellings_t2


SENSOR_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

MAHALANOBIS_DF: int = 5


@dataclass
class CovarianceDiagnostics:
    """Diagnostics for a single covariance matrix."""

    covariance_matrix: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    determinant: float
    is_positive_definite: bool
    is_numerically_suitable: bool


@dataclass
class Phase1T2Distribution:
    """Descriptive summary of Phase-I Hotelling's T^2 values."""

    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p90: float
    p95: float
    p99: float
    n_above_ucl: int
    pct_above_ucl: float


@dataclass
class ManovaGroupDiagnostics:
    """Covariance diagnostics for MANOVA groups plus imbalance info."""

    non_failure_n: int
    failure_n: int
    non_failure_covariance: CovarianceDiagnostics
    failure_covariance: CovarianceDiagnostics
    determinant_ratio: float


@dataclass
class NormalityDiagnostic:
    """Existing practical Mahalanobis-versus-chi-square Q-Q diagnostic."""

    mahalanobis_sq: np.ndarray
    chi2_quantiles: np.ndarray
    qq_correlation: float
    test_name: str
    limitations: str


@dataclass
class MardiaNormalityDiagnostic:
    """
    Formal Mardia multivariate skewness and kurtosis diagnostics.

    These calculations assess evidence relative to multivariate
    normal-reference behavior for the standardized healthy Phase-I
    observations. They are read-only diagnostics and do not alter the
    T^2 baseline, UCL, or any project result.
    """

    n: int
    p: int
    mardia_skewness_b1p: float
    skewness_chi_square: float
    skewness_degrees_of_freedom: int
    skewness_z: float
    skewness_p_value: float
    mardia_kurtosis_b2p: float
    normal_reference_kurtosis: float
    kurtosis_z: float
    kurtosis_p_value: float
    interpretation: str
    limitations: str


@dataclass
class BoxMResult:
    """
    Box's M covariance-homogeneity diagnostic for MANOVA groups.

    The test uses raw sensor values for Machine failure == 0 and
    Machine failure == 1. It is reported for assumption verification
    only and does not change the existing MANOVA implementation.
    """

    non_failure_n: int
    failure_n: int
    box_m_statistic: float
    correction_factor: float
    corrected_chi_square: float
    degrees_of_freedom: float
    p_value: float
    interpretation: str
    limitations: str


@dataclass
class LeveneBrownForsytheResult:
    """
    Median-centered Levene/Brown-Forsythe diagnostic for one sensor.

    A small p-value provides evidence that variance differs between the
    observed failure and non-failure groups for that individual sensor.
    """

    sensor: str
    statistic: float
    p_value: float
    interpretation: str


@dataclass
class DiagnosticsResult:
    """Top-level container bundling all diagnostic sections."""

    phase1_covariance: CovarianceDiagnostics
    phase1_t2_distribution: Phase1T2Distribution
    manova_diagnostics: ManovaGroupDiagnostics
    normality_diagnostic: NormalityDiagnostic
    mardia_normality: MardiaNormalityDiagnostic
    box_m: BoxMResult
    levene_brown_forsythe: List[LeveneBrownForsytheResult]
    top10_phase1_t2: pd.DataFrame


def _eigen_diagnostics(covariance_matrix: np.ndarray) -> CovarianceDiagnostics:
    """
    Compute eigenvalue-based diagnostics for a covariance matrix
    without modifying it in any way.

    Args:
        covariance_matrix: A square covariance matrix.

    Returns:
        A ``CovarianceDiagnostics`` object with eigenvalues,
        condition number, determinant, and positive-definiteness.
    """
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    min_eig = float(np.min(eigenvalues))
    max_eig = float(np.max(eigenvalues))

    condition_number = float(max_eig / min_eig) if min_eig > 0 else np.inf
    determinant = float(np.linalg.det(covariance_matrix))
    is_positive_definite = bool(min_eig > 0)

    is_numerically_suitable = bool(
        is_positive_definite
        and np.isfinite(determinant)
        and abs(determinant) > 1e-12
        and np.isfinite(condition_number)
        and condition_number < 1e10
    )

    return CovarianceDiagnostics(
        covariance_matrix=covariance_matrix,
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        determinant=determinant,
        is_positive_definite=is_positive_definite,
        is_numerically_suitable=is_numerically_suitable,
    )


def diagnose_phase1_covariance(phase_data: PhaseData) -> CovarianceDiagnostics:
    """
    Compute covariance diagnostics for the standardized Phase-I
    healthy baseline sensor variables.

    Args:
        phase_data: A ``PhaseData`` object from
            preprocessing.prepare_phase_data().

    Returns:
        A ``CovarianceDiagnostics`` object.
    """
    x = phase_data.phase1_healthy[phase_data.sensor_columns].values
    covariance_matrix = np.cov(x, rowvar=False)
    return _eigen_diagnostics(covariance_matrix)


def diagnose_phase1_t2_distribution(
    t2_result: HotellingsT2Result,
) -> Phase1T2Distribution:
    """
    Summarize the distribution of Phase-I Hotelling's T^2 values
    relative to the existing theoretically derived UCL.

    Args:
        t2_result: The ``HotellingsT2Result`` from
            hotellings_t2.run_hotellings_t2().

    Returns:
        A ``Phase1T2Distribution`` summary object.
    """
    t2_values = t2_result.phase1_t2["Hotelling_T2"]
    ucl = t2_result.ucl

    n_above = int((t2_values > ucl).sum())
    pct_above = float(n_above / len(t2_values) * 100) if len(t2_values) > 0 else 0.0

    return Phase1T2Distribution(
        count=int(len(t2_values)),
        mean=float(t2_values.mean()),
        median=float(t2_values.median()),
        std=float(t2_values.std()),
        minimum=float(t2_values.min()),
        maximum=float(t2_values.max()),
        p90=float(t2_values.quantile(0.90)),
        p95=float(t2_values.quantile(0.95)),
        p99=float(t2_values.quantile(0.99)),
        n_above_ucl=n_above,
        pct_above_ucl=pct_above,
    )


def diagnose_manova_groups(raw_df: pd.DataFrame) -> ManovaGroupDiagnostics:
    """
    Compute covariance diagnostics for the raw-sensor MANOVA groups
    (Machine failure == 0 vs Machine failure == 1), and report the
    determinant ratio and group-size imbalance.

    Args:
        raw_df: The raw dataset from load_dataset().

    Returns:
        A ``ManovaGroupDiagnostics`` object.
    """
    non_failure = raw_df.loc[
        raw_df["Machine failure"] == 0,
        SENSOR_COLUMNS,
    ]
    failure = raw_df.loc[
        raw_df["Machine failure"] == 1,
        SENSOR_COLUMNS,
    ]

    non_failure_cov = _eigen_diagnostics(non_failure.cov().values)
    failure_cov = _eigen_diagnostics(failure.cov().values)

    if non_failure_cov.determinant != 0:
        determinant_ratio = float(
            failure_cov.determinant / non_failure_cov.determinant
        )
    else:
        determinant_ratio = float("nan")

    return ManovaGroupDiagnostics(
        non_failure_n=int(len(non_failure)),
        failure_n=int(len(failure)),
        non_failure_covariance=non_failure_cov,
        failure_covariance=failure_cov,
        determinant_ratio=determinant_ratio,
    )


def diagnose_multivariate_normality(phase_data: PhaseData) -> NormalityDiagnostic:
    """
    Practical multivariate normality diagnostic using squared
    Mahalanobis distances of the Phase-I healthy baseline compared to
    the theoretical chi-square distribution.

    This is a descriptive Q-Q correlation diagnostic, not a formal
    hypothesis test. It is preserved exactly as an existing diagnostic
    and does not alter the statistical pipeline.

    Args:
        phase_data: A ``PhaseData`` object from
            preprocessing.prepare_phase_data().

    Returns:
        A ``NormalityDiagnostic`` object.
    """
    x = phase_data.phase1_healthy[phase_data.sensor_columns].values
    mean_vector = np.mean(x, axis=0)
    covariance_matrix = np.cov(x, rowvar=False)

    inv_cov = np.linalg.inv(covariance_matrix)
    deviations = x - mean_vector
    mahalanobis_sq = np.einsum(
        "ij,jk,ik->i",
        deviations,
        inv_cov,
        deviations,
    )

    sorted_distances = np.sort(mahalanobis_sq)
    n = len(sorted_distances)
    probs = (np.arange(1, n + 1) - 0.5) / n
    chi2_quantiles = stats.chi2.ppf(probs, df=MAHALANOBIS_DF)

    qq_correlation = float(
        np.corrcoef(sorted_distances, chi2_quantiles)[0, 1]
    )

    limitations = (
        "This is a descriptive Q-Q correlation between ordered squared "
        "Mahalanobis distances and theoretical chi-square quantiles "
        f"(df={MAHALANOBIS_DF}), not a formal hypothesis test. A high "
        "correlation is consistent with multivariate normality but does "
        "not prove it; it does not detect all departures from normality "
        "(e.g., certain skewness or tail patterns), and no single "
        "diagnostic can fully verify the multivariate normality assumption."
    )

    return NormalityDiagnostic(
        mahalanobis_sq=mahalanobis_sq,
        chi2_quantiles=chi2_quantiles,
        qq_correlation=qq_correlation,
        test_name="Mahalanobis distance vs chi-square Q-Q correlation",
        limitations=limitations,
    )


def diagnose_mardia_normality(
    phase_data: PhaseData,
) -> MardiaNormalityDiagnostic:
    """
    Calculate Mardia multivariate skewness and kurtosis diagnostics.

    The diagnostic uses the existing standardized healthy Phase-I sensor
    observations. It does not remove observations, transform data, alter
    the existing covariance estimate, or modify the Hotelling's T^2
    monitoring pipeline.

    The skewness statistic is:

        b1,p = (1 / n^2) * sum_i sum_j [d_i' S^-1 d_j]^3

    A chi-square-form statistic is calculated as:

        n * b1,p / 6 ~ chi-square(df)

    where:

        df = p(p + 1)(p + 2) / 6

    The kurtosis statistic is:

        b2,p = mean[(d_i' S^-1 d_i)^2]

    with normal-reference expectation p(p + 2).

    Args:
        phase_data: Existing standardized Phase-I healthy baseline.

    Returns:
        Mardia skewness/kurtosis statistics, standardized values,
        p-values, and reporting interpretation.
    """
    x = phase_data.phase1_healthy[phase_data.sensor_columns].values
    n, p = x.shape

    mean_vector = np.mean(x, axis=0)
    covariance_matrix = np.cov(x, rowvar=False)

    deviations = x - mean_vector

    # The covariance matrix is the existing Phase-I covariance estimate.
    # solve() avoids constructing an explicit inverse for the row-wise
    # Mahalanobis values used in the kurtosis diagnostic.
    solved_deviations = np.linalg.solve(covariance_matrix, deviations.T).T
    mahalanobis_sq = np.sum(deviations * solved_deviations, axis=1)

    # Pairwise standardized inner products are needed for Mardia b1,p.
    # This matrix is diagnostic-only and does not enter the T² pipeline.
    pairwise_inner_products = deviations @ solved_deviations.T
    mardia_skewness_b1p = float(
        np.sum(pairwise_inner_products ** 3) / (n**2)
    )

    skewness_degrees_of_freedom = int(
        p * (p + 1) * (p + 2) / 6
    )
    skewness_chi_square = float(n * mardia_skewness_b1p / 6)
    skewness_z = float(
        (
            skewness_chi_square - skewness_degrees_of_freedom
        )
        / np.sqrt(2 * skewness_degrees_of_freedom)
    )
    skewness_p_value = float(
        stats.chi2.sf(
            skewness_chi_square,
            df=skewness_degrees_of_freedom,
        )
    )

    mardia_kurtosis_b2p = float(np.mean(mahalanobis_sq**2))
    normal_reference_kurtosis = float(p * (p + 2))
    kurtosis_standard_error = float(
        np.sqrt(8 * p * (p + 2) / n)
    )
    kurtosis_z = float(
        (mardia_kurtosis_b2p - normal_reference_kurtosis)
        / kurtosis_standard_error
    )
    kurtosis_p_value = float(
        2 * stats.norm.sf(abs(kurtosis_z))
    )

    if skewness_p_value < 0.05 or kurtosis_p_value < 0.05:
        interpretation = (
            "Mardia diagnostics provide evidence of departure from "
            "multivariate normality in the standardized healthy Phase-I "
            "sensor observations. This is an assumption diagnostic only; "
            "it does not automatically invalidate descriptive or "
            "monitoring analysis, and it does not modify the existing "
            "Hotelling's T² pipeline."
        )
    else:
        interpretation = (
            "Mardia diagnostics do not provide statistically significant "
            "evidence of departure from multivariate normality at the "
            "0.05 level. This does not prove multivariate normality."
        )

    limitations = (
        "Mardia skewness and kurtosis are formal large-sample diagnostics "
        "relative to multivariate normal-reference behavior. Statistical "
        "significance can be highly sensitive with large samples. Results "
        "should be interpreted alongside the existing Mahalanobis-versus-"
        "chi-square Q-Q diagnostic and the substantive characteristics of "
        "the data. These diagnostics do not establish causal relationships "
        "and do not alter the existing monitoring methodology."
    )

    return MardiaNormalityDiagnostic(
        n=int(n),
        p=int(p),
        mardia_skewness_b1p=mardia_skewness_b1p,
        skewness_chi_square=skewness_chi_square,
        skewness_degrees_of_freedom=skewness_degrees_of_freedom,
        skewness_z=skewness_z,
        skewness_p_value=skewness_p_value,
        mardia_kurtosis_b2p=mardia_kurtosis_b2p,
        normal_reference_kurtosis=normal_reference_kurtosis,
        kurtosis_z=kurtosis_z,
        kurtosis_p_value=kurtosis_p_value,
        interpretation=interpretation,
        limitations=limitations,
    )


def diagnose_box_m(raw_df: pd.DataFrame) -> BoxMResult:
    """
    Calculate Box's M covariance-homogeneity diagnostic for the raw
    sensor variables in the two observed Machine failure groups.

    This is a read-only MANOVA assumption diagnostic. It does not
    alter the existing MANOVA model, group definitions, covariance
    matrices, sensor values, or any project result.

    The standard large-sample chi-square correction is used for the
    two-group case.

    Args:
        raw_df: Raw dataset containing sensor columns and Machine failure.

    Returns:
        Box's M statistic, correction, corrected chi-square statistic,
        degrees of freedom, p-value, and interpretation.

    Raises:
        ValueError: If required groups are empty or too small to compute
            sample covariance matrices.
    """
    non_failure = raw_df.loc[
        raw_df["Machine failure"] == 0,
        SENSOR_COLUMNS,
    ].to_numpy(dtype=float)

    failure = raw_df.loc[
        raw_df["Machine failure"] == 1,
        SENSOR_COLUMNS,
    ].to_numpy(dtype=float)

    group_arrays = [non_failure, failure]
    group_sizes = [len(group) for group in group_arrays]
    group_count = len(group_arrays)
    p = len(SENSOR_COLUMNS)
    total_n = sum(group_sizes)

    if any(size <= 1 for size in group_sizes):
        raise ValueError(
            "Box's M requires at least two observations in each group."
        )

    group_covariances = [
        np.cov(group, rowvar=False, ddof=1)
        for group in group_arrays
    ]

    pooled_covariance = sum(
        (group_sizes[index] - 1) * group_covariances[index]
        for index in range(group_count)
    ) / (total_n - group_count)

    pooled_sign, pooled_log_determinant = np.linalg.slogdet(
        pooled_covariance
    )
    group_log_determinants = []

    if pooled_sign <= 0:
        raise ValueError(
            "The pooled covariance matrix is not positive definite; "
            "Box's M cannot be calculated reliably."
        )

    for covariance_matrix in group_covariances:
        sign, log_determinant = np.linalg.slogdet(covariance_matrix)
        if sign <= 0:
            raise ValueError(
                "A group covariance matrix is not positive definite; "
                "Box's M cannot be calculated reliably."
            )
        group_log_determinants.append(log_determinant)

    box_m_statistic = float(
        (total_n - group_count) * pooled_log_determinant
        - sum(
            (group_sizes[index] - 1) * group_log_determinants[index]
            for index in range(group_count)
        )
    )

    correction_factor = float(
        (
            (2 * p**2 + 3 * p - 1)
            / (6 * (p + 1) * (group_count - 1))
        )
        * (
            sum(1 / (size - 1) for size in group_sizes)
            - 1 / (total_n - group_count)
        )
    )

    corrected_chi_square = float(
        box_m_statistic * (1 - correction_factor)
    )
    degrees_of_freedom = float(
        (group_count - 1) * p * (p + 1) / 2
    )
    p_value = float(
        stats.chi2.sf(
            corrected_chi_square,
            df=degrees_of_freedom,
        )
    )

    if p_value < 0.05:
        interpretation = (
            "Box's M provides evidence that covariance matrices differ "
            "between the observed non-failure and failure groups. The "
            "MANOVA covariance-homogeneity assumption is not supported "
            "by this diagnostic."
        )
    else:
        interpretation = (
            "Box's M does not provide statistically significant evidence "
            "of unequal covariance matrices at the 0.05 level. This does "
            "not prove covariance homogeneity."
        )

    limitations = (
        "Box's M is sensitive to multivariate normality departures and "
        "sample-size imbalance. The observed failure group is much smaller "
        "than the non-failure group, so results should be interpreted as "
        "assumption evidence rather than as a causal finding. This "
        "diagnostic does not alter the existing MANOVA analysis."
    )

    return BoxMResult(
        non_failure_n=int(group_sizes[0]),
        failure_n=int(group_sizes[1]),
        box_m_statistic=box_m_statistic,
        correction_factor=correction_factor,
        corrected_chi_square=corrected_chi_square,
        degrees_of_freedom=degrees_of_freedom,
        p_value=p_value,
        interpretation=interpretation,
        limitations=limitations,
    )


def diagnose_levene_brown_forsythe(
    raw_df: pd.DataFrame,
) -> List[LeveneBrownForsytheResult]:
    """
    Calculate median-centered Levene/Brown-Forsythe variance diagnostics
    for every raw sensor variable across observed failure groups.

    These are supplementary univariate diagnostics for the MANOVA
    covariance-homogeneity assumption. They do not alter the dataset,
    MANOVA model, or any monitoring result.

    Args:
        raw_df: Raw dataset containing sensors and Machine failure.

    Returns:
        One Levene/Brown-Forsythe result per sensor variable.
    """
    results: List[LeveneBrownForsytheResult] = []

    non_failure_mask = raw_df["Machine failure"] == 0
    failure_mask = raw_df["Machine failure"] == 1

    for sensor in SENSOR_COLUMNS:
        non_failure_values = raw_df.loc[
            non_failure_mask,
            sensor,
        ].to_numpy(dtype=float)

        failure_values = raw_df.loc[
            failure_mask,
            sensor,
        ].to_numpy(dtype=float)

        statistic, p_value = stats.levene(
            non_failure_values,
            failure_values,
            center="median",
        )

        if p_value < 0.05:
            interpretation = (
                "Evidence of unequal variance between observed "
                "non-failure and failure groups at the 0.05 level."
            )
        else:
            interpretation = (
                "No statistically significant evidence of unequal "
                "variance between observed non-failure and failure "
                "groups at the 0.05 level."
            )

        results.append(
            LeveneBrownForsytheResult(
                sensor=sensor,
                statistic=float(statistic),
                p_value=float(p_value),
                interpretation=interpretation,
            )
        )

    return results


def top_n_phase1_t2(
    t2_result: HotellingsT2Result,
    n: int = 10,
) -> pd.DataFrame:
    """
    Identify the top-N Phase-I observations by Hotelling's T^2 value,
    for diagnostic review only. No observations are removed.

    Args:
        t2_result: The ``HotellingsT2Result`` from
            hotellings_t2.run_hotellings_t2().
        n: Number of top observations to return.

    Returns:
        A DataFrame with UDI and Hotelling_T2, sorted descending.
    """
    top = t2_result.phase1_t2[
        ["UDI", "Hotelling_T2"]
    ].sort_values(
        "Hotelling_T2",
        ascending=False,
    )

    return top.head(n).reset_index(drop=True)


def run_diagnostics(
    raw_df: pd.DataFrame,
    phase_data: PhaseData,
    t2_result: HotellingsT2Result,
) -> DiagnosticsResult:
    """
    Run all read-only diagnostic sections and bundle results.

    Existing covariance, Phase-I T², MANOVA-group, Q-Q, and top-observation
    diagnostics are preserved. Mardia, Box's M, and Levene/Brown-Forsythe
    diagnostics are additive assumption checks only.

    Args:
        raw_df: Raw dataset from load_dataset().
        phase_data: PhaseData from preprocessing.prepare_phase_data().
        t2_result: HotellingsT2Result from hotellings_t2.run_hotellings_t2().

    Returns:
        A ``DiagnosticsResult`` containing all diagnostic outputs.
    """
    phase1_covariance = diagnose_phase1_covariance(phase_data)
    phase1_t2_distribution = diagnose_phase1_t2_distribution(t2_result)
    manova_diagnostics = diagnose_manova_groups(raw_df)
    normality_diagnostic = diagnose_multivariate_normality(phase_data)
    mardia_normality = diagnose_mardia_normality(phase_data)
    box_m = diagnose_box_m(raw_df)
    levene_brown_forsythe = diagnose_levene_brown_forsythe(raw_df)
    top10 = top_n_phase1_t2(t2_result, n=10)

    return DiagnosticsResult(
        phase1_covariance=phase1_covariance,
        phase1_t2_distribution=phase1_t2_distribution,
        manova_diagnostics=manova_diagnostics,
        normality_diagnostic=normality_diagnostic,
        mardia_normality=mardia_normality,
        box_m=box_m,
        levene_brown_forsythe=levene_brown_forsythe,
        top10_phase1_t2=top10,
    )


def plot_t2_histogram(t2_result: HotellingsT2Result) -> plt.Figure:
    """
    Build a histogram of Phase-II Hotelling's T^2 values with the
    UCL marked as a reference line.

    Args:
        t2_result: The ``HotellingsT2Result`` from
            hotellings_t2.run_hotellings_t2().

    Returns:
        A matplotlib Figure. Not displayed or saved automatically.
    """
    fig, ax = plt.subplots()
    ax.hist(
        t2_result.phase2_results["Hotelling_T2"],
        bins=40,
        alpha=0.7,
    )
    ax.axvline(
        t2_result.ucl,
        color="red",
        linestyle="--",
        label="UCL",
    )
    ax.set_xlabel("Hotelling's T2")
    ax.set_ylabel("Frequency")
    ax.set_title("Phase-II Hotelling's T2 Distribution")
    ax.legend()
    return fig


def plot_chi2_qq(
    normality_diagnostic: NormalityDiagnostic,
) -> plt.Figure:
    """
    Build a chi-square Q-Q plot comparing ordered Phase-I Mahalanobis
    squared distances to theoretical chi-square quantiles.

    Args:
        normality_diagnostic: The existing Q-Q diagnostic result.

    Returns:
        A matplotlib Figure. Not displayed or saved automatically.
    """
    fig, ax = plt.subplots()
    ax.scatter(
        normality_diagnostic.chi2_quantiles,
        np.sort(normality_diagnostic.mahalanobis_sq),
        alpha=0.6,
        s=15,
    )

    max_val = max(
        normality_diagnostic.chi2_quantiles.max(),
        normality_diagnostic.mahalanobis_sq.max(),
    )

    ax.plot(
        [0, max_val],
        [0, max_val],
        color="red",
        linestyle="--",
        label="y = x",
    )
    ax.set_xlabel(
        f"Theoretical Chi-Square Quantile (df={MAHALANOBIS_DF})"
    )
    ax.set_ylabel("Ordered Mahalanobis Distance Squared")
    ax.set_title("Chi-Square Q-Q Plot (Phase-I Healthy Baseline)")
    ax.legend()
    return fig


def plot_covariance_heatmap(
    covariance_matrix: np.ndarray,
    labels: List[str],
) -> plt.Figure:
    """
    Build a heatmap of a covariance matrix.

    Args:
        covariance_matrix: A square covariance matrix.
        labels: Variable names corresponding to matrix rows/columns.

    Returns:
        A matplotlib Figure. Not displayed or saved automatically.
    """
    fig, ax = plt.subplots()
    sns.heatmap(
        covariance_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="coolwarm",
        ax=ax,
    )
    ax.set_title("Covariance Matrix Heatmap")
    return fig


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    t2_result = run_hotellings_t2(prepared)
    diagnostics = run_diagnostics(raw_df, prepared, t2_result)

    print("=" * 60)
    print("STATISTICAL DIAGNOSTICS")
    print("=" * 60)

    pc = diagnostics.phase1_covariance
    print("\nPHASE-I COVARIANCE DIAGNOSTICS")
    print(f"Eigenvalues: {pc.eigenvalues}")
    print(f"Condition number: {pc.condition_number:.4f}")
    print(f"Minimum eigenvalue: {pc.min_eigenvalue:.6f}")
    print(f"Maximum eigenvalue: {pc.max_eigenvalue:.6f}")
    print(f"Determinant: {pc.determinant:.6e}")
    print(f"Positive definite: {pc.is_positive_definite}")
    print(f"Numerically suitable: {pc.is_numerically_suitable}")

    pt = diagnostics.phase1_t2_distribution
    print("\nPHASE-I T2 DISTRIBUTION")
    print(f"Count: {pt.count}")
    print(f"Mean: {pt.mean:.4f}")
    print(f"Median: {pt.median:.4f}")
    print(f"Std: {pt.std:.4f}")
    print(f"Min: {pt.minimum:.4f}")
    print(f"Max: {pt.maximum:.4f}")
    print(f"P90: {pt.p90:.4f}")
    print(f"P95: {pt.p95:.4f}")
    print(f"P99: {pt.p99:.4f}")
    print(f"Above UCL: {pt.n_above_ucl}")
    print(f"Percentage above UCL: {pt.pct_above_ucl:.2f}%")

    nd = diagnostics.normality_diagnostic
    print("\nEXISTING MAHALANOBIS Q-Q DIAGNOSTIC")
    print(f"Diagnostic used: {nd.test_name}")
    print(f"Q-Q correlation: {nd.qq_correlation:.6f}")
    print(f"Limitations: {nd.limitations}")

    mardia = diagnostics.mardia_normality
    print("\nMARDIA MULTIVARIATE NORMALITY DIAGNOSTICS")
    print(f"n: {mardia.n}")
    print(f"p: {mardia.p}")
    print(f"Mardia skewness b1,p: {mardia.mardia_skewness_b1p:.10f}")
    print(f"Skewness chi-square: {mardia.skewness_chi_square:.10f}")
    print(f"Skewness df: {mardia.skewness_degrees_of_freedom}")
    print(f"Skewness z: {mardia.skewness_z:.10f}")
    print(f"Skewness p-value: {mardia.skewness_p_value:.6e}")
    print(f"Mardia kurtosis b2,p: {mardia.mardia_kurtosis_b2p:.10f}")
    print(
        "Normal-reference kurtosis p(p+2): "
        f"{mardia.normal_reference_kurtosis:.6f}"
    )
    print(f"Kurtosis z: {mardia.kurtosis_z:.10f}")
    print(f"Kurtosis p-value: {mardia.kurtosis_p_value:.6e}")
    print(f"Interpretation: {mardia.interpretation}")

    box_m = diagnostics.box_m
    print("\nBOX'S M COVARIANCE-HOMOGENEITY DIAGNOSTIC")
    print(f"Non-failure n: {box_m.non_failure_n}")
    print(f"Failure n: {box_m.failure_n}")
    print(f"Box's M: {box_m.box_m_statistic:.10f}")
    print(f"Correction factor: {box_m.correction_factor:.10f}")
    print(
        "Corrected chi-square: "
        f"{box_m.corrected_chi_square:.10f}"
    )
    print(f"Degrees of freedom: {box_m.degrees_of_freedom:.2f}")
    print(f"p-value: {box_m.p_value:.6e}")
    print(f"Interpretation: {box_m.interpretation}")

    print("\nLEVENE/BROWN-FORSYTHE DIAGNOSTICS")
    for result in diagnostics.levene_brown_forsythe:
        print(f"{result.sensor}:")
        print(f"  Statistic: {result.statistic:.6f}")
        print(f"  p-value: {result.p_value:.6e}")
        print(f"  Interpretation: {result.interpretation}")

    print("\nTOP 10 PHASE-I T2 OBSERVATIONS")
    print(diagnostics.top10_phase1_t2.to_string(index=False))

    print("\n" + "=" * 60)