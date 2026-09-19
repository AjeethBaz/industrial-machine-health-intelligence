"""
backend/schemas.py

Pydantic response models and pure conversion helpers that turn the
existing pipeline's pandas/NumPy outputs into JSON-safe structures.

No statistical values are computed, altered, or re-derived here.
Every function in this file only reshapes already-computed results
from backend/pipeline.py for HTTP transport.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel


# ======================================================================
# CONVERSION HELPERS (pandas/NumPy -> JSON-safe Python)
# ======================================================================


def df_to_records(
    df: pd.DataFrame,
    index_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame into a list of JSON-safe row dictionaries,
    optionally including the index under the given column name.
    """
    working = df.copy()

    if index_name is not None:
        working = working.reset_index().rename(
            columns={working.index.name or "index": index_name}
        )

    records = working.to_dict(orient="records")
    return [_sanitize_record(record) for record in records]


def _sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NumPy scalar types within a record to native Python types."""
    return {
        key: _sanitize_value(value)
        for key, value in record.items()
    }


def _sanitize_value(value: Any) -> Any:
    """Recursively convert NumPy/pandas types to native JSON-safe Python types."""
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return [_sanitize_value(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _sanitize_value(item)
            for key, item in value.items()
        }

    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None

    return value


def matrix_to_nested_list(matrix: np.ndarray) -> List[List[float]]:
    """Convert a 2D NumPy array into a JSON-safe nested list of floats."""
    return [
        [float(value) for value in row]
        for row in matrix
    ]


def series_to_dict(series: pd.Series) -> Dict[str, float]:
    """Convert a pandas Series into a JSON-safe dict of float values."""
    return {
        str(key): float(value)
        for key, value in series.items()
    }


# ======================================================================
# RESPONSE MODELS
# ======================================================================


class ValidationSummary(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    total_missing: int


class HealthStatusCount(BaseModel):
    status: str
    count: int
    percentage: float


class OverviewResponse(BaseModel):
    total_observations: int
    phase1_healthy_count: int
    phase2_count: int
    ucl: float
    alert_rate_percent: float
    observed_failure_rate_percent: float
    health_status_distribution: List[HealthStatusCount]
    validation: ValidationSummary


class MachineListItem(BaseModel):
    udi: int
    health_status: str
    t2_value: float


class ContributingSensor(BaseModel):
    sensor: str
    z_score: float
    direction: str


class MachineHealthResponse(BaseModel):
    udi: int
    t2_value: float
    ucl: float
    t2_ratio: float
    health_status: str
    priority: str
    sensor_values: Dict[str, float]
    standardized_sensors: Dict[str, float]
    contributing_sensors: List[ContributingSensor]
    primary_recommendation: str
    additional_recommendations: List[str]
    evidence: List[str]
    limitations: List[str]
    failure_probability: Optional[float] = None
    failure_probability_note: str = (
        "Failure probability: Not available — T2 is a multivariate abnormality measure, "
        "not a probability."
    )


class SPCPoint(BaseModel):
    udi: int
    t2: float
    alert: bool
    health_status: str


class SPCResponse(BaseModel):
    ucl: float
    watch_threshold: float
    alert_threshold: float
    points: List[SPCPoint]


class PCAResponse(BaseModel):
    components: List[str]
    explained_variance_ratio: List[float]
    cumulative_explained_variance: List[float]
    loadings: Dict[str, Dict[str, float]]
    components_required_for_90_percent: int
    interpretation: str


class T2EvaluationResponse(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    specificity: float
    f1_score: float
    false_positive_rate: float
    alert_rate: float
    actual_failure_rate: float
    failure_capture_rate: float
    group_comparison: List[Dict[str, Any]]


class ManovaTestRow(BaseModel):
    test_name: str
    value: float
    f_value: float
    num_df: float
    den_df: float
    p_value: float


class ManovaResponse(BaseModel):
    group_sizes: Dict[str, int]
    descriptive_stats: List[Dict[str, Any]]
    mean_differences: Dict[str, float]
    multivariate_tests: List[ManovaTestRow]
    limitations: List[str]


class CovarianceDiagnosticsResponse(BaseModel):
    """
    Existing Phase-I covariance diagnostics.

    The new eigenvalues field is additive. Existing consumers may
    continue using the original fields unchanged.
    """

    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    determinant: float
    is_positive_definite: bool
    is_numerically_suitable: bool
    eigenvalues: Optional[List[float]] = None
    interpretation: Optional[str] = None


class Phase1T2DistributionResponse(BaseModel):
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


class NormalityDiagnosticResponse(BaseModel):
    """Existing Mahalanobis-versus-chi-square Q-Q diagnostic."""

    qq_correlation: float
    test_name: str
    limitations: str


class MardiaNormalityDiagnosticResponse(BaseModel):
    """
    Additive formal multivariate-normality assumption diagnostics.

    Optional fields preserve compatibility for any existing client that
    only consumes original /api/diagnostics sections.
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


class BoxMResponse(BaseModel):
    """
    Additive MANOVA covariance-homogeneity diagnostic response.
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


class LeveneBrownForsytheResponse(BaseModel):
    """
    Additive median-centered Levene/Brown-Forsythe result for one sensor.
    """

    sensor: str
    statistic: float
    p_value: float
    interpretation: str


class DiagnosticsResponse(BaseModel):
    """
    Diagnostics endpoint response.

    The original four fields are preserved exactly:
    - phase1_covariance
    - phase1_t2_distribution
    - normality_diagnostic
    - top10_phase1_t2

    New sections are optional additive fields for backward compatibility.
    """

    phase1_covariance: CovarianceDiagnosticsResponse
    phase1_t2_distribution: Phase1T2DistributionResponse
    normality_diagnostic: NormalityDiagnosticResponse
    top10_phase1_t2: List[Dict[str, Any]]

    mardia_normality: Optional[MardiaNormalityDiagnosticResponse] = None
    box_m: Optional[BoxMResponse] = None
    levene_brown_forsythe: Optional[
        List[LeveneBrownForsytheResponse]
    ] = None


class MaintenanceResponse(BaseModel):
    udi: int
    health_status: str
    priority: str
    contributing_sensors: List[ContributingSensor]
    primary_recommendation: str
    additional_recommendations: List[str]
    evidence: List[str]
    limitations: List[str]


class AssistantRequest(BaseModel):
    udi: int
    question: Optional[str] = None


class AssistantResponse(BaseModel):
    udi: int
    response: str
    failure_probability: Optional[float] = None
    failure_probability_note: str = (
        "Failure probability: Not available — T2 is a multivariate abnormality measure, "
        "not a probability."
    )