"""
backend/pipeline.py

Runs (and caches, in-process) the existing statistical pipeline from
src/*, and exposes plain-Python helper functions that backend/main.py
can call. This module does NOT reimplement or alter any statistical
methodology. It only reuses the already-tested functions/classes from
src/data_loader.py, src/preprocessing.py, src/pca_analysis.py,
src/hotellings_t2.py, src/t2_evaluation.py, src/manova_analysis.py,
src/diagnostics.py, src/health_assessment.py, and
src/maintenance_recommendation.py.

Caching strategy: a single module-level PipelineCache instance is
built lazily on first access and reused for the lifetime of the
FastAPI process (functionally equivalent to Streamlit's
@st.cache_resource, but implemented as a plain singleton since FastAPI
has no built-in caching decorator).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_loader import load_dataset, validate_dataset
from src.preprocessing import prepare_phase_data, PhaseData
from src.pca_analysis import run_pca, components_for_variance_threshold, PCAResult
from src.hotellings_t2 import run_hotellings_t2, HotellingsT2Result
from src.t2_evaluation import evaluate_t2_monitoring, T2EvaluationResult
from src.manova_analysis import run_manova_analysis, ManovaResult
from src.diagnostics import run_diagnostics, DiagnosticsResult
from src.health_assessment import (
    classify_health,
    calculate_t2_ratio,
    assess_dataset_health,
    summarize_health_statuses,
)
from src.maintenance_recommendation import generate_maintenance_recommendation

SENSOR_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


@dataclass
class PipelineCache:
    """Container holding every computed statistical result, built once."""

    raw_df: pd.DataFrame
    validation_summary: Dict[str, Any]
    phase_data: PhaseData
    pca_result: PCAResult
    t2_result: HotellingsT2Result
    t2_eval: T2EvaluationResult
    manova_result: ManovaResult
    diagnostics_result: DiagnosticsResult
    health_df: pd.DataFrame
    health_summary: pd.DataFrame


_cache: Optional[PipelineCache] = None


def get_pipeline() -> PipelineCache:
    """
    Return the cached pipeline, building it on first call.

    Returns:
        A PipelineCache instance with every computed statistical result.

    Raises:
        FileNotFoundError: If the dataset CSV is missing.
        ValueError: If any stage of the pipeline fails validation.
    """
    global _cache
    if _cache is not None:
        return _cache

    raw_df = load_dataset()
    validation_summary = validate_dataset(raw_df)

    phase_data = prepare_phase_data(raw_df)
    pca_result = run_pca(phase_data)
    t2_result = run_hotellings_t2(phase_data)
    t2_eval = evaluate_t2_monitoring(t2_result.phase2_results)
    manova_result = run_manova_analysis(raw_df)
    diagnostics_result = run_diagnostics(raw_df, phase_data, t2_result)

    health_df = assess_dataset_health(t2_result.phase2_results["Hotelling_T2"], t2_result.ucl)
    health_df.insert(0, "UDI", t2_result.phase2_results["UDI"].values)
    health_summary = summarize_health_statuses(health_df)

    _cache = PipelineCache(
        raw_df=raw_df,
        validation_summary=validation_summary,
        phase_data=phase_data,
        pca_result=pca_result,
        t2_result=t2_result,
        t2_eval=t2_eval,
        manova_result=manova_result,
        diagnostics_result=diagnostics_result,
        health_df=health_df,
        health_summary=health_summary,
    )
    return _cache


def reset_pipeline_cache() -> None:
    """Clear the cached pipeline, forcing a rebuild on next access."""
    global _cache
    _cache = None


def get_machine_row(cache: PipelineCache, udi: int) -> Optional[pd.Series]:
    """Retrieve the Phase-II row for a given UDI, or None if not found."""
    matches = cache.t2_result.phase2_results.loc[cache.t2_result.phase2_results["UDI"] == udi]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_machine_health(cache: PipelineCache, udi: int) -> Optional[Dict[str, Any]]:
    """
    Build the full machine-health record for a given Phase-II UDI,
    using only verified statistical outputs.

    Args:
        cache: The current PipelineCache.
        udi: The UDI to look up.

    Returns:
        A dictionary with T2, UCL, ratio, health status, priority,
        sensor values, standardized values, contributing sensors, and
        the maintenance recommendation. None if the UDI is not found
        in Phase II.
    """
    row = get_machine_row(cache, udi)
    if row is None:
        return None

    ucl = cache.t2_result.ucl
    t2_value = float(row["Hotelling_T2"])
    health_status = classify_health(t2_value, ucl)
    ratio = calculate_t2_ratio(t2_value, ucl)

    # phase2_results stores sensors in standardized form only (per
    # preprocessing.py); raw physical-unit values are not carried
    # through the existing pipeline output.
    standardized_values = {col: float(row[col]) for col in SENSOR_COLUMNS}
    sensor_values = standardized_values

    rec = generate_maintenance_recommendation(
        health_status, t2_value, ucl, sensor_values, standardized_values
    )

    return {
        "udi": int(udi),
        "t2_value": t2_value,
        "ucl": float(ucl),
        "t2_ratio": ratio,
        "health_status": health_status,
        "priority": rec["priority"],
        "sensor_values": sensor_values,
        "standardized_sensors": standardized_values,
        "contributing_sensors": rec["contributing_sensors"],
        "primary_recommendation": rec["primary_recommendation"],
        "additional_recommendations": rec["additional_recommendations"],
        "evidence": rec["evidence"],
        "limitations": rec["limitations"],
        "failure_probability": None,
    }


def build_pca_interpretation(cache: PipelineCache) -> str:
    """
    Build a cautious, non-causal PCA interpretation string based on
    the actual computed variance ratios.

    Args:
        cache: The current PipelineCache.

    Returns:
        A descriptive interpretation string.
    """
    pca_result = cache.pca_result
    n_components_90 = components_for_variance_threshold(pca_result, threshold=0.90)
    return (
        f"{n_components_90} principal component(s) explain at least 90% of the variance "
        f"in the Phase-I healthy baseline. PC1 explains "
        f"{pca_result.explained_variance_ratio[0] * 100:.2f}% of variance, reflecting a "
        "combination of sensor variation; this describes variance structure only and does "
        "not imply which variables cause abnormal states."
    )