"""
Phase I / Phase II data preparation for multivariate SPC.

This module builds the healthy Phase-I baseline and the Phase-II
monitoring dataset from the AI4I 2020 dataset, and fits a
StandardScaler on the Phase-I healthy baseline ONLY (to avoid data
leakage). No statistical monitoring (PCA, Hotelling's T^2, control
limits, MANOVA) or modeling is performed here.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset

SENSOR_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

REQUIRED_COLUMNS: List[str] = SENSOR_COLUMNS + ["UDI", "Machine failure"]

PHASE1_FRACTION: float = 0.70


@dataclass
class PhaseData:
    """Container for prepared Phase I / Phase II data and the fitted scaler."""

    phase1_healthy: pd.DataFrame
    phase2: pd.DataFrame
    scaler: StandardScaler
    sensor_columns: List[str]


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Ensure all required columns exist in the DataFrame.

    Args:
        df: Raw dataset loaded via ``load_dataset()``.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {missing}. "
            f"Required columns are: {REQUIRED_COLUMNS}."
        )


def _split_phases(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by UDI and split into Phase I (first 70%) and Phase II (remaining 30%).

    Args:
        df: Validated dataset.

    Returns:
        Tuple of (phase1_raw, phase2_raw) DataFrames, each a copy.
    """
    df_sorted = df.sort_values("UDI", ascending=True).reset_index(drop=True)
    split_index = int(len(df_sorted) * PHASE1_FRACTION)

    phase1_raw = df_sorted.iloc[:split_index].copy()
    phase2_raw = df_sorted.iloc[split_index:].copy()
    return phase1_raw, phase2_raw


def prepare_phase_data(df: pd.DataFrame) -> PhaseData:
    """
    Build the Phase-I healthy baseline and Phase-II monitoring dataset,
    then standardize sensor variables using a scaler fit only on the
    Phase-I healthy baseline.

    Args:
        df: Raw dataset as returned by ``load_dataset()``.

    Returns:
        A ``PhaseData`` object containing:
            - phase1_healthy: standardized healthy baseline (sensor columns only)
            - phase2: standardized sensor columns plus UDI and Machine failure
            - scaler: the StandardScaler fitted on Phase-I healthy data
            - sensor_columns: the list of five sensor column names

    Raises:
        ValueError: If required columns are missing, or if Phase I,
            Phase II, or the Phase-I healthy baseline end up empty.
    """
    _validate_columns(df)

    phase1_raw, phase2_raw = _split_phases(df)

    if phase1_raw.empty:
        raise ValueError("Phase I dataset is empty after splitting by UDI.")
    if phase2_raw.empty:
        raise ValueError("Phase II dataset is empty after splitting by UDI.")

    phase1_healthy_raw = phase1_raw[phase1_raw["Machine failure"] == 0].copy()
    if phase1_healthy_raw.empty:
        raise ValueError(
            "Phase-I healthy baseline is empty after filtering Machine failure == 0."
        )

    scaler = StandardScaler()
    scaler.fit(phase1_healthy_raw[SENSOR_COLUMNS])

    phase1_healthy = phase1_healthy_raw.copy()
    phase1_healthy[SENSOR_COLUMNS] = scaler.transform(phase1_healthy_raw[SENSOR_COLUMNS])
    phase1_healthy = phase1_healthy[["UDI"] + SENSOR_COLUMNS + ["Machine failure"]]

    phase2 = phase2_raw.copy()
    phase2[SENSOR_COLUMNS] = scaler.transform(phase2_raw[SENSOR_COLUMNS])
    phase2 = phase2[["UDI"] + SENSOR_COLUMNS + ["Machine failure"]]

    return PhaseData(
        phase1_healthy=phase1_healthy,
        phase2=phase2,
        scaler=scaler,
        sensor_columns=SENSOR_COLUMNS,
    )


def summarize_phase_data(df: pd.DataFrame, phase_data: PhaseData) -> Dict[str, Any]:
    """
    Build a summary report of the Phase I / Phase II preparation.

    Args:
        df: The original raw dataset used to compute Phase I size
            before healthy filtering.
        phase_data: The ``PhaseData`` object returned by
            ``prepare_phase_data()``.

    Returns:
        A dictionary summarizing observation counts across phases.
    """
    phase1_raw, phase2_raw = _split_phases(df)

    phase2_failures = int(phase_data.phase2["Machine failure"].sum())
    phase2_non_failures = int(len(phase_data.phase2) - phase2_failures)

    summary: Dict[str, Any] = {
        "total_observations": len(df),
        "phase1_before_healthy_filter": len(phase1_raw),
        "phase1_healthy_observations": len(phase_data.phase1_healthy),
        "phase2_observations": len(phase_data.phase2),
        "phase2_failures": phase2_failures,
        "phase2_non_failures": phase2_non_failures,
        "num_sensor_variables": len(phase_data.sensor_columns),
    }
    return summary


def print_preparation_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable Phase I / Phase II preparation summary."""
    print("=" * 60)
    print("PHASE I / PHASE II PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total observations: {summary['total_observations']}")
    print(f"Phase I observations (before healthy filter): {summary['phase1_before_healthy_filter']}")
    print(f"Phase I healthy observations: {summary['phase1_healthy_observations']}")
    print(f"Phase II observations: {summary['phase2_observations']}")
    print(f"Phase II failures: {summary['phase2_failures']}")
    print(f"Phase II non-failures: {summary['phase2_non_failures']}")
    print(f"Number of sensor variables: {summary['num_sensor_variables']}")
    print("=" * 60)


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    prep_summary = summarize_phase_data(raw_df, prepared)

    print_preparation_summary(prep_summary)

    print(f"\nPhase I healthy shape: {prepared.phase1_healthy.shape}")
    print(f"Phase II shape: {prepared.phase2.shape}")
    print(f"Sensor columns: {prepared.sensor_columns}")
