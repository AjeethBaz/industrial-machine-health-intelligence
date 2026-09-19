"""
Deterministic maintenance recommendation engine built on top of the
existing Hotelling's T^2 monitoring and health-assessment pipeline.

ACADEMIC TRANSPARENCY NOTES:

1. This module separates three distinct concepts and never merges
   them into a single unsupported claim:
     A. Statistical abnormality  -> Hotelling's T^2 relative to UCL
     B. Sensor extremeness       -> standardized sensor z-scores
     C. Maintenance recommendation -> an evidence-based operational
        response derived deterministically from A and B.

2. This engine does NOT use the observed "Machine failure" label to
   generate recommendations for any individual observation. That
   label is an outcome variable only; using it here would leak the
   target into a supposedly independent decision-support layer.

3. This module makes no claims of confirmed failure, physical fault,
   or component damage. It only flags statistically unusual
   multivariate/sensor states relative to the healthy Phase-I
   baseline and suggests what an operator/maintenance team should
   investigate next.

4. No failure probability is calculated anywhere in this module.

5. This is a purely deterministic, rule-based engine. No LLM, no
   machine-learning model, and no Streamlit code are used here.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import prepare_phase_data
from src.hotellings_t2 import run_hotellings_t2
from src.health_assessment import (
    get_health_thresholds,
    classify_health,
    calculate_t2_ratio,
    assess_dataset_health,
)

SENSOR_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

Z_THRESHOLD: float = 2.0

PRIORITY_BY_STATUS: Dict[str, str] = {
    "NORMAL": "ROUTINE",
    "WATCH": "MONITOR",
    "ALERT": "INVESTIGATE",
    "CRITICAL": "HIGH",
}

STATUS_RECOMMENDATION: Dict[str, str] = {
    "NORMAL": "Continue routine monitoring. No immediate multivariate abnormality is indicated.",
    "WATCH": "Increase monitoring attention and review recent sensor trends.",
    "ALERT": "Investigate the abnormal multivariate state and review the contributing sensor conditions.",
    "CRITICAL": "Prioritize prompt maintenance investigation of the abnormal multivariate state.",
}

LIMITATIONS: List[str] = [
    "The recommendation is based on statistical sensor abnormality and does not confirm a physical fault.",
    "Extreme standardized sensor values indicate unusual observations relative to the healthy baseline; they do not establish causation.",
    "The AI4I 2020 dataset is synthetic and contains only the five selected sensor variables used here.",
    "Additional physical inspection and domain-specific measurements would be required to confirm a mechanical or thermal fault.",
]


def _validate_health_status(health_status: str) -> None:
    """Raise ValueError if health_status is not a recognized status."""
    if health_status not in PRIORITY_BY_STATUS:
        raise ValueError(
            f"Unrecognized health_status: {health_status}. "
            f"Expected one of {list(PRIORITY_BY_STATUS.keys())}."
        )


def _validate_sensor_dict(values: Dict[str, float], name: str) -> None:
    """Raise ValueError if a sensor dictionary is missing required sensors."""
    missing = [col for col in SENSOR_COLUMNS if col not in values]
    if missing:
        raise ValueError(f"{name} is missing required sensor(s): {missing}.")


def identify_contributing_sensors(
    standardized_values: Dict[str, float], threshold: float = Z_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Identify sensors whose absolute standardized value meets or
    exceeds the given threshold.

    These are described as contributing/extreme sensor signals
    relative to the healthy Phase-I baseline. This function does not
    rank sensors as causal factors.

    Args:
        standardized_values: Mapping of sensor name to standardized value.
        threshold: Absolute z-score threshold (default 2.0).

    Returns:
        List of dicts with keys "sensor", "z_score", "direction"
        ("HIGH" or "LOW"), for sensors meeting or exceeding the
        threshold. Empty list if none do.
    """
    contributing: List[Dict[str, Any]] = []
    for sensor in SENSOR_COLUMNS:
        z = standardized_values.get(sensor)
        if z is None or not np.isfinite(z):
            continue
        if abs(z) >= threshold:
            direction = "HIGH" if z >= threshold else "LOW"
            contributing.append(
                {"sensor": sensor, "z_score": float(z), "direction": direction}
            )
    return contributing


def _sensor_z(standardized_values: Dict[str, float], sensor: str) -> Optional[float]:
    """Return the standardized value for a sensor, or None if unavailable/non-finite."""
    z = standardized_values.get(sensor)
    if z is None or not np.isfinite(z):
        return None
    return float(z)


def _apply_sensor_rules(standardized_values: Dict[str, float]) -> List[str]:
    """
    Apply Rules A-D (tool wear, torque, rotational speed, temperature)
    to generate individual sensor-specific recommendations.

    Args:
        standardized_values: Mapping of sensor name to standardized value.

    Returns:
        List of recommendation strings triggered by individual sensors.
    """
    recommendations: List[str] = []

    tool_wear = _sensor_z(standardized_values, "Tool wear [min]")
    torque = _sensor_z(standardized_values, "Torque [Nm]")
    rot_speed = _sensor_z(standardized_values, "Rotational speed [rpm]")
    air_temp = _sensor_z(standardized_values, "Air temperature [K]")
    process_temp = _sensor_z(standardized_values, "Process temperature [K]")

    # Rule A - Tool wear
    if tool_wear is not None and tool_wear >= Z_THRESHOLD:
        recommendations.append(
            "Inspect tool condition and assess whether tool replacement/service is required."
        )

    # Rule B - Torque
    if torque is not None and torque >= Z_THRESHOLD:
        recommendations.append(
            "Inspect mechanical load/torque conditions and check for abnormal loading or resistance."
        )
    elif torque is not None and torque <= -Z_THRESHOLD:
        recommendations.append(
            "Investigate unusually low torque/loading behavior and verify operating conditions."
        )

    # Rule C - Rotational speed
    if rot_speed is not None and rot_speed >= Z_THRESHOLD:
        recommendations.append(
            "Inspect rotational-speed operating conditions and verify that the machine is operating within its intended range."
        )
    elif rot_speed is not None and rot_speed <= -Z_THRESHOLD:
        recommendations.append(
            "Investigate unusually low rotational-speed behavior and verify operating conditions."
        )

    # Rule D - Temperature (air and/or process)
    high_temp = (air_temp is not None and air_temp >= Z_THRESHOLD) or (
        process_temp is not None and process_temp >= Z_THRESHOLD
    )
    low_temp = (air_temp is not None and air_temp <= -Z_THRESHOLD) or (
        process_temp is not None and process_temp <= -Z_THRESHOLD
    )
    if high_temp:
        recommendations.append(
            "Inspect thermal/process-temperature conditions and verify cooling, process settings, and operating environment."
        )
    if low_temp:
        recommendations.append(
            "Investigate unusually low temperature conditions and verify process/environmental settings."
        )

    return recommendations


def _apply_combination_rules(standardized_values: Dict[str, float]) -> List[str]:
    """
    Apply combination rules across multiple sensors to surface
    multivariate operating patterns.

    Args:
        standardized_values: Mapping of sensor name to standardized value.

    Returns:
        List of recommendation strings triggered by sensor combinations.
    """
    recommendations: List[str] = []

    tool_wear = _sensor_z(standardized_values, "Tool wear [min]")
    torque = _sensor_z(standardized_values, "Torque [Nm]")
    rot_speed = _sensor_z(standardized_values, "Rotational speed [rpm]")
    air_temp = _sensor_z(standardized_values, "Air temperature [K]")
    process_temp = _sensor_z(standardized_values, "Process temperature [K]")

    torque_high = torque is not None and torque >= Z_THRESHOLD
    rot_speed_low = rot_speed is not None and rot_speed <= -Z_THRESHOLD
    tool_wear_high = tool_wear is not None and tool_wear >= Z_THRESHOLD
    temp_high = (air_temp is not None and air_temp >= Z_THRESHOLD) or (
        process_temp is not None and process_temp >= Z_THRESHOLD
    )

    # Combination 1: high torque + low rotational speed
    if torque_high and rot_speed_low:
        recommendations.append(
            "Investigate a high-load/low-speed operating pattern and inspect mechanical operating conditions."
        )

    # Combination 2: high tool wear + high torque
    if tool_wear_high and torque_high:
        recommendations.append(
            "Prioritize inspection of tool condition together with the associated mechanical load."
        )

    # Combination 3: high temperature (air and/or process) + high torque
    if temp_high and torque_high:
        recommendations.append(
            "Investigate the combined thermal and load-related operating state."
        )

    return recommendations


def _build_evidence(
    t2_value: float, ucl: float, contributing_sensors: List[Dict[str, Any]]
) -> List[str]:
    """
    Build a list of objective evidence statements from the T^2 result
    and contributing sensor signals.

    Args:
        t2_value: The observation's Hotelling's T^2 value.
        ucl: The established T^2 upper control limit.
        contributing_sensors: Output of identify_contributing_sensors().

    Returns:
        List of evidence strings.
    """
    evidence: List[str] = []

    if t2_value > ucl:
        evidence.append(f"T2 = {t2_value:.4f} exceeds the established UCL of {ucl:.4f}.")
    else:
        evidence.append(f"T2 = {t2_value:.4f} does not exceed the established UCL of {ucl:.4f}.")

    for sensor_info in contributing_sensors:
        sensor = sensor_info["sensor"]
        z = sensor_info["z_score"]
        direction_word = "exceeding" if sensor_info["direction"] == "HIGH" else "below"
        evidence.append(
            f"{sensor} standardized value = {z:.4f}, {direction_word} the ±2 SD diagnostic threshold."
        )

    return evidence


def _build_explanation(
    health_status: str, contributing_sensors: List[Dict[str, Any]]
) -> str:
    """
    Build a concise, dynamically generated explanation connecting
    T^2 status, sensor signals, and recommended investigation.

    Args:
        health_status: One of NORMAL, WATCH, ALERT, CRITICAL.
        contributing_sensors: Output of identify_contributing_sensors().

    Returns:
        A single explanation string.
    """
    status_clause = {
        "NORMAL": "The machine is in a NORMAL state because its multivariate T2 statistic is within the established monitoring region.",
        "WATCH": "The machine is in a WATCH state because its multivariate T2 statistic is approaching the established control limit.",
        "ALERT": "The machine is in an ALERT state because its multivariate T2 statistic exceeds the established control limit.",
        "CRITICAL": "The machine is in a CRITICAL state because its multivariate T2 statistic substantially exceeds the established control limit.",
    }[health_status]

    if contributing_sensors:
        sensor_names = [s["sensor"] for s in contributing_sensors]
        if len(sensor_names) == 1:
            sensor_clause = f"{sensor_names[0]} also shows an unusually {contributing_sensors[0]['direction'].lower()} standardized value."
        else:
            joined = ", ".join(sensor_names[:-1]) + f" and {sensor_names[-1]}"
            sensor_clause = f"{joined} also show unusual standardized values relative to the healthy baseline."
    else:
        sensor_clause = "No individual sensor exceeded the ±2 SD diagnostic threshold."

    action_clause = {
        "NORMAL": "Routine monitoring should continue.",
        "WATCH": "Maintenance attention should focus on reviewing recent sensor trends.",
        "ALERT": "Maintenance attention should therefore focus on investigating the abnormal multivariate state and reviewing the contributing sensor conditions.",
        "CRITICAL": "Maintenance attention should therefore prioritize prompt investigation of the abnormal multivariate state and contributing sensor conditions.",
    }[health_status]

    return f"{status_clause} {sensor_clause} {action_clause}"


def generate_maintenance_recommendation(
    health_status: str,
    t2_value: float,
    ucl: float,
    sensor_values: Dict[str, float],
    standardized_values: Dict[str, float],
) -> Dict[str, Any]:
    """
    Generate a deterministic, explainable maintenance recommendation
    for a single observation.

    This function does NOT use the "Machine failure" label. It relies
    only on the health status (derived from T^2 vs UCL) and the raw
    and standardized sensor values.

    Args:
        health_status: One of NORMAL, WATCH, ALERT, CRITICAL.
        t2_value: The observation's Hotelling's T^2 value.
        ucl: The established T^2 upper control limit.
        sensor_values: Mapping of sensor name to raw (original units) value.
        standardized_values: Mapping of sensor name to standardized value.

    Returns:
        Dictionary with health_status, priority, primary_recommendation,
        additional_recommendations, contributing_sensors, evidence,
        explanation, and limitations.

    Raises:
        ValueError: If health_status is unrecognized or sensor
            dictionaries are missing required sensors.
    """
    _validate_health_status(health_status)
    _validate_sensor_dict(sensor_values, "sensor_values")
    _validate_sensor_dict(standardized_values, "standardized_values")

    contributing_sensors = identify_contributing_sensors(standardized_values, Z_THRESHOLD)

    sensor_recommendations = _apply_sensor_rules(standardized_values)
    combination_recommendations = _apply_combination_rules(standardized_values)

    additional_recommendations = sensor_recommendations + combination_recommendations
    # De-duplicate while preserving order
    seen = set()
    deduped_additional = []
    for rec in additional_recommendations:
        if rec not in seen:
            deduped_additional.append(rec)
            seen.add(rec)

    priority = PRIORITY_BY_STATUS[health_status]
    primary_recommendation = STATUS_RECOMMENDATION[health_status]

    evidence = _build_evidence(t2_value, ucl, contributing_sensors)
    explanation = _build_explanation(health_status, contributing_sensors)

    return {
        "health_status": health_status,
        "priority": priority,
        "primary_recommendation": primary_recommendation,
        "additional_recommendations": deduped_additional,
        "contributing_sensors": contributing_sensors,
        "evidence": evidence,
        "explanation": explanation,
        "limitations": list(LIMITATIONS),
    }


def generate_dataset_recommendations(
    health_dataframe: pd.DataFrame,
    sensor_dataframe: pd.DataFrame,
    standardized_dataframe: pd.DataFrame,
    ucl: float,
) -> pd.DataFrame:
    """
    Generate maintenance recommendations for every row in a dataset.

    Args:
        health_dataframe: DataFrame containing at least "T2",
            "Health_Status", and optionally "UDI" (e.g. output of
            health_assessment.assess_dataset_health()).
        sensor_dataframe: DataFrame with raw sensor columns, aligned
            by position/index with health_dataframe.
        standardized_dataframe: DataFrame with standardized sensor
            columns, aligned by position/index with health_dataframe.
        ucl: The established T^2 upper control limit.

    Returns:
        DataFrame with columns: UDI (if available), T2, T2_UCL,
        T2_Ratio, Health_Status, Priority, Primary_Recommendation,
        Contributing_Sensors.

    Raises:
        ValueError: If required columns are missing or row counts
            across the three input DataFrames do not match.
    """
    if "T2" not in health_dataframe.columns or "Health_Status" not in health_dataframe.columns:
        raise ValueError("health_dataframe must contain 'T2' and 'Health_Status' columns.")

    n = len(health_dataframe)
    if len(sensor_dataframe) != n or len(standardized_dataframe) != n:
        raise ValueError(
            "health_dataframe, sensor_dataframe, and standardized_dataframe "
            "must have the same number of rows."
        )

    health_reset = health_dataframe.reset_index(drop=True)
    sensor_reset = sensor_dataframe.reset_index(drop=True)
    standardized_reset = standardized_dataframe.reset_index(drop=True)

    records: List[Dict[str, Any]] = []

    for i in range(n):
        t2_value = float(health_reset.loc[i, "T2"])
        health_status = str(health_reset.loc[i, "Health_Status"])
        sensor_values = {col: float(sensor_reset.loc[i, col]) for col in SENSOR_COLUMNS}
        standardized_values = {
            col: float(standardized_reset.loc[i, col]) for col in SENSOR_COLUMNS
        }

        result = generate_maintenance_recommendation(
            health_status, t2_value, ucl, sensor_values, standardized_values
        )

        record = {
            "T2": t2_value,
            "T2_UCL": ucl,
            "T2_Ratio": calculate_t2_ratio(t2_value, ucl),
            "Health_Status": health_status,
            "Priority": result["priority"],
            "Primary_Recommendation": result["primary_recommendation"],
            "Contributing_Sensors": result["contributing_sensors"],
        }
        if "UDI" in health_reset.columns:
            record = {"UDI": health_reset.loc[i, "UDI"], **record}

        records.append(record)

    return pd.DataFrame(records)


def _print_example(
    label: str,
    row: pd.Series,
    sensor_values: Dict[str, float],
    standardized_values: Dict[str, float],
    ucl: float,
) -> None:
    """Print a full recommendation example for one Phase-II observation."""
    t2_value = float(row["Hotelling_T2"])
    health_status = classify_health(t2_value, ucl)
    result = generate_maintenance_recommendation(
        health_status, t2_value, ucl, sensor_values, standardized_values
    )

    print(f"\n--- {label} example (UDI={int(row['UDI'])}) ---")
    print(f"UDI: {int(row['UDI'])}")
    print(f"T2: {t2_value:.4f}")
    print(f"T2/UCL: {calculate_t2_ratio(t2_value, ucl):.4f}")
    print(f"Health status: {result['health_status']}")
    print(f"Priority: {result['priority']}")
    print(f"Contributing sensors: {result['contributing_sensors']}")
    print(f"Primary recommendation: {result['primary_recommendation']}")
    print(f"Additional recommendations: {result['additional_recommendations']}")
    print(f"Evidence: {result['evidence']}")
    print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    t2_result = run_hotellings_t2(prepared)

    ucl = t2_result.ucl
    phase2_results = t2_result.phase2_results  # UDI, sensors (standardized), Machine failure, Hotelling_T2, T2_Alert

    health_df = assess_dataset_health(phase2_results["Hotelling_T2"], ucl)
    health_df.insert(0, "UDI", phase2_results["UDI"].values)

    print("=" * 60)
    print("MAINTENANCE RECOMMENDATION ENGINE")
    print("=" * 60)

    print(f"\nT2 UCL: {ucl:.4f}")
    print(f"\nTotal Phase-II observations: {len(phase2_results)}")

    priority_counts = health_df["Health_Status"].map(PRIORITY_BY_STATUS).value_counts()
    print("\nPriority summary:\n")
    for status, priority in PRIORITY_BY_STATUS.items():
        count = int(priority_counts.get(priority, 0))
        print(f"{priority}: {count}")

    for status in ["NORMAL", "WATCH", "ALERT", "CRITICAL"]:
        matches = health_df[health_df["Health_Status"] == status]
        if matches.empty:
            print(f"\nNo {status} observations found in Phase-II results.")
            continue

        sample_udi = matches.iloc[0]["UDI"]
        sample_row = phase2_results.loc[phase2_results["UDI"] == sample_udi].iloc[0]

        standardized_values = {col: float(sample_row[col]) for col in SENSOR_COLUMNS}
        sensor_values = standardized_values  # phase2_results stores standardized sensors

        _print_example(status, sample_row, sensor_values, standardized_values, ucl)

    print("\n" + "=" * 60)