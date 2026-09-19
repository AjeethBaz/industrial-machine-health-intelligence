"""
Transparent machine-health assessment layer built on top of the
existing Hotelling's T^2 monitoring pipeline.

ACADEMIC TRANSPARENCY NOTES (read before using this module):

1. Hotelling's T^2 is a multivariate abnormality statistic. It
   measures how far a standardized observation is from the Phase-I
   healthy baseline mean, accounting for covariance among the five
   sensor variables. It is NOT a probability of failure.

2. The UCL (upper control limit) used here is the theoretically
   derived Phase-II control limit already established in
   hotellings_t2.py (F-distribution based, alpha = 0.05). This
   module does not recompute, refit, or alter that UCL.

3. The NORMAL / WATCH / ALERT / CRITICAL bands below are operational
   severity bands derived from simple multiples of the UCL. They are
   NOT empirically calibrated failure probabilities and must not be
   presented as such.

4. ALERT or CRITICAL status means the observation's multivariate
   sensor pattern is statistically unusual relative to the healthy
   baseline. It does NOT mean machine failure is confirmed. Likewise,
   NORMAL does NOT guarantee the machine is healthy — the existing
   Phase-II evaluation shows T^2 captures about 70% of observed
   failures while also generating false alerts.

5. "Extreme sensor" flags (Part 7) describe association with the
   unusual multivariate state only. They are not a causal attribution
   of failure to any single sensor.

This module reuses the existing Hotelling's T^2 implementation from
hotellings_t2.py and does not duplicate its mathematics.
"""

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import prepare_phase_data
from src.hotellings_t2 import run_hotellings_t2

NumericArray = Union[Sequence[float], np.ndarray, pd.Series]

HEALTH_STATUSES: List[str] = ["NORMAL", "WATCH", "ALERT", "CRITICAL"]

INTERPRETATION_TEXT: Dict[str, str] = {
    "NORMAL": (
        "Multivariate sensor behavior is within the normal monitoring "
        "region relative to the established T2 control limit."
    ),
    "WATCH": (
        "Multivariate sensor behavior is approaching the established T2 "
        "control limit. Continued monitoring is recommended."
    ),
    "ALERT": (
        "Multivariate sensor behavior exceeds the established T2 control "
        "limit and indicates an abnormal multivariate state requiring "
        "investigation."
    ),
    "CRITICAL": (
        "Multivariate sensor behavior is substantially above the "
        "established T2 control limit and warrants prompt investigation."
    ),
}


def _validate_ucl(ucl: float) -> None:
    """
    Validate that a UCL value is a finite, positive number.

    Args:
        ucl: The upper control limit to validate.

    Raises:
        ValueError: If ucl is not finite or not positive.
    """
    if ucl is None or not np.isfinite(ucl) or ucl <= 0:
        raise ValueError(f"UCL must be a finite, positive number. Got: {ucl}.")


def _validate_t2_value(t2_value: float) -> None:
    """
    Validate that a single T^2 value is a finite, non-negative number.

    Args:
        t2_value: The T^2 value to validate.

    Raises:
        ValueError: If t2_value is not finite or negative.
    """
    if t2_value is None or not np.isfinite(t2_value) or t2_value < 0:
        raise ValueError(f"T2 value must be a finite, non-negative number. Got: {t2_value}.")


def get_health_thresholds(ucl: float) -> Dict[str, float]:
    """
    Compute the operational severity band thresholds from the
    established T^2 UCL.

    These bands are simple multiples of the UCL (0.75x and 1.50x) and
    are NOT empirically calibrated probability thresholds.

    Args:
        ucl: The established Hotelling's T^2 upper control limit.

    Returns:
        Dictionary with keys: "ucl", "watch_threshold", "alert_threshold".

    Raises:
        ValueError: If ucl is invalid.
    """
    _validate_ucl(ucl)
    return {
        "ucl": float(ucl),
        "watch_threshold": float(0.75 * ucl),
        "alert_threshold": float(1.50 * ucl),
    }


def classify_health(t2_value: float, ucl: float) -> str:
    """
    Classify a single T^2 value into an operational health status
    band relative to the UCL.

    Bands:
        NORMAL:   T2 <= 0.75 * UCL
        WATCH:    0.75 * UCL < T2 <= 1.00 * UCL
        ALERT:    1.00 * UCL < T2 <= 1.50 * UCL
        CRITICAL: T2 > 1.50 * UCL

    Args:
        t2_value: The observation's Hotelling's T^2 value.
        ucl: The established T^2 upper control limit.

    Returns:
        One of "NORMAL", "WATCH", "ALERT", "CRITICAL".

    Raises:
        ValueError: If t2_value or ucl is invalid.
    """
    _validate_ucl(ucl)
    _validate_t2_value(t2_value)

    thresholds = get_health_thresholds(ucl)
    watch_threshold = thresholds["watch_threshold"]
    alert_threshold = thresholds["alert_threshold"]

    if t2_value <= watch_threshold:
        return "NORMAL"
    if t2_value <= ucl:
        return "WATCH"
    if t2_value <= alert_threshold:
        return "ALERT"
    return "CRITICAL"


def calculate_t2_ratio(t2_value: float, ucl: float) -> float:
    """
    Calculate the T^2-to-UCL abnormality ratio.

    This ratio expresses how far an observation's T^2 is from the
    control limit. It is an abnormality ratio, NOT a probability of
    failure.

    Args:
        t2_value: The observation's Hotelling's T^2 value.
        ucl: The established T^2 upper control limit.

    Returns:
        T2 / UCL as a float.

    Raises:
        ValueError: If t2_value or ucl is invalid.
    """
    _validate_ucl(ucl)
    _validate_t2_value(t2_value)
    return float(t2_value / ucl)


def identify_extreme_sensors(
    standardized_values: Dict[str, float], sensor_names: List[str], threshold: float = 2.0
) -> Dict[str, float]:
    """
    Identify sensors whose absolute standardized value meets or
    exceeds the given threshold.

    These sensors are associated with the unusual multivariate state
    for this observation. This is NOT a causal attribution of failure
    to any single sensor.

    Args:
        standardized_values: Mapping of sensor name to standardized value.
        sensor_names: The list of sensor names to check.
        threshold: Absolute standardized-value threshold (default 2.0).

    Returns:
        Dictionary mapping sensor name to standardized value, for
        sensors meeting or exceeding the threshold. Empty if none do.
    """
    extreme_sensors: Dict[str, float] = {}
    for name in sensor_names:
        value = standardized_values.get(name)
        if value is not None and np.isfinite(value) and abs(value) >= threshold:
            extreme_sensors[name] = float(value)
    return extreme_sensors


def assess_machine_health(
    sensor_values: Dict[str, float],
    standardized_values: Dict[str, float],
    t2_value: float,
    ucl: float,
) -> Dict[str, Any]:
    """
    Produce a structured, transparent health assessment for a single
    observation.

    Args:
        sensor_values: Mapping of sensor name to raw (original units) value.
        standardized_values: Mapping of sensor name to standardized value.
        t2_value: The observation's Hotelling's T^2 value.
        ucl: The established T^2 upper control limit.

    Returns:
        Dictionary with health_status, t2_value, ucl, t2_ratio,
        distance_from_ucl, sensor_values, standardized_values, and
        interpretation text.

    Raises:
        ValueError: If t2_value or ucl is invalid.
    """
    _validate_ucl(ucl)
    _validate_t2_value(t2_value)

    health_status = classify_health(t2_value, ucl)
    t2_ratio = calculate_t2_ratio(t2_value, ucl)
    distance_from_ucl = float(t2_value - ucl)

    return {
        "health_status": health_status,
        "t2_value": float(t2_value),
        "ucl": float(ucl),
        "t2_ratio": t2_ratio,
        "distance_from_ucl": distance_from_ucl,
        "sensor_values": dict(sensor_values),
        "standardized_values": dict(standardized_values),
        "interpretation": INTERPRETATION_TEXT[health_status],
    }


def assess_dataset_health(t2_values: NumericArray, ucl: float) -> pd.DataFrame:
    """
    Assess health status for an array/Series of T^2 values against
    the established UCL.

    Args:
        t2_values: Array or Series of Hotelling's T^2 values.
        ucl: The established T^2 upper control limit.

    Returns:
        DataFrame with columns: T2, T2_UCL, T2_Ratio, Health_Status.
        The original index is preserved if t2_values is a pandas Series.

    Raises:
        ValueError: If ucl is invalid or t2_values is empty.
    """
    _validate_ucl(ucl)

    if isinstance(t2_values, pd.Series):
        index = t2_values.index
        values = t2_values.values
    else:
        values = np.asarray(t2_values, dtype=float)
        index = pd.RangeIndex(start=0, stop=len(values))

    if len(values) == 0:
        raise ValueError("t2_values must contain at least one observation.")

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("t2_values must contain only finite, non-negative numbers.")

    ratios = values / ucl
    statuses = [classify_health(float(v), ucl) for v in values]

    result = pd.DataFrame(
        {
            "T2": values,
            "T2_UCL": ucl,
            "T2_Ratio": ratios,
            "Health_Status": statuses,
        },
        index=index,
    )
    return result


def summarize_health_statuses(health_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize counts and percentages of each health status in a
    health-assessment DataFrame.

    Args:
        health_dataframe: A DataFrame containing a "Health_Status" column,
            such as the output of assess_dataset_health().

    Returns:
        DataFrame indexed by health status (NORMAL, WATCH, ALERT,
        CRITICAL) with "Count" and "Percentage" columns. Percentages
        sum to approximately 100%.

    Raises:
        ValueError: If "Health_Status" column is missing.
    """
    if "Health_Status" not in health_dataframe.columns:
        raise ValueError("health_dataframe must contain a 'Health_Status' column.")

    total = len(health_dataframe)
    counts = health_dataframe["Health_Status"].value_counts()

    summary = pd.DataFrame(
        {
            "Count": [int(counts.get(status, 0)) for status in HEALTH_STATUSES],
            "Percentage": [
                float(counts.get(status, 0) / total * 100) if total > 0 else 0.0
                for status in HEALTH_STATUSES
            ],
        },
        index=HEALTH_STATUSES,
    )
    return summary


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    t2_result = run_hotellings_t2(prepared)

    ucl = t2_result.ucl
    phase2_results = t2_result.phase2_results

    thresholds = get_health_thresholds(ucl)
    health_df = assess_dataset_health(phase2_results["Hotelling_T2"], ucl)
    health_df.insert(0, "UDI", phase2_results["UDI"].values)

    summary = summarize_health_statuses(health_df)

    print("=" * 60)
    print("MACHINE HEALTH ASSESSMENT")
    print("=" * 60)

    print(f"\nT2 UCL: {thresholds['ucl']:.4f}")
    print(f"\nNormal threshold: {thresholds['watch_threshold']:.4f}")
    print(f"Watch threshold: {thresholds['ucl']:.4f}")
    print(f"Alert threshold: {thresholds['alert_threshold']:.4f}")

    print("\nHEALTH STATUS SUMMARY\n")
    for status in HEALTH_STATUSES:
        count = int(summary.loc[status, "Count"])
        pct = float(summary.loc[status, "Percentage"])
        print(f"{status}: {count} ({pct:.2f}%)")

    print("\nSample assessments (first 10 Phase-II observations):\n")
    print(
        health_df[["UDI", "T2", "T2_Ratio", "Health_Status"]]
        .head(10)
        .to_string(index=False)
    )

    sensor_columns = prepared.sensor_columns

    normal_rows = health_df[health_df["Health_Status"] == "NORMAL"]
    critical_rows = health_df[health_df["Health_Status"] == "CRITICAL"]

    print("\nExample interpretation:\n")

    if not normal_rows.empty:
        sample_udi = normal_rows.iloc[0]["UDI"]
        sample_row = phase2_results.loc[phase2_results["UDI"] == sample_udi].iloc[0]
        sensor_values = {col: float(sample_row[col]) for col in sensor_columns}
        standardized_values = sensor_values  # phase2_results sensors are already standardized
        t2_value = float(sample_row["Hotelling_T2"])

        assessment = assess_machine_health(sensor_values, standardized_values, t2_value, ucl)
        print(f"NORMAL example (UDI={int(sample_udi)}):")
        print(f"  T2 = {assessment['t2_value']:.4f}, Ratio = {assessment['t2_ratio']:.4f}")
        print(f"  Interpretation: {assessment['interpretation']}")
    else:
        print("No NORMAL observations found in Phase-II results.")

    if not critical_rows.empty:
        sample_udi = critical_rows.iloc[0]["UDI"]
        sample_row = phase2_results.loc[phase2_results["UDI"] == sample_udi].iloc[0]
        sensor_values = {col: float(sample_row[col]) for col in sensor_columns}
        standardized_values = sensor_values
        t2_value = float(sample_row["Hotelling_T2"])

        assessment = assess_machine_health(sensor_values, standardized_values, t2_value, ucl)
        extreme_sensors = identify_extreme_sensors(standardized_values, sensor_columns, threshold=2.0)

        print(f"\nCRITICAL example (UDI={int(sample_udi)}):")
        print(f"  T2 = {assessment['t2_value']:.4f}, Ratio = {assessment['t2_ratio']:.4f}")
        print(f"  Interpretation: {assessment['interpretation']}")
        if extreme_sensors:
            print(f"  Extreme sensors (associated with this unusual state): {extreme_sensors}")
        else:
            print("  No individual sensor exceeded the |z| >= 2.0 threshold.")
    else:
        print("\nNo CRITICAL observations found in Phase-II results.")

    print("\n" + "=" * 60)