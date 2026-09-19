"""
Evaluation of Phase-II Hotelling's T-squared alerts against the
observed Machine failure outcome.

This module does NOT alter the Hotelling's T^2 methodology, UCL, or
alert definitions produced by hotellings_t2.py. It only measures how
well the existing T2_Alert flag aligns with actual Machine failure
labels, using standard classification-style metrics and descriptive
group comparisons. No causal claims are made or implied; results
only support statements about association (e.g., failure
observations having higher/lower T^2 on average).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data_loader import load_dataset
from src.preprocessing import prepare_phase_data
from src.hotellings_t2 import run_hotellings_t2

REQUIRED_COLUMNS: List[str] = ["Hotelling_T2", "T2_Alert", "Machine failure"]


@dataclass
class T2EvaluationResult:
    """Container for Hotelling's T^2 evaluation results."""

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
    group_comparison: pd.DataFrame
    failure_t2_values: np.ndarray
    non_failure_t2_values: np.ndarray


def _validate_phase2_results(phase2_results: pd.DataFrame) -> None:
    """
    Validate that the Phase-II results DataFrame has the columns
    required for evaluation and that both failure classes are present.

    Args:
        phase2_results: Phase-II results from run_hotellings_t2().

    Raises:
        ValueError: If required columns are missing or only one
            Machine failure class is present.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in phase2_results.columns]
    if missing:
        raise ValueError(
            f"Phase-II results are missing required column(s): {missing}."
        )

    unique_classes = phase2_results["Machine failure"].unique()
    if len(unique_classes) < 2:
        raise ValueError(
            "Phase-II results must contain both Machine failure classes "
            f"(0 and 1) for evaluation. Found only: {unique_classes}."
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def calculate_confusion_matrix(phase2_results: pd.DataFrame):
    """
    Build the confusion matrix comparing T2_Alert (predicted) against
    Machine failure (actual).

    Args:
        phase2_results: Phase-II results containing T2_Alert and
            Machine failure columns.

    Returns:
        Tuple of (tp, fp, tn, fn) as integers.
    """
    y_true = phase2_results["Machine failure"].astype(int)
    y_pred = phase2_results["T2_Alert"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(tn), int(fn)


def calculate_metrics(tp: int, fp: int, tn: int, fn: int, total: int) -> dict:
    """
    Calculate evaluation metrics from confusion matrix counts, with
    safe handling of division-by-zero cases.

    Args:
        tp: True positives.
        fp: False positives.
        tn: True negatives.
        fn: False negatives.
        total: Total number of Phase-II observations.

    Returns:
        Dictionary of metric name to value.
    """
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = _safe_divide(fp, fp + tn)
    alert_rate = _safe_divide(tp + fp, total)
    actual_failure_rate = _safe_divide(tp + fn, total)
    failure_capture_rate = recall  # equivalent definition: TP / (TP + FN)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "alert_rate": alert_rate,
        "actual_failure_rate": actual_failure_rate,
        "failure_capture_rate": failure_capture_rate,
    }


def _describe_group(t2_values: pd.Series, alerts: pd.Series) -> dict:
    """Compute descriptive statistics for a group of T2 values and alerts."""
    count = int(len(t2_values))
    n_alerts = int(alerts.sum())
    alert_pct = _safe_divide(n_alerts, count) * 100

    return {
        "count": count,
        "mean": float(t2_values.mean()) if count > 0 else np.nan,
        "median": float(t2_values.median()) if count > 0 else np.nan,
        "std": float(t2_values.std()) if count > 0 else np.nan,
        "min": float(t2_values.min()) if count > 0 else np.nan,
        "max": float(t2_values.max()) if count > 0 else np.nan,
        "q25": float(t2_values.quantile(0.25)) if count > 0 else np.nan,
        "q75": float(t2_values.quantile(0.75)) if count > 0 else np.nan,
        "alerts": n_alerts,
        "alert_percentage": alert_pct,
    }


def compare_t2_by_failure_status(phase2_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Hotelling's T^2 distributions between Phase-II failure
    and non-failure observations.

    Args:
        phase2_results: Phase-II results containing Hotelling_T2,
            T2_Alert, and Machine failure columns.

    Returns:
        A DataFrame with one row per group ("Failure", "Non-failure")
        and descriptive statistics as columns.
    """
    failure_mask = phase2_results["Machine failure"] == 1
    non_failure_mask = phase2_results["Machine failure"] == 0

    failure_stats = _describe_group(
        phase2_results.loc[failure_mask, "Hotelling_T2"],
        phase2_results.loc[failure_mask, "T2_Alert"],
    )
    non_failure_stats = _describe_group(
        phase2_results.loc[non_failure_mask, "Hotelling_T2"],
        phase2_results.loc[non_failure_mask, "T2_Alert"],
    )

    comparison = pd.DataFrame(
        [failure_stats, non_failure_stats],
        index=["Failure", "Non-failure"],
    )
    return comparison


def evaluate_t2_monitoring(phase2_results: pd.DataFrame) -> T2EvaluationResult:
    """
    Run the full evaluation of Phase-II T^2 alerts against Machine
    failure outcomes.

    Args:
        phase2_results: Phase-II results from run_hotellings_t2().

    Returns:
        A ``T2EvaluationResult`` containing confusion matrix values,
        metrics, and group comparison statistics.
    """
    _validate_phase2_results(phase2_results)

    tp, fp, tn, fn = calculate_confusion_matrix(phase2_results)
    total = len(phase2_results)
    metrics = calculate_metrics(tp, fp, tn, fn, total)
    group_comparison = compare_t2_by_failure_status(phase2_results)

    failure_t2_values = phase2_results.loc[
        phase2_results["Machine failure"] == 1, "Hotelling_T2"
    ].values
    non_failure_t2_values = phase2_results.loc[
        phase2_results["Machine failure"] == 0, "Hotelling_T2"
    ].values

    return T2EvaluationResult(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=metrics["precision"],
        recall=metrics["recall"],
        specificity=metrics["specificity"],
        f1_score=metrics["f1_score"],
        false_positive_rate=metrics["false_positive_rate"],
        alert_rate=metrics["alert_rate"],
        actual_failure_rate=metrics["actual_failure_rate"],
        failure_capture_rate=metrics["failure_capture_rate"],
        group_comparison=group_comparison,
        failure_t2_values=failure_t2_values,
        non_failure_t2_values=non_failure_t2_values,
    )


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    t2_result = run_hotellings_t2(prepared)
    evaluation = evaluate_t2_monitoring(t2_result.phase2_results)

    print("=" * 60)
    print("HOTELLING'S T-SQUARED EVALUATION")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(f"TP: {evaluation.true_positives}")
    print(f"FP: {evaluation.false_positives}")
    print(f"TN: {evaluation.true_negatives}")
    print(f"FN: {evaluation.false_negatives}")

    print("\nMetrics:")
    print(f"Precision: {evaluation.precision:.4f}")
    print(f"Recall: {evaluation.recall:.4f}")
    print(f"Specificity: {evaluation.specificity:.4f}")
    print(f"F1 Score: {evaluation.f1_score:.4f}")
    print(f"False Positive Rate: {evaluation.false_positive_rate:.4f}")
    print(f"Alert Rate: {evaluation.alert_rate:.4f}")
    print(f"Actual Failure Rate: {evaluation.actual_failure_rate:.4f}")
    print(f"Failure Capture Rate: {evaluation.failure_capture_rate:.4f}")

    gc = evaluation.group_comparison

    print("\nT2 by Failure Status:")
    print("Failure observations:")
    print(f"  count: {gc.loc['Failure', 'count']}")
    print(f"  mean: {gc.loc['Failure', 'mean']:.4f}")
    print(f"  median: {gc.loc['Failure', 'median']:.4f}")
    print(f"  std: {gc.loc['Failure', 'std']:.4f}")
    print(f"  min: {gc.loc['Failure', 'min']:.4f}")
    print(f"  max: {gc.loc['Failure', 'max']:.4f}")
    print(f"  Q25: {gc.loc['Failure', 'q25']:.4f}")
    print(f"  Q75: {gc.loc['Failure', 'q75']:.4f}")
    print(f"  alerts: {gc.loc['Failure', 'alerts']}")
    print(f"  alert percentage: {gc.loc['Failure', 'alert_percentage']:.2f}%")

    print("\nNon-failure observations:")
    print(f"  count: {gc.loc['Non-failure', 'count']}")
    print(f"  mean: {gc.loc['Non-failure', 'mean']:.4f}")
    print(f"  median: {gc.loc['Non-failure', 'median']:.4f}")
    print(f"  std: {gc.loc['Non-failure', 'std']:.4f}")
    print(f"  min: {gc.loc['Non-failure', 'min']:.4f}")
    print(f"  max: {gc.loc['Non-failure', 'max']:.4f}")
    print(f"  Q25: {gc.loc['Non-failure', 'q25']:.4f}")
    print(f"  Q75: {gc.loc['Non-failure', 'q75']:.4f}")
    print(f"  alerts: {gc.loc['Non-failure', 'alerts']}")
    print(f"  alert percentage: {gc.loc['Non-failure', 'alert_percentage']:.2f}%")

    mean_diff = gc.loc["Failure", "mean"] - gc.loc["Non-failure", "mean"]
    print(f"\nMean T2 difference (failure - non-failure): {mean_diff:.4f}")