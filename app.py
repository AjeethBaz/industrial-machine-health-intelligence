"""
Industrial Machine Health Intelligence Platform
Multivariate Statistical Process Control for Predictive Maintenance

Integrated Streamlit application. This file orchestrates the existing,
already-tested statistical modules (src/*) and the Gemini AI assistant
(agent/gemini_agent.py). It does NOT reimplement or alter any
statistical methodology, formula, or calculation. All numeric results
shown here are computed by the underlying modules.

IMPORTANT STATISTICAL DISCLAIMERS (enforced throughout the UI):
- Hotelling's T^2 is a multivariate abnormality score, not a failure
  probability.
- T^2 / UCL is an abnormality ratio; it is never converted into a
  probability via sigmoid, percentage scaling, or any other ad hoc
  transformation.
- No calibrated failure-probability model exists in this project.
  Wherever a probability might be expected, the app explicitly states
  it is not available.
- MANOVA and PCA results are descriptive/inferential about group mean
  differences and variance structure; they are never presented as
  causal evidence of failure.
"""

import os
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import load_dataset, validate_dataset
from src.preprocessing import prepare_phase_data, summarize_phase_data
from src.pca_analysis import run_pca, components_for_variance_threshold
from src.hotellings_t2 import run_hotellings_t2
from src.t2_evaluation import evaluate_t2_monitoring
from src.manova_analysis import run_manova_analysis
from src.diagnostics import run_diagnostics
from src.health_assessment import (
    get_health_thresholds,
    classify_health,
    calculate_t2_ratio,
    assess_dataset_health,
    summarize_health_statuses,
)
from src.maintenance_recommendation import (
    generate_maintenance_recommendation,
    identify_contributing_sensors,
)

try:
    from agent.gemini_agent import analyze_machine_health, answer_followup

    AGENT_AVAILABLE = True
except Exception:  # noqa: BLE001
    AGENT_AVAILABLE = False

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

HEALTH_COLORS = {
    "NORMAL": "#2ecc71",
    "WATCH": "#f1c40f",
    "ALERT": "#e67e22",
    "CRITICAL": "#e74c3c",
}


# ======================================================================
# CACHED PIPELINE LOADERS
# ======================================================================


@st.cache_data(show_spinner=False)
def load_and_validate_data():
    """Load the raw dataset and run validation. Cached for performance."""
    raw_df = load_dataset()
    validation_summary = validate_dataset(raw_df)
    return raw_df, validation_summary


@st.cache_resource(show_spinner=False)
def build_phase_data(raw_df: pd.DataFrame):
    """Run Phase I / Phase II preprocessing. Cached as a resource (holds a fitted scaler)."""
    return prepare_phase_data(raw_df)


@st.cache_resource(show_spinner=False)
def build_pca_result(_phase_data):
    """Run PCA on the Phase-I healthy baseline. Cached as a resource (holds a fitted PCA object)."""
    return run_pca(_phase_data)


@st.cache_resource(show_spinner=False)
def build_t2_result(_phase_data):
    """Run Hotelling's T^2 monitoring. Cached as a resource (holds numpy arrays/matrices)."""
    return run_hotellings_t2(_phase_data)


@st.cache_data(show_spinner=False)
def build_t2_evaluation(_phase2_results_hash, phase2_results: pd.DataFrame):
    """Evaluate T^2 alerts against Machine failure. Cached on the phase2 results content."""
    return evaluate_t2_monitoring(phase2_results)


@st.cache_resource(show_spinner=False)
def build_manova_result(_raw_df: pd.DataFrame):
    """Run MANOVA on raw sensor values. Cached as a resource."""
    return run_manova_analysis(_raw_df)


@st.cache_resource(show_spinner=False)
def build_diagnostics_result(_raw_df: pd.DataFrame, _phase_data, _t2_result):
    """Run statistical diagnostics. Cached as a resource."""
    return run_diagnostics(_raw_df, _phase_data, _t2_result)


@st.cache_data(show_spinner=False)
def build_health_dataframe(_phase2_hotelling_t2: pd.Series, ucl: float, _udis: pd.Series):
    """Assess health status for every Phase-II observation. Cached on inputs."""
    health_df = assess_dataset_health(_phase2_hotelling_t2, ucl)
    health_df.insert(0, "UDI", _udis.values)
    return health_df


def load_full_pipeline():
    """
    Run (or retrieve from cache) the full statistical pipeline and
    return every intermediate result needed by the UI.

    Returns:
        A dictionary bundling all pipeline outputs, or None if a
        fatal error occurred (in which case an error is already
        shown via st.error).
    """
    try:
        raw_df, validation_summary = load_and_validate_data()
    except FileNotFoundError as exc:
        st.error(
            "Dataset not found. Please ensure 'ai4i2020.csv' exists inside the "
            f"'data/' folder at the project root.\n\nDetails: {exc}"
        )
        return None
    except ValueError as exc:
        st.error(f"Dataset validation failed: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error while loading the dataset: {exc}")
        return None

    if raw_df.empty:
        st.error("The loaded dataset is empty. Cannot proceed with analysis.")
        return None

    try:
        phase_data = build_phase_data(raw_df)
    except ValueError as exc:
        st.error(f"Phase I / Phase II preparation failed: {exc}")
        return None

    try:
        pca_result = build_pca_result(phase_data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"PCA analysis failed: {exc}")
        return None

    try:
        t2_result = build_t2_result(phase_data)
    except ValueError as exc:
        st.error(f"Hotelling's T2 calculation failed: {exc}")
        return None

    try:
        phase2_hash = pd.util.hash_pandas_object(t2_result.phase2_results).sum()
        t2_eval = build_t2_evaluation(phase2_hash, t2_result.phase2_results)
    except ValueError as exc:
        st.error(f"T2 evaluation failed: {exc}")
        return None

    try:
        manova_result = build_manova_result(raw_df)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"MANOVA analysis could not be completed: {exc}")
        manova_result = None

    try:
        diagnostics_result = build_diagnostics_result(raw_df, phase_data, t2_result)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Diagnostics could not be completed: {exc}")
        diagnostics_result = None

    try:
        health_df = build_health_dataframe(
            t2_result.phase2_results["Hotelling_T2"],
            t2_result.ucl,
            t2_result.phase2_results["UDI"],
        )
        health_summary = summarize_health_statuses(health_df)
    except ValueError as exc:
        st.error(f"Health assessment failed: {exc}")
        return None

    return {
        "raw_df": raw_df,
        "validation_summary": validation_summary,
        "phase_data": phase_data,
        "pca_result": pca_result,
        "t2_result": t2_result,
        "t2_eval": t2_eval,
        "manova_result": manova_result,
        "diagnostics_result": diagnostics_result,
        "health_df": health_df,
        "health_summary": health_summary,
    }


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================


def format_p_value(p: float) -> str:
    """Format a p-value, using 'p < 0.001' below the standard reporting threshold."""
    if p is None or not np.isfinite(p):
        return "unavailable"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def get_machine_row(t2_result, udi: int) -> Optional[pd.Series]:
    """Retrieve the Phase-II row for a given UDI, or None if not found."""
    matches = t2_result.phase2_results.loc[t2_result.phase2_results["UDI"] == udi]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_machine_context(pipeline: Dict[str, Any], udi: int) -> Optional[Dict[str, Any]]:
    """
    Build the structured machine_context dictionary consumed by the
    Gemini agent, using only verified statistical outputs.
    """
    t2_result = pipeline["t2_result"]
    row = get_machine_row(t2_result, udi)
    if row is None:
        return None

    ucl = t2_result.ucl
    t2_value = float(row["Hotelling_T2"])
    health_status = classify_health(t2_value, ucl)
    ratio = calculate_t2_ratio(t2_value, ucl)

    standardized_values = {col: float(row[col]) for col in SENSOR_COLUMNS}
    sensor_values = standardized_values  # phase2_results stores standardized sensors only

    rec = generate_maintenance_recommendation(
        health_status, t2_value, ucl, sensor_values, standardized_values
    )

    pca_result = pipeline["pca_result"]
    n_components_90 = components_for_variance_threshold(pca_result, threshold=0.90)
    pca_interpretation = (
        f"{n_components_90} principal component(s) explain at least 90% of the variance "
        f"in the Phase-I healthy baseline. PC1 explains "
        f"{pca_result.explained_variance_ratio[0] * 100:.2f}% of variance, reflecting a "
        "combination of sensor variation; this describes variance structure only and does "
        "not imply which variables cause abnormal states."
    )

    return {
        "udi": int(udi),
        "t2_value": t2_value,
        "ucl": float(ucl),
        "t2_ratio": ratio,
        "health_status": health_status,
        "priority": rec["priority"],
        "contributing_sensors": rec["contributing_sensors"],
        "pca_interpretation": pca_interpretation,
        "diagnostics": None,
        "primary_recommendation": rec["primary_recommendation"],
        "additional_recommendations": rec["additional_recommendations"],
        "evidence": rec["evidence"],
        "limitations": rec["limitations"],
        "sensor_values": sensor_values,
        "standardized_sensors": standardized_values,
        "failure_probability": None,
    }


def render_failure_probability_notice():
    """Render the mandatory failure-probability unavailability notice."""
    st.info(
        "Failure probability: Not available — T² is a multivariate abnormality measure, "
        "not a probability."
    )


def health_badge(status: str) -> str:
    """Return a colored HTML badge for a health status, for use with st.markdown."""
    color = HEALTH_COLORS.get(status, "#7f8c8d")
    return (
        f"<span style='background-color:{color};color:white;padding:4px 10px;"
        f"border-radius:6px;font-weight:600;'>{status}</span>"
    )


# ======================================================================
# PAGE RENDERERS
# ======================================================================


def render_dashboard(pipeline: Dict[str, Any]):
    st.header("Dashboard")
    st.caption("System-wide overview of the AI4I 2020 dataset and monitoring pipeline.")

    raw_df = pipeline["raw_df"]
    phase_data = pipeline["phase_data"]
    t2_result = pipeline["t2_result"]
    health_summary = pipeline["health_summary"]

    n_total = len(raw_df)
    n_phase1_healthy = len(phase_data.phase1_healthy)
    n_phase2 = len(phase_data.phase2)
    ucl = t2_result.ucl
    alert_rate = float(t2_result.phase2_results["T2_Alert"].mean() * 100)
    observed_failure_rate = float(raw_df["Machine failure"].mean() * 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", f"{n_total:,}")
    col2.metric("Phase-I Healthy Baseline", f"{n_phase1_healthy:,}")
    col3.metric("Phase-II Observations", f"{n_phase2:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("T² UCL", f"{ucl:.4f}")
    col5.metric("T² Alert Rate (Phase-II)", f"{alert_rate:.2f}%")
    col6.metric("Observed Failure Rate (full dataset)", f"{observed_failure_rate:.2f}%")

    st.caption(
        "Note: T² alert rate reflects the proportion of Phase-II observations flagged as "
        "statistically abnormal. It is a monitoring signal, not the observed failure rate."
    )

    st.subheader("Health Status Distribution (Phase-II)")
    summary_reset = health_summary.reset_index().rename(columns={"index": "Status"})
    fig = px.bar(
        summary_reset,
        x="Status",
        y="Count",
        color="Status",
        color_discrete_map=HEALTH_COLORS,
        text="Count",
    )
    fig.update_layout(showlegend=False, yaxis_title="Number of Observations")
    st.plotly_chart(fig, use_container_width=True)


def render_machine_health(pipeline: Dict[str, Any], selected_udi: int):
    st.header("Machine Health")
    st.caption("Detailed health status for a selected Phase-II observation.")

    t2_result = pipeline["t2_result"]
    row = get_machine_row(t2_result, selected_udi)

    if row is None:
        st.warning(f"UDI {selected_udi} was not found in the Phase-II dataset.")
        return

    ucl = t2_result.ucl
    t2_value = float(row["Hotelling_T2"])
    health_status = classify_health(t2_value, ucl)
    ratio = calculate_t2_ratio(t2_value, ucl)

    standardized_values = {col: float(row[col]) for col in SENSOR_COLUMNS}
    rec = generate_maintenance_recommendation(
        health_status, t2_value, ucl, standardized_values, standardized_values
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("UDI", int(selected_udi))
    col2.metric("Hotelling's T²", f"{t2_value:.4f}")
    col3.metric("T² UCL", f"{ucl:.4f}")

    col4, col5 = st.columns(2)
    col4.metric("T² / UCL Ratio", f"{ratio:.4f}")
    col5.markdown(f"**Health Status:** {health_badge(health_status)}", unsafe_allow_html=True)

    st.markdown(f"**Maintenance Priority:** `{rec['priority']}`")
    render_failure_probability_notice()

    st.subheader("Standardized Sensor Values")
    sensor_df = pd.DataFrame(
        {
            "Sensor": SENSOR_COLUMNS,
            "Standardized Value (z-score)": [standardized_values[c] for c in SENSOR_COLUMNS],
        }
    )
    sensor_df["Extreme (|z| ≥ 2.0)"] = sensor_df["Standardized Value (z-score)"].abs() >= 2.0
    st.dataframe(sensor_df, use_container_width=True, hide_index=True)

    contributing = rec["contributing_sensors"]
    st.subheader("Contributing Sensor Pattern")
    if contributing:
        for item in contributing:
            direction_word = "above" if item["direction"] == "HIGH" else "below"
            st.markdown(
                f"- **{item['sensor']}**: {item['z_score']:+.4f} SD "
                f"({direction_word} the healthy baseline mean) — associated with the abnormal state."
            )
    else:
        st.info("No individual sensor exceeded the ±2 SD diagnostic threshold for this observation.")


def render_spc_monitoring(pipeline: Dict[str, Any]):
    st.header("SPC Monitoring")
    st.caption(
        "Hotelling's T² identifies observations whose multivariate sensor profile is unusual "
        "relative to the healthy Phase-I baseline."
    )

    t2_result = pipeline["t2_result"]
    phase2 = t2_result.phase2_results
    ucl = t2_result.ucl
    thresholds = get_health_thresholds(ucl)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=phase2["UDI"],
            y=phase2["Hotelling_T2"],
            mode="markers",
            marker=dict(
                color=phase2["T2_Alert"].map({0: "#2ecc71", 1: "#e74c3c"}),
                size=6,
            ),
            name="Phase-II T²",
            text=phase2["Health_Status"] if "Health_Status" in phase2.columns else None,
        )
    )
    fig.add_hline(y=ucl, line_dash="dash", line_color="black", annotation_text="UCL")
    fig.add_hline(
        y=thresholds["watch_threshold"],
        line_dash="dot",
        line_color="orange",
        annotation_text="Watch threshold",
    )
    fig.add_hline(
        y=thresholds["alert_threshold"],
        line_dash="dot",
        line_color="red",
        annotation_text="Alert threshold",
    )
    fig.update_layout(
        xaxis_title="UDI (observation sequence)",
        yaxis_title="Hotelling's T²",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Red points exceed the established UCL (T² alerts). This indicates a statistically "
        "unusual multivariate operating state, not a confirmed failure."
    )


def render_pca_analysis(pipeline: Dict[str, Any]):
    st.header("PCA Analysis")
    st.caption("Variance structure of the five standardized sensor variables (Phase-I healthy baseline).")

    pca_result = pipeline["pca_result"]
    n_required = components_for_variance_threshold(pca_result, threshold=0.90)

    pc_labels = [f"PC{i + 1}" for i in range(len(pca_result.explained_variance_ratio))]
    variance_df = pd.DataFrame(
        {
            "Component": pc_labels,
            "Explained Variance Ratio": pca_result.explained_variance_ratio,
            "Cumulative Explained Variance": pca_result.cumulative_explained_variance,
        }
    )

    st.dataframe(variance_df.style.format({
        "Explained Variance Ratio": "{:.2%}",
        "Cumulative Explained Variance": "{:.2%}",
    }), use_container_width=True, hide_index=True)

    st.info(f"{n_required} principal component(s) are required to reach at least 90% cumulative explained variance.")

    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(x=pc_labels, y=pca_result.explained_variance_ratio, name="Explained Variance"))
    fig_scree.add_trace(
        go.Scatter(
            x=pc_labels,
            y=pca_result.cumulative_explained_variance,
            name="Cumulative",
            yaxis="y2",
            mode="lines+markers",
        )
    )
    fig_scree.update_layout(
        yaxis=dict(title="Explained Variance Ratio", tickformat=".0%"),
        yaxis2=dict(title="Cumulative Variance", overlaying="y", side="right", tickformat=".0%"),
        height=450,
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    st.subheader("Loadings")
    st.dataframe(pca_result.loadings.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Interpretation")
    st.warning(
        "PCA describes variance structure among sensor variables. It does not identify "
        "causes of machine failure. Loadings indicate which sensors co-vary most strongly "
        "with each component, e.g., 'PC1 represents a dominant combination of temperature, "
        "rotational speed, and torque variation.' This is a descriptive statement about "
        "variance, not a causal claim."
    )


def render_t2_evaluation(pipeline: Dict[str, Any]):
    st.header("T² Evaluation")
    st.caption("Comparison of T² alerts against the observed Machine failure outcome (Phase-II).")

    t2_eval = pipeline["t2_eval"]

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        {
            "Predicted: Alert": [t2_eval.true_positives, t2_eval.false_positives],
            "Predicted: No Alert": [t2_eval.false_negatives, t2_eval.true_negatives],
        },
        index=["Actual: Failure", "Actual: No Failure"],
    )
    st.dataframe(cm_df, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{t2_eval.precision * 100:.2f}%")
    col2.metric("Recall", f"{t2_eval.recall * 100:.2f}%")
    col3.metric("Specificity", f"{t2_eval.specificity * 100:.2f}%")
    col4.metric("F1 Score", f"{t2_eval.f1_score * 100:.2f}%")

    col5, col6, col7 = st.columns(3)
    col5.metric("False Positive Rate", f"{t2_eval.false_positive_rate * 100:.2f}%")
    col6.metric("Alert Rate", f"{t2_eval.alert_rate * 100:.2f}%")
    col7.metric("Failure Capture Rate", f"{t2_eval.failure_capture_rate * 100:.2f}%")

    st.warning(
        "T² is a multivariate monitoring method, so an alert indicates an unusual "
        "multivariate operating state. It does not automatically mean the machine will fail."
    )

    st.subheader("T² by Observed Failure Status")
    st.dataframe(t2_eval.group_comparison.style.format("{:.4f}"), use_container_width=True)


def render_manova(pipeline: Dict[str, Any]):
    st.header("MANOVA")
    st.caption(
        "MANOVA tests whether the joint mean vector of the sensor variables differs between "
        "the observed failure and non-failure groups."
    )

    manova_result = pipeline["manova_result"]
    if manova_result is None:
        st.error("MANOVA results are unavailable for this session.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Non-failure group size", manova_result.group_sizes["Non-failure"])
    col2.metric("Failure group size", manova_result.group_sizes["Failure"])

    st.subheader("Descriptive Statistics by Group")
    st.dataframe(manova_result.descriptive_stats.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Group Mean Differences (Failure − Non-failure)")
    st.dataframe(
        manova_result.mean_differences.to_frame("Mean Difference").style.format("{:.4f}"),
        use_container_width=True,
    )

    st.subheader("Multivariate Test Statistics")
    tests_display = manova_result.multivariate_tests.copy()
    if "Pr > F" in tests_display.columns:
        tests_display["Pr > F"] = tests_display["Pr > F"].apply(format_p_value)
    st.dataframe(tests_display, use_container_width=True)

    st.warning(
        "A statistically significant MANOVA result indicates the multivariate sensor means "
        "differ between the observed failure and non-failure groups. It does not establish "
        "that these sensors cause machine failure."
    )

    with st.expander("Assumption diagnostics and limitations"):
        for note in manova_result.limitations:
            st.markdown(f"- {note}")


def render_diagnostics(pipeline: Dict[str, Any]):
    st.header("Diagnostics")
    st.caption("Assumption checks and numerical diagnostics supporting the analyses above.")

    diagnostics_result = pipeline["diagnostics_result"]
    if diagnostics_result is None:
        st.error("Diagnostics are unavailable for this session.")
        return

    pc = diagnostics_result.phase1_covariance
    st.subheader("Phase-I Covariance Diagnostics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Condition Number", f"{pc.condition_number:.4f}")
    col2.metric("Min Eigenvalue", f"{pc.min_eigenvalue:.6f}")
    col3.metric("Max Eigenvalue", f"{pc.max_eigenvalue:.6f}")

    col4, col5 = st.columns(2)
    col4.metric("Determinant", f"{pc.determinant:.6e}")
    col5.metric("Positive Definite", "Yes" if pc.is_positive_definite else "No")

    if not pc.is_numerically_suitable:
        st.warning("The Phase-I covariance matrix did not pass all numerical suitability checks.")

    pt = diagnostics_result.phase1_t2_distribution
    st.subheader("Phase-I T² Distribution")
    col6, col7, col8, col9 = st.columns(4)
    col6.metric("Mean", f"{pt.mean:.4f}")
    col7.metric("Median", f"{pt.median:.4f}")
    col8.metric("P95", f"{pt.p95:.4f}")
    col9.metric("% Above UCL", f"{pt.pct_above_ucl:.2f}%")

    nd = diagnostics_result.normality_diagnostic
    st.subheader("Multivariate Normality Diagnostic")
    st.metric("Q-Q Correlation (Mahalanobis vs Chi-square)", f"{nd.qq_correlation:.6f}")
    st.caption(nd.limitations)
    st.info(
        "This diagnostic is descriptive evidence only. A high correlation is consistent with "
        "multivariate normality but does not prove it."
    )

    st.subheader("Top 10 Phase-I Observations by T²")
    st.dataframe(diagnostics_result.top10_phase1_t2, use_container_width=True, hide_index=True)


def render_maintenance(pipeline: Dict[str, Any], selected_udi: int):
    st.header("Maintenance")
    st.caption("Deterministic maintenance recommendation for the selected machine.")

    t2_result = pipeline["t2_result"]
    row = get_machine_row(t2_result, selected_udi)

    if row is None:
        st.warning(f"UDI {selected_udi} was not found in the Phase-II dataset.")
        return

    ucl = t2_result.ucl
    t2_value = float(row["Hotelling_T2"])
    health_status = classify_health(t2_value, ucl)
    standardized_values = {col: float(row[col]) for col in SENSOR_COLUMNS}

    rec = generate_maintenance_recommendation(
        health_status, t2_value, ucl, standardized_values, standardized_values
    )

    st.markdown(f"**Health Status:** {health_badge(rec['health_status'])}", unsafe_allow_html=True)
    st.markdown(f"**Priority:** `{rec['priority']}`")

    st.subheader("Primary Recommendation")
    st.success(rec["primary_recommendation"])

    if rec["additional_recommendations"]:
        st.subheader("Additional Recommendations")
        for item in rec["additional_recommendations"]:
            st.markdown(f"- {item}")

    st.subheader("Contributing Sensor Patterns")
    if rec["contributing_sensors"]:
        for item in rec["contributing_sensors"]:
            st.markdown(f"- **{item['sensor']}**: {item['z_score']:+.4f} SD ({item['direction']})")
    else:
        st.info("No individual sensor exceeded the diagnostic threshold.")

    st.subheader("Evidence")
    for item in rec["evidence"]:
        st.markdown(f"- {item}")

    with st.expander("Limitations"):
        for item in rec["limitations"]:
            st.markdown(f"- {item}")


def render_ai_assistant(pipeline: Dict[str, Any], selected_udi: int):
    st.header("AI Maintenance Assistant")
    st.caption("Explanations grounded in verified statistical outputs. Powered by Gemini.")

    if not AGENT_AVAILABLE:
        st.error(
            "The Gemini AI agent module could not be loaded. Check that agent/gemini_agent.py "
            "is present and its dependencies are installed."
        )
        return

    if not os.environ.get("GEMINI_API_KEY"):
        st.warning(
            "GEMINI_API_KEY is not set. Set it in a .env file at the project root or as a "
            "system environment variable, then restart the app to enable the AI assistant."
        )
        return

    machine_context = build_machine_context(pipeline, selected_udi)
    if machine_context is None:
        st.warning(f"UDI {selected_udi} was not found in the Phase-II dataset.")
        return

    with st.expander("View verified machine context sent to the AI assistant"):
        st.json(machine_context)

    render_failure_probability_notice()

    if st.button("Generate AI Health Explanation", type="primary"):
        with st.spinner("Contacting Gemini..."):
            try:
                explanation = analyze_machine_health(machine_context)
            except Exception as exc:  # noqa: BLE001
                explanation = f"AI Assistant error: {exc}"
        st.markdown(explanation)

    st.divider()
    st.subheader("Ask a Follow-up Question")
    st.caption(
        "Examples: \"Why is this machine critical?\", \"What should maintenance inspect first?\", "
        "\"Which sensors are contributing to the abnormal state?\", \"Is this machine guaranteed "
        "to fail?\", \"What does the T² score mean?\""
    )

    question = st.text_input("Your question", key="followup_question")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Contacting Gemini..."):
                try:
                    answer = answer_followup(question, machine_context)
                except Exception as exc:  # noqa: BLE001
                    answer = f"AI Assistant error: {exc}"
            st.markdown(answer)


# ======================================================================
# MAIN APP
# ======================================================================


def main():
    st.set_page_config(
        page_title="Industrial Machine Health Intelligence Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Industrial Machine Health Intelligence Platform")
    st.caption("Multivariate Statistical Process Control for Predictive Maintenance")

    try:
        pipeline = load_full_pipeline()
    except Exception as exc:  # noqa: BLE001
        st.error(f"A fatal error occurred while initializing the statistical pipeline: {exc}")
        st.text(traceback.format_exc())
        st.stop()

    if pipeline is None:
        st.stop()

    with st.sidebar:
        st.header("Dataset")
        st.write(f"Rows: {pipeline['validation_summary']['row_count']:,}")
        st.write(f"Columns: {pipeline['validation_summary']['column_count']}")
        st.write(f"Duplicate rows: {pipeline['validation_summary']['duplicate_row_count']}")
        st.write(f"Total missing values: {pipeline['validation_summary']['total_missing']}")

        st.divider()
        st.header("Machine Selector")
        phase2_udis = pipeline["t2_result"].phase2_results["UDI"].tolist()
        selected_udi = st.selectbox("Select Phase-II UDI", options=phase2_udis, index=0)

        st.divider()
        st.header("Navigation")
        page = st.radio(
            "Go to",
            [
                "Dashboard",
                "Machine Health",
                "SPC Monitoring",
                "PCA Analysis",
                "T² Evaluation",
                "MANOVA",
                "Diagnostics",
                "Maintenance",
                "AI Maintenance Assistant",
            ],
        )

    if page == "Dashboard":
        render_dashboard(pipeline)
    elif page == "Machine Health":
        render_machine_health(pipeline, selected_udi)
    elif page == "SPC Monitoring":
        render_spc_monitoring(pipeline)
    elif page == "PCA Analysis":
        render_pca_analysis(pipeline)
    elif page == "T² Evaluation":
        render_t2_evaluation(pipeline)
    elif page == "MANOVA":
        render_manova(pipeline)
    elif page == "Diagnostics":
        render_diagnostics(pipeline)
    elif page == "Maintenance":
        render_maintenance(pipeline, selected_udi)
    elif page == "AI Maintenance Assistant":
        render_ai_assistant(pipeline, selected_udi)


if __name__ == "__main__":
    main()
