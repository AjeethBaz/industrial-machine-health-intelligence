"""
backend/main.py

FastAPI application exposing the existing statistical pipeline
(src/*) and the existing Gemini agent (agent/gemini_agent.py) over
HTTP/JSON for the React frontend.

This module performs NO statistical calculations itself. Every numeric
value returned comes from backend/pipeline.py, which calls the
already-tested src/ modules directly.

The additional diagnostics exposed by /api/diagnostics are read-only
assumption diagnostics. They do not modify the existing PCA,
Hotelling's T², UCL, MANOVA, health assessment, or maintenance logic.

GEMINI_API_KEY is read only inside agent/gemini_agent.py via os.environ;
it is never included in any response body and is never read directly in
this file. The .env file lives at agent/.env, so it is loaded explicitly
here before agent.gemini_agent is imported.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Anchor .env loading to agent/.env, not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AGENT_ENV = _PROJECT_ROOT / "agent" / ".env"
load_dotenv(dotenv_path=_AGENT_ENV)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import (
    get_pipeline,
    build_machine_health,
    build_pca_interpretation,
    SENSOR_COLUMNS,
)
from backend.schemas import (
    df_to_records,
    matrix_to_nested_list,
    series_to_dict,
    OverviewResponse,
    ValidationSummary,
    HealthStatusCount,
    MachineListItem,
    MachineHealthResponse,
    ContributingSensor,
    SPCResponse,
    SPCPoint,
    PCAResponse,
    T2EvaluationResponse,
    ManovaResponse,
    ManovaTestRow,
    DiagnosticsResponse,
    CovarianceDiagnosticsResponse,
    Phase1T2DistributionResponse,
    NormalityDiagnosticResponse,
    MardiaNormalityDiagnosticResponse,
    BoxMResponse,
    LeveneBrownForsytheResponse,
    MaintenanceResponse,
    AssistantRequest,
    AssistantResponse,
)

try:
    from agent.gemini_agent import analyze_machine_health, answer_followup

    AGENT_AVAILABLE = True
except Exception:  # noqa: BLE001
    AGENT_AVAILABLE = False


app = FastAPI(
    title="Industrial Machine Health Intelligence Platform API",
    description=(
        "FastAPI backend exposing the existing multivariate SPC statistical "
        "pipeline (PCA, Hotelling's T-squared, MANOVA, diagnostics, health "
        "assessment, maintenance recommendations) and the Gemini AI "
        "maintenance assistant. All statistics are computed by src/*; this "
        "API only serializes results to JSON."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pipeline_or_500():
    """Load the cached pipeline, converting pipeline errors to HTTP 500s."""
    try:
        return get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset not found: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Statistical pipeline error: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected pipeline error: {exc}",
        ) from exc


@app.get("/api/overview", response_model=OverviewResponse)
def get_overview():
    """Return dataset-wide KPI summary and health status distribution."""
    cache = _pipeline_or_500()

    n_total = len(cache.raw_df)
    n_phase1_healthy = len(cache.phase_data.phase1_healthy)
    n_phase2 = len(cache.phase_data.phase2)
    ucl = cache.t2_result.ucl
    alert_rate = float(
        cache.t2_result.phase2_results["T2_Alert"].mean() * 100
    )
    observed_failure_rate = float(
        cache.raw_df["Machine failure"].mean() * 100
    )

    distribution = [
        HealthStatusCount(
            status=status,
            count=int(cache.health_summary.loc[status, "Count"]),
            percentage=float(
                cache.health_summary.loc[status, "Percentage"]
            ),
        )
        for status in ["NORMAL", "WATCH", "ALERT", "CRITICAL"]
    ]

    validation = ValidationSummary(
        row_count=cache.validation_summary["row_count"],
        column_count=cache.validation_summary["column_count"],
        duplicate_row_count=cache.validation_summary["duplicate_row_count"],
        total_missing=cache.validation_summary["total_missing"],
    )

    return OverviewResponse(
        total_observations=n_total,
        phase1_healthy_count=n_phase1_healthy,
        phase2_count=n_phase2,
        ucl=float(ucl),
        alert_rate_percent=alert_rate,
        observed_failure_rate_percent=observed_failure_rate,
        health_status_distribution=distribution,
        validation=validation,
    )


@app.get("/api/machines", response_model=List[MachineListItem])
def list_machines():
    """Return the list of Phase-II UDIs with their current health status."""
    from src.health_assessment import classify_health

    cache = _pipeline_or_500()
    phase2 = cache.t2_result.phase2_results
    ucl = cache.t2_result.ucl

    items = []

    for _, row in phase2.iterrows():
        t2_value = float(row["Hotelling_T2"])

        items.append(
            MachineListItem(
                udi=int(row["UDI"]),
                health_status=classify_health(t2_value, ucl),
                t2_value=t2_value,
            )
        )

    return items


@app.get(
    "/api/machines/{udi}",
    response_model=MachineHealthResponse,
)
def get_machine(udi: int):
    """Return the full machine-health record for a given Phase-II UDI."""
    cache = _pipeline_or_500()
    result = build_machine_health(cache, udi)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"UDI {udi} not found in Phase-II dataset.",
        )

    result["contributing_sensors"] = [
        ContributingSensor(**sensor)
        for sensor in result["contributing_sensors"]
    ]

    return MachineHealthResponse(**result)


@app.get("/api/spc", response_model=SPCResponse)
def get_spc():
    """Return Hotelling's T2 control-chart data for all Phase-II observations."""
    from src.health_assessment import get_health_thresholds, classify_health

    cache = _pipeline_or_500()
    phase2 = cache.t2_result.phase2_results
    ucl = cache.t2_result.ucl
    thresholds = get_health_thresholds(ucl)

    points = [
        SPCPoint(
            udi=int(row["UDI"]),
            t2=float(row["Hotelling_T2"]),
            alert=bool(row["T2_Alert"]),
            health_status=classify_health(
                float(row["Hotelling_T2"]),
                ucl,
            ),
        )
        for _, row in phase2.iterrows()
    ]

    return SPCResponse(
        ucl=float(ucl),
        watch_threshold=thresholds["watch_threshold"],
        alert_threshold=thresholds["alert_threshold"],
        points=points,
    )


@app.get("/api/pca", response_model=PCAResponse)
def get_pca():
    """Return PCA variance structure and loadings for the Phase-I healthy baseline."""
    from src.pca_analysis import components_for_variance_threshold

    cache = _pipeline_or_500()
    pca_result = cache.pca_result

    n_components = len(pca_result.explained_variance_ratio)
    component_names = [
        f"PC{index + 1}"
        for index in range(n_components)
    ]

    loadings_dict = {
        sensor: {
            component: float(
                pca_result.loadings.loc[sensor, component]
            )
            for component in pca_result.loadings.columns
        }
        for sensor in pca_result.loadings.index
    }

    return PCAResponse(
        components=component_names,
        explained_variance_ratio=[
            float(value)
            for value in pca_result.explained_variance_ratio
        ],
        cumulative_explained_variance=[
            float(value)
            for value in pca_result.cumulative_explained_variance
        ],
        loadings=loadings_dict,
        components_required_for_90_percent=(
            components_for_variance_threshold(
                pca_result,
                threshold=0.90,
            )
        ),
        interpretation=build_pca_interpretation(cache),
    )


@app.get(
    "/api/t2-evaluation",
    response_model=T2EvaluationResponse,
)
def get_t2_evaluation():
    """Return confusion matrix and metrics for T² alerts vs Machine failure."""
    cache = _pipeline_or_500()
    evaluation = cache.t2_eval

    group_comparison = df_to_records(
        evaluation.group_comparison,
        index_name="group",
    )

    return T2EvaluationResponse(
        true_positives=evaluation.true_positives,
        false_positives=evaluation.false_positives,
        true_negatives=evaluation.true_negatives,
        false_negatives=evaluation.false_negatives,
        precision=evaluation.precision,
        recall=evaluation.recall,
        specificity=evaluation.specificity,
        f1_score=evaluation.f1_score,
        false_positive_rate=evaluation.false_positive_rate,
        alert_rate=evaluation.alert_rate,
        actual_failure_rate=evaluation.actual_failure_rate,
        failure_capture_rate=evaluation.failure_capture_rate,
        group_comparison=group_comparison,
    )


@app.get("/api/manova", response_model=ManovaResponse)
def get_manova():
    """Return MANOVA sizes, descriptive statistics, and test statistics."""
    cache = _pipeline_or_500()
    manova_result = cache.manova_result

    descriptive_stats = df_to_records(
        manova_result.descriptive_stats.reset_index()
    )

    tests_df = manova_result.multivariate_tests
    test_rows = []

    for test_name, row in tests_df.iterrows():
        test_rows.append(
            ManovaTestRow(
                test_name=str(test_name),
                value=float(row["Value"]),
                f_value=float(row["F Value"]),
                num_df=float(row["Num DF"]),
                den_df=float(row["Den DF"]),
                p_value=float(row["Pr > F"]),
            )
        )

    return ManovaResponse(
        group_sizes=manova_result.group_sizes,
        descriptive_stats=descriptive_stats,
        mean_differences=series_to_dict(
            manova_result.mean_differences
        ),
        multivariate_tests=test_rows,
        limitations=manova_result.limitations,
    )


@app.get(
    "/api/diagnostics",
    response_model=DiagnosticsResponse,
)
def get_diagnostics():
    """
    Return existing diagnostics plus additive assumption diagnostics.

    Existing response sections are preserved:
    - phase1_covariance
    - phase1_t2_distribution
    - normality_diagnostic
    - top10_phase1_t2

    New sections are read-only:
    - mardia_normality
    - box_m
    - levene_brown_forsythe
    """
    cache = _pipeline_or_500()
    diagnostics = cache.diagnostics_result

    phase1_covariance = diagnostics.phase1_covariance

    covariance_interpretation = (
        "The standardized Phase-I healthy covariance matrix is positive "
        "definite and numerically suitable for the existing Hotelling's "
        "T² calculation."
        if phase1_covariance.is_numerically_suitable
        else
        "The covariance diagnostics indicate potential numerical concerns "
        "for the existing Hotelling's T² calculation."
    )

    covariance_diag = CovarianceDiagnosticsResponse(
        condition_number=float(
            phase1_covariance.condition_number
        ),
        min_eigenvalue=float(
            phase1_covariance.min_eigenvalue
        ),
        max_eigenvalue=float(
            phase1_covariance.max_eigenvalue
        ),
        determinant=float(
            phase1_covariance.determinant
        ),
        is_positive_definite=bool(
            phase1_covariance.is_positive_definite
        ),
        is_numerically_suitable=bool(
            phase1_covariance.is_numerically_suitable
        ),
        eigenvalues=[
            float(value)
            for value in phase1_covariance.eigenvalues
        ],
        interpretation=covariance_interpretation,
    )

    phase1_t2_distribution = diagnostics.phase1_t2_distribution

    t2_dist = Phase1T2DistributionResponse(
        count=phase1_t2_distribution.count,
        mean=phase1_t2_distribution.mean,
        median=phase1_t2_distribution.median,
        std=phase1_t2_distribution.std,
        minimum=phase1_t2_distribution.minimum,
        maximum=phase1_t2_distribution.maximum,
        p90=phase1_t2_distribution.p90,
        p95=phase1_t2_distribution.p95,
        p99=phase1_t2_distribution.p99,
        n_above_ucl=phase1_t2_distribution.n_above_ucl,
        pct_above_ucl=phase1_t2_distribution.pct_above_ucl,
    )

    normality_diagnostic = diagnostics.normality_diagnostic

    normality = NormalityDiagnosticResponse(
        qq_correlation=float(
            normality_diagnostic.qq_correlation
        ),
        test_name=normality_diagnostic.test_name,
        limitations=normality_diagnostic.limitations,
    )

    mardia_diagnostic = diagnostics.mardia_normality

    mardia_normality = MardiaNormalityDiagnosticResponse(
        n=mardia_diagnostic.n,
        p=mardia_diagnostic.p,
        mardia_skewness_b1p=float(
            mardia_diagnostic.mardia_skewness_b1p
        ),
        skewness_chi_square=float(
            mardia_diagnostic.skewness_chi_square
        ),
        skewness_degrees_of_freedom=(
            mardia_diagnostic.skewness_degrees_of_freedom
        ),
        skewness_z=float(
            mardia_diagnostic.skewness_z
        ),
        skewness_p_value=float(
            mardia_diagnostic.skewness_p_value
        ),
        mardia_kurtosis_b2p=float(
            mardia_diagnostic.mardia_kurtosis_b2p
        ),
        normal_reference_kurtosis=float(
            mardia_diagnostic.normal_reference_kurtosis
        ),
        kurtosis_z=float(
            mardia_diagnostic.kurtosis_z
        ),
        kurtosis_p_value=float(
            mardia_diagnostic.kurtosis_p_value
        ),
        interpretation=mardia_diagnostic.interpretation,
        limitations=mardia_diagnostic.limitations,
    )

    box_m_diagnostic = diagnostics.box_m

    box_m = BoxMResponse(
        non_failure_n=box_m_diagnostic.non_failure_n,
        failure_n=box_m_diagnostic.failure_n,
        box_m_statistic=float(
            box_m_diagnostic.box_m_statistic
        ),
        correction_factor=float(
            box_m_diagnostic.correction_factor
        ),
        corrected_chi_square=float(
            box_m_diagnostic.corrected_chi_square
        ),
        degrees_of_freedom=float(
            box_m_diagnostic.degrees_of_freedom
        ),
        p_value=float(box_m_diagnostic.p_value),
        interpretation=box_m_diagnostic.interpretation,
        limitations=box_m_diagnostic.limitations,
    )

    levene_brown_forsythe = [
        LeveneBrownForsytheResponse(
            sensor=result.sensor,
            statistic=float(result.statistic),
            p_value=float(result.p_value),
            interpretation=result.interpretation,
        )
        for result in diagnostics.levene_brown_forsythe
    ]

    top10 = df_to_records(diagnostics.top10_phase1_t2)

    return DiagnosticsResponse(
        phase1_covariance=covariance_diag,
        phase1_t2_distribution=t2_dist,
        normality_diagnostic=normality,
        top10_phase1_t2=top10,
        mardia_normality=mardia_normality,
        box_m=box_m,
        levene_brown_forsythe=levene_brown_forsythe,
    )


@app.get(
    "/api/maintenance/{udi}",
    response_model=MaintenanceResponse,
)
def get_maintenance(udi: int):
    """Return maintenance recommendation for a given Phase-II UDI."""
    cache = _pipeline_or_500()
    result = build_machine_health(cache, udi)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"UDI {udi} not found in Phase-II dataset.",
        )

    return MaintenanceResponse(
        udi=result["udi"],
        health_status=result["health_status"],
        priority=result["priority"],
        contributing_sensors=[
            ContributingSensor(**sensor)
            for sensor in result["contributing_sensors"]
        ],
        primary_recommendation=result["primary_recommendation"],
        additional_recommendations=result[
            "additional_recommendations"
        ],
        evidence=result["evidence"],
        limitations=result["limitations"],
    )


@app.post(
    "/api/assistant",
    response_model=AssistantResponse,
)
def post_assistant(request: AssistantRequest):
    """
    Generate AI explanation or follow-up answer for one machine.

    If no question is provided, analyze_machine_health() is called.
    If a question is provided, answer_followup() is called.
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="The Gemini AI agent module is unavailable on the server.",
        )

    cache = _pipeline_or_500()
    machine_data = build_machine_health(cache, request.udi)

    if machine_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"UDI {request.udi} not found in Phase-II dataset.",
        )

    machine_context: Dict[str, Any] = {
        "udi": machine_data["udi"],
        "t2_value": machine_data["t2_value"],
        "ucl": machine_data["ucl"],
        "t2_ratio": machine_data["t2_ratio"],
        "health_status": machine_data["health_status"],
        "priority": machine_data["priority"],
        "contributing_sensors": machine_data[
            "contributing_sensors"
        ],
        "pca_interpretation": build_pca_interpretation(cache),
        "diagnostics": None,
        "primary_recommendation": machine_data[
            "primary_recommendation"
        ],
        "additional_recommendations": machine_data[
            "additional_recommendations"
        ],
        "evidence": machine_data["evidence"],
        "limitations": machine_data["limitations"],
        "sensor_values": machine_data["sensor_values"],
        "standardized_sensors": machine_data[
            "standardized_sensors"
        ],
        "failure_probability": None,
    }

    try:
        if request.question and request.question.strip():
            response_text = answer_followup(
                request.question,
                machine_context,
            )
        else:
            response_text = analyze_machine_health(machine_context)
    except Exception as exc:  # noqa: BLE001
        response_text = f"AI Assistant error: {exc}"

    return AssistantResponse(
        udi=request.udi,
        response=response_text,
        failure_probability=None,
    )


@app.get("/")
def root():
    """Basic health-check endpoint."""
    return {
        "status": "ok",
        "service": "Industrial Machine Health Intelligence Platform API",
    }