"""
Gemini-based AI Maintenance Assistant.

This module is an EXPLANATION layer only. It does not calculate,
recalculate, or override any statistical value produced by the
existing Python pipeline (data_loader, preprocessing, pca_analysis,
hotellings_t2, t2_evaluation, manova_analysis, diagnostics,
health_assessment, maintenance_recommendation). All numeric outputs
(T^2, UCL, ratios, health status, contributing sensors, priority,
recommendations) are treated as authoritative, verified facts
supplied by the caller in `machine_context`.

Gemini's role is strictly to explain and contextualize those
already-computed results in plain language for a maintenance
audience. The assistant must not:
  - invent or restate numbers that were not provided in the context
  - claim T^2/UCL is a "failure probability"
  - claim any sensor "caused" a failure
  - assert that the machine "will fail" or "has failed"

Uses the current official Google GenAI Python SDK (`google-genai`),
not the deprecated `google-generativeai` package.
"""

import os
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; if unavailable, environment variables
    # must already be set in the shell/OS.
    pass

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

DEFAULT_MODEL: str = "gemini-2.5-flash"

SYSTEM_INSTRUCTION: str = """You are an AI Maintenance Assistant embedded in an industrial
machine-health monitoring platform built on multivariate Statistical Process Control (SPC).

AUTHORITATIVE SOURCE OF TRUTH:
All statistical values you are given (Hotelling's T-squared, the T-squared/UCL ratio,
UCL, health status, contributing/extreme sensors, PCA information, diagnostics, and the
maintenance recommendation/priority) were computed by a verified Python statistical
pipeline. You must treat these values as ground truth. Do NOT recalculate, estimate,
guess, or invent any numeric value that was not explicitly provided to you. If a value
is missing or unavailable, say so plainly instead of filling it in.

STRICT RULES YOU MUST FOLLOW:
1. Hotelling's T-squared (T^2) is a MULTIVARIATE ABNORMALITY SCORE relative to a healthy
   baseline. It is NOT a probability of failure. Never describe T^2, the T^2/UCL ratio,
   or any severity band (NORMAL/WATCH/ALERT/CRITICAL) as a "failure probability",
   "chance of failure", or any percentage likelihood of failure. No calibrated failure
   probability model exists in this system.
2. Never claim that a sensor "caused" the abnormal state or "caused" a failure. Use
   wording such as "contributing sensor pattern", "associated with the abnormal state",
   or "observed extreme value relative to the healthy baseline".
3. Never claim the machine "will fail", "has failed", or that a specific component
   (bearing, motor, tool, etc.) "is damaged" or "is overheating" unless that exact
   wording/finding was explicitly provided in the context. Prefer "investigate",
   "review", "inspect", "unusual", "abnormal multivariate state".
4. Do not fabricate sensor readings, UDI values, dates, thresholds, or statistics that
   were not present in the provided context.
5. Be concise, structured, and evidence-based. Ground every claim in the specific
   numbers and labels provided.

WHEN EXPLAINING A MACHINE'S HEALTH, ADDRESS:
- What is happening (the current health status and what it means operationally)
- Why the machine is considered abnormal or normal (T^2 vs UCL relationship)
- Which sensor patterns contribute to the abnormality (contributing/extreme sensors)
- What the statistical evidence indicates (referencing the actual provided values)
- What maintenance personnel should investigate next (based on the provided
  recommendation/priority)
- What information is uncertain, missing, or unavailable in the given context

Keep responses focused, professional, and appropriate for a maintenance engineer or
plant operator audience. Avoid speculative or alarmist language.
"""


class GeminiAgentError(Exception):
    """Raised for agent-level errors that should be shown to the user as text, not crash the app."""


def _get_api_key() -> Optional[str]:
    """
    Read the Gemini API key from the environment.

    Returns:
        The API key string, or None if not set.
    """
    return os.environ.get("GEMINI_API_KEY")


def _get_client() -> genai.Client:
    """
    Build a Gemini client using the API key from the environment.

    Returns:
        A configured genai.Client instance.

    Raises:
        GeminiAgentError: If the API key is missing.
    """
    api_key = _get_api_key()
    if not api_key:
        raise GeminiAgentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it in your .env file or system environment before using the AI assistant."
        )
    return genai.Client(api_key=api_key)


def _format_missing(value: Any) -> str:
    """Return a display-safe string for a possibly-missing value."""
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_contributing_sensors(contributing_sensors: Any) -> str:
    """
    Format contributing sensor information into a readable string.

    Args:
        contributing_sensors: Expected list of dicts with keys
            "sensor", "z_score", "direction", or None/empty if absent.

    Returns:
        A human-readable summary string.
    """
    if not contributing_sensors:
        return "None reported at or above the diagnostic threshold."

    lines = []
    for item in contributing_sensors:
        if not isinstance(item, dict):
            continue
        sensor = item.get("sensor", "unknown sensor")
        z_score = item.get("z_score")
        direction = item.get("direction", "")
        z_str = f"{z_score:.4f}" if isinstance(z_score, (int, float)) else "unavailable"
        lines.append(f"- {sensor}: standardized value = {z_str} ({direction})")
    return "\n".join(lines) if lines else "None reported at or above the diagnostic threshold."


def _build_context_block(machine_context: Dict[str, Any]) -> str:
    """
    Build a structured, plain-text context block from machine_context,
    tolerant of missing fields.

    Args:
        machine_context: Dictionary of verified statistical results.
            Expected keys (all optional) include: udi, t2_value, ucl,
            t2_ratio, health_status, priority, contributing_sensors,
            pca_interpretation, diagnostics, recommendation,
            primary_recommendation, additional_recommendations,
            evidence, explanation, limitations.

    Returns:
        A formatted string summarizing the available context.
    """
    ctx = machine_context or {}

    contributing = _format_contributing_sensors(ctx.get("contributing_sensors"))

    additional_recs = ctx.get("additional_recommendations")
    additional_recs_str = (
        "\n".join(f"- {r}" for r in additional_recs) if additional_recs else "None"
    )

    evidence = ctx.get("evidence")
    evidence_str = "\n".join(f"- {e}" for e in evidence) if evidence else "None provided"

    limitations = ctx.get("limitations")
    limitations_str = "\n".join(f"- {l}" for l in limitations) if limitations else "None provided"

    block = f"""VERIFIED MACHINE HEALTH CONTEXT (source of truth, computed by the statistical pipeline):

UDI: {_format_missing(ctx.get('udi'))}
Hotelling's T-squared (T2): {_format_missing(ctx.get('t2_value'))}
Established UCL: {_format_missing(ctx.get('ucl'))}
T2 / UCL ratio (abnormality ratio, NOT a probability): {_format_missing(ctx.get('t2_ratio'))}
Health status: {_format_missing(ctx.get('health_status'))}
Priority: {_format_missing(ctx.get('priority'))}

Contributing / extreme sensor signals (association only, not causation):
{contributing}

PCA interpretation notes: {_format_missing(ctx.get('pca_interpretation'))}

Additional diagnostics: {_format_missing(ctx.get('diagnostics'))}

Primary maintenance recommendation: {_format_missing(ctx.get('primary_recommendation') or ctx.get('recommendation'))}

Additional recommendations:
{additional_recs_str}

Evidence:
{evidence_str}

Stated limitations:
{limitations_str}
"""
    return block


def _call_gemini(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Call the Gemini API with the given prompt and system instruction,
    handling errors defensively.

    Args:
        prompt: The user-facing prompt content to send.
        model: The Gemini model name to use.

    Returns:
        The model's text response, or a clear error message string
        (never raises to the caller).
    """
    try:
        client = _get_client()
    except GeminiAgentError as exc:
        return f"AI Assistant unavailable: {exc}"

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
    except genai_errors.APIError as exc:
        return (
            "AI Assistant error: the Gemini API returned an error "
            f"({getattr(exc, 'code', 'unknown code')}). "
            "This may indicate an invalid API key, quota limit, or network issue. "
            f"Details: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 - defensive catch-all for local reliability
        return f"AI Assistant error: an unexpected error occurred while contacting Gemini ({exc})."

    text = getattr(response, "text", None)
    if not text or not text.strip():
        return (
            "AI Assistant returned an empty response. This can happen if the request "
            "was blocked by safety filters or the model produced no output. "
            "Please try rephrasing the question or check the machine context provided."
        )

    return text.strip()


def analyze_machine_health(machine_context: Dict[str, Any]) -> str:
    """
    Generate a plain-language explanation of a machine's current
    health status using verified statistical results as the sole
    source of truth.

    Args:
        machine_context: Dictionary of verified statistical results
            (see _build_context_block for expected keys). Missing
            fields are handled gracefully.

    Returns:
        A explanatory text response from the AI assistant, or a
        clear error message string if the request could not be
        completed. Never raises an exception.
    """
    if not isinstance(machine_context, dict):
        return "AI Assistant error: machine_context must be a dictionary of verified statistical results."

    context_block = _build_context_block(machine_context)

    prompt = f"""{context_block}

TASK:
Using only the verified information above, explain this machine's current health
status to a maintenance engineer. Structure your answer to address:
1. What is happening (current status in plain language)
2. Why the machine is considered abnormal or normal
3. Which sensor patterns contribute to the abnormality (if any)
4. What the statistical evidence indicates
5. What maintenance personnel should investigate next
6. What information is uncertain or unavailable

Do not introduce any numeric value not present in the context above.
"""
    return _call_gemini(prompt)


def answer_followup(question: str, machine_context: Dict[str, Any]) -> str:
    """
    Answer a follow-up question about a machine's health, grounded
    strictly in the verified statistical context provided.

    Args:
        question: The user's follow-up question.
        machine_context: Dictionary of verified statistical results
            (see _build_context_block for expected keys).

    Returns:
        A text response from the AI assistant, or a clear error
        message string if the request could not be completed. Never
        raises an exception.
    """
    if not question or not question.strip():
        return "Please provide a question for the AI assistant to answer."

    if not isinstance(machine_context, dict):
        return "AI Assistant error: machine_context must be a dictionary of verified statistical results."

    context_block = _build_context_block(machine_context)

    prompt = f"""{context_block}

MAINTENANCE PERSONNEL QUESTION:
{question.strip()}

TASK:
Answer the question above using only the verified information provided in the context.
If the answer requires information not present in the context, clearly state that the
information is unavailable rather than guessing or fabricating it. Do not describe T2
or the T2/UCL ratio as a failure probability, and do not claim any sensor caused a
failure.
"""
    return _call_gemini(prompt)


if __name__ == "__main__":
    print("=" * 60)
    print("GEMINI AGENT CONNECTION TEST")
    print("=" * 60)

    api_key = _get_api_key()
    if not api_key:
        print("\nGEMINI_API_KEY is not set. Set it in a .env file or your environment, then re-run.")
    else:
        print(f"\nGEMINI_API_KEY detected (length={len(api_key)}). Testing connection...")

        sample_context = {
            "udi": 7012,
            "t2_value": 83.78,
            "ucl": 11.0854,
            "t2_ratio": 7.5577,
            "health_status": "CRITICAL",
            "priority": "HIGH",
            "contributing_sensors": [
                {"sensor": "Rotational speed [rpm]", "z_score": 6.4507, "direction": "HIGH"},
                {"sensor": "Torque [Nm]", "z_score": -2.8823, "direction": "LOW"},
            ],
            "pca_interpretation": "Not supplied in this test.",
            "diagnostics": "Not supplied in this test.",
            "primary_recommendation": "Prioritize prompt maintenance investigation of the abnormal multivariate state.",
            "additional_recommendations": [
                "Inspect rotational-speed operating conditions and verify that the machine is operating within its intended range.",
                "Investigate a high-load/low-speed operating pattern and inspect mechanical operating conditions.",
            ],
            "evidence": [
                "T2 = 83.7800 exceeds the established UCL of 11.0854.",
                "Rotational speed [rpm] standardized value = 6.4507, exceeding the +/-2 SD diagnostic threshold.",
            ],
            "limitations": [
                "The recommendation is based on statistical sensor abnormality and does not confirm a physical fault.",
            ],
        }

        print("\n--- analyze_machine_health() ---\n")
        print(analyze_machine_health(sample_context))

        print("\n--- answer_followup() ---\n")
        print(answer_followup("What should the maintenance team check first?", sample_context))

    print("\n" + "=" * 60)