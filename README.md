# Industrial Machine Health Intelligence Platform Using Multivariate SPC for Predictive Maintenance

> **MSc Computational Statistics & Applied AI**  
> **Course:** MCAI513B-3 Multivariate Techniques  
> **Assessment:** CIA 3 Individual Project

A local machine-health monitoring and predictive-maintenance intelligence platform using multivariate statistical techniques. The application combines a React/Vite frontend, FastAPI backend, Python statistical analysis engine, and Gemini-based AI explanation layer.

The platform analyzes multivariate machine sensor measurements to provide:

- Machine-health monitoring
- Hotelling's \(T^2\) multivariate anomaly monitoring
- Principal Component Analysis (PCA)
- MANOVA group comparison
- Statistical diagnostics and assumption checks
- Maintenance-oriented recommendations
- AI-assisted explanations of verified machine-health outputs

> **Important:** Hotelling's \(T^2\) identifies multivariate abnormality relative to a healthy Phase-I baseline. It is **not** a calibrated failure probability.

---

## Contents

- [Project Context](#project-context)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Statistical Diagnostics](#statistical-diagnostics)
- [Health Assessment](#health-assessment)
- [Maintenance Recommendation Layer](#maintenance-recommendation-layer)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Local Setup](#local-setup)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Academic Integrity and AI Disclosure](#academic-integrity-and-ai-disclosure)
- [Repository and Deployment Placeholders](#repository-and-deployment-placeholders)
- [References](#references)

---

## Project Context

Industrial systems generate multiple sensor measurements simultaneously. A univariate threshold on one sensor may fail to capture an abnormal operating pattern that appears only when several variables are considered together.

This project applies multivariate statistical process control to a predictive-maintenance dataset. It establishes a healthy operating baseline from Phase-I observations, then evaluates Phase-II observations using Hotelling's \(T^2\) statistic.

The platform includes both statistical analysis and a software layer for communicating results through a web dashboard and AI-assisted explanation interface.

---

## Research Question

> How can multivariate statistical techniques, including PCA, Hotelling's \(T^2\), MANOVA, and statistical diagnostics, support machine-health monitoring and maintenance decision support using multivariate sensor data?

The project investigates whether the combined pattern of five machine sensor variables differs from the established healthy baseline and whether observed failure and non-failure groups differ in their joint mean sensor profile.

---

## Dataset

### AI4I 2020 Predictive Maintenance Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- **Source:** [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc)
- **Dataset type:** Synthetic predictive-maintenance dataset
- **Observations:** 10,000
- **Columns:** 14
- **Missing values:** 0
- **Duplicate rows:** 0
- **Observed machine failures:** 339
- **Observed non-failures:** 9,661
- **Observed failure rate:** 3.39%

### Sensor Variables Used for Multivariate Monitoring

The following five continuous sensor/process variables are used in the multivariate monitoring pipeline:

| Variable | Description |
|---|---|
| `Air temperature [K]` | Air temperature in Kelvin |
| `Process temperature [K]` | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Rotational speed in revolutions per minute |
| `Torque [Nm]` | Torque in Newton metres |
| `Tool wear [min]` | Tool wear time in minutes |

The following fields are **not** used as Hotelling's \(T^2\) sensor predictors:

- `UDI`
- `Product ID`
- `Type`
- `Machine failure`
- Failure-mode flags: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`

`Machine failure` is retained as an observed outcome label for retrospective evaluation and MANOVA group comparison.

---

## Key Results

| Result | Verified value |
|---|---:|
| Healthy Phase-I observations | 6,722 |
| Phase-II observations | 3,000 |
| PCA cumulative variance through PC3 | 96.34% |
| Hotelling's \(T^2\) UCL | 11.0854 |
| Phase-II \(T^2\) alerts | 406 / 3,000 |
| Phase-II alert rate | 13.53% |
| \(T^2\) precision against observed failure labels | 10.59% |
| \(T^2\) recall against observed failure labels | 70.49% |
| \(T^2\) specificity against observed failure labels | 87.65% |
| \(T^2\) F1 score | 18.42% |
| MANOVA Wilks' lambda | 0.875719 |
| MANOVA F statistic | 283.6664 |
| MANOVA p-value | < 0.001 |
| Phase-I covariance condition number | 26.9323 |
| Phase-I covariance matrix | Positive definite |

---

## Methodology

### 1. Phase-I and Phase-II Split

Records are sorted by `UDI` in ascending order.

| Stage | Definition | Observations |
|---|---|---:|
| Full dataset | All available records | 10,000 |
| Phase I | First 70% of sorted observations | 7,000 |
| Healthy Phase-I baseline | Phase-I records with `Machine failure == 0` | 6,722 |
| Phase II | Remaining 30% of sorted observations | 3,000 |
| Phase-II failures | Observed `Machine failure == 1` | 61 |
| Phase-II non-failures | Observed `Machine failure == 0` | 2,939 |

The healthy Phase-I baseline is used to estimate the reference mean vector and covariance matrix.

### 2. Standardization

A `StandardScaler` is fitted **only** on the healthy Phase-I sensor observations.

```text
Fit scaler on healthy Phase-I sensor data
→ transform healthy Phase-I sensor data
→ apply the same fitted scaler to Phase-II sensor data
```

This prevents Phase-II observations, including observed failures, from influencing the healthy baseline scaling parameters.

### 3. Principal Component Analysis

PCA is applied to the standardized healthy Phase-I sensor variables.

| Principal component | Explained variance |
|---|---:|
| PC1 | 39.69% |
| PC2 | 36.66% |
| PC3 | 19.99% |
| Cumulative through PC3 | 96.34% |

Three principal components explain more than 90% of the variance in the standardized healthy baseline.

Interpretive descriptions of the PCA structure are:

- **PC1:** dominant thermal and operating-load variation pattern
- **PC2:** speed, torque, and temperature variation pattern
- **PC3:** predominantly tool-wear variation

These loading-based descriptions summarize covariance and variance structure. They do **not** establish causal effects.

### 4. Hotelling's \(T^2\) Monitoring

For an observation vector \(x\), healthy baseline mean vector \(\mu\), and Phase-I covariance matrix \(S\), Hotelling's statistic is:

\[
T^2 = (x - \mu)^T S^{-1}(x - \mu)
\]

The project uses:

| Parameter | Value |
|---|---:|
| Number of monitored variables \(p\) | 5 |
| Healthy Phase-I baseline size \(n\) | 6,722 |
| Significance level \(\alpha\) | 0.05 |
| Phase-II UCL | 11.0854 |

The Phase-II upper control limit is calculated from the classical \(F\)-distribution form:

\[
UCL =
\frac{p(n + 1)(n - 1)}
{n(n-p)}
F_{p,n-p;1-\alpha}
\]

A Phase-II observation is flagged when:

\[
T^2 > UCL
\]

### 5. T² Evaluation Against Observed Failure Labels

Hotelling's \(T^2\) alerts are compared retrospectively with the observed `Machine failure` labels in Phase II.

| Measure | Value |
|---|---:|
| True positives | 43 |
| False positives | 363 |
| True negatives | 2,576 |
| False negatives | 18 |
| Precision | 10.59% |
| Recall | 70.49% |
| Specificity | 87.65% |
| F1 score | 18.42% |
| Phase-II alerts | 406 / 3,000 |
| Alert rate | 13.53% |

The \(T^2\) monitor captures 70.49% of observed Phase-II failures, but precision is low. Therefore, it should be interpreted as a multivariate abnormality monitor rather than as a standalone binary failure classifier or calibrated failure-probability model.

### 6. MANOVA

MANOVA compares the joint mean vector of the five raw sensor variables between:

- `Machine failure = 0` — non-failure group
- `Machine failure = 1` — failure group

The model concept is:

\[
Y_1 + Y_2 + Y_3 + Y_4 + Y_5
\sim
\text{Machine failure}
\]

where the \(Y\) variables are the five sensor measurements.

| MANOVA result | Value |
|---|---:|
| Wilks' lambda | 0.875719 |
| F statistic | 283.6664 |
| Degrees of freedom | 5, 9994 |
| p-value | < 0.001 |

There is strong statistical evidence that the joint mean sensor vector differs between observed failure and non-failure groups.

This result does **not** prove that any sensor variable causes machine failure.

---

## Statistical Diagnostics

The project includes read-only assumption diagnostics for academic verification and reporting. These diagnostics do not remove observations, transform the dataset, regularize covariance matrices, modify the UCL, or change the core monitoring methodology.

### Data Quality

| Diagnostic | Result |
|---|---:|
| Rows | 10,000 |
| Columns | 14 |
| Missing values | 0 |
| Duplicate rows | 0 |

### Phase-I Covariance Diagnostics

Diagnostics are calculated from the standardized healthy Phase-I sensor observations.

| Diagnostic | Value |
|---|---:|
| Minimum eigenvalue | 0.073700 |
| Maximum eigenvalue | 1.984917 |
| Determinant | 0.02926574 |
| Condition number | 26.9323 |
| Positive definite | True |
| Numerically suitable for existing \(T^2\) calculation | True |

The covariance matrix is positive definite and numerically suitable for the existing Hotelling's \(T^2\) calculation.

### Multivariate Normality Diagnostics

The project retains a Mahalanobis-distance-versus-chi-square Q-Q correlation diagnostic and adds Mardia multivariate skewness and kurtosis diagnostics.

| Diagnostic | Value |
|---|---:|
| Mahalanobis/chi-square Q-Q correlation | 0.865879 |
| Mardia skewness \(b_{1,p}\) | 6.831525 |
| Skewness chi-square statistic | 7653.5855 |
| Skewness degrees of freedom | 35 |
| Mardia kurtosis \(b_{2,p}\) | 43.736944 |
| Normal-reference kurtosis \(p(p+2)\) | 35 |
| Skewness p-value | Effectively 0 |
| Kurtosis p-value | Effectively 0 |

Mardia diagnostics provide strong evidence of departure from multivariate normality in the standardized healthy Phase-I observations.

The classical multivariate-normal reference assumptions are therefore not well supported by these diagnostics. This is reported as a limitation for statistical interpretation; it does not automatically invalidate descriptive or monitoring analysis.

### MANOVA Covariance Homogeneity

Box's M assesses whether the covariance matrices are equal for the observed non-failure and failure groups.

| Box's M diagnostic | Value |
|---|---:|
| Non-failure observations | 9,661 |
| Failure observations | 339 |
| Box's M statistic | 1601.2963 |
| Corrected chi-square statistic | 1592.8640 |
| Degrees of freedom | 15 |
| p-value | Effectively 0 |

Box's M provides strong evidence that covariance homogeneity is not supported between the observed machine-failure groups.

### Levene/Brown–Forsythe Variance Diagnostics

Median-centered Levene/Brown–Forsythe tests provide supplementary univariate variance diagnostics.

| Sensor variable | p-value | Interpretation |
|---|---:|---|
| Air temperature [K] | 0.334863 | No significant evidence of unequal variance |
| Process temperature [K] | 0.000128 | Evidence of unequal variance |
| Rotational speed [rpm] | \(6.60 \times 10^{-15}\) | Evidence of unequal variance |
| Torque [Nm] | \(4.62 \times 10^{-34}\) | Evidence of unequal variance |
| Tool wear [min] | \(1.55 \times 10^{-6}\) | Evidence of unequal variance |

These diagnostics are limitation and validation evidence. They do not change the existing MANOVA model or the Hotelling's \(T^2\) monitoring pipeline.

---

## Health Assessment

Phase-II observations are placed into operational severity bands using the Hotelling's \(T^2\) UCL.

| Health status | Rule |
|---|---|
| `NORMAL` | \(T^2 \leq 0.75 \times UCL\) |
| `WATCH` | \(T^2 > 0.75 \times UCL\) and \(T^2 \leq UCL\) |
| `ALERT` | \(T^2 > UCL\) and \(T^2 \leq 1.5 \times UCL\) |
| `CRITICAL` | \(T^2 > 1.5 \times UCL\) |

### Phase-II Health Distribution

| Health status | Count | Percentage |
|---|---:|---:|
| Normal | 2,131 | 71.03% |
| Watch | 463 | 15.43% |
| Alert | 326 | 10.87% |
| Critical | 80 | 2.67% |
| Total | 3,000 | 100.00% |

These categories are operational severity bands for decision support. They are **not probabilities** and should not be interpreted as calibrated failure-risk classes.

---

## Maintenance Recommendation Layer

Maintenance recommendations are generated from:

- Operational health status
- Hotelling's \(T^2\) abnormality level
- Standardized sensor extremes relative to the healthy Phase-I baseline
- Direction of contributing sensor deviations: unusually high or unusually low

Examples of recommendation logic include:

| Sensor pattern | Example maintenance focus |
|---|---|
| High tool wear | Inspect tool condition and consider replacement |
| Unusual torque | Investigate mechanical loading or torque-related conditions |
| Unusual rotational speed | Verify operating conditions and speed-control behavior |
| Unusual temperatures | Inspect thermal or process operating conditions |
| High torque with low speed | Investigate focused mechanical loading or operating-condition issues |

These recommendations are investigation guidance based on statistical sensor abnormality. They do not confirm a physical fault and do not estimate calibrated failure probability.

---

## Architecture

```text
React/Vite Frontend
        ↓ Axios HTTP requests
FastAPI Backend
        ↓
Python Statistical Analysis Engine
        ↓
PCA / Hotelling's T² / MANOVA / Diagnostics / Maintenance Logic
        ↓
JSON API responses
        ↓
React Dashboard and AI Assistant
```

### Frontend

- React 18
- Vite
- TypeScript
- React Router
- Axios
- Recharts

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Statistical Engine

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- statsmodels
- matplotlib
- seaborn

### AI Assistant

- Google Gemini through the `google-genai` SDK
- Gemini API key loaded server-side from `agent/.env`
- API key must never be exposed in frontend source, browser code, API responses, or GitHub commits
- AI explanations are grounded in backend-generated statistical results

---

## Project Structure

```text
industrial-machine-health/
├── agent/
│   ├── gemini_agent.py
│   └── .env                  # Local only; ignored by Git
├── backend/
│   ├── main.py
│   ├── pipeline.py
│   └── schemas.py
├── data/
│   └── ai4i2020.csv
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── pca_analysis.py
│   ├── hotellings_t2.py
│   ├── t2_evaluation.py
│   ├── manova_analysis.py
│   ├── diagnostics.py
│   ├── health_assessment.py
│   └── maintenance_recommendation.py
├── app.py
├── package.json
├── package-lock.json
├── requirements.txt
└── .gitignore
```

The structure above documents the intended repository organization. Local virtual environments, Node dependency folders, build outputs, cache folders, logs, and `agent/.env` should remain excluded from Git.

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/overview` | Dataset-level KPIs, validation summary, and health distribution |
| `GET` | `/api/machines/{udi}` | Detailed health assessment for one Phase-II machine UDI |
| `GET` | `/api/spc` | Phase-II Hotelling's \(T^2\) SPC/control-chart data |
| `GET` | `/api/pca` | PCA variance structure, cumulative variance, loadings, and interpretation |
| `GET` | `/api/t2-evaluation` | T² alert evaluation against observed `Machine failure` labels |
| `GET` | `/api/manova` | MANOVA group sizes, descriptive statistics, mean differences, and multivariate tests |
| `GET` | `/api/diagnostics` | Covariance, T² distribution, Q-Q, Mardia, Box's M, and Levene/Brown–Forsythe diagnostics |
| `GET` | `/api/maintenance/{udi}` | Maintenance recommendation for one Phase-II machine |
| `POST` | `/api/assistant` | AI-assisted explanation for a selected machine and optional follow-up question |

---

## Local Setup

### Prerequisites

Install locally:

- Python
- Node.js and npm
- Git
- A Google Gemini API key for the optional AI assistant functionality

### Clone the Repository

```powershell
git clone <REPOSITORY_URL>
cd industrial-machine-health
```

Replace `<REPOSITORY_URL>` with the actual GitHub repository URL.

### Python Environment and Backend Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Gemini API Key

Create this local file:

```text
agent/.env
```

Add a placeholder/value in this format:

```text
GEMINI_API_KEY=your_key_here
```

Do not commit this file. It is intended to be ignored by `.gitignore`.

### Start the FastAPI Backend

The FastAPI application is defined in `backend/main.py`. The expected local development command is:

```powershell
uvicorn backend.main:app --reload
```

If your local project uses a root `app.py` wrapper or a different startup target, verify the correct local Uvicorn command before running it.

The backend is expected to run locally at:

```text
http://127.0.0.1:8000
```

### Start the React Frontend

Open a second PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite normally provides a local frontend URL such as:

```text
http://localhost:5173
```

---

## Reproducibility

The repository is structured to support local reproducibility through:

- Pinned direct Python dependencies in `requirements.txt`
- Frontend `package.json` and `package-lock.json`
- Source code for preprocessing, PCA, Hotelling's \(T^2\), MANOVA, diagnostics, health assessment, and maintenance logic
- Included AI4I 2020 dataset file
- Documented API structure
- Documented preprocessing decisions and verified summary results

Results depend on the specified preprocessing choice:

```text
Sort by UDI
→ first 70% of records as Phase I
→ retain Phase-I records where Machine failure == 0
→ fit StandardScaler on the healthy Phase-I baseline only
→ use remaining 30% as Phase II
```

Changing the split, baseline definition, sensor variables, scaling procedure, or control-limit configuration will change results.

---

## Limitations

- The AI4I 2020 dataset is synthetic and does not represent a live industrial deployment.
- Only five sensor/process variables are used, limiting machine-state observability.
- The Phase-I healthy baseline definition is a modeling choice and influences the reference mean, covariance, and T² control limit.
- Mardia diagnostics show evidence of departure from multivariate normality.
- Box's M indicates covariance homogeneity is not supported between observed failure and non-failure groups.
- Four of the five Levene/Brown–Forsythe sensor diagnostics indicate unequal group variances.
- Hotelling's \(T^2\) has low precision when retrospectively evaluated against observed failure labels.
- Hotelling's \(T^2\) is not a calibrated failure-probability model.
- The current application uses the provided historical dataset rather than real-time industrial sensor streaming.
- PCA and MANOVA findings describe statistical association and group difference, not causal relationships.
- AI assistant responses are explanations of available statistical outputs and should not replace physical inspection, engineering judgment, or site safety procedures.
- The project is a local academic implementation and is not presented as a production-ready industrial control system.

---

## Academic Integrity and AI Disclosure

> AI-assisted development tools were used during implementation for code generation, debugging, refactoring, and frontend/backend development. Statistical methodology, analytical decisions, validation, interpretation, and final project evaluation were independently reviewed and verified by the student.

The AI4I 2020 Predictive Maintenance Dataset is credited to the UCI Machine Learning Repository.

---

## Repository and Deployment Placeholders

| Item | Value |
|---|---|
| GitHub repository URL | `<REPOSITORY_URL>` |
| Deployed application URL | `<DEPLOYED_APPLICATION_URL>` |
| Student Developer Pack benefit/tool used | `<STUDENT_DEVELOPER_PACK_BENEFIT_OR_TOOL>` |

No deployment URL is claimed in this README. Replace placeholders only when a real repository URL, deployment URL, or verified student-developer benefit/tool is available.

---

## References

1. M. Matzka, “Explainable Artificial Intelligence for Predictive Maintenance Applications,” AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository. Available at: [https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc](https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenanc)

2. Hotelling, H. (1947). Multivariate Quality Control. In *Techniques of Statistical Analysis*. McGraw-Hill.

3. Jackson, J. E. (2005). *A User's Guide to Principal Components*. Wiley.

4. Rencher, A. C., & Christensen, W. F. (2012). *Methods of Multivariate Analysis* (3rd ed.). Wiley.

5. Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with applications. *Biometrika*, 57(3), 519–530.

6. Box, G. E. P. (1949). A general distribution theory for a class of likelihood criteria. *Biometrika*, 36(3/4), 317–346.