"""
Principal Component Analysis on the standardized Phase-I healthy
sensor baseline.

PCA is fitted exclusively on Phase-I healthy, standardized sensor
data (produced by preprocessing.prepare_phase_data). This module
only computes PCA structure (variance, loadings, scores) and exposes
plotting helpers; it does not compute Hotelling's T^2, control
limits, or any monitoring statistics.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from src.data_loader import load_dataset
from src.preprocessing import PhaseData, prepare_phase_data


@dataclass
class PCAResult:
    """Container for PCA outputs computed on the Phase-I healthy baseline."""

    pca: PCA
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    loadings: pd.DataFrame
    phase1_scores: pd.DataFrame
    sensor_columns: List[str]


def run_pca(phase_data: PhaseData) -> PCAResult:
    """
    Fit PCA on the standardized Phase-I healthy sensor data and
    build a structured result.

    Args:
        phase_data: A ``PhaseData`` object from
            ``preprocessing.prepare_phase_data()``.

    Returns:
        A ``PCAResult`` containing the fitted PCA object, variance
        statistics, loadings, and Phase-I principal component scores.
    """
    sensor_columns = phase_data.sensor_columns
    x_phase1 = phase_data.phase1_healthy[sensor_columns].values

    n_components = len(sensor_columns)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(x_phase1)

    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    pc_names = [f"PC{i + 1}" for i in range(n_components)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=sensor_columns,
        columns=pc_names,
    )

    phase1_scores = pd.DataFrame(scores, columns=pc_names)
    phase1_scores.insert(0, "UDI", phase_data.phase1_healthy["UDI"].values)

    return PCAResult(
        pca=pca,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        loadings=loadings,
        phase1_scores=phase1_scores,
        sensor_columns=sensor_columns,
    )


def components_for_variance_threshold(
    pca_result: PCAResult, threshold: float = 0.90
) -> int:
    """
    Determine the minimum number of principal components required to
    reach at least the given cumulative explained variance threshold.

    Args:
        pca_result: The ``PCAResult`` from ``run_pca()``.
        threshold: Target cumulative explained variance (default 0.90).

    Returns:
        The smallest number of components whose cumulative explained
        variance is >= threshold.
    """
    cumulative = pca_result.cumulative_explained_variance
    n_components = int(np.argmax(cumulative >= threshold) + 1)
    return n_components


def plot_scree(pca_result: PCAResult) -> plt.Figure:
    """
    Build a scree plot (explained variance per component).

    Args:
        pca_result: The ``PCAResult`` from ``run_pca()``.

    Returns:
        A matplotlib Figure. Not displayed or saved automatically.
    """
    pc_labels = [f"PC{i + 1}" for i in range(len(pca_result.explained_variance_ratio))]

    fig, ax = plt.subplots()
    ax.plot(pc_labels, pca_result.explained_variance_ratio, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Scree Plot")
    return fig


def plot_cumulative_variance(pca_result: PCAResult) -> plt.Figure:
    """
    Build a cumulative explained variance plot.

    Args:
        pca_result: The ``PCAResult`` from ``run_pca()``.

    Returns:
        A matplotlib Figure. Not displayed or saved automatically.
    """
    pc_labels = [f"PC{i + 1}" for i in range(len(pca_result.cumulative_explained_variance))]

    fig, ax = plt.subplots()
    ax.plot(pc_labels, pca_result.cumulative_explained_variance, marker="o")
    ax.axhline(y=0.90, color="red", linestyle="--", label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Explained Variance Plot")
    ax.legend()
    return fig


def plot_pc1_pc2_scores(pca_result: PCAResult) -> plt.Figure:
    """
    Build a PC1 vs PC2 score scatter plot using Phase-I healthy
    observations only.

    Args:
        pca_result: The ``PCAResult`` from ``run_pca()``.

    Returns:
        A matplotlib Figure. Not displayed or saved automatically.
    """
    fig, ax = plt.subplots()
    ax.scatter(
        pca_result.phase1_scores["PC1"],
        pca_result.phase1_scores["PC2"],
        alpha=0.6,
        s=15,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Phase-I Healthy: PC1 vs PC2 Scores")
    return fig


if __name__ == "__main__":
    raw_df = load_dataset()
    prepared = prepare_phase_data(raw_df)
    pca_result = run_pca(prepared)

    print("=" * 60)
    print("PCA RESULTS (Phase-I Healthy Baseline)")
    print("=" * 60)

    print("\nExplained variance ratio (PC1-PC5):")
    for i, ratio in enumerate(pca_result.explained_variance_ratio, start=1):
        print(f"  PC{i}: {ratio:.4f}")

    print("\nCumulative explained variance:")
    for i, cum in enumerate(pca_result.cumulative_explained_variance, start=1):
        print(f"  PC{i}: {cum:.4f}")

    n_required = components_for_variance_threshold(pca_result, threshold=0.90)
    print(f"\nNumber of PCs required to reach >= 90% cumulative variance: {n_required}")

    print("\nLoading matrix (rows = sensors, columns = PCs):")
    print(pca_result.loadings)