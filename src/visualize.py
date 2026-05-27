"""
# ID: MOD-VIZ-001
# Requirement: Produce 2-D scatter plots of cluster assignments using either
#              PCA or UMAP for dimensionality reduction.
# Purpose: Allow human inspection of cluster separation quality and save plots
#          to disk for reporting.
# Rationale: High-dimensional embeddings cannot be directly visualised; PCA is
#             fast and deterministic while UMAP preserves local neighbourhood
#             structure better for non-linear manifolds.
# Inputs:
#   embeddings   - np.ndarray (N, D) float32.
#   labels       - np.ndarray (N,) int  : cluster assignments.
#   texts        - list[str] of length N (used for medoid annotation).
#   medoid_indices - list[int]: indices of medoid samples.
#   method       - str : 'pca' or 'umap'.
#   output_path  - str : file path for the saved PNG (None = show interactively).
#   title        - str : plot title.
# Outputs:
#   Saves PNG at output_path; also returns the 2-D coordinate array.
# Preconditions: embeddings.shape[0] == len(labels).
# Postconditions: PNG written to output_path when specified.
# Assumptions: matplotlib backend supports writing PNG without a display when
#              output_path is set (uses Agg backend automatically).
# Side Effects: File I/O to output_path; optional UMAP import.
# Failure Modes: ImportError if umap-learn not installed and method='umap'.
# Error Handling: Falls back to PCA with a warning if UMAP import fails.
# Verification: tests/test_visualize.py
# References: https://umap-learn.readthedocs.io/en/latest/
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import matplotlib
import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reduce_to_2d(embeddings: np.ndarray, method: str) -> np.ndarray:
    """
    # ID: HELP-REDUCE-001
    # Purpose: Project N-D embeddings down to 2 dimensions for plotting.
    # Inputs:  embeddings - (N, D) array; method - 'pca' or 'umap'.
    # Outputs: (N, 2) float array.
    # Failure Modes: ImportError for UMAP handled by graceful PCA fallback.
    """
    method = method.lower()

    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, random_state=42)
            print("[visualize] Running UMAP dimensionality reduction ...")
            return reducer.fit_transform(embeddings)
        except ImportError:
            warnings.warn(
                "umap-learn not found; falling back to PCA. "
                "Install with: pip install umap-learn",
                stacklevel=3,
            )

    # PCA path (also fallback)
    print("[visualize] Running PCA dimensionality reduction ...")
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    medoid_indices: Optional[List[int]] = None,
    method: str = "pca",
    output_path: Optional[str] = None,
    title: str = "K-Medoids Clustering",
) -> np.ndarray:
    """
    # ID: FUNC-PLOT-001
    # Requirement: Render a 2-D scatter plot coloured by cluster label and save
    #              or display it.
    # Purpose: Primary visualisation function for the clustering pipeline.
    # Inputs:
    #   embeddings     - (N, D) float32 embedding matrix.
    #   labels         - (N,) int cluster assignment array.
    #   texts          - list[str] of raw text (used for medoid annotation).
    #   medoid_indices - list[int] of medoid row indices (marked with a star).
    #   method         - reduction method: 'pca' or 'umap'.
    #   output_path    - save path for PNG (None = interactive display).
    #   title          - figure title string.
    # Outputs:
    #   coords_2d - (N, 2) float array of projected coordinates.
    # Postconditions: PNG written to output_path when not None.
    # Error Handling: Falls back to Agg backend when no display is available.
    """
    # --- Use non-interactive backend when saving to file ---
    if output_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt  # imported after backend selection

    # --- Reduce dimensions ---
    coords_2d = _reduce_to_2d(embeddings, method)

    # --- Plot ---
    n_clusters = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=18,
        linewidths=0,
    )

    # --- Annotate medoids with a star and short text snippet ---
    if medoid_indices:
        for idx in medoid_indices:
            ax.scatter(
                coords_2d[idx, 0],
                coords_2d[idx, 1],
                marker="*",
                s=260,
                color="black",
                zorder=5,
                label="_nolegend_",
            )
            snippet = texts[idx][:40] + "..." if len(texts[idx]) > 40 else texts[idx]
            ax.annotate(
                snippet,
                (coords_2d[idx, 0], coords_2d[idx, 1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="black",
            )

    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{title}\n({method.upper()} projection, {n_clusters} clusters)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[visualize] Plot saved to '{output_path}'.")
    else:
        plt.show()

    plt.close(fig)
    return coords_2d
