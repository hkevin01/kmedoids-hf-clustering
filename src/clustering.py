"""
# ID: MOD-CLUSTER-001
# Requirement: Cluster a float32 embedding matrix with K-Medoids and return
#              cluster labels plus a silhouette score.
# Purpose: Provide a reusable, configurable K-Medoids clustering step that
#          produces both cluster assignments and a quantitative quality metric.
# Rationale: K-Medoids is more robust to outliers than K-Means because cluster
#             centres are always actual data points (medoids). The cosine metric
#             is appropriate for normalised text embeddings.
# Inputs:
#   embeddings  - np.ndarray (N, D) float32 : embedding matrix.
#   n_clusters  - int  : number of clusters K.
#   metric      - str  : distance metric for KMedoids (e.g. 'cosine').
#   method      - str  : KMedoids algorithm variant ('alternate' or 'pam').
#   random_state- int  : RNG seed for reproducibility.
# Outputs:
#   labels      - np.ndarray (N,) int   : cluster index per sample.
#   silhouette  - float                 : silhouette score in [-1, +1].
#   model       - KMedoids              : fitted model instance.
# Preconditions: embeddings.shape[0] > n_clusters >= 2.
# Postconditions: len(labels) == N; -1 <= silhouette <= 1.
# Assumptions: Embeddings are L2-normalised for cosine metric.
# Failure Modes: ValueError if n_clusters >= N; MemoryError on very large N.
# Error Handling: Raises ValueError with diagnostic message.
# Verification: tests/test_clustering.py
# References: https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
"""

from __future__ import annotations

from typing import Tuple, Any

import numpy as np
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_kmedoids(
    embeddings: np.ndarray,
    n_clusters: int = 4,
    metric: str = "cosine",
    method: str = "alternate",
    random_state: int = 42,
) -> Tuple[np.ndarray, float, Any]:
    """
    # ID: FUNC-CLUSTER-001
    # Requirement: Fit K-Medoids on embeddings and return (labels, silhouette, model).
    # Purpose: Core clustering step called by main.py after embedding.
    # Inputs:
    #   embeddings   - shape (N, D) float32 embedding matrix.
    #   n_clusters   - number of clusters K (must satisfy 2 <= K < N).
    #   metric       - pairwise distance metric string accepted by KMedoids.
    #   method       - 'alternate' (fast) or 'pam' (exact).
    #   random_state - integer seed for reproducible initialisation.
    # Outputs:
    #   labels      - integer cluster index array of shape (N,).
    #   silhouette  - float score measuring cluster separation quality.
    #   model       - fitted KMedoids instance (exposes .medoid_indices_).
    # Preconditions: 2 <= n_clusters < embeddings.shape[0].
    # Postconditions: model.labels_ == labels.
    # Error Handling: Raises ValueError on invalid n_clusters.
    """
    # --- Input validation ---
    n_samples = embeddings.shape[0]
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}.")
    if n_clusters >= n_samples:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be < number of samples ({n_samples})."
        )

    # --- Fit K-Medoids ---
    from sklearn_extra.cluster import KMedoids  # deferred: avoids distutils on import
    print(
        f"[clustering] Running K-Medoids: k={n_clusters}, "
        f"metric={metric}, method={method} ..."
    )
    kmed = KMedoids(
        n_clusters=n_clusters,
        metric=metric,
        method=method,
        random_state=random_state,
    )
    kmed.fit(embeddings)
    labels: np.ndarray = kmed.labels_

    # --- Silhouette score (uses same metric) ---
    # Silhouette requires at least 2 distinct labels and >= 2 samples per cluster.
    sil_score: float = silhouette_score(embeddings, labels, metric=metric)
    print(f"[clustering] Silhouette score ({metric}): {sil_score:.4f}")
    print(f"[clustering] Medoid indices: {kmed.medoid_indices_.tolist()}")

    return labels, sil_score, kmed
