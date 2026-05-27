"""
# ID: TEST-CLUSTER-001
# Purpose: Unit tests for src/clustering.py - validates K-Medoids on synthetic
#          data without requiring HF dataset or embedding model.
"""

import numpy as np
import pytest
from src.clustering import run_kmedoids


def make_blobs(n_per_cluster=20, n_clusters=3, n_dim=8, seed=0):
    """Generate clearly separated synthetic blobs."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(n_clusters):
        centre = np.zeros(n_dim)
        centre[k % n_dim] = 10.0 * (k + 1)
        blob = rng.normal(loc=centre, scale=0.5, size=(n_per_cluster, n_dim))
        parts.append(blob)
    X = np.vstack(parts).astype(np.float32)
    # L2 normalise so cosine metric works properly
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


class TestRunKMedoids:
    def test_basic_shape(self):
        X = make_blobs()
        labels, score, model = run_kmedoids(X, n_clusters=3)
        assert labels.shape == (X.shape[0],)
        assert len(model.medoid_indices_) == 3

    def test_silhouette_range(self):
        X = make_blobs()
        _, score, _ = run_kmedoids(X, n_clusters=3)
        assert -1.0 <= score <= 1.0

    def test_n_clusters_too_small(self):
        X = make_blobs()
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            run_kmedoids(X, n_clusters=1)

    def test_n_clusters_too_large(self):
        X = make_blobs(n_per_cluster=5, n_clusters=2)
        with pytest.raises(ValueError, match="must be < number of samples"):
            run_kmedoids(X, n_clusters=100)

    def test_euclidean_metric(self):
        X = make_blobs()
        labels, score, _ = run_kmedoids(X, n_clusters=3, metric="euclidean")
        assert labels.shape == (X.shape[0],)
