"""
# ID: MAIN-001
# Requirement: Provide a runnable end-to-end K-Medoids clustering pipeline
#              that loads a Hugging Face dataset, embeds it, clusters it,
#              evaluates the result, and saves visualisation plots.
# Purpose: Single entry-point demonstrating the full pipeline.
# Rationale: All heavy logic lives in src/; main.py only orchestrates the
#             steps and owns CLI-style configuration constants for easy tuning.
# Inputs:  None (configured via PIPELINE_CONFIG dict below).
# Outputs:
#   - Console summary with silhouette score and per-cluster sample counts.
#   - PNG plots saved to outputs/ folder.
# Preconditions: All requirements installed (pip install -r requirements.txt).
# Side Effects: HF dataset cache, model weight cache, outputs/ PNG files.
# Failure Modes: Network required for first run; raises on bad config values.
# Verification: Run `python main.py` and check outputs/ folder.
# References: src/data_loader.py, src/embedding.py, src/clustering.py,
#             src/visualize.py
"""

from __future__ import annotations

import os
from collections import Counter
from typing import List

import numpy as np

from src.data_loader import load_hf_dataset
from src.embedding import embed_texts
from src.clustering import run_kmedoids
from src.visualize import plot_clusters


# ---------------------------------------------------------------------------
# Pipeline Configuration
# Edit these values to try different datasets, models, or cluster counts.
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = {
    # ---- Data ----
    "dataset_name":  "ag_news",   # Any HF dataset slug
    "split":         "train",
    "text_column":   "text",
    "label_column":  "label",     # Set to None if no labels available
    "max_samples":   1000,        # Reduce for quick tests; increase for quality

    # ---- Embedding ----
    # Lightweight 384-d model; swap for 'all-mpnet-base-v2' for higher quality
    "model_name":    "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size":    64,
    "normalize":     True,        # L2 normalise - required for cosine metric

    # ---- Clustering ----
    "n_clusters":    4,           # ag_news has 4 ground-truth classes
    "metric":        "cosine",    # 'cosine' or 'euclidean'
    "method":        "alternate", # 'alternate' (fast) or 'pam' (slow/exact)
    "random_state":  42,

    # ---- Visualisation ----
    # Use 'umap' for better structure preservation (requires umap-learn)
    "viz_method":    "pca",
    "output_dir":    "outputs",
}


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def summarise_clusters(
    labels: np.ndarray,
    texts: List[str],
    ground_truth: List[int],
    medoid_indices: List[int],
    silhouette: float,
) -> None:
    """
    # ID: FUNC-SUMMARY-001
    # Purpose: Print a human-readable cluster summary to stdout.
    # Inputs:
    #   labels        - predicted cluster labels (N,).
    #   texts         - raw text samples (N,).
    #   ground_truth  - true integer labels (N,) or empty list.
    #   medoid_indices - row indices of medoids.
    #   silhouette    - silhouette score float.
    # Outputs: console output only.
    """
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"  Silhouette score : {silhouette:.4f}")
    print(f"  Cluster sizes    : {dict(sorted(Counter(labels.tolist()).items()))}")

    print("\n  --- Medoid Samples (cluster representatives) ---")
    for rank, idx in enumerate(medoid_indices):
        cluster_id = labels[idx]
        snippet = texts[idx][:100].replace("\n", " ")
        gt_info = f"  (true label={ground_truth[idx]})" if ground_truth else ""
        print(f"  [Cluster {cluster_id}] medoid idx={idx}{gt_info}")
        print(f"    '{snippet}...'")

    print("=" * 60 + "\n")


def main() -> None:
    """
    # ID: FUNC-MAIN-001
    # Requirement: Execute the full pipeline using PIPELINE_CONFIG.
    # Purpose: Orchestrate data loading, embedding, clustering, evaluation,
    #          and visualisation in sequence.
    # Preconditions: requirements.txt packages installed.
    # Postconditions: PNGs saved in outputs/; summary printed to stdout.
    """
    cfg = PIPELINE_CONFIG

    # ---- Step 1: Create output directory ----
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ---- Step 2: Load dataset ----
    texts, labels_gt = load_hf_dataset(
        dataset_name=cfg["dataset_name"],
        split=cfg["split"],
        text_column=cfg["text_column"],
        label_column=cfg["label_column"],
        max_samples=cfg["max_samples"],
    )

    # ---- Step 3: Embed text ----
    embeddings = embed_texts(
        texts=texts,
        model_name=cfg["model_name"],
        batch_size=cfg["batch_size"],
        normalize=cfg["normalize"],
    )

    # ---- Step 4: K-Medoids clustering ----
    cluster_labels, sil_score, kmed_model = run_kmedoids(
        embeddings=embeddings,
        n_clusters=cfg["n_clusters"],
        metric=cfg["metric"],
        method=cfg["method"],
        random_state=cfg["random_state"],
    )

    medoid_indices: List[int] = kmed_model.medoid_indices_.tolist()

    # ---- Step 5: Print summary ----
    summarise_clusters(
        labels=cluster_labels,
        texts=texts,
        ground_truth=labels_gt,
        medoid_indices=medoid_indices,
        silhouette=sil_score,
    )

    # ---- Step 6: PCA visualisation ----
    pca_path = os.path.join(cfg["output_dir"], "clusters_pca.png")
    plot_clusters(
        embeddings=embeddings,
        labels=cluster_labels,
        texts=texts,
        medoid_indices=medoid_indices,
        method="pca",
        output_path=pca_path,
        title=f"K-Medoids on {cfg['dataset_name']} (PCA)",
    )

    # ---- Step 7: UMAP visualisation (optional - falls back to PCA if missing) ----
    umap_path = os.path.join(cfg["output_dir"], "clusters_umap.png")
    plot_clusters(
        embeddings=embeddings,
        labels=cluster_labels,
        texts=texts,
        medoid_indices=medoid_indices,
        method="umap",
        output_path=umap_path,
        title=f"K-Medoids on {cfg['dataset_name']} (UMAP)",
    )

    print("[main] Pipeline complete.")
    print(f"[main] Plots saved to '{cfg['output_dir']}/'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
