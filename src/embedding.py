"""
# ID: MOD-EMBED-001
# Requirement: Convert a list of text strings into a 2-D numpy array of
#              dense float32 embeddings using SentenceTransformers.
# Purpose: Decouple embedding logic from clustering so models can be swapped
#          without touching downstream code.
# Rationale: SentenceTransformers all-MiniLM-L6-v2 is fast (< 1 s / 1 000
#             sentences on CPU), lightweight (80 MB), and produces high-quality
#             384-d embeddings suitable for cosine-distance clustering.
# Inputs:
#   texts        - list[str] : Raw or lightly preprocessed text samples.
#   model_name   - str       : SentenceTransformer model identifier.
#   batch_size   - int       : Encoding batch size (trades memory for speed).
#   normalize    - bool      : L2-normalise embeddings (recommended for cosine).
# Outputs:
#   np.ndarray of shape (N, D) float32 - one embedding row per input text.
# Preconditions: texts is non-empty; model downloadable from HF Hub.
# Postconditions: Output rows are L2-normalised when normalize=True.
# Assumptions: CPU inference acceptable for <= 5 000 samples.
# Side Effects: Model weights cached in ~/.cache/torch/sentence_transformers/.
# Failure Modes: RuntimeError if CUDA OOM; OSError if model not found.
# Error Handling: Propagates library exceptions with context message.
# Verification: tests/test_embedding.py
# References: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
"""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    # ID: FUNC-EMBED-001
    # Requirement: Produce a float32 embedding matrix for the given text list.
    # Purpose: Uniform entry-point for the embedding step of the pipeline.
    # Inputs:
    #   texts      - list[str] of N samples (non-empty).
    #   model_name - HF model identifier or local path.
    #   batch_size - rows per forward pass through the encoder.
    #   normalize  - if True, embeddings are L2-normalised (unit vectors).
    # Outputs:
    #   np.ndarray shape (N, D) dtype float32.
    # Preconditions: len(texts) >= 1.
    # Postconditions: shape[0] == len(texts); dtype == float32.
    # Error Handling: Raises ValueError on empty input.
    """
    # --- Input validation ---
    if not texts:
        raise ValueError("embed_texts received an empty list of texts.")

    # --- Load model ---
    from sentence_transformers import SentenceTransformer  # deferred
    print(f"[embedding] Loading model '{model_name}' ...")
    model = SentenceTransformer(model_name)

    # --- Encode ---
    print(f"[embedding] Encoding {len(texts)} texts (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(np.float32)
    print(f"[embedding] Embedding matrix shape: {embeddings.shape}")

    return embeddings
