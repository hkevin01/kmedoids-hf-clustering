"""
# ID: MOD-DATA-001
# Requirement: Load a Hugging Face dataset and return a flat list of text samples
#              together with their ground-truth labels (when available).
# Purpose: Centralise all data-loading and light preprocessing so the rest of
#          the pipeline consumes a clean (texts, labels) tuple.
# Rationale: Keeping I/O separate from modelling logic makes unit-testing and
#             dataset swapping trivial.
# Inputs:
#   dataset_name  - str  : Hugging Face dataset identifier (e.g. "ag_news").
#   split         - str  : Dataset split to use (default "train").
#   text_column   - str  : Column that contains the raw text.
#   label_column  - str  : Column that contains integer class labels (or None).
#   max_samples   - int  : Maximum number of rows to load (keeps memory sane).
# Outputs:
#   texts  - list[str]  : Cleaned text samples.
#   labels - list[int]  : Corresponding integer labels (empty list if none).
# Preconditions: `datasets` package installed; internet access for first download.
# Postconditions: len(texts) == len(labels) when labels are present.
# Assumptions: Text column contains plain strings; labels are integers.
# Side Effects: Hugging Face cache written to ~/.cache/huggingface/datasets/.
# Failure Modes: ValueError if columns not found; ConnectionError on no network.
# Error Handling: Raises descriptive ValueError with available column names.
# Verification: tests/test_data_loader.py
# References: https://huggingface.co/docs/datasets
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    # ID: HELP-CLEAN-001
    # Purpose: Strip leading/trailing whitespace and collapse internal
    #          whitespace runs to a single space.
    # Inputs:  text - raw string
    # Outputs: normalised string
    # Failure Modes: Non-string input raises AttributeError (intentional - caller
    #                must ensure correct column is selected).
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hf_dataset(
    dataset_name: str = "ag_news",
    split: str = "train",
    text_column: str = "text",
    label_column: Optional[str] = "label",
    max_samples: int = 2000,
) -> Tuple[List[str], List[int]]:
    """
    # ID: FUNC-LOAD-001
    # Requirement: Return a (texts, labels) tuple for the requested HF dataset.
    # Purpose: Single entry-point for dataset acquisition used by main.py and
    #          the embedding module.
    # Inputs:
    #   dataset_name  - Hugging Face dataset slug.
    #   split         - Which split to load (train / test / validation).
    #   text_column   - Name of the column holding raw text.
    #   label_column  - Name of the integer label column, or None.
    #   max_samples   - Hard cap on rows to process (avoids OOM on large sets).
    # Outputs:
    #   texts  - list[str] of cleaned text samples (length <= max_samples).
    #   labels - list[int] of ground-truth labels (same length), or [] if None.
    # Preconditions: Network available for first download.
    # Postconditions: Each element of texts is a non-empty string.
    # Error Handling: Raises ValueError with column list on bad column name.
    """
    # --- Input validation ---
    if max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")

    # --- Load dataset ---
    from datasets import load_dataset  # deferred: not needed for unit tests
    print(f"[data_loader] Loading '{dataset_name}' (split='{split}') ...")
    dataset = load_dataset(dataset_name, split=split)

    # --- Validate columns ---
    available = dataset.column_names
    if text_column not in available:
        raise ValueError(
            f"text_column='{text_column}' not found. "
            f"Available columns: {available}"
        )
    if label_column is not None and label_column not in available:
        raise ValueError(
            f"label_column='{label_column}' not found. "
            f"Available columns: {available}"
        )

    # --- Subsample ---
    n = min(max_samples, len(dataset))
    dataset = dataset.select(range(n))
    print(f"[data_loader] Using {n} samples from '{dataset_name}'.")

    # --- Extract & clean text ---
    texts: List[str] = [_clean_text(str(row[text_column])) for row in dataset]

    # --- Extract labels (if present) ---
    labels: List[int] = []
    if label_column is not None:
        labels = [int(row[label_column]) for row in dataset]

    return texts, labels
