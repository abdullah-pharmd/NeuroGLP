"""
src/neuroglp/features/builder.py

Production-grade feature matrix builder for Vigi-Pheno pharmacovigilance platform.
Constructs high-dimensional binary presence-absence indicator matrices from patient
co-medication records, applies low-frequency noise pruning, and strictly enforces
zero-NaN and non-infinite mathematical invariants.
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from neuroglp.features.normalizer import (
    DOSAGE_UNITS,
    RE_COMBO_SPLIT,
    RE_DOSAGE,
    normalize_drug_name,
)

logger = logging.getLogger(__name__)

RE_UNIT_ONLY = re.compile(r"^\d*(?:\.\d+)?\s*(?:ml|l|hr|day|dose|actuation)\b", re.IGNORECASE)


def extract_patient_drugs(concomitant_drugs: Any) -> Set[str]:
    """
    Extracts and standardizes the set of active drug generic entities for a single patient report.
    Handles lists, tuples, sets, strings, stringified lists, None, and splits multi-active combination
    products into discrete active moieties.

    Parameters
    ----------
    concomitant_drugs : Any
        Raw concomitant medication field from patient record.

    Returns
    -------
    Set[str]
        Deduplicated set of standardized generic drug entities for the patient.
    """
    drugs_set: Set[str] = set()
    if concomitant_drugs is None or (isinstance(concomitant_drugs, float) and np.isnan(concomitant_drugs)):
        return drugs_set

    raw_list: List[str] = []
    if isinstance(concomitant_drugs, (list, tuple, set, np.ndarray)):
        for item in concomitant_drugs:
            if item is not None and not (isinstance(item, float) and np.isnan(item)):
                raw_list.append(str(item))
    elif isinstance(concomitant_drugs, str):
        s = concomitant_drugs.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    raw_list.extend(
                        str(x) for x in parsed
                        if x is not None and not (isinstance(x, float) and np.isnan(x))
                    )
                else:
                    raw_list.append(s)
            except Exception:
                raw_list.append(s)
        else:
            raw_list.append(s)
    else:
        raw_list.append(str(concomitant_drugs))

    for raw_d in raw_list:
        raw_str = str(raw_d).strip()
        if not raw_str:
            continue

        # Strip dosages first so compound units like 100 units/ml are not split across slash
        cleaned_raw = RE_DOSAGE.sub(" ", raw_str).strip()

        # Split combinations across /, \, and, &, +
        sub_items = RE_COMBO_SPLIT.split(cleaned_raw) if RE_COMBO_SPLIT.search(cleaned_raw) else [cleaned_raw]

        for sub in sub_items:
            sub_trimmed = sub.strip()
            if not sub_trimmed or RE_UNIT_ONLY.match(sub_trimmed):
                continue
            norm = normalize_drug_name(sub_trimmed)
            if not norm or norm in DOSAGE_UNITS:
                continue

            # If brand mapping itself resolved to a combination, split into constituent entities
            if " / " in norm or " and " in norm:
                for part in RE_COMBO_SPLIT.split(norm):
                    part_norm = normalize_drug_name(part.strip())
                    if part_norm and part_norm not in DOSAGE_UNITS:
                        drugs_set.add(part_norm)
            else:
                drugs_set.add(norm)

    return drugs_set


def validate_feature_matrix_invariants(
    matrix: np.ndarray,
    feature_names: List[str],
    clean_df: pd.DataFrame,
) -> bool:
    """
    Mathematically validates that the feature matrix strictly satisfies all Milestone 2 invariants.

    Parameters
    ----------
    matrix : np.ndarray
        Binary presence-absence feature matrix.
    feature_names : List[str]
        Standardized feature column names.
    clean_df : pd.DataFrame
        Aligned patient demographic and clinical DataFrame.

    Returns
    -------
    bool
        True if all invariants hold; raises AssertionError otherwise.
    """
    assert isinstance(matrix, np.ndarray), f"Expected np.ndarray, got {type(matrix)}"
    assert isinstance(feature_names, list), f"Expected list of feature names, got {type(feature_names)}"
    assert isinstance(clean_df, pd.DataFrame), f"Expected pd.DataFrame, got {type(clean_df)}"
    assert matrix.ndim == 2, f"Expected 2D matrix, got shape {matrix.shape}"
    assert matrix.shape[0] == len(clean_df), (
        f"Row alignment mismatch: matrix has {matrix.shape[0]} rows, clean_df has {len(clean_df)}"
    )
    assert matrix.shape[1] == len(feature_names), (
        f"Column alignment mismatch: matrix has {matrix.shape[1]} cols, feature_names has {len(feature_names)}"
    )
    assert matrix.dtype == np.uint8, f"Expected uint8 dtype, got {matrix.dtype}"

    # Strict Zero-NaN invariant
    nan_count = int(np.isnan(matrix).sum())
    assert nan_count == 0, f"STRICT INVARIANT VIOLATION: Found {nan_count} NaNs in feature matrix!"
    assert not np.isinf(matrix).any(), "STRICT INVARIANT VIOLATION: Found infinite values in feature matrix!"

    # Binary value invariant
    if matrix.size > 0:
        unique_vals = set(np.unique(matrix))
        assert unique_vals.issubset({0, 1}), f"Non-binary values detected in feature matrix: {unique_vals}"

    return True


def build_feature_matrix(
    df: pd.DataFrame,
    min_freq: float = 0.005,
    drug_column: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Builds a high-dimensional presence-absence binary feature matrix from patient co-medications.

    Satisfies M2 ↔ M3 interface contract:
    - Normalizes raw co-medication entities.
    - Discards noise drugs appearing in fewer than min_freq of patient reports.
    - Constructs an (N x D) uint8 indicator matrix.
    - Mathematically asserts zero NaN and non-infinite values.
    - Returns clean_df aligned 1:1 with matrix rows.

    Parameters
    ----------
    df : pd.DataFrame
        Structured patient DataFrame (e.g. from flatten_reports()).
    min_freq : float, default 0.005
        Minimum reporting frequency threshold (fraction of total cohort, e.g. 0.005 = 0.5%).
    drug_column : Optional[str], default None
        Column name containing concomitant drug lists. If None, auto-resolves.

    Returns
    -------
    Tuple[np.ndarray, List[str], pd.DataFrame]
        - feature_matrix: np.ndarray of dtype uint8 with shape (N, D).
        - feature_names: List[str] of length D with alphabetically sorted generic drug names.
        - clean_df: pd.DataFrame aligned 1:1 with feature_matrix rows.
    """
    clean_df = df.copy().reset_index(drop=True)
    total_patients = len(clean_df)

    # Boundary Case 1: Empty DataFrame
    if total_patients == 0:
        empty_matrix = np.zeros((0, 0), dtype=np.uint8)
        validate_feature_matrix_invariants(empty_matrix, [], clean_df)
        return empty_matrix, [], clean_df

    # Resolve drug column
    if drug_column is None:
        candidates = ["concomitant_drugs", "concomitant", "co_medications", "comedications", "drugs"]
        for candidate in candidates:
            if candidate in clean_df.columns:
                drug_column = candidate
                break
        if drug_column is None:
            lower_cols = {c.lower(): c for c in clean_df.columns}
            for candidate in candidates:
                if candidate in lower_cols:
                    drug_column = lower_cols[candidate]
                    break

    if drug_column is None or drug_column not in clean_df.columns:
        raise ValueError(
            f"Input DataFrame must contain 'concomitant_drugs' column. Found columns: {list(clean_df.columns)}"
        )

    # Extract patient drugs and compute global frequency counts
    patient_drugs_list: List[Set[str]] = []
    drug_counts: Counter[str] = Counter()

    for val in clean_df[drug_column]:
        p_drugs = extract_patient_drugs(val)
        patient_drugs_list.append(p_drugs)
        drug_counts.update(p_drugs)

    # Noise pruning filter (< min_freq)
    threshold_count = total_patients * float(min_freq)
    vocab = sorted([
        drug for drug, count in drug_counts.items()
        if count >= (threshold_count - 1e-9)
    ])

    # Boundary Case 2: No surviving drugs after pruning
    if len(vocab) == 0:
        empty_matrix = np.zeros((total_patients, 0), dtype=np.uint8)
        feature_names: List[str] = []
        validate_feature_matrix_invariants(empty_matrix, feature_names, clean_df)
        return empty_matrix, feature_names, clean_df

    # Filter patient drug sets to only retained vocabulary to suppress sklearn UserWarning
    vocab_set = set(vocab)
    filtered_drugs = [
        [d for d in p_drugs if d in vocab_set]
        for p_drugs in patient_drugs_list
    ]

    # Binary Presence/Absence indicator matrix via MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=vocab)
    matrix = mlb.fit_transform(filtered_drugs).astype(np.uint8)
    feature_names = list(mlb.classes_)

    # Strictly validate mathematical invariants before returning
    validate_feature_matrix_invariants(matrix, feature_names, clean_df)

    return matrix, feature_names, clean_df


def get_feature_frequencies(
    matrix: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Computes reporting frequency and prevalence statistics for each feature column.

    Parameters
    ----------
    matrix : np.ndarray
        Binary feature matrix (N x D).
    feature_names : List[str]
        List of feature names corresponding to matrix columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['drug', 'patient_count', 'prevalence_pct'] sorted descending.
    """
    if matrix.size == 0 or len(feature_names) == 0:
        return pd.DataFrame(columns=["drug", "patient_count", "prevalence_pct"])

    counts = matrix.sum(axis=0)
    total = matrix.shape[0]
    pcts = (counts / total) * 100.0

    df_freq = pd.DataFrame({
        "drug": feature_names,
        "patient_count": counts.astype(int),
        "prevalence_pct": np.round(pcts, 2),
    }).sort_values(by="patient_count", ascending=False).reset_index(drop=True)

    return df_freq
