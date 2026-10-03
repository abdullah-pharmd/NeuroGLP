"""
tests/test_builder_stress.py

Adversarial Stress Test Suite for Milestone 2: Feature Matrix Builder.
Authored by challenger_m2_2 (Feature Matrix Adversarial Challenger).

Covers:
1. Boundary Cohorts: N=0, N=1, 100% empty co-medications, D=0 (all pruned).
2. Extreme min_freq values: 0.0, 0.5, 1.0, 1.5, negative min_freq (-0.1).
3. Hostile inputs: pd.NA, np.nan, float literals, malformed AST strings,
   null bytes, emojis, dirty mixed types in list, SQL/XSS injections.
4. Duplicate drugs per patient (deduplication & indicator boundedness).
5. Scale & Memory: N=10,000 patients with polypharmacy benchmark.
6. Invariant verification across all stress outputs.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from neuroglp.features.builder import (
    build_feature_matrix,
    extract_patient_drugs,
    get_feature_frequencies,
    validate_feature_matrix_invariants,
)
from neuroglp.features.normalizer import normalize_drug_name


# =====================================================================
# 1. Boundary Cohorts (N=0, N=1, 100% Empty Co-meds, D=0)
# =====================================================================

def test_stress_empty_cohort_n0() -> None:
    """Stress test: N=0 patients."""
    df = pd.DataFrame(columns=["concomitant_drugs", "safetyreportid", "patientonsetage"])
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.005)

    assert matrix.shape == (0, 0)
    assert feature_names == []
    assert len(clean_df) == 0
    assert matrix.dtype == np.uint8
    assert np.isnan(matrix).sum() == 0
    assert not np.isinf(matrix).any()


def test_stress_single_patient_with_drugs_n1() -> None:
    """Stress test: N=1 patient with valid drugs."""
    df = pd.DataFrame([{
        "safetyreportid": "P1",
        "concomitant_drugs": ["metformin 500mg", "zoloft 50mg"]
    }])
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.0)

    assert matrix.shape == (1, 2)
    assert feature_names == ["metformin", "sertraline"]
    assert len(clean_df) == 1
    assert matrix.dtype == np.uint8
    assert np.all(matrix == 1)
    assert np.isnan(matrix).sum() == 0


def test_stress_single_patient_without_drugs_n1() -> None:
    """Stress test: N=1 patient with zero co-medications."""
    df = pd.DataFrame([{
        "safetyreportid": "P1",
        "concomitant_drugs": []
    }])
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.005)

    assert matrix.shape == (1, 0)
    assert feature_names == []
    assert len(clean_df) == 1
    assert matrix.dtype == np.uint8
    assert np.isnan(matrix).sum() == 0


def test_stress_cohort_100pct_zero_comedications() -> None:
    """Stress test: N=150 patients where 100% have empty or None co-meds."""
    data = []
    for i in range(150):
        concomitant = [] if i % 2 == 0 else None
        data.append({
            "safetyreportid": f"P-{i}",
            "concomitant_drugs": concomitant,
            "patientonsetage": float(30 + i % 40),
        })
    df = pd.DataFrame(data)

    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.005)

    assert matrix.shape == (150, 0)
    assert feature_names == []
    assert len(clean_df) == 150
    assert matrix.dtype == np.uint8
    assert np.isnan(matrix).sum() == 0
    assert not np.isinf(matrix).any()


def test_stress_all_unique_drugs_pruned_d0() -> None:
    """Stress test: N=500 patients, each taking 1 unique rare drug, min_freq=0.02 (D=0)."""
    data = [{
        "safetyreportid": f"P-{i}",
        "concomitant_drugs": [f"experimental_drug_{i}"]
    } for i in range(500)]
    df = pd.DataFrame(data)

    # 2% threshold requires 10 occurrences; each has 1 -> all pruned
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.02)

    assert matrix.shape == (500, 0)
    assert feature_names == []
    assert len(clean_df) == 500
    assert matrix.dtype == np.uint8
    assert np.isnan(matrix).sum() == 0


# =====================================================================
# 2. Extreme min_freq Values (0.0, 0.5, 1.0, 1.5, negative)
# =====================================================================

@pytest.mark.parametrize("min_freq,expected_drugs", [
    (0.0, ["atorvastatin", "metformin", "sertraline"]),  # Keeps all drugs >= 1 case
    (0.5, ["metformin", "sertraline"]),                  # Keeps drugs in >= 50% cases
    (1.0, ["metformin"]),                                # Keeps drugs in 100% cases
    (1.5, []),                                           # Impossible threshold -> D=0
    (-0.1, ["atorvastatin", "metformin", "sertraline"]), # Negative threshold -> keeps all
])
def test_stress_extreme_min_freq(min_freq: float, expected_drugs: list[str]) -> None:
    """Stress test: Extreme frequency thresholds on controlled cohort."""
    # 4 patients:
    # P0: metformin, sertraline, atorvastatin
    # P1: metformin, sertraline
    # P2: metformin
    # P3: metformin
    # Metformin: 4/4 (100%)
    # Sertraline: 2/4 (50%)
    # Atorvastatin: 1/4 (25%)
    data = [
        {"safetyreportid": "P0", "concomitant_drugs": ["metformin", "sertraline", "atorvastatin"]},
        {"safetyreportid": "P1", "concomitant_drugs": ["metformin", "sertraline"]},
        {"safetyreportid": "P2", "concomitant_drugs": ["metformin"]},
        {"safetyreportid": "P3", "concomitant_drugs": ["metformin"]},
    ]
    df = pd.DataFrame(data)

    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=min_freq)

    assert feature_names == expected_drugs
    assert matrix.shape == (4, len(expected_drugs))
    assert len(clean_df) == 4
    assert matrix.dtype == np.uint8
    assert np.isnan(matrix).sum() == 0
    if matrix.size > 0:
        assert set(np.unique(matrix)).issubset({0, 1})


# =====================================================================
# 3. Hostile DataFrame Inputs & Dirty Types
# =====================================================================

def test_stress_hostile_concomitant_types() -> None:
    """Stress test: Dirty types in concomitant column (pd.NA, np.nan, float, int, malformed AST)."""
    data = [
        # 1. Normal list
        {"safetyreportid": "P1", "concomitant_drugs": ["metformin 500mg"]},
        # 2. pd.NA literal
        {"safetyreportid": "P2", "concomitant_drugs": pd.NA},
        # 3. np.nan float
        {"safetyreportid": "P3", "concomitant_drugs": np.nan},
        # 4. None literal
        {"safetyreportid": "P4", "concomitant_drugs": None},
        # 5. Stringified list (valid Python AST)
        {"safetyreportid": "P5", "concomitant_drugs": "['metformin', 'zoloft']"},
        # 6. JSON stringified list (double quotes)
        {"safetyreportid": "P6", "concomitant_drugs": '["metformin", "lipitor"]'},
        # 7. Malformed unclosed bracket string
        {"safetyreportid": "P7", "concomitant_drugs": "['metformin', 'incomplete"},
        # 8. Unquoted bracket string
        {"safetyreportid": "P8", "concomitant_drugs": "[metformin, sertraline]"},
        # 9. List containing dirty nested primitives
        {"safetyreportid": "P9", "concomitant_drugs": [None, np.nan, pd.NA, 12345, True, False]},
        # 10. String with null byte, newlines, tabs, SQL injection, XSS
        {"safetyreportid": "P10", "concomitant_drugs": ["\u0000", "' OR 1=1 --", "<script>alert(1)</script>", "  \n\t  "]},
        # 11. Large emoji repetitions and whitespace
        {"safetyreportid": "P11", "concomitant_drugs": ["🔥" * 50, "   "]},
        # 12. Non-list scalar string (single drug without brackets)
        {"safetyreportid": "P12", "concomitant_drugs": "metformin"},
    ]
    df = pd.DataFrame(data)

    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.0)

    # Must not crash, must satisfy all invariants
    assert len(clean_df) == 12
    assert matrix.shape[0] == 12
    assert matrix.shape[1] == len(feature_names)
    assert matrix.dtype == np.uint8
    assert np.isnan(matrix).sum() == 0
    assert not np.isinf(matrix).any()
    if matrix.size > 0:
        assert set(np.unique(matrix)).issubset({0, 1})

    # 'metformin' was provided in P1, P5, P6, P12 (and possibly P8 depending on fallback)
    assert "metformin" in feature_names


def test_stress_patient_drug_deduplication() -> None:
    """Stress test: A patient reporting identical or synonymous drug multiple times."""
    # P1 has metformin 3 times under different salt/brand/dose representations
    df = pd.DataFrame([{
        "safetyreportid": "P1",
        "concomitant_drugs": [
            "METFORMIN HCL 500MG",
            "Glucophage 1000mg",
            "metformin hydrochloride",
            "METFORMIN",
        ]
    }])
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.0)

    assert feature_names == ["metformin"]
    assert matrix.shape == (1, 1)
    # Binary indicator must be strictly 1, never > 1 despite 4 mentions
    assert matrix[0, 0] == 1


def test_stress_non_standard_dataframe_index() -> None:
    """Stress test: Discontinuous, string, or multi-index DataFrame retains 1:1 alignment."""
    data = [
        {"safetyreportid": "P100", "concomitant_drugs": ["metformin"]},
        {"safetyreportid": "P200", "concomitant_drugs": ["sertraline"]},
        {"safetyreportid": "P300", "concomitant_drugs": ["duloxetine"]},
    ]
    df = pd.DataFrame(data, index=[100, 200, 300])

    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.0)

    assert len(clean_df) == 3
    assert list(clean_df.index) == [0, 1, 2]  # Reset index guarantee
    assert clean_df.iloc[0]["safetyreportid"] == "P100"
    assert clean_df.iloc[1]["safetyreportid"] == "P200"
    assert clean_df.iloc[2]["safetyreportid"] == "P300"
    # Row 0 must correspond to metformin only
    metformin_col = feature_names.index("metformin")
    assert matrix[0, metformin_col] == 1
    assert matrix[1, metformin_col] == 0


def test_stress_auto_discovery_of_column_names() -> None:
    """Stress test: Auto-discovery of alternative column names for co-medications."""
    for col_name in ["concomitant", "co_medications", "comedications", "drugs", "CONCOMITANT_DRUGS"]:
        df = pd.DataFrame({
            "safetyreportid": ["P1", "P2"],
            col_name: [["metformin"], ["lisinopril"]]
        })
        matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.0)
        assert len(feature_names) == 2
        assert matrix.shape == (2, 2)


def test_stress_missing_concomitant_column_error() -> None:
    """Stress test: DataFrame missing any recognizable drug column raises ValueError."""
    df = pd.DataFrame({
        "safetyreportid": ["P1"],
        "unrelated_column": [123]
    })
    with pytest.raises(ValueError, match="Input DataFrame must contain 'concomitant_drugs'"):
        build_feature_matrix(df)


# =====================================================================
# 4. Large-Scale Synthetic Cohort Benchmark (N=10,000)
# =====================================================================

def test_stress_large_scale_cohort_n10000() -> None:
    """
    Stress test: N=10,000 patients with synthetic polypharmacy distribution.
    Verifies:
    1. Execution time is under 10 seconds.
    2. Memory efficiency: uint8 matrix representation.
    3. Invariants: zero-NaN, non-inf, binary {0, 1}, row alignment.
    4. Frequency computation helper accuracy.
    """
    np.random.seed(42)
    n_patients = 10000
    drug_pool = [
        "metformin", "sertraline", "atorvastatin", "lisinopril", "duloxetine",
        "escitalopram", "bupropion", "gabapentin", "omeprazole", "levothyroxine",
        "amlodipine", "losartan", "fluoxetine", "citalopram", "quetiapine",
        "aripiprazole", "mirtazapine", "trazodone", "venlafaxine", "lamotrigine"
    ]
    # Add 50 rare noise drugs
    rare_drugs = [f"rare_compound_{k}" for k in range(50)]

    records = []
    for i in range(n_patients):
        # 10% patients have no comedications (monotherapy)
        if i % 10 == 0:
            comed = []
        else:
            # 1-6 common drugs
            num_common = np.random.randint(1, 6)
            chosen = list(np.random.choice(drug_pool, size=num_common, replace=False))
            # 5% chance of taking a rare noise drug
            if np.random.rand() < 0.05:
                chosen.append(np.random.choice(rare_drugs))
            comed = chosen

        records.append({
            "safetyreportid": f"PATIENT-{i:06d}",
            "patientonsetage": float(np.random.randint(20, 85)),
            "patientsex": int(np.random.choice([1, 2])),
            "concomitant_drugs": comed,
        })

    df = pd.DataFrame(records)

    start_time = time.perf_counter()
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.005)
    elapsed = time.perf_counter() - start_time

    # Performance Gate
    assert elapsed < 10.0, f"Performance bottleneck: build_feature_matrix took {elapsed:.2f}s on N=10,000"

    # Shape and alignment
    assert matrix.shape[0] == n_patients
    assert len(clean_df) == n_patients
    assert matrix.shape[1] == len(feature_names)
    assert matrix.dtype == np.uint8

    # Memory Gate: 10,000 x D uint8 matrix should be < 5 MB
    matrix_mb = matrix.nbytes / (1024 * 1024)
    assert matrix_mb < 5.0, f"Excessive memory: matrix is {matrix_mb:.2f} MB"

    # Invariants
    assert np.isnan(matrix).sum() == 0
    assert not np.isinf(matrix).any()
    unique_vals = set(np.unique(matrix))
    assert unique_vals.issubset({0, 1})

    # Frequency analysis helper verification
    freq_df = get_feature_frequencies(matrix, feature_names)
    assert len(freq_df) == len(feature_names)
    assert list(freq_df.columns) == ["drug", "patient_count", "prevalence_pct"]
    assert freq_df["patient_count"].max() <= n_patients
    assert freq_df["prevalence_pct"].min() >= 0.5  # min_freq is 0.005 (0.5%)


def test_stress_curated_benchmark_dataset() -> None:
    """
    Stress test: Evaluates build_feature_matrix on the full real-world
    curated OpenFDA GLP-1 benchmark dataset (6,680 records).
    """
    import gzip
    import json
    from pathlib import Path
    from neuroglp.data.parser import flatten_reports

    candidates = [
        Path("src/neuroglp/data/curated_glp1_psychiatric.json.gz"),
        Path("data/curated_glp1_psychiatric.json.gz"),
        Path(__file__).resolve().parent.parent / "src" / "neuroglp" / "data" / "curated_glp1_psychiatric.json.gz",
    ]
    gz_path = next((p for p in candidates if p.exists()), None)
    if gz_path is None:
        pytest.skip("curated_glp1_psychiatric.json.gz not found")

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("results", data)
    df = flatten_reports(records)

    start_time = time.perf_counter()
    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.005)
    elapsed = time.perf_counter() - start_time

    assert elapsed < 5.0, f"Benchmark execution too slow: {elapsed:.2f}s"
    assert matrix.shape[0] == 6680
    assert len(clean_df) == 6680
    assert len(feature_names) == 150
    assert matrix.shape[1] == 150
    assert matrix.dtype == np.uint8

    # Strict invariant validation
    validate_feature_matrix_invariants(matrix, feature_names, clean_df)

    # Invariants
    assert np.isnan(matrix).sum() == 0
    assert not np.isinf(matrix).any()
    unique_vals = set(np.unique(matrix))
    assert unique_vals.issubset({0, 1})

    # Validate top clinical co-medications
    freq_df = get_feature_frequencies(matrix, feature_names)
    top_drugs = list(freq_df["drug"].head(5))
    assert "metformin" in top_drugs
    assert "atorvastatin" in top_drugs

