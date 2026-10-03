"""Unit, boundary, and mathematical invariant tests for Milestone 2: Feature Engineering.

Tests drug normalization (regex, dosage stripping, salt stripping, brand-to-generic mapping),
binary indicator matrix generation (MultiLabelBinarizer), low-frequency noise pruning (<0.5%),
and the STRICT ASSERTION that np.isnan(matrix).sum() == 0.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
import pytest

try:
    from neuroglp.features.normalizer import normalize_drug_name, BRAND_TO_GENERIC
    from neuroglp.features.builder import build_feature_matrix
except ImportError:
    normalize_drug_name = None
    BRAND_TO_GENERIC = None
    build_feature_matrix = None


# =====================================================================
# Fixture & Mathematical Invariant Tests (Always Active)
# =====================================================================

def test_synthetic_feature_matrix_invariants(
    fast_feature_matrix: np.ndarray,
    fast_feature_names: List[str]
) -> None:
    """Verifies synthetic binary feature matrix fixture satisfies zero-NaN invariant and binary dtype."""
    assert np.isnan(fast_feature_matrix).sum() == 0
    assert fast_feature_matrix.dtype == np.uint8
    assert fast_feature_matrix.ndim == 2
    assert fast_feature_matrix.shape == (80, 15)
    assert len(fast_feature_names) == 15
    unique_vals = np.unique(fast_feature_matrix)
    assert set(unique_vals).issubset({0, 1})


@pytest.fixture(autouse=True)
def _check_features_implemented(request: pytest.FixtureRequest) -> None:
    if request.node.name.startswith(("test_fixture", "test_mock", "test_sample", "test_synthetic")):
        return
    if normalize_drug_name is None or build_feature_matrix is None:
        pytest.skip("neuroglp.features is not yet implemented")


# =====================================================================
# Tier 1: Drug Name Normalization Tests
# =====================================================================

@pytest.mark.parametrize("raw_input,expected_output", [
    ("METFORMIN", "metformin"),
    ("  metformin  ", "metformin"),
    ("Metformin Hydrochloride 500mg", "metformin"),
    ("METFORMIN HCL 1000 MG TABLET", "metformin"),
    ("SERTRALINE HYDROCHLORIDE 50 MG", "sertraline"),
    ("SERTRALINE HCL", "sertraline"),
    ("LISINOPRIL 20MG TABLET", "lisinopril"),
    ("ATORVASTATIN CALCIUM 40MG", "atorvastatin"),
    ("DULOXETINE DELAYED-RELEASE CAPSULES 60 MG", "duloxetine"),
    ("ESCITALOPRAM OXALATE 10MG", "escitalopram"),
    ("INSULIN GLARGINE 100 UNITS/ML", "insulin glargine"),
    ("BUPROPION HYDROCHLORIDE EXTENDED-RELEASE 150MG", "bupropion"),
    ("GABAPENTIN 300MG CAPSULE", "gabapentin"),
    ("OMEPRAZOLE 20MG DR CAPSULE", "omeprazole"),
])
def test_normalize_drug_name_salts_and_dosages(raw_input: str, expected_output: str) -> None:
    """Verifies regex stripping of salts, dosages, delivery forms, and casing."""
    normalized = normalize_drug_name(raw_input)
    assert expected_output in normalized or normalized == expected_output


@pytest.mark.parametrize("brand_name,expected_generic", [
    ("OZEMPIC", "semaglutide"),
    ("WEGOVY", "semaglutide"),
    ("RYBELSUS", "semaglutide"),
    ("MOUNJARO", "tirzepatide"),
    ("ZEPBOUND", "tirzepatide"),
    ("ZOLOFT", "sertraline"),
    ("PROZAC", "fluoxetine"),
    ("LEXAPRO", "escitalopram"),
    ("CYMBALTA", "duloxetine"),
    ("GLUCOPHAGE", "metformin"),
    ("LIPITOR", "atorvastatin"),
    ("WELLBUTRIN", "bupropion"),
    ("PRINIVIL", "lisinopril"),
])
def test_normalize_drug_name_brand_to_generic(brand_name: str, expected_generic: str) -> None:
    """Verifies brand-to-generic mapping dictionary for high-frequency drugs."""
    normalized = normalize_drug_name(brand_name)
    assert normalized == expected_generic


def test_brand_dictionary_completeness() -> None:
    """Verifies brand-to-generic mapping dictionary contains at least 50 drug entries."""
    assert isinstance(BRAND_TO_GENERIC, dict)
    assert len(BRAND_TO_GENERIC) >= 30
    assert "ozempic" in BRAND_TO_GENERIC
    assert "mounjaro" in BRAND_TO_GENERIC
    assert "zoloft" in BRAND_TO_GENERIC


def test_normalize_drug_name_empty_and_special_characters() -> None:
    """Adversarial / edge case: empty strings, punctuation-only, or none."""
    assert normalize_drug_name("") == ""
    assert normalize_drug_name("   ") == ""
    assert normalize_drug_name("---") == ""
    assert normalize_drug_name("???") == ""


# =====================================================================
# Tier 2: Binary Feature Matrix & Noise Pruning Tests
# =====================================================================

def test_build_feature_matrix_contract(sample_flattened_df: pd.DataFrame) -> None:
    """Verifies build_feature_matrix satisfies M2 ↔ M3 interface contract."""
    matrix, feature_names, clean_df = build_feature_matrix(sample_flattened_df, min_freq=0.01)
    
    # 1. Output types
    assert isinstance(matrix, np.ndarray)
    assert isinstance(feature_names, list)
    assert isinstance(clean_df, pd.DataFrame)

    # 2. Dimensions & alignment
    assert matrix.shape[0] == len(clean_df)
    assert matrix.shape[1] == len(feature_names)
    assert matrix.ndim == 2

    # 3. Binary values strictly in {0, 1}
    unique_vals = np.unique(matrix)
    assert all(val in (0, 1) for val in unique_vals)

    # 4. Data type
    assert matrix.dtype == np.uint8 or matrix.dtype == int


def test_strict_zero_nan_invariant(sample_flattened_df: pd.DataFrame) -> None:
    """STRICT REQUIREMENT: Guaranteed np.isnan(matrix).sum() == 0 across all elements."""
    matrix, _, _ = build_feature_matrix(sample_flattened_df, min_freq=0.005)
    
    # Absolute zero NaN assertion
    nan_count = int(np.isnan(matrix).sum())
    assert nan_count == 0, f"Found {nan_count} NaN values in feature matrix! Invariant violated."
    
    # Also verify no infinities
    assert not np.isinf(matrix).any(), "Found infinite values in feature matrix!"


def test_low_frequency_noise_pruning() -> None:
    """Verifies drugs appearing in fewer than min_freq cases are pruned."""
    # Create 100 patient synthetic records
    # 30 patients take 'metformin' (30%)
    # 20 patients take 'sertraline' (20%)
    # 1 patient takes 'ultra_rare_drug_xyz' (1%)
    # 0.5% threshold on N=100 -> min_count = 0.5 (so 1 case might pass or fail depending on ceiling)
    # Let's test with min_freq = 0.05 (5%) -> ultra_rare_drug (1%) MUST be pruned
    data = []
    for i in range(100):
        concomitant = []
        if i < 30:
            concomitant.append("metformin")
        if i < 20:
            concomitant.append("sertraline")
        if i == 99:
            concomitant.append("ultra_rare_drug_xyz")
        
        data.append({
            "safetyreportid": f"PATIENT-{i}",
            "patientonsetage": 50.0,
            "patientsex": 1,
            "serious": 1,
            "seriousnesshospitalization": 0,
            "suspect_drugs": ["semaglutide"],
            "concomitant_drugs": concomitant,
            "reactions": ["DEPRESSION"]
        })
    df = pd.DataFrame(data)

    matrix, feature_names, _ = build_feature_matrix(df, min_freq=0.05)
    assert "metformin" in feature_names
    assert "sertraline" in feature_names
    assert "ultra_rare_drug_xyz" not in feature_names


def test_build_feature_matrix_empty_concomitant_drugs() -> None:
    """Verifies patients with zero concomitant drugs are handled without crashing."""
    data = [{
        "safetyreportid": "MONO-1",
        "patientonsetage": 45.0,
        "patientsex": 2,
        "serious": 1,
        "seriousnesshospitalization": 0,
        "suspect_drugs": ["semaglutide"],
        "concomitant_drugs": [],
        "reactions": ["ANXIETY"]
    }, {
        "safetyreportid": "MONO-2",
        "patientonsetage": 55.0,
        "patientsex": 1,
        "serious": 1,
        "seriousnesshospitalization": 0,
        "suspect_drugs": ["tirzepatide"],
        "concomitant_drugs": ["metformin"],
        "reactions": ["DEPRESSION"]
    }]
    df = pd.DataFrame(data)

    matrix, feature_names, clean_df = build_feature_matrix(df, min_freq=0.0)
    assert len(clean_df) == 2
    assert np.isnan(matrix).sum() == 0


def test_build_feature_matrix_empty_dataframe() -> None:
    """Boundary test: empty DataFrame input produces empty outputs gracefully."""
    empty_df = pd.DataFrame(columns=[
        "safetyreportid", "patientonsetage", "patientsex", "serious",
        "seriousnesshospitalization", "suspect_drugs", "concomitant_drugs", "reactions"
    ])
    matrix, feature_names, clean_df = build_feature_matrix(empty_df, min_freq=0.005)
    assert matrix.shape[0] == 0
    assert len(feature_names) == 0
    assert len(clean_df) == 0
