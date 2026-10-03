"""Adversarial stress-testing suite for normalizer.py.

Evaluates normalize_drug_name and normalize_drug_list under hostile, dirty,
boundary, and malformed inputs. Authored by challenger_m2_1.
"""

from __future__ import annotations

import time
from typing import Any, List
import numpy as np
import pandas as pd
import pytest

from neuroglp.features.normalizer import (
    BRAND_TO_GENERIC,
    DOSAGE_UNITS,
    RE_COMBO_SPLIT,
    RE_DOSAGE,
    RE_FORMS,
    RE_RELEASE,
    RE_SALTS,
    get_brand_mapping,
    is_known_drug,
    normalize_drug_list,
    normalize_drug_name,
)


# =====================================================================
# Category A: Null, Undefined, NaN & Empty States
# =====================================================================

@pytest.mark.parametrize("null_input", [
    None,
    np.nan,
    float("nan"),
    pd.NA,
    "",
    "   ",
    "\t\r\n",
    "      \n\t   ",
])
def test_category_a_null_and_empty_scalar_inputs(null_input: Any) -> None:
    """Verifies normalize_drug_name gracefully handles all null, NaN, and empty string types."""
    res = normalize_drug_name(null_input)
    assert res == "", f"Expected empty string for null input {repr(null_input)}, got {repr(res)}"
    assert isinstance(res, str)


@pytest.mark.parametrize("empty_collection", [
    None,
    [],
    (),
    set(),
    [None],
    ["", "   "],
    [None, ""],
])
def test_category_a_null_and_empty_list_inputs(empty_collection: Any) -> None:
    """Verifies normalize_drug_list returns empty list when given null or empty collections."""
    res = normalize_drug_list(empty_collection)
    assert res == [], f"Expected [] for {repr(empty_collection)}, got {repr(res)}"
    assert isinstance(res, list)


def test_category_a_nan_leakage_in_drug_list() -> None:
    """
    CRITICAL FINDING DEFECT TEST:
    Demonstrates whether np.nan, float('nan'), or pd.NA inside a drug list
    leak as literal string entities ('nan', '<na>') into normalized output.
    """
    mixed_list = ["metformin 500mg", np.nan, None, ""]
    res = normalize_drug_list(mixed_list)
    assert "metformin" in res
    # Empirically verifies the NaN leakage defect: 'nan' should not be present
    assert "nan" not in res and "<na>" not in res, (
        f"DEFECT CONFIRMED: np.nan leaked into normalized output as a drug entity! Got: {res}"
    )


# =====================================================================
# Category B: Primitive Boundary Limits & Pure Punctuation
# =====================================================================

@pytest.mark.parametrize("punct_input", [
    "---",
    "???",
    ".,;:",
    "///",
    "\\\\\\",
    "((()))",
    "[[[]]]",
    "{{{}}}",
    "\"\"\"'''",
    "!@#$%^&*()_+=-~`<>|\\",
    "  - - . , ; ? !  ",
])
def test_category_b_pure_punctuation_yields_empty_string(punct_input: str) -> None:
    """Verifies strings consisting solely of punctuation characters yield empty strings without crashing."""
    res = normalize_drug_name(punct_input)
    assert res == "", f"Expected empty string for pure punctuation {repr(punct_input)}, got {repr(res)}"


@pytest.mark.parametrize("nested_input,expected_drug", [
    ("((metformin [hcl] {500mg}))", "metformin"),
    ("[[[OZEMPIC]]]", "semaglutide"),
    ("\"\"\"zoloft\"\"\"", "sertraline"),
    ("<<prozac>>", "fluoxetine"),  # Cleaned of <> and mapped to generic fluoxetine
    ("((  metformin  ))", "metformin"),
])
def test_category_b_nested_brackets_and_punctuation(nested_input: str, expected_drug: str) -> None:
    """Verifies deeply nested brackets and quotations are stripped around valid drug entities."""
    res = normalize_drug_name(nested_input)
    assert expected_drug in res or res == BRAND_TO_GENERIC.get(expected_drug, expected_drug), (
        f"Expected {expected_drug} in normalized result {repr(res)}"
    )


# =====================================================================
# Category C: Casing, Dosages, Salts & Formulations
# =====================================================================

@pytest.mark.parametrize("mixed_case_input,expected_drug", [
    ("MeTfOrMiN hCl 500 MG", "metformin"),
    ("oZeMpIc", "semaglutide"),
    ("zOlOfT 50 Mg TaBlEt", "sertraline"),
    ("WeGoVy InJeCtIoN", "semaglutide"),
    ("lUpIn-MeTfOrMiN hYdRoChLoRiDe", "lupin metformin"),
])
def test_category_c_mixed_capitalization(mixed_case_input: str, expected_drug: str) -> None:
    """Verifies mixed and erratic casing is normalized to clean lowercase generic entities."""
    res = normalize_drug_name(mixed_case_input)
    assert res == expected_drug, f"Expected {expected_drug}, got {repr(res)}"


@pytest.mark.parametrize("bizarre_dose_input,expected_drug", [
    ("semaglutide 0.005 mcg / 0.1 ml", "semaglutide"),
    ("metformin .5mg", "metformin"),
    ("INSULIN GLARGINE 100 UNITS/ML", "insulin glargine"),
    ("fentanyl 25 mcg/hr patch", "fentanyl"),
    ("albuterol 90 mcg/actuation inhaler", "albuterol"),
    ("potassium chloride 20 meq", ""),  # Both are in RE_SALTS
    ("vitamin d3 50000 iu capsule", "vitamin d3"),
])
def test_category_c_bizarre_and_fractional_dosages(bizarre_dose_input: str, expected_drug: str) -> None:
    """Verifies bizarre dosages (micrograms, units/ml, mcg/hr, fractional) are stripped."""
    res = normalize_drug_name(bizarre_dose_input)
    assert res == expected_drug, f"Expected {expected_drug}, got {repr(res)}"


def test_category_c_multiple_consecutive_salts() -> None:
    """Verifies multiple consecutive salt modifiers are stripped without leaving residuals."""
    res = normalize_drug_name("metformin hydrochloride sodium tartrate maleate 500mg")
    assert res == "metformin", f"Expected 'metformin', got {repr(res)}"

    # Pure salts without active ingredient
    res_salts_only = normalize_drug_name("hydrochloride sodium tartrate maleate")
    assert res_salts_only == "", f"Expected empty string for pure salts, got {repr(res_salts_only)}"


# =====================================================================
# Category C: Malformed Strings, Injections & Unicode
# =====================================================================

@pytest.mark.parametrize("injection_payload", [
    "metformin'; DROP TABLE patients; --",
    "1 OR 1=1",
    "<script>alert('xss')</script>metformin",
    "metformin\x00hcl 500mg",
    "metformin\n\thcl\r\n500mg",
    "met\u200bformin 500mg",
])
def test_category_c_injection_and_malformed_strings(injection_payload: str) -> None:
    """Verifies SQL, XSS, null bytes, and control characters do not crash the engine."""
    res = normalize_drug_name(injection_payload)
    assert isinstance(res, str)
    # Ensure null byte does not cause termination or crash
    assert "\x00" not in res or isinstance(res, str)


@pytest.mark.parametrize("unicode_input,expected_substring", [
    ("semaglutide 500 \u03bcg", "semaglutide"),   # Greek small mu
    ("semaglutide 500 \u00b5g", "semaglutide"),   # Micro sign
    ("caf\u00e9ine 100mg", "caf\u00e9ine"),       # Latin accented e
    ("ibuprofeno espa\u00f1ol", "ibuprofeno"),    # Spanish n with tilde
    ("\U0001f48a metformin 500mg \U0001f48a", "metformin"), # Pill emoji
])
def test_category_c_unicode_and_diacritics(unicode_input: str, expected_substring: str) -> None:
    """Verifies international characters, symbols, and emojis do not crash normalization."""
    res = normalize_drug_name(unicode_input)
    assert isinstance(res, str)
    assert expected_substring in res, f"Expected {expected_substring} in {repr(res)}"


# =====================================================================
# Category C & D: Combinations & normalize_drug_list
# =====================================================================

@pytest.mark.parametrize("combo_input,expected_entities", [
    (["metformin 500mg / sitagliptin 50mg"], ["metformin", "sitagliptin"]),
    (["aspirin and dipyridamole"], ["aspirin", "dipyridamole"]),
    (["drug1 + drug2"], ["drug1", "drug2"]),
    (["drug1 \\ drug2"], ["drug1", "drug2"]),
    (["drug1 / drug2 and drug3 + drug4"], ["drug1", "drug2", "drug3", "drug4"]),
    (["JANUMET"], ["metformin", "sitagliptin"]),
    (["ADDERALL"], ["amphetamine", "dextroamphetamine"]),
    (["SYNJARDY"], ["empagliflozin", "metformin"]),
    (["ADVAIR"], ["fluticasone", "salmeterol"]),
    (["metformin 500mg", "metformin hcl 1000mg", "glucophage"], ["metformin"]),
])
def test_category_c_combinations_and_deduplication(
    combo_input: List[str], expected_entities: List[str]
) -> None:
    """Verifies combination products are split into constituent entities and deduplicated."""
    res = normalize_drug_list(combo_input)
    assert res == expected_entities, f"Expected {expected_entities}, got {res}"


def test_category_c_combo_without_spaces_defect() -> None:
    """
    DEFECT DEMONSTRATION:
    Tests whether combination separator '+' without surrounding spaces ('drug1+drug2')
    is properly split into separate drugs.
    """
    res = normalize_drug_list(["drug1+drug2"])
    assert res == ["drug1", "drug2"], f"DEFECT CONFIRMED: '+' without spaces was not split! Got: {res}"


@pytest.mark.parametrize("iter_input", [
    (d for d in ["ozempic", "zoloft"]),
    np.array(["ozempic", "zoloft"]),
    pd.Series(["ozempic", "zoloft"]),
    {"ozempic", "zoloft"},
])
def test_category_d_various_iterable_types(iter_input: Any) -> None:
    """Verifies normalize_drug_list accepts generators, numpy arrays, sets, and pandas Series."""
    res = normalize_drug_list(iter_input)
    assert set(res) == {"semaglutide", "sertraline"}


@pytest.mark.parametrize("non_iter", [
    12345,
    123.45,
    float("nan"),
])
def test_category_d_non_iterable_inputs_return_empty_list(non_iter: Any) -> None:
    """Verifies non-iterable scalar inputs return empty list without raising TypeError."""
    res = normalize_drug_list(non_iter)
    assert res == []


# =====================================================================
# Stress & ReDoS / Catastrophic Backtracking Performance
# =====================================================================

def test_stress_and_redos_performance() -> None:
    """Verifies that pathological regex inputs do not cause ReDoS and complete in < 1 second."""
    long_string = "metformin " * 1000  # 10,000 characters
    pathological_regex = "a" * 5000 + " 500mg"
    pathological_slashes = "/ " * 2000
    large_list = ["metformin 500mg"] * 10000

    t0 = time.perf_counter()
    res1 = normalize_drug_name(long_string)
    res2 = normalize_drug_name(pathological_regex)
    res3 = normalize_drug_name(pathological_slashes)
    res4 = normalize_drug_list(large_list)
    total_time = time.perf_counter() - t0

    assert "metformin" in res1
    assert res3 == ""
    assert res4 == ["metformin"]
    assert total_time < 1.0, f"Stress test exceeded 1.0s limit: {total_time:.3f}s"
