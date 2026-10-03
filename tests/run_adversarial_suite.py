"""
Adversarial stress-test runner for Vigi-Pheno normalizer.
Runs an extensive battery of hostile, dirty, malformed, and boundary inputs.
Outputs detailed results and logs any unhandled exceptions or data corruption.
"""

import sys
import time
import unicodedata
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, "src")

from neuroglp.features.normalizer import (
    normalize_drug_name,
    normalize_drug_list,
    is_known_drug,
    get_brand_mapping,
    BRAND_TO_GENERIC,
    DOSAGE_UNITS,
    RE_DOSAGE,
    RE_SALTS,
    RE_FORMS,
    RE_RELEASE,
    RE_COMBO_SPLIT,
)


def run_tests():
    print("=" * 80)
    print("VIGI-PHENO NORMALIZER ADVERSARIAL STRESS TEST SUITE")
    print("=" * 80)

    results = {
        "passed": 0,
        "failed": 0,
        "exceptions": [],
        "findings": [],
    }

    # -------------------------------------------------------------------------
    # Battery 1: Null, None, NaN, Primitive Types (Category A & D)
    # -------------------------------------------------------------------------
    print("\n--- BATTERY 1: NULL, NONE, NAN, AND DYNAMIC TYPES ---")
    null_cases = [
        ("None literal", None, ""),
        ("np.nan float", np.nan, ""),
        ("float('nan')", float("nan"), ""),
        ("pd.NA", pd.NA, ""),
        ("Empty string", "", ""),
        ("Whitespace string", "   \t\r\n   ", ""),
        ("Numeric zero int", 0, ""),
        ("Numeric zero float", 0.0, ""),
        ("Integer 500", 500, ""),
        ("Float 123.45", 123.45, ""),
        ("Float inf", float("inf"), "inf"),
        ("Float -inf", float("-inf"), "inf"),
        ("Boolean True", True, "true"),
        ("Boolean False", False, "false"),
        ("Empty list []", [], ""),
        ("Empty dict {}", {}, ""),
        ("Arbitrary object", object(), ""),
    ]

    for label, val, expected in null_cases:
        try:
            res = normalize_drug_name(val)
            # Must return string and never raise exception
            assert isinstance(res, str), f"Result must be str, got {type(res)}"
            print(f"  [OK] {label:25s}: input={repr(val):18s} -> output={repr(res):10s}")
            results["passed"] += 1
        except Exception as e:
            print(f"  [FAIL] {label:25s}: input={repr(val)} -> EXCEPTION: {e}")
            results["failed"] += 1
            results["exceptions"].append((label, val, str(e)))

    # -------------------------------------------------------------------------
    # Battery 2: Pure Punctuation & Special Characters (Category B)
    # -------------------------------------------------------------------------
    print("\n--- BATTERY 2: PURE PUNCTUATION AND SYMBOLS ---")
    punct_cases = [
        ("Dashes", "---"),
        ("Questions", "???"),
        ("Colons/dots/commas", ".,;:"),
        ("Slashes", "///"),
        ("Backslashes", "\\\\\\"),
        ("Parentheses", "((()))"),
        ("Brackets", "[[[]]]"),
        ("Braces", "{{{}}}"),
        ("Quotes", "\"\"\"'''"),
        ("Punctuation soup", "!?@#$%^&*()_+=-~`<>|\\"),
        ("Mixed spaces & punct", "  - - . , ; ? !  "),
    ]

    for label, val in punct_cases:
        try:
            res = normalize_drug_name(val)
            assert res == "", f"Expected empty string for pure punct, got {repr(res)}"
            print(f"  [OK] {label:25s}: input={repr(val):18s} -> output={repr(res):10s}")
            results["passed"] += 1
        except Exception as e:
            print(f"  [FAIL] {label:25s}: input={repr(val)} -> {e}")
            results["failed"] += 1
            results["exceptions"].append((label, val, str(e)))

    # -------------------------------------------------------------------------
    # Battery 3: Mixed Capitalization, Salts, Dosages (Category C)
    # -------------------------------------------------------------------------
    print("\n--- BATTERY 3: MIXED CASE, SALTS, DOSAGES, FORMS ---")
    clinical_cases = [
        ("Mixed case salt/dose", "MeTfOrMiN hCl 500 MG", "metformin"),
        ("All caps brand", "OZEMPIC", "semaglutide"),
        ("Alternating caps brand", "oZeMpIc", "semaglutide"),
        ("Nested punct drug", "((metformin [hcl] {500mg}))", "metformin"),
        ("Punctuation prefix/suffix", "***METFORMIN HCL 500MG***", "metformin"),
        ("Micro dosage decimal", "semaglutide 0.005 mcg / 0.1 ml", "semaglutide"),
        ("Decimal without leading zero", "metformin .5mg", "metformin"),
        ("Insulin compound units", "INSULIN GLARGINE 100 UNITS/ML", "insulin glargine"),
        ("IU compound units", "ergocalciferol 50,000 iu", "ergocalciferol"),
        ("Fentanyl patch dose/hr", "Fentanyl 25 mcg/hr patch", "fentanyl"),
        ("Albuterol actuation", "albuterol 90 mcg/actuation inhaler", "albuterol"),
        ("Multiple consecutive salts", "metformin hydrochloride sodium tartrate maleate 500mg", "metformin"),
        ("Multiple salts no drug", "hydrochloride sodium tartrate maleate", ""),
        ("Salt name as brand", "depakote er", "divalproex sodium"),
        ("Brand with salt in generic", "depakene", "valproic acid"),
        ("Brand with salt in brand", "klor-con", None),  # might not be in brand dict
        ("Electrolyte salt: potassium chloride", "potassium chloride 20 meq", ""), # Note: both salts
        ("Electrolyte salt: calcium carbonate", "calcium carbonate 500 mg", ""), # Note: both salts
        ("Electrolyte salt: sodium bicarbonate", "sodium bicarbonate 650 mg", ""), # Note: both salts
    ]

    for label, val, expected in clinical_cases:
        try:
            res = normalize_drug_name(val)
            print(f"  [RES] {label:32s}: {repr(val):45s} -> {repr(res):20s}")
            if expected is not None:
                if res == expected:
                    results["passed"] += 1
                else:
                    print(f"        -> DISCREPANCY: expected {repr(expected)}, got {repr(res)}")
                    results["findings"].append((label, val, f"Expected {expected}, got {res}"))
            else:
                results["passed"] += 1
        except Exception as e:
            print(f"  [FAIL] {label:32s}: input={repr(val)} -> {e}")
            results["failed"] += 1
            results["exceptions"].append((label, val, str(e)))

    # -------------------------------------------------------------------------
    # Battery 4: Unicode, Diacritics, Emojis, Injection Payloads
    # -------------------------------------------------------------------------
    print("\n--- BATTERY 4: UNICODE, DIACRITICS, EMOJIS, INJECTIONS ---")
    hostile_cases = [
        ("SQL Injection 1", "metformin'; DROP TABLE patients; --"),
        ("SQL Injection 2", "1 OR 1=1"),
        ("XSS Payload", "<script>alert('xss')</script>metformin"),
        ("Null byte injection", "metformin\x00hcl 500mg"),
        ("Newline & Tab injection", "metformin\n\thcl\r\n500mg"),
        ("Zero-width space", "met\u200bformin 500mg"),
        ("Greek small mu (U+03BC)", "semaglutide 500 \u03bcg"),
        ("Micro sign (U+00B5)", "semaglutide 500 \u00b5g"),
        ("Accented french (caféine)", "caféine 100mg"),
        ("Spanish tilde (español)", "ibuprofeno español"),
        ("Fullwidth ASCII (ＭＥＴＦＯＲＭＩＮ)", "\uff2d\uff25\uff34\uff26\uff2f\uff32\uff2d\uff29\uff2e"),
        ("Emoji sandwich", "💊 metformin 500mg 💊"),
        ("Right-to-left mark", "\u200fmetformin 500mg\u200f"),
        ("Unbalanced quotes", "metformin\" 500mg"),
        ("HTML entities", "&lt;metformin&gt;"),
    ]

    for label, val in hostile_cases:
        try:
            res = normalize_drug_name(val)
            print(f"  [RES] {label:28s}: {repr(val):40s} -> {repr(res):20s}")
            assert isinstance(res, str)
            results["passed"] += 1
        except Exception as e:
            print(f"  [FAIL] {label:28s}: input={repr(val)} -> {e}")
            results["failed"] += 1
            results["exceptions"].append((label, val, str(e)))

    # -------------------------------------------------------------------------
    # Battery 5: Combinations & normalize_drug_list
    # -------------------------------------------------------------------------
    print("\n--- BATTERY 5: COMBINATIONS & NORMALIZE_DRUG_LIST ---")
    list_cases = [
        ("None input", None, []),
        ("Empty list", [], []),
        ("List of empty & None", [None, "", "   ", np.nan], []),
        ("Single valid drug", ["METFORMIN 500 MG"], ["metformin"]),
        ("Deduplication identical", ["METFORMIN 500 MG", "metformin hcl 1000mg", "glucophage"], ["metformin"]),
        ("Slash combo with doses", ["metformin 500mg / sitagliptin 50mg"], ["metformin", "sitagliptin"]),
        ("And combo", ["aspirin and dipyridamole"], ["aspirin", "dipyridamole"]),
        ("Plus combo with spaces", ["drug1 + drug2"], ["drug1", "drug2"]),
        ("Plus combo without spaces", ["drug1+drug2"], None),
        ("Ampersand combo with spaces", ["drug1 & drug2"], ["drug1", "drug2"]),
        ("Ampersand combo no spaces", ["drug1&drug2"], None),
        ("Backslash combo", ["drug1 \\ drug2"], ["drug1", "drug2"]),
        ("Triple combo / and +", ["drug1 / drug2 and drug3 + drug4"], ["drug1", "drug2", "drug3", "drug4"]),
        ("Brand combo Janumet", ["JANUMET"], ["metformin", "sitagliptin"]),
        ("Brand combo Adderall", ["ADDERALL"], ["amphetamine", "dextroamphetamine"]),
        ("Brand combo Synjardy", ["SYNJARDY"], ["empagliflozin", "metformin"]),
        ("Brand combo Advair", ["ADVAIR"], ["fluticasone", "salmeterol"]),
        ("Compound dosage preserved", ["INSULIN GLARGINE 100 UNITS/ML"], ["insulin glargine"]),
        ("Iterable generator", (d for d in ["ozempic", "zoloft"]), ["semaglutide", "sertraline"]),
        ("Numpy array input", np.array(["ozempic", "zoloft"]), ["semaglutide", "sertraline"]),
        ("Pandas series input", pd.Series(["ozempic", "zoloft"]), ["semaglutide", "sertraline"]),
        ("Non-iterable integer", 12345, []),
        ("Non-iterable float nan", float("nan"), []),
    ]

    for label, val, expected in list_cases:
        try:
            res = normalize_drug_list(val)
            print(f"  [RES] {label:30s}: input={repr(val)[:40]:40s} -> {repr(res):30s}")
            assert isinstance(res, list)
            if expected is not None:
                if res == expected:
                    results["passed"] += 1
                else:
                    print(f"        -> DISCREPANCY: expected {expected}, got {res}")
                    results["findings"].append((label, str(val), f"Expected {expected}, got {res}"))
            else:
                results["passed"] += 1
        except Exception as e:
            print(f"  [FAIL] {label:30s}: input={repr(val)} -> {e}")
            results["failed"] += 1
            results["exceptions"].append((label, val, str(e)))

    # -------------------------------------------------------------------------
    # Battery 6: Stress & ReDoS / Catastrophic Backtracking Tests
    # -------------------------------------------------------------------------
    print("\n--- BATTERY 6: STRESS, THROUGHPUT, AND REDOS CHECKS ---")
    stress_payloads = [
        ("Long string 10,000 chars", "metformin " * 1000),
        ("Pathological regex input", "a" * 5000 + " 500mg"),
        ("Pathological nested slashes", "/ " * 2000),
        ("Pathological numbers", "123456789.987654321 " * 1000),
        ("10,000 item drug list", ["metformin 500mg"] * 10000),
    ]

    for label, payload in stress_payloads:
        t0 = time.perf_counter()
        try:
            if isinstance(payload, list):
                res = normalize_drug_list(payload)
            else:
                res = normalize_drug_name(payload)
            elapsed = time.perf_counter() - t0
            print(f"  [OK] {label:30s}: executed in {elapsed*1000:.2f} ms")
            assert elapsed < 2.0, f"Execution took too long: {elapsed:.2f}s (potential ReDoS)"
            results["passed"] += 1
        except Exception as e:
            print(f"  [FAIL] {label:30s}: -> {e}")
            results["failed"] += 1
            results["exceptions"].append((label, "large_payload", str(e)))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADVERSARIAL STRESS TEST SUMMARY")
    print("=" * 80)
    print(f"Total Passed: {results['passed']}")
    print(f"Total Failed: {results['failed']}")
    print(f"Total Exceptions Caught: {len(results['exceptions'])}")
    print(f"Total Discrepancies/Findings: {len(results['findings'])}")

    if results["findings"]:
        print("\nFindings Details:")
        for name, val, desc in results["findings"]:
            print(f" - [{name}]: {desc}")

    return results


if __name__ == "__main__":
    run_tests()
