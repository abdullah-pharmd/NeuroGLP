"""Shared fixtures and test configuration for Vigi-Pheno test suite.

Provides fast, reproducible, and offline-compatible synthetic datasets
and OpenFDA API payloads designed to run all tests in < 30 seconds.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def mock_openfda_response() -> Dict[str, Any]:
    """Realistic mock OpenFDA JSON response payload with 80 adverse event records."""
    records = []
    
    # 3 clinical clusters to ensure downstream HDBSCAN discovery of >=3 clusters:
    # Group 1 (SSRI/SNRI Polypharmacy)
    group1_drugs = [
        {"medicinalproduct": "OZEMPIC", "drugcharacterization": "1",
         "openfda": {"generic_name": ["SEMAGLUTIDE"], "brand_name": ["OZEMPIC"]}},
        {"medicinalproduct": "SERTRALINE HYDROCHLORIDE 50MG", "drugcharacterization": "2",
         "openfda": {"generic_name": ["SERTRALINE"], "brand_name": ["ZOLOFT"]}},
        {"medicinalproduct": "ESCITALOPRAM OXALATE 10MG", "drugcharacterization": "2",
         "openfda": {"generic_name": ["ESCITALOPRAM"], "brand_name": ["LEXAPRO"]}},
        {"medicinalproduct": "DULOXETINE DELAYED-RELEASE", "drugcharacterization": "2",
         "openfda": {"generic_name": ["DULOXETINE"], "brand_name": ["CYMBALTA"]}},
    ]
    
    # Group 2 (Cardiometabolic Comorbidities)
    group2_drugs = [
        {"medicinalproduct": "WEGOVY", "drugcharacterization": "1",
         "openfda": {"generic_name": ["SEMAGLUTIDE"]}},
        {"medicinalproduct": "METFORMIN HYDROCHLORIDE 500MG", "drugcharacterization": "2",
         "openfda": {"generic_name": ["METFORMIN"], "brand_name": ["GLUCOPHAGE"]}},
        {"medicinalproduct": "LISINOPRIL 20MG TABLET", "drugcharacterization": "2",
         "openfda": {"generic_name": ["LISINOPRIL"], "brand_name": ["PRINIVIL"]}},
        {"medicinalproduct": "ATORVASTATIN CALCIUM 40MG", "drugcharacterization": "2",
         "openfda": {"generic_name": ["ATORVASTATIN"], "brand_name": ["LIPITOR"]}},
        {"medicinalproduct": "INSULIN GLARGINE 100 U/ML", "drugcharacterization": "2",
         "openfda": {"generic_name": ["INSULIN GLARGINE"], "brand_name": ["LANTUS"]}},
    ]
    
    # Group 3 (Oral Contraceptive & Endocrine)
    group3_drugs = [
        {"medicinalproduct": "MOUNJARO", "drugcharacterization": "1",
         "openfda": {"generic_name": ["TIRZEPATIDE"], "brand_name": ["MOUNJARO"]}},
        {"medicinalproduct": "ETHINYL ESTRADIOL AND LEVONORGESTREL", "drugcharacterization": "2",
         "openfda": {"generic_name": ["ETHINYL ESTRADIOL / LEVONORGESTREL"], "brand_name": ["ALESSE"]}},
        {"medicinalproduct": "DROSPIRENONE AND ETHINYL ESTRADIOL", "drugcharacterization": "2",
         "openfda": {"generic_name": ["DROSPIRENONE / ETHINYL ESTRADIOL"], "brand_name": ["YASMIN"]}},
        {"medicinalproduct": "SPIRONOLACTONE 25MG TABLET", "drugcharacterization": "2",
         "openfda": {"generic_name": ["SPIRONOLACTONE"], "brand_name": ["ALDACTONE"]}},
    ]

    # Generate 80 records: 25 group 1, 25 group 2, 25 group 3, 5 noise/outliers
    for i in range(80):
        report_id = f"FDA-{1000000 + i}"
        if i < 25:
            drugs = group1_drugs[:]
            age = str(42 + (i % 15))
            age_unit = "801"  # Year
            sex = "2"  # Female
            serious = "1"
            hospitalization = "1" if (i % 3 == 0) else "0"
            reactions = [{"reactionmeddrapt": "SUICIDAL IDEATION"}, {"reactionmeddrapt": "DEPRESSION"}]
        elif i < 50:
            drugs = group2_drugs[:]
            age = str(58 + (i % 20))
            age_unit = "801"
            sex = "1" if (i % 2 == 0) else "2"
            serious = "1"
            hospitalization = "1" if (i % 2 == 0) else "0"
            reactions = [{"reactionmeddrapt": "ANXIETY"}, {"reactionmeddrapt": "DEPRESSED MOOD"}]
        elif i < 75:
            drugs = group3_drugs[:]
            age = str(28 + (i % 12))
            age_unit = "801"
            sex = "2"  # Predominantly female
            serious = "1" if (i % 2 == 0) else "2"
            hospitalization = "0"
            reactions = [{"reactionmeddrapt": "ANXIETY"}, {"reactionmeddrapt": "PANIC ATTACK"}]
        else:
            # Noise points / edge cases
            if i == 75:
                # Age in months
                age = "24"
                age_unit = "802"
                drugs = [{"medicinalproduct": "RYBELSUS", "drugcharacterization": "1"}]
            elif i == 76:
                # Missing age, unknown sex
                age = None
                age_unit = None
                sex = "0"
                drugs = [{"medicinalproduct": "ZEPBOUND", "drugcharacterization": "1"}]
            elif i == 77:
                # Age in decades
                age = "6"
                age_unit = "800"
                drugs = [
                    {"medicinalproduct": "SEMAGLUTIDE", "drugcharacterization": "1"},
                    {"medicinalproduct": "GABAPENTIN 300MG", "drugcharacterization": "2"}
                ]
            else:
                age = "65"
                age_unit = "801"
                drugs = [
                    {"medicinalproduct": "TIRZEPATIDE", "drugcharacterization": "1"},
                    {"medicinalproduct": "OMEPRAZOLE 20MG", "drugcharacterization": "2"},
                    {"medicinalproduct": "ASPIRIN 81MG", "drugcharacterization": "2"}
                ]
            sex = "2" if i % 2 == 0 else "1"
            serious = "1"
            hospitalization = "1" if i % 2 == 0 else "0"
            reactions = [{"reactionmeddrapt": "SUICIDAL BEHAVIOUR"}]

        record = {
            "safetyreportid": report_id,
            "serious": serious,
            "seriousnesshospitalization": hospitalization,
            "patient": {
                "patientonsetage": age,
                "patientonsetageunit": age_unit,
                "patientsex": sex,
                "drug": drugs,
                "reaction": reactions
            }
        }
        records.append(record)

    return {
        "meta": {
            "disclaimer": "Do not rely on openFDA to make decisions regarding medical care.",
            "terms": "https://open.fda.gov/terms/",
            "license": "https://open.fda.gov/license/",
            "last_updated": "2026-09-30",
            "results": {
                "skip": 0,
                "limit": 100,
                "total": 7074
            }
        },
        "results": records
    }


@pytest.fixture
def sample_raw_records(mock_openfda_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List of 80 raw OpenFDA adverse event records."""
    return mock_openfda_response["results"]


@pytest.fixture
def sample_flattened_df(sample_raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Pre-flattened DataFrame satisfying M1 ↔ M2 interface contract."""
    rows = []
    for r in sample_raw_records:
        patient = r.get("patient", {})
        
        # Age conversion
        raw_age = patient.get("patientonsetage")
        raw_unit = patient.get("patientonsetageunit", "801")
        if raw_age is not None:
            try:
                age_val = float(raw_age)
                if raw_unit == "800":  # Decade
                    age_val *= 10.0
                elif raw_unit == "802":  # Month
                    age_val /= 12.0
                elif raw_unit == "803":  # Week
                    age_val /= 52.0
                elif raw_unit == "804":  # Day
                    age_val /= 365.25
            except (ValueError, TypeError):
                age_val = 52.0
        else:
            age_val = 52.0  # Median imputed

        # Sex mapping
        raw_sex = str(patient.get("patientsex", "0"))
        if raw_sex == "1":
            sex_val = 1
        elif raw_sex == "2":
            sex_val = 2
        else:
            sex_val = 0

        # Serious & Hospitalization
        serious_val = 1 if str(r.get("serious")) == "1" else 2
        hosp_val = 1 if str(r.get("seriousnesshospitalization")) == "1" else 0

        # Drug classification
        drugs = patient.get("drug", [])
        suspect_drugs = []
        concomitant_drugs = []
        for d in drugs:
            char = str(d.get("drugcharacterization", "2"))
            name = d.get("medicinalproduct") or ""
            if d.get("openfda") and d["openfda"].get("generic_name"):
                name = d["openfda"]["generic_name"][0]
            if char == "1":
                suspect_drugs.append(name.lower())
            else:
                concomitant_drugs.append(name)

        # Reactions
        reactions = [
            rx.get("reactionmeddrapt", "").upper()
            for rx in patient.get("reaction", [])
            if rx.get("reactionmeddrapt")
        ]

        rows.append({
            "safetyreportid": r.get("safetyreportid"),
            "patientonsetage": float(age_val),
            "patientsex": int(sex_val),
            "serious": int(serious_val),
            "seriousnesshospitalization": int(hosp_val),
            "suspect_drugs": suspect_drugs,
            "concomitant_drugs": concomitant_drugs,
            "reactions": reactions
        })

    return pd.DataFrame(rows)


@pytest.fixture
def fast_feature_matrix_tuple() -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Fast synthetic binary feature matrix (80 x 15) ensuring UMAP runs in < 1.5s.
    
    Guarantees:
    - dtype: uint8
    - shape: (80, 15)
    - np.isnan(matrix).sum() == 0
    - Multi-modal cluster structure (Group 0, Group 1, Group 2, Noise)
    """
    n_samples = 80
    feature_names = [
        "sertraline", "escitalopram", "duloxetine", "fluoxetine",
        "metformin", "lisinopril", "atorvastatin", "insulin glargine",
        "ethinyl estradiol", "levonorgestrel", "drospirenone", "spironolactone",
        "bupropion", "gabapentin", "omeprazole"
    ]
    n_features = len(feature_names)
    matrix = np.zeros((n_samples, n_features), dtype=np.uint8)

    # Group 1 (rows 0-24): SSRI/SNRI enriched (cols 0, 1, 2, 3)
    for i in range(25):
        matrix[i, 0] = 1  # sertraline
        matrix[i, 1] = 1  # escitalopram
        if i % 2 == 0:
            matrix[i, 2] = 1  # duloxetine
        if i % 3 == 0:
            matrix[i, 3] = 1  # fluoxetine

    # Group 2 (rows 25-49): Cardiometabolic enriched (cols 4, 5, 6, 7)
    for i in range(25, 50):
        matrix[i, 4] = 1  # metformin
        matrix[i, 5] = 1  # lisinopril
        if i % 2 == 0:
            matrix[i, 6] = 1  # atorvastatin
        if i % 3 == 0:
            matrix[i, 7] = 1  # insulin glargine

    # Group 3 (rows 50-74): Contraceptive / Endocrine enriched (cols 8, 9, 10, 11)
    for i in range(50, 75):
        matrix[i, 8] = 1   # ethinyl estradiol
        matrix[i, 9] = 1   # levonorgestrel
        if i % 2 == 0:
            matrix[i, 10] = 1  # drospirenone
        if i % 3 == 0:
            matrix[i, 11] = 1  # spironolactone

    # Outliers / Noise (rows 75-79): Sparse single hits
    matrix[75, 12] = 1  # bupropion
    matrix[76, 13] = 1  # gabapentin
    matrix[77, 14] = 1  # omeprazole
    matrix[78, 12] = 1
    matrix[79, 13] = 1

    # Corresponding clean_df
    ids = [f"FDA-{1000000 + i}" for i in range(n_samples)]
    ages = [42.0 + (i % 15) if i < 25 else 58.0 + (i % 20) if i < 50 else 28.0 + (i % 12) for i in range(n_samples)]
    sexes = [2 if i < 25 else (1 if i % 2 == 0 else 2) if i < 50 else 2 for i in range(n_samples)]
    hosp = [1 if i % 3 == 0 else 0 for i in range(n_samples)]
    suspect = [["semaglutide"] if i < 50 else ["tirzepatide"] for i in range(n_samples)]

    clean_df = pd.DataFrame({
        "safetyreportid": ids,
        "patientonsetage": ages,
        "patientsex": sexes,
        "serious": [1] * n_samples,
        "seriousnesshospitalization": hosp,
        "suspect_drugs": suspect,
        "reactions": [["SUICIDAL IDEATION", "DEPRESSION"]] * n_samples
    })

    return matrix, feature_names, clean_df


@pytest.fixture
def fast_feature_matrix(fast_feature_matrix_tuple: Tuple[np.ndarray, List[str], pd.DataFrame]) -> np.ndarray:
    """Convenience fixture returning the numpy binary feature matrix."""
    return fast_feature_matrix_tuple[0]


@pytest.fixture
def fast_feature_names(fast_feature_matrix_tuple: Tuple[np.ndarray, List[str], pd.DataFrame]) -> List[str]:
    """Convenience fixture returning the feature names list."""
    return fast_feature_matrix_tuple[1]


@pytest.fixture
def fast_clean_df(fast_feature_matrix_tuple: Tuple[np.ndarray, List[str], pd.DataFrame]) -> pd.DataFrame:
    """Convenience fixture returning the aligned clean DataFrame."""
    return fast_feature_matrix_tuple[2]


@pytest.fixture
def synthetic_clustered_df(fast_clean_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-computed clustered DataFrame with deterministic 3D coordinates and HDBSCAN labels."""
    df = fast_clean_df.copy()
    n = len(df)
    
    # Coordinates centered around 3 separated clusters
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    labels = np.zeros(n, dtype=int)

    rng = np.random.RandomState(42)
    # Cluster 0
    x[0:25] = rng.normal(loc=-5.0, scale=0.3, size=25)
    y[0:25] = rng.normal(loc=2.0, scale=0.3, size=25)
    z[0:25] = rng.normal(loc=0.0, scale=0.3, size=25)
    labels[0:25] = 0

    # Cluster 1
    x[25:50] = rng.normal(loc=5.0, scale=0.3, size=25)
    y[25:50] = rng.normal(loc=-2.0, scale=0.3, size=25)
    z[25:50] = rng.normal(loc=3.0, scale=0.3, size=25)
    labels[25:50] = 1

    # Cluster 2
    x[50:75] = rng.normal(loc=0.0, scale=0.3, size=25)
    y[50:75] = rng.normal(loc=6.0, scale=0.3, size=25)
    z[50:75] = rng.normal(loc=-4.0, scale=0.3, size=25)
    labels[50:75] = 2

    # Noise (-1)
    x[75:80] = rng.uniform(low=-8.0, high=8.0, size=5)
    y[75:80] = rng.uniform(low=-8.0, high=8.0, size=5)
    z[75:80] = rng.uniform(low=-8.0, high=8.0, size=5)
    labels[75:80] = -1

    df["umap_x"] = x
    df["umap_y"] = y
    df["umap_z"] = z
    df["cluster_label"] = labels
    return df


@pytest.fixture
def sample_cluster_stats() -> Dict[int, Dict[str, Any]]:
    """Sample cluster profile statistics dictionary matching M3 ↔ M4/M5 contract."""
    return {
        -1: {
            "patient_count": 5,
            "percentage": 6.25,
            "mean_age": 61.2,
            "female_pct": 60.0,
            "hospitalization_rate": 40.0,
            "top_drugs": [("omeprazole", 40.0, 1.2), ("bupropion", 40.0, 1.5)],
            "top_reactions": [("SUICIDAL BEHAVIOUR", 60.0)]
        },
        0: {
            "patient_count": 25,
            "percentage": 31.25,
            "mean_age": 44.5,
            "female_pct": 88.0,
            "hospitalization_rate": 36.0,
            "top_drugs": [("sertraline", 100.0, 8.5), ("escitalopram", 100.0, 7.8), ("duloxetine", 52.0, 4.2)],
            "top_reactions": [("SUICIDAL IDEATION", 80.0), ("DEPRESSION", 68.0)]
        },
        1: {
            "patient_count": 25,
            "percentage": 31.25,
            "mean_age": 62.1,
            "female_pct": 52.0,
            "hospitalization_rate": 48.0,
            "top_drugs": [("metformin", 100.0, 9.2), ("lisinopril", 100.0, 8.0), ("atorvastatin", 52.0, 4.5)],
            "top_reactions": [("ANXIETY", 72.0), ("DEPRESSED MOOD", 56.0)]
        },
        2: {
            "patient_count": 25,
            "percentage": 31.25,
            "mean_age": 31.4,
            "female_pct": 96.0,
            "hospitalization_rate": 12.0,
            "top_drugs": [("ethinyl estradiol", 100.0, 12.4), ("levonorgestrel", 100.0, 11.0), ("drospirenone", 52.0, 6.1)],
            "top_reactions": [("ANXIETY", 84.0), ("PANIC ATTACK", 44.0)]
        }
    }


@pytest.fixture
def curated_benchmark_dataset(sample_raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Curated benchmark dataset fixture (at least 80 realistic adverse event records)."""
    return sample_raw_records
