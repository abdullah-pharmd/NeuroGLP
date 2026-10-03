"""
src/neuroglp/features/__init__.py

High-dimensional sparse feature engineering module for Vigi-Pheno.
Exports drug normalization, presence-absence binary feature matrix construction,
noise filtering, and zero-NaN validation.
"""

from neuroglp.features.builder import (
    build_feature_matrix,
    extract_patient_drugs,
    get_feature_frequencies,
    validate_feature_matrix_invariants,
)
from neuroglp.features.normalizer import (
    BRAND_TO_GENERIC,
    DOSAGE_UNITS,
    get_brand_mapping,
    is_known_drug,
    normalize_drug_list,
    normalize_drug_name,
)

__all__ = [
    "build_feature_matrix",
    "extract_patient_drugs",
    "get_feature_frequencies",
    "validate_feature_matrix_invariants",
    "normalize_drug_name",
    "normalize_drug_list",
    "is_known_drug",
    "get_brand_mapping",
    "BRAND_TO_GENERIC",
    "DOSAGE_UNITS",
]
