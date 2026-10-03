"""Adversarial stress-test suite for Milestone 3: HDBSCAN & Profiler (challenger_m3_2).

Thoroughly probes:
- Hostile coordinates: collinear 3D points, all-identical points, extreme outliers, non-finite values.
- Odds Ratio edge cases: zero division (b=0 or c=0), all-zero contingency table (0,0,0,0), overflow.
- Adaptive ladder guarantees: discovering >= 3 distinct non-noise clusters on multi-modal cohorts.
- Robustness against malformed demographics, monotherapy clusters, and boundary sample sizes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from neuroglp.models.hdbscan_clusterer import cluster_embeddings, cluster_hdbscan
from neuroglp.models.profiler import compute_odds_ratio, profile_clusters, ClusterProfile
from neuroglp.models.pipeline import fit_phenotyping_pipeline, PhenotypingResults


# =============================================================================
# 1. Hostile Coordinates & HDBSCAN Boundary Tests
# =============================================================================

def test_adversarial_collinear_3d_coordinates() -> None:
    """Probes HDBSCAN behavior when 3D points lie strictly along a 1D line."""
    # 100 points along the X axis
    x = np.linspace(-10.0, 10.0, 100)
    collinear_coords = np.column_stack([x, np.zeros(100), np.zeros(100)])

    labels, meta = cluster_embeddings(collinear_coords, min_cluster_size=5)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 100
    assert "n_clusters" in meta
    assert "n_noise" in meta
    assert meta["n_noise"] >= 0
    assert meta["noise_ratio"] <= 1.0


def test_adversarial_identical_coordinates() -> None:
    """Probes HDBSCAN behavior when all points have identical coordinates (zero distance matrix)."""
    identical_coords = np.ones((60, 3), dtype=float) * 100.0

    labels, meta = cluster_embeddings(identical_coords, min_cluster_size=5)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 60
    # All points are either grouped into 0 clusters (all noise) or 1 cluster
    assert meta["n_clusters"] in (0, 1)
    assert not np.isnan(labels).any()


def test_adversarial_extreme_outliers_isolated() -> None:
    """Verifies extreme spatial outliers (+/- 1e12) are strictly classified as noise (-1)."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 3) + np.array([-5.0, 0.0, 0.0])
    c2 = rng.randn(30, 3) + np.array([5.0, 0.0, 0.0])
    c3 = rng.randn(30, 3) + np.array([0.0, 5.0, 5.0])

    # Inject extreme spatial outliers
    outlier_pos = np.array([[1e12, 1e12, 1e12]])
    outlier_neg = np.array([[-1e12, -1e12, -1e12]])

    coords = np.vstack([c1, c2, c3, outlier_pos, outlier_neg])
    labels, meta = cluster_embeddings(coords, min_cluster_size=10, min_clusters_target=3)

    # Check the two injected outliers (indices -2 and -1)
    assert labels[-2] == -1, f"Positive extreme outlier not labeled as noise: {labels[-2]}"
    assert labels[-1] == -1, f"Negative extreme outlier not labeled as noise: {labels[-1]}"
    assert meta["n_clusters"] >= 3


def test_adversarial_nan_and_inf_coordinates_rejected() -> None:
    """Verifies that non-finite coordinates strictly raise ValueError."""
    nan_coords = np.array([[1.0, 2.0, np.nan], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        cluster_embeddings(nan_coords)

    inf_coords = np.array([[1.0, 2.0, np.inf], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        cluster_embeddings(inf_coords)


def test_adversarial_empty_coordinates_handling() -> None:
    """Verifies empty coordinate arrays return empty outputs without crashing."""
    empty_coords = np.empty((0, 3), dtype=float)
    labels, meta = cluster_embeddings(empty_coords)

    assert len(labels) == 0
    assert meta["n_clusters"] == 0
    assert meta["n_noise"] == 0
    assert meta["noise_ratio"] == 0.0
    assert meta["converged"] is False


def test_adversarial_small_cohorts_boundary() -> None:
    """Evaluates cohorts with 2 to 5 points (below min_clusters_target * 2)."""
    for n in [2, 3, 4, 5]:
        pts = np.zeros((n, 3), dtype=float)
        labels, meta = cluster_embeddings(pts, min_clusters_target=3)
        assert len(labels) == n
        assert meta["n_clusters"] == 0
        assert meta["n_noise"] == n


def test_adversarial_single_point_cohort_boundary() -> None:
    """Exposes N=1 boundary failure where HDBSCAN receives min_cluster_size > n_samples."""
    pts = np.zeros((1, 3), dtype=float)
    # HDBSCAN throws ValueError when k (min_cluster_size=2) > n_samples=1.
    # We verify whether it raises or handles gracefully:
    try:
        labels, meta = cluster_embeddings(pts, min_clusters_target=3)
        assert len(labels) == 1
        assert labels[0] == -1
    except ValueError as exc:
        # Documented failure mode: HDBSCAN nearest-neighbor requirement k <= n_samples
        assert "k must be less than or equal to the number of training points" in str(exc)


# =============================================================================
# 2. Odds Ratio & Haldane-Anscombe Edge Case Tests
# =============================================================================

@pytest.mark.parametrize(
    "a,b,c,d,expected_predicate",
    [
        (0, 0, 0, 0, lambda or_val: or_val == 1.0),
        (10, 0, 5, 20, lambda or_val: or_val > 1.0 and not np.isnan(or_val) and not np.isinf(or_val)),
        (10, 5, 0, 20, lambda or_val: or_val > 1.0 and not np.isnan(or_val) and not np.isinf(or_val)),
        (10, 0, 0, 20, lambda or_val: or_val > 1.0 and not np.isnan(or_val) and not np.isinf(or_val)),
        (0, 10, 20, 0, lambda or_val: 0.0 < or_val < 1.0 and not np.isnan(or_val) and not np.isinf(or_val)),
        (0, 0, 10, 20, lambda or_val: or_val == 1.0),
        (10, 20, 0, 0, lambda or_val: or_val == 1.0),
        (1e150, 1e150, 1e150, 1e150, lambda or_val: or_val == pytest.approx(1.0, rel=1e-3)),
        (1e200, 1, 1, 1e200, lambda or_val: or_val == 1.0),  # Overflow product fallback
        (1.5, 2.5, 3.5, 4.5, lambda or_val: 0.0 < or_val < 1.0),
    ],
)
def test_odds_ratio_mathematical_edge_cases(a, b, c, d, expected_predicate) -> None:
    """Verifies that compute_odds_ratio strictly returns finite non-NaN float across all edge cases."""
    or_val = compute_odds_ratio(a, b, c, d)
    assert isinstance(or_val, float)
    assert not np.isnan(or_val)
    assert not np.isinf(or_val)
    assert expected_predicate(or_val)


def test_odds_ratio_negative_inputs_rejected() -> None:
    """Verifies compute_odds_ratio strictly rejects negative contingency entries."""
    with pytest.raises(ValueError, match="cannot be negative"):
        compute_odds_ratio(-1, 10, 10, 10)

    with pytest.raises(ValueError, match="cannot be negative"):
        compute_odds_ratio(10, -5, 10, 10)


# =============================================================================
# 3. Clinical Profiler Robustness & Monotherapy Tests
# =============================================================================

def test_profile_clusters_monotherapy_cluster() -> None:
    """Verifies clinical profiler handles a monotherapy cluster (0 concomitant drugs) cleanly."""
    n_patients = 30
    df = pd.DataFrame({
        "patientonsetage": [45.0 + i for i in range(n_patients)],
        "patientsex": [1 if i % 2 == 0 else 2 for i in range(n_patients)],
        "seriousnesshospitalization": [1 if i % 3 == 0 else 0 for i in range(n_patients)],
        "reactions": [["DEPRESSION"] for _ in range(n_patients)],
    })

    # All-zero feature matrix (patients on GLP-1 monotherapy, no co-medications)
    mat = np.zeros((n_patients, 5), dtype=np.uint8)
    feat_names = ["metformin", "lisinopril", "atorvastatin", "sertraline", "omeprazole"]
    cluster_labels = np.zeros(n_patients, dtype=int)

    profiles = profile_clusters(
        df=df,
        cluster_labels=cluster_labels,
        feature_matrix=mat,
        feature_names=feat_names,
    )

    assert 0 in profiles
    prof = profiles[0]
    assert prof.patient_count == n_patients
    assert prof.percentage == 100.0
    assert prof.top_drugs == []  # Clean empty list, no NaN or crash
    assert len(prof.top_reactions) == 1
    assert prof.top_reactions[0][0] == "DEPRESSION"


def test_profile_clusters_malformed_and_missing_demographics() -> None:
    """Verifies profiler does not crash when columns contain corrupted or missing values."""
    df = pd.DataFrame([
        {"patientonsetage": None, "patientsex": None, "seriousnesshospitalization": None, "reactions": None},
        {"patientonsetage": "invalid_age", "patientsex": 99, "seriousnesshospitalization": "unknown", "reactions": 12345},
        {"patientonsetage": np.nan, "patientsex": np.nan, "seriousnesshospitalization": np.nan, "reactions": "[unclosed"},
        {"patientonsetage": 55.0, "patientsex": 2, "seriousnesshospitalization": 1, "reactions": ["ANXIETY", None, "SUICIDAL IDEATION"]},
    ])
    mat = np.zeros((4, 2), dtype=np.uint8)
    feat_names = ["drugA", "drugB"]
    labels = np.array([0, 0, 1, 1])

    profiles = profile_clusters(
        df=df,
        cluster_labels=labels,
        feature_matrix=mat,
        feature_names=feat_names,
    )

    assert 0 in profiles
    assert 1 in profiles
    assert profiles[0].mean_age == 52.0  # Default imputed age
    assert profiles[1].female_pct == 50.0
    assert not np.isnan(profiles[0].hospitalization_rate)


# =============================================================================
# 4. Adaptive Ladder Multi-Modal Guarantees
# =============================================================================

def test_adaptive_ladder_guarantees_target_clusters_on_multimodal_cohort() -> None:
    """Verifies that adaptive ladder steps down from overly restrictive min_cluster_size to discover >= 3 clusters."""
    rng = np.random.RandomState(42)
    # 3 distinct clusters of 25 points each
    c1 = rng.randn(25, 3) + np.array([-12.0, 0.0, 0.0])
    c2 = rng.randn(25, 3) + np.array([0.0, 12.0, 0.0])
    c3 = rng.randn(25, 3) + np.array([12.0, 0.0, 12.0])
    noise = rng.uniform(-15, 15, size=(10, 3))
    pts = np.vstack([c1, c2, c3, noise])

    # Initial min_cluster_size=30 is larger than cluster sizes (25 points).
    # Standard HDBSCAN without stepping down would not discover 3 clusters at mcs=30.
    # The adaptive ladder descends to mcs=22 within 9 steps and discovers all 3 clusters.
    labels, meta = cluster_embeddings(pts, min_cluster_size=30, min_clusters_target=3)

    assert meta["n_clusters"] >= 3, f"Expected >= 3 clusters, got {meta['n_clusters']}"
    assert meta["converged"] is True
    assert meta["ladder_steps"] > 1
    assert meta["min_cluster_size"] < 30


def test_adaptive_ladder_step_budget_boundary() -> None:
    """Probes ladder behavior when initial min_cluster_size=50 requires >20 steps to converge."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(25, 3) + np.array([-12.0, 0.0, 0.0])
    c2 = rng.randn(25, 3) + np.array([0.0, 12.0, 0.0])
    c3 = rng.randn(25, 3) + np.array([12.0, 0.0, 12.0])
    noise = rng.uniform(-15, 15, size=(10, 3))
    pts = np.vstack([c1, c2, c3, noise])

    # With default max_ladder_steps=20, mcs=50 exhausts the 20-step budget before reaching mcs < 27
    _, meta_default = cluster_embeddings(pts, min_cluster_size=50, min_clusters_target=3, max_ladder_steps=20)
    # With expanded budget max_ladder_steps=30, the ladder reaches step 23 and converges
    _, meta_expanded = cluster_embeddings(pts, min_cluster_size=50, min_clusters_target=3, max_ladder_steps=30)

    assert meta_expanded["converged"] is True
    assert meta_expanded["n_clusters"] >= 3
    assert meta_expanded["ladder_steps"] == 23



def test_adaptive_ladder_graceful_stop_when_fewer_than_target_clusters() -> None:
    """Verifies ladder terminates gracefully without uncaught exceptions on 2-cluster cohort."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(25, 3) + np.array([-15.0, 0.0, 0.0])
    c2 = rng.randn(25, 3) + np.array([15.0, 0.0, 0.0])
    pts = np.vstack([c1, c2])

    labels, meta = cluster_embeddings(pts, min_cluster_size=10, min_clusters_target=3, max_ladder_steps=5)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 50
    assert "converged" in meta
    # Cohort only has 2 dense clusters, so converged may be False
    assert meta["n_clusters"] <= 3


# =============================================================================
# 5. Phenotyping Pipeline Interface & Dataclass Contract Tests
# =============================================================================

def test_pipeline_results_contract(fast_feature_matrix: np.ndarray, fast_clean_df: pd.DataFrame, fast_feature_names: List[str]) -> None:
    """Verifies PhenotypingResults supports attribute access, item access, in operator, and to_dict."""
    results = fit_phenotyping_pipeline(
        feature_matrix=fast_feature_matrix,
        clean_df=fast_clean_df,
        feature_names=fast_feature_names,
        n_epochs=20,
        random_state=42,
    )

    # Attribute access
    assert isinstance(results.embeddings_3d, np.ndarray)
    assert isinstance(results.cluster_labels, np.ndarray)
    assert isinstance(results.cluster_stats, dict)
    assert isinstance(results.clustered_df, pd.DataFrame)
    assert isinstance(results.n_clusters, int)

    # Item subscript access
    assert np.array_equal(results["embeddings_3d"], results.embeddings_3d)
    assert results["n_clusters"] == results.n_clusters

    # 'in' operator
    assert "embeddings_3d" in results
    assert "cluster_stats" in results
    assert "nonexistent_key" not in results

    # to_dict conversion
    d = results.to_dict()
    assert isinstance(d, dict)
    assert "embeddings_3d" in d
    assert "cluster_labels" in d


def test_pipeline_dimension_mismatch_validation(fast_feature_matrix: np.ndarray, fast_clean_df: pd.DataFrame) -> None:
    """Verifies dimension mismatches raise descriptive ValueError."""
    # Row mismatch: pass df with 1 fewer row
    truncated_df = fast_clean_df.iloc[:-1]
    with pytest.raises(ValueError, match="Row count mismatch"):
        fit_phenotyping_pipeline(fast_feature_matrix, truncated_df)

    # Column mismatch: pass wrong number of feature names
    with pytest.raises(ValueError, match="feature_names length"):
        fit_phenotyping_pipeline(
            fast_feature_matrix,
            fast_clean_df,
            feature_names=["only_one_name"],
        )
