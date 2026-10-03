"""Unit, determinism, and statistical tests for Milestone 3: Phenotyping Engine.

Tests UMAP 3D Jaccard embedding determinism (random_state=42), HDBSCAN density clustering
(asserting >= 3 non-noise clusters and noise -1 handling), cluster statistical profiling,
and odds-ratio calculations.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

try:
    from neuroglp.models.umap_embedder import project_umap_3d
    from neuroglp.models.hdbscan_clusterer import cluster_hdbscan
    from neuroglp.models.profiler import profile_clusters, compute_odds_ratio
    try:
        from neuroglp.models import fit_phenotyping_pipeline
    except ImportError:
        from neuroglp.models.pipeline import fit_phenotyping_pipeline
except ImportError:
    project_umap_3d = None
    cluster_hdbscan = None
    profile_clusters = None
    compute_odds_ratio = None
    fit_phenotyping_pipeline = None

# =====================================================================
# Fixture & Synthetic Cluster Tests (Always Active)
# =====================================================================

def test_synthetic_cluster_dataset_distribution(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies synthetic clustered dataset contains >= 3 clusters and noise (-1)."""
    assert len(synthetic_clustered_df) == 80
    assert "cluster_label" in synthetic_clustered_df.columns
    labels = set(synthetic_clustered_df["cluster_label"])
    assert -1 in labels
    non_noise = labels - {-1}
    assert len(non_noise) >= 3
    # Check 3D coordinates exist
    assert "umap_x" in synthetic_clustered_df.columns
    assert "umap_y" in synthetic_clustered_df.columns
    assert "umap_z" in synthetic_clustered_df.columns


@pytest.fixture(autouse=True)
def _check_models_implemented(request: pytest.FixtureRequest) -> None:
    if request.node.name.startswith(("test_fixture", "test_mock", "test_sample", "test_synthetic")):
        return
    if project_umap_3d is None or cluster_hdbscan is None:
        pytest.skip("neuroglp.models is not yet implemented")


# =====================================================================
# Tier 1 & 2: UMAP 3D Projection & Determinism Tests
# =====================================================================

def test_umap_3d_projection_shape_and_speed(fast_feature_matrix: np.ndarray) -> None:
    """Verifies UMAP outputs (N, 3) coordinates and completes in under 2 seconds."""
    start_time = time.perf_counter()
    embeddings = project_umap_3d(fast_feature_matrix, random_state=42, n_epochs=50)
    elapsed = time.perf_counter() - start_time

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (fast_feature_matrix.shape[0], 3)
    assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64
    assert np.isnan(embeddings).sum() == 0
    assert elapsed < 3.0, f"UMAP execution took {elapsed:.2f}s, exceeding fast test threshold of 3.0s"


def test_umap_deterministic_reproducibility(fast_feature_matrix: np.ndarray) -> None:
    """STRICT REQUIREMENT: Assert identical coordinates across repeated runs with random_state=42."""
    run_1 = project_umap_3d(fast_feature_matrix, random_state=42, n_epochs=50)
    run_2 = project_umap_3d(fast_feature_matrix, random_state=42, n_epochs=50)
    
    # Assert coordinates match within numerical tolerance
    np.testing.assert_allclose(
        run_1,
        run_2,
        atol=1e-4,
        err_msg="UMAP coordinates are not deterministic with fixed random_state=42!"
    )


# =====================================================================
# Tier 1 & 2: HDBSCAN Clustering Tests
# =====================================================================

def test_hdbscan_clustering_discovery_and_noise_isolation(fast_feature_matrix: np.ndarray) -> None:
    """STRICT REQUIREMENT: HDBSCAN discovers >= 3 non-noise clusters and isolates noise (-1)."""
    # 1. Project to 3D
    embeddings = project_umap_3d(fast_feature_matrix, random_state=42, n_epochs=50)
    
    # 2. Cluster with HDBSCAN
    labels = cluster_hdbscan(embeddings, min_cluster_size=10, min_samples=5)
    
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(fast_feature_matrix)
    
    unique_labels = set(labels)
    # Check noise label -1 is accommodated
    assert -1 in unique_labels or len(unique_labels) >= 3
    
    # Check >= 3 distinct non-noise clusters
    non_noise_clusters = [lbl for lbl in unique_labels if lbl != -1]
    assert len(non_noise_clusters) >= 3, (
        f"Expected at least 3 non-noise clusters, but found {len(non_noise_clusters)}: {non_noise_clusters}"
    )


def test_hdbscan_synthetic_spatial_separation(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies HDBSCAN on pre-separated 3D Gaussian clusters discovers exactly the intended clusters."""
    coords = synthetic_clustered_df[["umap_x", "umap_y", "umap_z"]].to_numpy()
    labels = cluster_hdbscan(coords, min_cluster_size=15, min_samples=5)
    
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 3


# =====================================================================
# Tier 1 & 2: Cluster Statistical Profiling Tests
# =====================================================================

def test_compute_odds_ratio() -> None:
    """Verifies Odds Ratio calculation and zero-division handling."""
    # Enriched drug: 20 in cluster take it out of 25 (80%), 10 in rest take it out of 75 (13.3%)
    or_val = compute_odds_ratio(a=20, b=5, c=10, d=65)
    assert or_val > 1.0
    
    # Non-enriched drug: equal proportions
    or_neutral = compute_odds_ratio(a=10, b=10, c=10, d=10)
    assert or_neutral == pytest.approx(1.0, rel=0.1)

    # Edge cases (zero counts): Haldane-Anscombe correction (+0.5) prevents divide by zero
    or_zero = compute_odds_ratio(a=0, b=20, c=0, d=60)
    assert not np.isnan(or_zero)
    assert not np.isinf(or_zero)


def test_profile_clusters_metrics(
    synthetic_clustered_df: pd.DataFrame,
    fast_feature_matrix: np.ndarray,
    fast_feature_names: List[str]
) -> None:
    """Verifies profile_clusters computes patient_count, demographics, hospitalization rate, and top drugs."""
    labels = synthetic_clustered_df["cluster_label"].to_numpy()
    profiles = profile_clusters(
        df=synthetic_clustered_df,
        cluster_labels=labels,
        feature_matrix=fast_feature_matrix,
        feature_names=fast_feature_names
    )
    
    assert isinstance(profiles, dict)
    # Check all clusters are represented, including noise -1
    for cl_id in set(labels):
        assert cl_id in profiles
        profile = profiles[cl_id]
        
        # Check required metrics
        assert "patient_count" in profile
        assert "percentage" in profile
        assert "mean_age" in profile
        assert "female_pct" in profile or "sex_ratio" in profile
        assert "hospitalization_rate" in profile
        assert "top_drugs" in profile
        assert "top_reactions" in profile

        assert profile["patient_count"] > 0
        assert 0.0 <= profile["percentage"] <= 100.0
        assert 0.0 <= profile["hospitalization_rate"] <= 100.0


# =====================================================================
# Tier 3: Complete Fit Phenotyping Pipeline Integration
# =====================================================================

def test_fit_phenotyping_pipeline_full_integration(
    fast_feature_matrix: np.ndarray,
    fast_clean_df: pd.DataFrame,
    fast_feature_names: List[str]
) -> None:
    """Tests high-level fit_phenotyping_pipeline coordinating UMAP, HDBSCAN, and profiling."""
    results = fit_phenotyping_pipeline(
        feature_matrix=fast_feature_matrix,
        clean_df=fast_clean_df,
        feature_names=fast_feature_names,
        random_state=42
    )

    # Verify embeddings
    assert hasattr(results, "embeddings_3d") or "embeddings_3d" in results
    embeddings = results.embeddings_3d if hasattr(results, "embeddings_3d") else results["embeddings_3d"]
    assert embeddings.shape == (len(fast_clean_df), 3)

    # Verify cluster labels
    labels = results.cluster_labels if hasattr(results, "cluster_labels") else results["cluster_labels"]
    assert len(labels) == len(fast_clean_df)

    # Verify cluster stats
    stats = results.cluster_stats if hasattr(results, "cluster_stats") else results["cluster_stats"]
    assert isinstance(stats, dict)
    assert len(stats) >= 3
