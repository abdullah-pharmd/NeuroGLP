"""
tests/test_umap_stress.py

Adversarial Stress Test Suite for Milestone 3: UMAP 3D Jaccard Embedder.
Authored by challenger_m3_1 (UMAP Adversarial Challenger).

Covers:
1. Boundary Cohort Sizes:
   - N=0 (empty cohort, D>0 and D=0)
   - N=1 (single patient)
   - N=2 (two patients)
   - N=3 (three patients: boundary condition N <= n_components)
   - N=4 (four patients: boundary condition N == n_components + 1, UMAP triggers)
   - Small cohorts where n_neighbors > n_samples
2. Degenerate Feature Matrices:
   - All rows identical (all pairwise Jaccard distances = 0)
   - All rows all-zero vectors (0/0 Jaccard division)
   - Single all-zero row among non-zero patients
   - All ones matrix
3. Disconnected Subgraphs & Orthogonal Vectors:
   - Identity matrix: completely disconnected singletons (all pairwise Jaccard distances = 1.0)
   - Two disjoint isolated islands (distance within island = 0, across islands = 1.0)
   - Multi-island disconnected graph with singleton outliers
4. Strict Zero-NaN & Zero-Inf Verification:
   - np.isnan(coords).sum() == 0 across all scenarios
   - not np.isinf(coords).any() across all scenarios
   - Valid continuous coordinates
5. Strict Deterministic Reproducibility:
   - Repeated runs with random_state=42 produce identical coordinates
   - Seed sensitivity (random_state=42 vs random_state=99)
   - Functional equivalence between embed_patients_3d and project_umap_3d
6. Dynamic Type & Sparse Representation Coercion:
   - scipy.sparse.csr_matrix and csc_matrix
   - pandas DataFrame
   - boolean arrays, float arrays, uint8 arrays
7. Class & Method Invariants (UMAPEmbedder):
   - fit() followed by transform()
   - fit_transform() equivalences
   - preservation of n_features_in_ and embedding_
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from neuroglp.models.umap_embedder import (
    UMAPEmbedder,
    embed_patients_3d,
    project_umap_3d,
)


# =====================================================================
# Helper Invariant Validator
# =====================================================================

def assert_valid_embedding(coords: np.ndarray, expected_n: int, expected_dim: int = 3) -> None:
    """Rigorous invariant checks for 3D coordinates."""
    assert isinstance(coords, np.ndarray), f"Expected np.ndarray, got {type(coords)}"
    assert coords.shape == (expected_n, expected_dim), (
        f"Expected shape ({expected_n}, {expected_dim}), got {coords.shape}"
    )
    assert coords.dtype in (np.float32, np.float64), f"Unexpected dtype: {coords.dtype}"
    assert np.isnan(coords).sum() == 0, f"Found {np.isnan(coords).sum()} NaN values in coords!"
    assert not np.isinf(coords).any(), "Found Infinite values in coords!"


# =====================================================================
# 1. Boundary Cohort Sizes (N=0, N=1, N=2, N=3, N=4, N < n_neighbors)
# =====================================================================

def test_stress_empty_matrix_n0() -> None:
    """Stress test: N=0 patients with D=10 features (both interfaces)."""
    X = np.zeros((0, 10), dtype=np.uint8)
    coords_proj = project_umap_3d(X, random_state=42)
    assert_valid_embedding(coords_proj, expected_n=0, expected_dim=3)

    coords_emb = embed_patients_3d(X, random_state=42)
    assert_valid_embedding(coords_emb, expected_n=0, expected_dim=3)


def test_stress_empty_matrix_n0_d0() -> None:
    """Stress test: N=0 patients with D=0 features."""
    X = np.zeros((0, 0), dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42)
    assert_valid_embedding(coords, expected_n=0, expected_dim=3)


def test_stress_single_patient_n1() -> None:
    """Stress test: N=1 single patient with D=5 features (both interfaces)."""
    X = np.array([[1, 0, 1, 0, 1]], dtype=np.uint8)
    coords_proj = project_umap_3d(X, random_state=42)
    assert_valid_embedding(coords_proj, expected_n=1, expected_dim=3)

    coords_emb = embed_patients_3d(X, random_state=42)
    assert_valid_embedding(coords_emb, expected_n=1, expected_dim=3)
    np.testing.assert_array_equal(coords_proj, coords_emb)


def test_stress_single_patient_zero_features_n1_d0() -> None:
    """Stress test: N=1 single patient with D=0 features."""
    X = np.zeros((1, 0), dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42)
    assert_valid_embedding(coords, expected_n=1, expected_dim=3)


def test_stress_two_patients_n2() -> None:
    """Stress test: N=2 patients with distinct features (both interfaces)."""
    X = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    coords_proj = project_umap_3d(X, random_state=42)
    assert_valid_embedding(coords_proj, expected_n=2, expected_dim=3)

    coords_emb = embed_patients_3d(X, random_state=42)
    assert_valid_embedding(coords_emb, expected_n=2, expected_dim=3)
    np.testing.assert_array_equal(coords_proj, coords_emb)


def test_stress_three_patients_boundary_n3() -> None:
    """Stress test: N=3 patients (exact boundary condition N == n_components)."""
    X = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
    ], dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42)
    assert_valid_embedding(coords, expected_n=3, expected_dim=3)


def test_stress_four_patients_boundary_n4() -> None:
    """Stress test: N=4 patients (exact boundary condition N == n_components + 1, UMAP triggers)."""
    X = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=4, expected_dim=3)


def test_stress_small_cohort_neighbors_clamped_n5() -> None:
    """Stress test: N=5 with default n_neighbors=15 (should clamp to max(2, 5-1) = 4)."""
    X = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.uint8)
    coords = project_umap_3d(X, n_neighbors=15, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=5, expected_dim=3)


# =====================================================================
# 2. Degenerate Feature Matrices
# =====================================================================

def test_stress_all_rows_identical() -> None:
    """Stress test: 30 patients with identical drug vectors (all pairwise Jaccard dists = 0.0)."""
    single_row = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    X = np.tile(single_row, (30, 1))
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=30, expected_dim=3)


def test_stress_all_rows_all_zero_vectors() -> None:
    """Stress test: 25 patients with all-zero co-medications (Jaccard 0/0 division)."""
    X = np.zeros((25, 8), dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=25, expected_dim=3)


def test_stress_single_all_zero_row_among_active_patients() -> None:
    """Stress test: 20 active patients and 1 completely blank patient (all zeros)."""
    rng = np.random.RandomState(42)
    X = (rng.rand(21, 10) > 0.6).astype(np.uint8)
    X[0, :] = 0  # Patient 0 has zero co-medications
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=21, expected_dim=3)


def test_stress_all_ones_matrix() -> None:
    """Stress test: 20 patients taking all drugs (all ones)."""
    X = np.ones((20, 10), dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=20, expected_dim=3)


# =====================================================================
# 3. Disconnected Subgraphs & Orthogonal Vectors (Jaccard Distance 1.0)
# =====================================================================

def test_stress_identity_matrix_isolated_singletons() -> None:
    """Stress test: Identity matrix where every patient takes a distinct drug (all pairwise Jaccard = 1.0)."""
    N = 20
    X = np.eye(N, dtype=np.uint8)
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=N, expected_dim=3)


def test_stress_disconnected_islands() -> None:
    """Stress test: Two disjoint patient groups with zero overlap (Jaccard distance across groups = 1.0)."""
    # Group 1: 15 patients take drugs 0-3
    g1 = np.zeros((15, 8), dtype=np.uint8)
    g1[:, 0:4] = 1
    # Group 2: 15 patients take drugs 4-7
    g2 = np.zeros((15, 8), dtype=np.uint8)
    g2[:, 4:8] = 1

    X = np.vstack([g1, g2])
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=30, expected_dim=3)


def test_stress_multi_islands_with_singletons() -> None:
    """Stress test: 3 distinct islands + 5 completely isolated singletons."""
    island_a = np.tile([1, 1, 0, 0, 0, 0, 0, 0], (10, 1))
    island_b = np.tile([0, 0, 1, 1, 0, 0, 0, 0], (10, 1))
    island_c = np.tile([0, 0, 0, 0, 1, 1, 0, 0], (10, 1))
    singleton_1 = np.array([[0, 0, 0, 0, 0, 0, 1, 0]])
    singleton_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

    X = np.vstack([island_a, island_b, island_c, singleton_1, singleton_2]).astype(np.uint8)
    coords = project_umap_3d(X, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=32, expected_dim=3)


# =====================================================================
# 4. Strict Determinism Verification
# =====================================================================

def test_stress_determinism_repeated_runs() -> None:
    """STRICT DETERMINISM: Assert identical coordinates across repeated runs with random_state=42."""
    rng = np.random.RandomState(123)
    X = (rng.rand(35, 12) > 0.7).astype(np.uint8)

    run_1 = project_umap_3d(X, random_state=42, n_epochs=30)
    run_2 = project_umap_3d(X, random_state=42, n_epochs=30)
    run_3 = project_umap_3d(X, random_state=42, n_epochs=30)

    np.testing.assert_allclose(run_1, run_2, atol=1e-5, err_msg="Run 1 and Run 2 diverged!")
    np.testing.assert_allclose(run_2, run_3, atol=1e-5, err_msg="Run 2 and Run 3 diverged!")


def test_stress_determinism_small_cohorts() -> None:
    """Verify determinism even on fallback small cohorts (N=1, N=2, N=3)."""
    for n in [1, 2, 3]:
        X = np.ones((n, 4), dtype=np.uint8)
        run_1 = project_umap_3d(X, random_state=42)
        run_2 = project_umap_3d(X, random_state=42)
        np.testing.assert_array_equal(run_1, run_2, err_msg=f"N={n} runs are not deterministic!")


def test_stress_random_state_sensitivity() -> None:
    """Verify different seeds produce different projections (stochasticity properly controlled)."""
    rng = np.random.RandomState(456)
    X = (rng.rand(30, 8) > 0.6).astype(np.uint8)

    run_seed_42 = project_umap_3d(X, random_state=42, n_epochs=25)
    run_seed_99 = project_umap_3d(X, random_state=99, n_epochs=25)

    assert not np.allclose(run_seed_42, run_seed_99, atol=1e-3), (
        "Projections with seed 42 and seed 99 should not be identical!"
    )


def test_stress_interface_equivalence() -> None:
    """Verify embed_patients_3d and project_umap_3d return identical coordinates."""
    rng = np.random.RandomState(789)
    X = (rng.rand(25, 6) > 0.5).astype(np.uint8)

    c_emb = embed_patients_3d(X, random_state=42, n_epochs=20)
    c_proj = project_umap_3d(X, random_state=42, n_epochs=20)

    np.testing.assert_allclose(c_emb, c_proj, atol=1e-5)


# =====================================================================
# 5. Dynamic Types & Sparse Representations
# =====================================================================

def test_stress_scipy_sparse_csr_matrix() -> None:
    """Stress test: scipy.sparse.csr_matrix binary input."""
    dense = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.uint8)
    sparse_csr = sp.csr_matrix(dense)
    coords = project_umap_3d(sparse_csr, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=5, expected_dim=3)


def test_stress_scipy_sparse_csc_matrix() -> None:
    """Stress test: scipy.sparse.csc_matrix binary input."""
    dense = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.uint8)
    sparse_csc = sp.csc_matrix(dense)
    coords = project_umap_3d(sparse_csc, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=5, expected_dim=3)


def test_stress_pandas_dataframe_input() -> None:
    """Stress test: pandas DataFrame input."""
    df = pd.DataFrame({
        "drug_a": [1, 0, 1, 0, 1],
        "drug_b": [0, 1, 1, 0, 0],
        "drug_c": [1, 1, 0, 1, 0],
        "drug_d": [0, 0, 0, 1, 1],
    })
    coords = project_umap_3d(df, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=5, expected_dim=3)


def test_stress_boolean_input_matrix() -> None:
    """Stress test: Boolean numpy array."""
    bool_arr = np.array([
        [True, False, True],
        [False, True, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
    ], dtype=bool)
    coords = project_umap_3d(bool_arr, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=5, expected_dim=3)


def test_stress_float_binary_matrix() -> None:
    """Stress test: Float32 representation of binary indicators."""
    float_arr = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    coords = project_umap_3d(float_arr, random_state=42, n_epochs=20)
    assert_valid_embedding(coords, expected_n=5, expected_dim=3)


# =====================================================================
# 6. Class & Pipeline Invariants (UMAPEmbedder)
# =====================================================================

def test_stress_embedder_class_fit_and_transform() -> None:
    """Verify UMAPEmbedder class scikit-learn API compatibility."""
    X_train = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.uint8)

    embedder = UMAPEmbedder(n_components=3, random_state=42, n_epochs=20, n_neighbors=3)
    coords = embedder.fit_transform(X_train)

    assert embedder.n_features_in_ == 4
    assert embedder.embedding_ is not None
    assert embedder.embedding_.shape == (6, 3)
    assert_valid_embedding(coords, expected_n=6, expected_dim=3)


def test_stress_embedder_transform_before_fit_raises() -> None:
    """Verify transform() before fit() raises clean RuntimeError."""
    embedder = UMAPEmbedder(n_components=3)
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        embedder.transform(np.array([[1, 0, 1]]))


def test_stress_high_sparsity_and_features() -> None:
    """Stress test: 50 patients, 200 features, 98% sparse."""
    rng = np.random.RandomState(42)
    sparse_data = (rng.rand(50, 200) > 0.98).astype(np.uint8)
    coords = project_umap_3d(sparse_data, random_state=42, n_epochs=25)
    assert_valid_embedding(coords, expected_n=50, expected_dim=3)
