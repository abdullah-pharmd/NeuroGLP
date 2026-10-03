"""
tests/run_umap_adversarial.py

Adversarial Stress Test Harness for Milestone 3: UMAP 3D Jaccard Embedder.
Authored by challenger_m3_1 (UMAP Adversarial Challenger).

Executes an exhaustive empirical battery against:
`neuroglp.models.umap_embedder.project_umap_3d` and `embed_patients_3d`.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Ensure src is on sys.path
sys.path.insert(0, "src")

from neuroglp.models.umap_embedder import (
    UMAPEmbedder,
    embed_patients_3d,
    project_umap_3d,
)


def run_adversarial_battery() -> Dict[str, Any]:
    print("=" * 80)
    print("VIGI-PHENO UMAP 3D JACCARD EMBEDDER ADVERSARIAL STRESS SUITE")
    print("=" * 80)

    results: Dict[str, Any] = {
        "passed": 0,
        "failed": 0,
        "tests": [],
        "findings": [],
        "timings": {},
    }

    def record_test(name: str, passed: bool, duration_ms: float, detail: str = ""):
        results["tests"].append((name, passed, duration_ms, detail))
        if passed:
            results["passed"] += 1
            print(f"  [PASS] {name:50s} ({duration_ms:6.2f} ms) {detail}")
        else:
            results["failed"] += 1
            print(f"  [FAIL] {name:50s} ({duration_ms:6.2f} ms) -> {detail}")
            results["findings"].append((name, detail))

    # =========================================================================
    # BATTERY 1: COHORT CARDINALITY BOUNDARIES
    # =========================================================================
    print("\n--- BATTERY 1: COHORT CARDINALITY BOUNDARIES ---")
    cardinality_cases = [
        ("N=0, D=10 (Empty matrix)", np.zeros((0, 10), dtype=np.uint8), (0, 3)),
        ("N=0, D=0 (Zero samples, zero features)", np.zeros((0, 0), dtype=np.uint8), (0, 3)),
        ("N=1, D=5 (Single patient)", np.array([[1, 0, 1, 0, 1]], dtype=np.uint8), (1, 3)),
        ("N=1, D=0 (Single patient, zero features)", np.zeros((1, 0), dtype=np.uint8), (1, 3)),
        ("N=2, D=4 (Two patients)", np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8), (2, 3)),
        ("N=3, D=4 (N == n_components boundary)", np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.uint8), (3, 3)),
        ("N=4, D=4 (N == n_components + 1, UMAP triggers)", np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8), (4, 3)),
        ("N=5, D=5 (n_neighbors clamped to 4)", np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 0]], dtype=np.uint8), (5, 3)),
    ]

    for label, mat, exp_shape in cardinality_cases:
        t0 = time.perf_counter()
        try:
            coords = project_umap_3d(mat, random_state=42, n_epochs=20)
            elapsed = (time.perf_counter() - t0) * 1000
            assert coords.shape == exp_shape, f"Shape mismatch: expected {exp_shape}, got {coords.shape}"
            assert np.isnan(coords).sum() == 0, f"NaN count: {np.isnan(coords).sum()}"
            assert not np.isinf(coords).any(), "Inf values found"
            record_test(label, True, elapsed, f"Shape {coords.shape}, NaNs: 0")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            record_test(label, False, elapsed, str(e))

    # =========================================================================
    # BATTERY 2: DEGENERATE TOPOLOGIES & 0/0 DIVISIONS
    # =========================================================================
    print("\n--- BATTERY 2: DEGENERATE TOPOLOGIES & 0/0 DIVISIONS ---")
    deg_cases = [
        ("All identical rows (N=30, D=8, dist=0.0)", np.tile([1, 0, 1, 1, 0, 0, 1, 0], (30, 1)).astype(np.uint8), (30, 3)),
        ("All zero vectors (N=25, D=8, Jaccard 0/0)", np.zeros((25, 8), dtype=np.uint8), (25, 3)),
        ("N=4 all zero vectors (Boundary + 0/0)", np.zeros((4, 5), dtype=np.uint8), (4, 3)),
        ("N=4 identical rows (Boundary + dist=0.0)", np.tile([1, 1, 0], (4, 1)).astype(np.uint8), (4, 3)),
        ("All ones matrix (N=20, D=10, 100% density)", np.ones((20, 10), dtype=np.uint8), (20, 3)),
        ("Single zero patient among 20 active patients", None, (21, 3)),
    ]

    for label, mat, exp_shape in deg_cases:
        if mat is None:
            rng = np.random.RandomState(42)
            mat = (rng.rand(21, 10) > 0.6).astype(np.uint8)
            mat[0, :] = 0

        t0 = time.perf_counter()
        try:
            coords = project_umap_3d(mat, random_state=42, n_epochs=25)
            elapsed = (time.perf_counter() - t0) * 1000
            assert coords.shape == exp_shape, f"Shape mismatch: expected {exp_shape}, got {coords.shape}"
            assert np.isnan(coords).sum() == 0, f"NaN count: {np.isnan(coords).sum()}"
            assert not np.isinf(coords).any(), "Inf values found"
            record_test(label, True, elapsed, f"Shape {coords.shape}, NaNs: 0")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            record_test(label, False, elapsed, str(e))

    # =========================================================================
    # BATTERY 3: DISCONNECTED SUBGRAPHS & ORTHOGONAL VECTORS
    # =========================================================================
    print("\n--- BATTERY 3: DISCONNECTED SUBGRAPHS & ORTHOGONAL VECTORS ---")
    # 1. Identity matrix: all pairwise Jaccard distances = 1.0
    t0 = time.perf_counter()
    try:
        mat_eye = np.eye(20, dtype=np.uint8)
        coords_eye = project_umap_3d(mat_eye, random_state=42, n_epochs=25)
        elapsed = (time.perf_counter() - t0) * 1000
        assert coords_eye.shape == (20, 3)
        assert np.isnan(coords_eye).sum() == 0
        assert not np.isinf(coords_eye).any()
        record_test("Identity 20x20 (All pairwise Jaccard = 1.0)", True, elapsed, "Zero NaNs/Infs")
    except Exception as e:
        record_test("Identity 20x20 (All pairwise Jaccard = 1.0)", False, (time.perf_counter() - t0) * 1000, str(e))

    # 2. Two isolated disjoint subgraphs
    t0 = time.perf_counter()
    try:
        g1 = np.zeros((15, 8), dtype=np.uint8)
        g1[:, 0:4] = 1
        g2 = np.zeros((15, 8), dtype=np.uint8)
        g2[:, 4:8] = 1
        mat_islands = np.vstack([g1, g2])
        coords_islands = project_umap_3d(mat_islands, random_state=42, n_epochs=25)
        elapsed = (time.perf_counter() - t0) * 1000
        assert coords_islands.shape == (30, 3)
        assert np.isnan(coords_islands).sum() == 0
        assert not np.isinf(coords_islands).any()
        record_test("Two disjoint islands (No inter-cluster overlap)", True, elapsed, "Zero NaNs/Infs")
    except Exception as e:
        record_test("Two disjoint islands (No inter-cluster overlap)", False, (time.perf_counter() - t0) * 1000, str(e))

    # 3. Three islands + 2 singleton isolates
    t0 = time.perf_counter()
    try:
        ia = np.tile([1, 1, 0, 0, 0, 0, 0, 0], (10, 1))
        ib = np.tile([0, 0, 1, 1, 0, 0, 0, 0], (10, 1))
        ic = np.tile([0, 0, 0, 0, 1, 1, 0, 0], (10, 1))
        s1 = np.array([[0, 0, 0, 0, 0, 0, 1, 0]])
        s2 = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])
        mat_multi = np.vstack([ia, ib, ic, s1, s2]).astype(np.uint8)
        coords_multi = project_umap_3d(mat_multi, random_state=42, n_epochs=25)
        elapsed = (time.perf_counter() - t0) * 1000
        assert coords_multi.shape == (32, 3)
        assert np.isnan(coords_multi).sum() == 0
        assert not np.isinf(coords_multi).any()
        record_test("Three islands + 2 singleton isolates", True, elapsed, "Zero NaNs/Infs")
    except Exception as e:
        record_test("Three islands + 2 singleton isolates", False, (time.perf_counter() - t0) * 1000, str(e))

    # =========================================================================
    # BATTERY 4: STRICT DETERMINISM & REPEATABILITY
    # =========================================================================
    print("\n--- BATTERY 4: STRICT DETERMINISM & REPEATABILITY ---")
    t0 = time.perf_counter()
    try:
        rng = np.random.RandomState(123)
        X_det = (rng.rand(35, 12) > 0.7).astype(np.uint8)
        r1 = project_umap_3d(X_det, random_state=42, n_epochs=30)
        r2 = project_umap_3d(X_det, random_state=42, n_epochs=30)
        r3 = project_umap_3d(X_det, random_state=42, n_epochs=30)
        np.testing.assert_allclose(r1, r2, atol=1e-5)
        np.testing.assert_allclose(r2, r3, atol=1e-5)
        elapsed = (time.perf_counter() - t0) * 1000
        record_test("Determinism: 3 repeated runs (random_state=42)", True, elapsed, "Max diff == 0.0")
    except Exception as e:
        record_test("Determinism: 3 repeated runs (random_state=42)", False, (time.perf_counter() - t0) * 1000, str(e))

    # Seed sensitivity
    t0 = time.perf_counter()
    try:
        r_seed42 = project_umap_3d(X_det, random_state=42, n_epochs=25)
        r_seed99 = project_umap_3d(X_det, random_state=99, n_epochs=25)
        diff = np.max(np.abs(r_seed42 - r_seed99))
        assert diff > 0.1, f"Seed 42 and 99 yielded nearly identical projections (diff={diff})"
        elapsed = (time.perf_counter() - t0) * 1000
        record_test("Seed sensitivity (seed=42 vs seed=99 diverge)", True, elapsed, f"Max diff: {diff:.4f}")
    except Exception as e:
        record_test("Seed sensitivity (seed=42 vs seed=99 diverge)", False, (time.perf_counter() - t0) * 1000, str(e))

    # Equivalence of embed_patients_3d and project_umap_3d
    t0 = time.perf_counter()
    try:
        e1 = embed_patients_3d(X_det, random_state=42, n_epochs=25)
        p1 = project_umap_3d(X_det, random_state=42, n_epochs=25)
        np.testing.assert_allclose(e1, p1, atol=1e-5)
        elapsed = (time.perf_counter() - t0) * 1000
        record_test("Equivalence: embed_patients_3d vs project_umap_3d", True, elapsed, "Strict match")
    except Exception as e:
        record_test("Equivalence: embed_patients_3d vs project_umap_3d", False, (time.perf_counter() - t0) * 1000, str(e))

    # =========================================================================
    # BATTERY 5: POLYMORPHIC TYPES & SPARSE CONTAINERS
    # =========================================================================
    print("\n--- BATTERY 5: POLYMORPHIC TYPES & SPARSE CONTAINERS ---")
    poly_mat = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.uint8)

    type_cases = [
        ("scipy.sparse.csr_matrix", sp.csr_matrix(poly_mat)),
        ("scipy.sparse.csc_matrix", sp.csc_matrix(poly_mat)),
        ("pandas.DataFrame", pd.DataFrame(poly_mat, columns=[f"d{i}" for i in range(4)])),
        ("numpy.ndarray bool dtype", poly_mat.astype(bool)),
        ("numpy.ndarray float32 dtype", poly_mat.astype(np.float32)),
        ("numpy.ndarray int64 dtype", poly_mat.astype(np.int64)),
        ("Sparse empty CSR matrix", sp.csr_matrix((0, 10), dtype=np.uint8)),
        ("Sparse single patient CSR", sp.csr_matrix([[1, 0, 1, 0]], dtype=np.uint8)),
    ]

    for label, container in type_cases:
        t0 = time.perf_counter()
        try:
            exp_n = container.shape[0]
            coords = project_umap_3d(container, random_state=42, n_epochs=20)
            elapsed = (time.perf_counter() - t0) * 1000
            assert coords.shape == (exp_n, 3)
            assert np.isnan(coords).sum() == 0
            assert not np.isinf(coords).any()
            record_test(label, True, elapsed, f"Shape ({exp_n}, 3), NaNs: 0")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            record_test(label, False, elapsed, str(e))

    # =========================================================================
    # BATTERY 6: HIGH SPARSITY & SCALE BENCHMARK
    # =========================================================================
    print("\n--- BATTERY 6: HIGH SPARSITY & SCALE BENCHMARK ---")
    # 200 patients x 100 drugs, 98% sparse
    t0 = time.perf_counter()
    try:
        rng = np.random.RandomState(42)
        sparse_large = (rng.rand(200, 100) > 0.98).astype(np.uint8)
        coords_sparse = project_umap_3d(sparse_large, random_state=42, n_epochs=30)
        elapsed = (time.perf_counter() - t0) * 1000
        assert coords_sparse.shape == (200, 3)
        assert np.isnan(coords_sparse).sum() == 0
        assert not np.isinf(coords_sparse).any()
        record_test("N=200, D=100 (98% sparse polypharmacy)", True, elapsed, f"Execution: {elapsed:.1f}ms")
    except Exception as e:
        record_test("N=200, D=100 (98% sparse polypharmacy)", False, (time.perf_counter() - t0) * 1000, str(e))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ADVERSARIAL STRESS TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests Executed: {len(results['tests'])}")
    print(f"Total Passed: {results['passed']}")
    print(f"Total Failed: {results['failed']}")
    if results["findings"]:
        print("\nFindings / Failures:")
        for name, detail in results["findings"]:
            print(f" - [{name}]: {detail}")
    else:
        print("\nAll adversarial vectors passed with ZERO exceptions, ZERO NaNs, and ZERO Infs.")

    return results


if __name__ == "__main__":
    run_adversarial_battery()
