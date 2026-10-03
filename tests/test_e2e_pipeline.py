"""Tier 4: End-to-End Pipeline Integration Test.

Executes complete pipeline from raw OpenFDA adverse event records to:
1. Tabular record flattening & demographic parsing
2. Sparse binary presence-absence feature matrix (<0.5% filter, zero-NaN guarantee)
3. UMAP 3D Jaccard manifold embedding & HDBSCAN clustering (>=3 clusters)
4. Cluster epidemiological profiling & polypharmacy risk analysis
5. Interactive 3D Plotly figure generation
6. Research Abstract generation (ABSTRACT.md)

Enforces STRICT EXECUTION RUNTIME CEILING: total execution must complete in < 30.0 seconds.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

try:
    from neuroglp.data.parser import flatten_reports
    from neuroglp.features.builder import build_feature_matrix
    try:
        from neuroglp.models import fit_phenotyping_pipeline
    except ImportError:
        from neuroglp.models.pipeline import fit_phenotyping_pipeline
    from neuroglp.app.plots import render_3d_phenotype_scatter
    from neuroglp.app.risk_rules import evaluate_polypharmacy_risks
    from neuroglp.reporting.abstract_generator import generate_abstract
except ImportError:
    flatten_reports = None
    build_feature_matrix = None
    fit_phenotyping_pipeline = None
    render_3d_phenotype_scatter = None
    evaluate_polypharmacy_risks = None
    generate_abstract = None

if any(m is None for m in [flatten_reports, build_feature_matrix, fit_phenotyping_pipeline, render_3d_phenotype_scatter, evaluate_polypharmacy_risks, generate_abstract]):
    pytestmark = pytest.mark.skip(reason="Full neuroglp pipeline not yet implemented")


def test_full_e2e_pipeline_execution_under_30s(
    sample_raw_records: List[Dict[str, Any]],
    tmp_path: Path
) -> None:
    """Full End-to-End Pipeline test verifying seamless stage handoffs and <30s runtime."""
    start_time = time.perf_counter()

    # -------------------------------------------------------------
    # Stage 1: Flatten Raw OpenFDA Records
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    df_flattened = flatten_reports(sample_raw_records)
    t1 = time.perf_counter()
    assert len(df_flattened) == len(sample_raw_records)
    assert "safetyreportid" in df_flattened.columns
    assert "patientonsetage" in df_flattened.columns
    print(f"\n[E2E] Stage 1 (Flattening): {t1 - t0:.3f}s")

    # -------------------------------------------------------------
    # Stage 2: High-Dimensional Binary Feature Engineering
    # -------------------------------------------------------------
    matrix, feature_names, clean_df = build_feature_matrix(df_flattened, min_freq=0.01)
    t2 = time.perf_counter()
    assert matrix.shape[0] == len(clean_df)
    assert matrix.shape[1] == len(feature_names)
    # Mathematical Invariant
    assert np.isnan(matrix).sum() == 0
    print(f"[E2E] Stage 2 (Feature Matrix): {t2 - t1:.3f}s ({matrix.shape[0]}x{matrix.shape[1]})")

    # -------------------------------------------------------------
    # Stage 3: Unsupervised Phenotyping (UMAP 3D + HDBSCAN)
    # -------------------------------------------------------------
    results = fit_phenotyping_pipeline(
        feature_matrix=matrix,
        clean_df=clean_df,
        feature_names=feature_names,
        random_state=42
    )
    t3 = time.perf_counter()
    embeddings = results.embeddings_3d if hasattr(results, "embeddings_3d") else results["embeddings_3d"]
    labels = results.cluster_labels if hasattr(results, "cluster_labels") else results["cluster_labels"]
    stats = results.cluster_stats if hasattr(results, "cluster_stats") else results["cluster_stats"]

    assert embeddings.shape == (len(clean_df), 3)
    assert np.isnan(embeddings).sum() == 0
    assert len(labels) == len(clean_df)
    
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) >= 3, f"Expected >=3 clusters, found {len(unique_clusters)}"
    print(f"[E2E] Stage 3 (UMAP & HDBSCAN): {t3 - t2:.3f}s (Discovered {len(unique_clusters)} clusters)")

    # -------------------------------------------------------------
    # Stage 4: App Visuals & Risk Rules Simulation
    # -------------------------------------------------------------
    clustered_df = clean_df.copy()
    clustered_df["umap_x"] = embeddings[:, 0]
    clustered_df["umap_y"] = embeddings[:, 1]
    clustered_df["umap_z"] = embeddings[:, 2]
    clustered_df["cluster_label"] = labels

    fig = render_3d_phenotype_scatter(clustered_df)
    assert fig is not None

    # Polypharmacy rules evaluation across clusters
    total_alerts = 0
    for cl_id, profile in stats.items():
        if cl_id != -1 and "top_drugs" in profile:
            alerts = evaluate_polypharmacy_risks(profile["top_drugs"])
            total_alerts += len(alerts)
    t4 = time.perf_counter()
    print(f"[E2E] Stage 4 (Plotly & Polypharmacy Rules): {t4 - t3:.3f}s ({total_alerts} alerts)")

    # -------------------------------------------------------------
    # Stage 5: Research Abstract Generation
    # -------------------------------------------------------------
    abstract_out = tmp_path / "ABSTRACT.md"
    abstract_text = generate_abstract(stats, output_path=str(abstract_out))
    t5 = time.perf_counter()
    assert abstract_out.exists()
    assert "Background" in abstract_text
    assert "Results" in abstract_text
    print(f"[E2E] Stage 5 (Abstract Generation): {t5 - t4:.3f}s")

    # -------------------------------------------------------------
    # HARD GATE: Total Pipeline Runtime < 30.0 Seconds
    # -------------------------------------------------------------
    total_elapsed = time.perf_counter() - start_time
    print(f"[E2E] TOTAL PIPELINE EXECUTION TIME: {total_elapsed:.3f}s")
    assert total_elapsed < 30.0, (
        f"Pipeline breached the 30.0s hard ceiling: took {total_elapsed:.2f}s"
    )
