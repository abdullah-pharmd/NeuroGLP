"""Vigi-Pheno Unsupervised Phenotyping & Clinical Modeling Module.

Provides:
- UMAP 3D Jaccard manifold projection (project_umap_3d, embed_patients_3d, UMAPEmbedder)
- HDBSCAN density clustering & noise isolation (cluster_hdbscan, cluster_embeddings)
- Clinical statistical profiler & Odds Ratio engine (compute_odds_ratio, profile_clusters, ClusterProfile)
- Unified phenotyping pipeline coordinator (fit_phenotyping_pipeline, PhenotypingResults)
"""

from __future__ import annotations

from neuroglp.models.umap_embedder import (
    UMAPEmbedder,
    embed_patients_3d,
    project_umap_3d,
)
from neuroglp.models.hdbscan_clusterer import (
    cluster_embeddings,
    cluster_hdbscan,
)
from neuroglp.models.profiler import (
    ClusterProfile,
    compute_odds_ratio,
    profile_clusters,
)
from neuroglp.models.pipeline import (
    PhenotypingResults,
    fit_phenotyping_pipeline,
)

__all__ = [
    # UMAP Embedder
    "UMAPEmbedder",
    "project_umap_3d",
    "embed_patients_3d",
    # HDBSCAN Clusterer
    "cluster_hdbscan",
    "cluster_embeddings",
    # Clinical Profiler
    "compute_odds_ratio",
    "profile_clusters",
    "ClusterProfile",
    # End-to-End Pipeline
    "fit_phenotyping_pipeline",
    "PhenotypingResults",
]
