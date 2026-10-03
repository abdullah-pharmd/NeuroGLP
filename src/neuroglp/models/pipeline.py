"""High-level phenotyping pipeline coordinator for Vigi-Pheno.

Coordinates UMAP 3D manifold projection, HDBSCAN phenotyping, and clinical
statistical profiling into a unified, reproducible pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from neuroglp.models.umap_embedder import embed_patients_3d, project_umap_3d
from neuroglp.models.hdbscan_clusterer import cluster_hdbscan
from neuroglp.models.profiler import ClusterProfile, profile_clusters

logger = logging.getLogger(__name__)


@dataclass
class PhenotypingResults:
    """Container holding end-to-end unsupervised phenotyping pipeline results.

    Supports both dataclass attribute access (e.g. `results.embeddings_3d`)
    and dictionary subscript access (e.g. `results['embeddings_3d']`).

    Attributes
    ----------
    embeddings_3d : np.ndarray
        Array of shape (N, 3) containing 3D UMAP coordinates.
    cluster_labels : np.ndarray
        Integer array of shape (N,) containing HDBSCAN cluster IDs.
    cluster_stats : Dict[int, ClusterProfile]
        Mapping from cluster ID to clinical profile statistics.
    clean_df : pd.DataFrame
        Aligned patient demographic dataframe.
    feature_names : List[str]
        List of standardized drug feature names.
    n_clusters : int
        Count of discovered non-noise clusters (>= 0).
    noise_count : int
        Count of unclustered noise patients (label -1).
    noise_pct : float
        Percentage of noise patients in cohort.
    clustered_df : pd.DataFrame
        Convenience merged DataFrame containing clean_df plus umap_x, umap_y,
        umap_z, and cluster_label columns.
    params : Dict[str, Any]
        Dictionary of pipeline configuration parameters and execution metrics.
    """

    embeddings_3d: np.ndarray
    cluster_labels: np.ndarray
    cluster_stats: Dict[int, ClusterProfile]
    clean_df: pd.DataFrame
    feature_names: List[str]
    n_clusters: int
    noise_count: int
    noise_pct: float
    clustered_df: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'PhenotypingResults' object has no attribute or key '{key}'")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        return [
            "embeddings_3d",
            "cluster_labels",
            "cluster_stats",
            "clean_df",
            "feature_names",
            "n_clusters",
            "noise_count",
            "noise_pct",
            "clustered_df",
            "params",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert results container into standard dictionary."""
        return {
            "embeddings_3d": self.embeddings_3d,
            "cluster_labels": self.cluster_labels,
            "cluster_stats": self.cluster_stats,
            "clean_df": self.clean_df,
            "feature_names": self.feature_names,
            "n_clusters": self.n_clusters,
            "noise_count": self.noise_count,
            "noise_pct": self.noise_pct,
            "clustered_df": self.clustered_df,
            "params": self.params,
        }


def fit_phenotyping_pipeline(
    feature_matrix: np.ndarray,
    clean_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    n_components: int = 3,
    metric: str = "jaccard",
    min_cluster_size: int = 10,
    min_samples: Optional[int] = 5,
    n_epochs: Optional[int] = 50,
    top_n_drugs: int = 10,
    top_n_reactions: int = 10,
    **kwargs: Any,
) -> PhenotypingResults:
    """Executes the complete unsupervised phenotyping and clinical profiling pipeline.

    Workflow:
    1. Projects binary feature matrix to 3D via UMAP with Jaccard metric and random_state.
    2. Clusters 3D embeddings via HDBSCAN with adaptive parameter ladder (targeting >= 3 clusters).
    3. Profiles epidemiological demographics, drug odds ratios, and reactions per cluster.
    4. Packages results into a unified PhenotypingResults container.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Binary presence-absence matrix of shape (N, D).
    clean_df : pd.DataFrame
        Aligned patient demographic DataFrame of length N.
    feature_names : Optional[List[str]], default=None
        List of D drug names. Defaults to generic names if None.
    random_state : int, default=42
        Random seed for deterministic UMAP reproducibility.
    n_components : int, default=3
        Dimensionality of embedding projection.
    metric : str, default="jaccard"
        Distance metric for binary co-medication matrix.
    min_cluster_size : int, default=10
        Initial HDBSCAN minimum cluster size.
    min_samples : Optional[int], default=5
        HDBSCAN neighborhood density samples.
    n_epochs : Optional[int], default=50
        UMAP optimization epochs (calibrated for fast test execution).
    top_n_drugs : int, default=10
        Top enriched drugs to include per cluster.
    top_n_reactions : int, default=10
        Top adverse reactions to include per cluster.

    Returns
    -------
    PhenotypingResults
        Complete results container with embeddings, cluster labels, and profiles.
    """
    t_start = time.perf_counter()

    if not isinstance(feature_matrix, np.ndarray):
        feature_matrix = np.asarray(feature_matrix)

    n_samples, n_features = feature_matrix.shape
    if n_samples != len(clean_df):
        raise ValueError(
            f"Row count mismatch: feature_matrix has {n_samples} rows, "
            f"clean_df has {len(clean_df)} rows."
        )

    if feature_names is None:
        feature_names = [f"drug_{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"feature_matrix columns ({n_features})."
        )

    # 1. Dimensionality Reduction (UMAP 3D Jaccard)
    logger.info("Projecting %d patients to 3D manifold via UMAP (metric=%s)...", n_samples, metric)
    embeddings_3d = embed_patients_3d(
        feature_matrix,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        n_epochs=n_epochs,
        **kwargs,
    )

    # 2. Phenotype Discovery (HDBSCAN Clustering)
    logger.info("Clustering 3D coordinates via HDBSCAN...")
    cluster_labels = cluster_hdbscan(
        embeddings_3d,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        min_clusters_target=3,
    )

    # 3. Clinical Profiling
    logger.info("Generating epidemiological profiles per cluster...")
    cluster_stats = profile_clusters(
        df=clean_df,
        cluster_labels=cluster_labels,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        top_n_drugs=top_n_drugs,
        top_n_reactions=top_n_reactions,
    )

    # 4. Merged clustered DataFrame for Streamlit/Plotly consumption
    clustered_df = clean_df.copy()
    clustered_df["umap_x"] = embeddings_3d[:, 0]
    clustered_df["umap_y"] = embeddings_3d[:, 1]
    clustered_df["umap_z"] = embeddings_3d[:, 2]
    clustered_df["cluster_label"] = cluster_labels

    non_noise_clusters = set(cluster_labels) - {-1}
    n_clusters = len(non_noise_clusters)
    noise_count = int(np.sum(cluster_labels == -1))
    noise_pct = round(float((noise_count / n_samples) * 100.0), 2) if n_samples > 0 else 0.0

    t_elapsed = round(float(time.perf_counter() - t_start), 4)
    params = {
        "random_state": random_state,
        "n_components": n_components,
        "metric": metric,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_epochs": n_epochs,
        "runtime_seconds": t_elapsed,
    }

    return PhenotypingResults(
        embeddings_3d=embeddings_3d,
        cluster_labels=cluster_labels,
        cluster_stats=cluster_stats,
        clean_df=clean_df,
        feature_names=feature_names,
        n_clusters=n_clusters,
        noise_count=noise_count,
        noise_pct=noise_pct,
        clustered_df=clustered_df,
        params=params,
    )


__all__ = ["PhenotypingResults", "fit_phenotyping_pipeline"]
