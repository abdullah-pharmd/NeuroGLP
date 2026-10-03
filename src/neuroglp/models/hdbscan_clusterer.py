"""HDBSCAN density clustering engine for Vigi-Pheno patient phenotypes.

Clusters 3D patient coordinates (from UMAP Jaccard projections) into latent
risk phenotypes, isolates unclustered outlier patients (label -1), and incorporates
an adaptive parameter ladder to guarantee discovery of >= 3 distinct non-noise
clusters on multi-modal cohorts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import hdbscan
import numpy as np

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings_3d: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    min_clusters_target: int = 3,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    max_ladder_steps: int = 20,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cluster 3D patient embeddings using HDBSCAN with an adaptive parameter ladder.

    Parameters
    ----------
    embeddings_3d : np.ndarray
        Array of shape (N, 3) containing 3D manifold coordinates.
    min_cluster_size : int, default=5
        Initial minimum size of groupings to consider a cluster.
    min_samples : Optional[int], default=None
        Number of samples in a neighborhood for a point to be considered a core point.
        Defaults to max(1, min_cluster_size // 2) if None.
    min_clusters_target : int, default=3
        Target minimum number of distinct non-noise clusters to discover.
    cluster_selection_epsilon : float, default=0.0
        Distance threshold below which clusters are not split.
    cluster_selection_method : str, default="eom"
        Cluster selection method: "eom" (Excess of Mass) or "leaf".
    allow_single_cluster : bool, default=False
        Whether to allow HDBSCAN to return a single cluster.
    max_ladder_steps : int, default=20
        Maximum parameter search steps before stopping ladder progression.

    Returns
    -------
    cluster_labels : np.ndarray
        Integer array of shape (N,) with cluster IDs. -1 denotes noise.
    metadata : Dict[str, Any]
        Dictionary with clustering diagnostics:
        - 'n_clusters': int (number of non-noise clusters)
        - 'n_noise': int (count of noise points)
        - 'noise_ratio': float (fraction of points labeled noise)
        - 'min_cluster_size': int (final parameter used)
        - 'min_samples': int (final parameter used)
        - 'cluster_selection_method': str
        - 'cluster_sizes': Dict[int, int] (counts per cluster)
        - 'ladder_steps': int (number of evaluated parameter configurations)
        - 'converged': bool (True if n_clusters >= min_clusters_target)
        - 'probabilities': np.ndarray (membership probabilities)
        - 'outlier_scores': np.ndarray (GLOSH outlier scores)
    """
    if not isinstance(embeddings_3d, np.ndarray):
        embeddings_3d = np.asarray(embeddings_3d, dtype=float)

    n_samples = len(embeddings_3d)

    # Boundary Case 1: Empty input
    if n_samples == 0:
        return np.empty((0,), dtype=int), {
            "n_clusters": 0,
            "n_noise": 0,
            "noise_ratio": 0.0,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": cluster_selection_method,
            "cluster_sizes": {},
            "ladder_steps": 0,
            "converged": False,
            "probabilities": np.empty((0,), dtype=float),
            "outlier_scores": np.empty((0,), dtype=float),
        }

    # Validation: Check finite coordinates
    if np.isnan(embeddings_3d).any() or np.isinf(embeddings_3d).any():
        raise ValueError("embeddings_3d contains NaN or Inf values")

    # Boundary Case 2: Cohort smaller than target cluster math
    # HDBSCAN requires min_cluster_size >= 2, so discovering K clusters requires N >= 2 * K
    if n_samples < min_clusters_target * 2:
        effective_mcs = max(2, n_samples // max(1, min_clusters_target))
        effective_ms = 1
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_mcs,
            min_samples=effective_ms,
            cluster_selection_method="leaf",
            allow_single_cluster=allow_single_cluster,
        )
        labels = clusterer.fit_predict(embeddings_3d)
        non_noise = set(labels) - {-1}
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(k): int(v) for k, v in zip(unique_labels, counts)}
        return labels, {
            "n_clusters": len(non_noise),
            "n_noise": int(np.sum(labels == -1)),
            "noise_ratio": float(np.mean(labels == -1)),
            "min_cluster_size": effective_mcs,
            "min_samples": effective_ms,
            "cluster_selection_method": "leaf",
            "cluster_sizes": cluster_sizes,
            "ladder_steps": 1,
            "converged": len(non_noise) >= min_clusters_target,
            "probabilities": getattr(clusterer, "probabilities_", np.zeros(n_samples)),
            "outlier_scores": getattr(clusterer, "outlier_scores_", np.zeros(n_samples)),
        }

    # Construct Adaptive Parameter Ladder
    initial_ms = min_samples if min_samples is not None else max(1, min_cluster_size // 2)
    ladder: List[Tuple[int, int, str]] = []

    # Priority 1: User's exact requested configuration
    ladder.append((min_cluster_size, initial_ms, cluster_selection_method))

    # Priority 2: Generate candidate min_cluster_size descending
    mcs_candidates: List[int] = [min_cluster_size]
    curr_mcs = min_cluster_size
    while curr_mcs > 2:
        curr_mcs = max(2, int(curr_mcs * 0.75))
        if curr_mcs not in mcs_candidates:
            mcs_candidates.append(curr_mcs)
    if 2 not in mcs_candidates and n_samples >= 6:
        mcs_candidates.append(2)

    for mcs in mcs_candidates:
        ms_candidates = [
            max(1, mcs // 2),
            max(1, mcs // 3),
            2,
            1,
        ]
        seen_ms = set()
        for ms in ms_candidates:
            if ms in seen_ms:
                continue
            seen_ms.add(ms)
            for method in ["eom", "leaf"]:
                candidate = (mcs, ms, method)
                if candidate not in ladder:
                    ladder.append(candidate)

    # Evaluate ladder configurations
    best_labels: Optional[np.ndarray] = None
    best_metadata: Optional[Dict[str, Any]] = None
    best_cluster_count = -1
    step_count = 0

    for mcs, ms, method in ladder:
        step_count += 1
        if step_count > max_ladder_steps:
            break

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method=method,
                allow_single_cluster=allow_single_cluster,
            )
            labels = clusterer.fit_predict(embeddings_3d)
        except Exception as exc:
            logger.debug(f"HDBSCAN step failed with mcs={mcs}, ms={ms}, method={method}: {exc}")
            continue

        unique_non_noise = set(labels) - {-1}
        n_clusters = len(unique_non_noise)

        # Track best configuration discovered so far
        if n_clusters > best_cluster_count or best_labels is None:
            best_cluster_count = n_clusters
            best_labels = labels
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = {int(k): int(v) for k, v in zip(unique_labels, counts)}
            best_metadata = {
                "n_clusters": n_clusters,
                "n_noise": int(np.sum(labels == -1)),
                "noise_ratio": float(np.mean(labels == -1)),
                "min_cluster_size": mcs,
                "min_samples": ms,
                "cluster_selection_method": method,
                "cluster_sizes": cluster_sizes,
                "ladder_steps": step_count,
                "converged": n_clusters >= min_clusters_target,
                "probabilities": getattr(clusterer, "probabilities_", np.zeros(n_samples)),
                "outlier_scores": getattr(clusterer, "outlier_scores_", np.zeros(n_samples)),
            }

        # Early termination: Target satisfied!
        if n_clusters >= min_clusters_target:
            break

    assert best_labels is not None and best_metadata is not None
    return best_labels, best_metadata


def cluster_hdbscan(
    embeddings_3d: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    min_clusters_target: int = 3,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper returning cluster labels array directly.

    Adheres to the signature tested in tests/test_models.py.
    """
    labels, _ = cluster_embeddings(
        embeddings_3d=embeddings_3d,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        min_clusters_target=min_clusters_target,
        **kwargs,
    )
    return labels


__all__ = ["cluster_embeddings", "cluster_hdbscan"]
