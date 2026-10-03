"""Clinical statistical profiler and Odds Ratio calculation engine for Vigi-Pheno.

Calculates cluster demographic summaries (age, female percentage, hospitalization rate),
co-medication prevalence and exposure Odds Ratios (OR) with Haldane-Anscombe correction,
and adverse psychiatric reaction frequencies.
"""

from __future__ import annotations

import ast
from collections import Counter
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_odds_ratio(
    a: int | float,
    b: int | float,
    c: int | float,
    d: int | float,
) -> float:
    """Computes the exposure Odds Ratio (OR) for a 2x2 contingency table.

    Applies the Haldane-Anscombe correction (+0.5 added to each cell) if any cell
    is zero or if division by zero would occur, eliminating infinite or NaN values.

    Parameters
    ----------
    a : int | float
        Count of patients in cluster taking the drug (Exposed Cases).
    b : int | float
        Count of patients in cluster NOT taking the drug (Unexposed Cases).
    c : int | float
        Count of patients outside cluster taking the drug (Exposed Non-Cases).
    d : int | float
        Count of patients outside cluster NOT taking the drug (Unexposed Non-Cases).

    Returns
    -------
    float
        Calculated Odds Ratio. Returns 1.0 if both margins are zero.
    """
    if a < 0 or b < 0 or c < 0 or d < 0:
        raise ValueError(f"Contingency counts cannot be negative: a={a}, b={b}, c={c}, d={d}")

    # No comparison possible if either margin is zero
    if (a + b) == 0 or (c + d) == 0:
        return 1.0

    # Haldane-Anscombe correction applied when any cell is zero
    if a == 0 or b == 0 or c == 0 or d == 0:
        or_val = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    else:
        denom = float(b * c)
        if denom == 0.0:
            or_val = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        else:
            or_val = float(a * d) / denom

    if np.isnan(or_val) or np.isinf(or_val):
        return 1.0
    return float(or_val)


class ClusterProfile(dict):
    """Container for cluster clinical metrics supporting both dict and attribute access.

    Attributes / Keys
    -----------------
    patient_count : int
        Number of patients in cluster.
    percentage : float
        Percentage of total cohort (0.0 to 100.0).
    cohort_pct : float
        Alias of percentage.
    mean_age : float
        Mean patient onset age in years.
    female_pct : float
        Percentage of female patients (0.0 to 100.0).
    sex_ratio : float
        Alias of female_pct.
    hospitalization_rate : float
        Percentage of hospitalized cases (0.0 to 100.0).
    hosp_delta : float
        Delta compared to background cohort hospitalization rate.
    top_drugs : List[Tuple[str, float, float]]
        List of (drug_name, prevalence_pct, odds_ratio) tuples.
    top_reactions : List[Tuple[str, float]]
        List of (reaction_name, prevalence_pct) tuples.
    cluster_id : int
        HDBSCAN cluster ID (-1 for noise, >=0 for phenotypes).
    label_name : str
        Human-readable cluster name.
    is_noise : bool
        True if cluster_id == -1.
    """

    def __init__(
        self,
        patient_count: int,
        percentage: float,
        mean_age: float,
        female_pct: float,
        hospitalization_rate: float,
        top_drugs: List[Tuple[str, float, float]],
        top_reactions: List[Tuple[str, float]],
        cluster_id: int = -1,
        label_name: str = "",
        is_noise: bool = False,
        hosp_delta: float = 0.0,
        **kwargs: Any,
    ) -> None:
        data: Dict[str, Any] = {
            "patient_count": int(patient_count),
            "percentage": float(percentage),
            "cohort_pct": float(percentage),
            "mean_age": float(mean_age),
            "female_pct": float(female_pct),
            "sex_ratio": float(female_pct),
            "hospitalization_rate": float(hospitalization_rate),
            "hosp_delta": float(hosp_delta),
            "top_drugs": top_drugs,
            "top_reactions": top_reactions,
            "cluster_id": int(cluster_id),
            "label_name": label_name or (f"Cluster {cluster_id}" if cluster_id != -1 else "Noise (Outliers)"),
            "is_noise": bool(is_noise or cluster_id == -1),
        }
        data.update(kwargs)
        super().__init__(data)
        self.__dict__.update(data)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'ClusterProfile' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
        self.__dict__[key] = value


def profile_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    top_n_drugs: int = 10,
    top_n_reactions: int = 10,
    cohort_hosp_baseline: Optional[float] = None,
) -> Dict[int, ClusterProfile]:
    """Computes comprehensive clinical and polypharmacy profiles for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned patient demographic dataframe aligned with rows of feature_matrix.
    cluster_labels : np.ndarray
        Array of shape (N,) with cluster IDs.
    feature_matrix : np.ndarray
        Binary presence-absence feature matrix of shape (N, D).
    feature_names : List[str]
        List of D drug names corresponding to feature_matrix columns.
    top_n_drugs : int, default=10
        Maximum number of enriched drugs to return per cluster.
    top_n_reactions : int, default=10
        Maximum number of frequent adverse reactions to return per cluster.
    cohort_hosp_baseline : Optional[float], default=None
        Optional precomputed cohort hospitalization rate. Calculated from df if None.

    Returns
    -------
    Dict[int, ClusterProfile]
        Mapping from cluster ID (including -1 for noise) to ClusterProfile.
    """
    total_patients = len(df)
    if total_patients == 0:
        return {}

    if not isinstance(cluster_labels, np.ndarray):
        cluster_labels = np.asarray(cluster_labels)

    unique_labels = sorted(set(cluster_labels))
    profiles: Dict[int, ClusterProfile] = {}

    # Calculate background cohort hospitalization baseline
    if cohort_hosp_baseline is None:
        if "seriousnesshospitalization" in df.columns:
            hosp_num = pd.to_numeric(df["seriousnesshospitalization"], errors="coerce").fillna(0)
            cohort_hosp_baseline = float((hosp_num == 1).mean() * 100.0)
        else:
            cohort_hosp_baseline = 0.0

    for cl_id in unique_labels:
        mask = (cluster_labels == cl_id)
        cl_count = int(np.sum(mask))
        if cl_count == 0:
            continue

        cl_df = df.iloc[mask] if isinstance(df, pd.DataFrame) else df[mask]

        # 1. Cohort percentage
        percentage = round(float((cl_count / total_patients) * 100.0), 2)

        # 2. Mean age
        if "patientonsetage" in cl_df.columns:
            ages = pd.to_numeric(cl_df["patientonsetage"], errors="coerce").dropna()
            mean_age = round(float(ages.mean()), 1) if not ages.empty else 52.0
        else:
            mean_age = 52.0

        # 3. Female percentage (OpenFDA sex: 1=Male, 2=Female, 0=Unknown)
        if "patientsex" in cl_df.columns:
            sex_series = pd.to_numeric(cl_df["patientsex"], errors="coerce").fillna(0)
            female_count = int(np.sum(sex_series == 2))
            female_pct = round(float((female_count / cl_count) * 100.0), 1)
        else:
            female_pct = 50.0

        # 4. Hospitalization rate & Delta
        if "seriousnesshospitalization" in cl_df.columns:
            hosp_series = pd.to_numeric(cl_df["seriousnesshospitalization"], errors="coerce").fillna(0)
            hosp_count = int(np.sum(hosp_series == 1))
            hosp_rate = round(float((hosp_count / cl_count) * 100.0), 1)
        else:
            hosp_rate = 0.0

        hosp_delta = round(float(hosp_rate - cohort_hosp_baseline), 1)

        # 5. Top co-medications with prevalence and Odds Ratio
        cl_matrix = feature_matrix[mask]
        rest_matrix = feature_matrix[~mask]
        rest_count = total_patients - cl_count

        drug_candidates: List[Tuple[str, float, float]] = []
        for j, drug in enumerate(feature_names):
            a = int(np.sum(cl_matrix[:, j]))
            b = cl_count - a
            c = int(np.sum(rest_matrix[:, j])) if rest_count > 0 else 0
            d = rest_count - c if rest_count > 0 else 0
            prev_pct = round(float((a / cl_count) * 100.0), 1)

            if a > 0:
                or_val = round(compute_odds_ratio(a, b, c, d), 2)
                drug_candidates.append((drug, prev_pct, or_val))

        # Sort primarily by prevalence descending, then by Odds Ratio descending
        drug_candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        top_drugs = drug_candidates[:top_n_drugs]

        # 6. Top adverse psychiatric reactions
        rx_counts: Counter[str] = Counter()
        if "reactions" in cl_df.columns:
            for val in cl_df["reactions"]:
                rx_set = set()
                if isinstance(val, (list, tuple, set)):
                    for r in val:
                        if r and isinstance(r, str):
                            rx_set.add(r.strip().upper())
                elif isinstance(val, str):
                    s = val.strip()
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            parsed = ast.literal_eval(s)
                            if isinstance(parsed, list):
                                for r in parsed:
                                    if r and isinstance(r, str):
                                        rx_set.add(r.strip().upper())
                        except Exception:
                            rx_set.add(s.upper())
                    else:
                        rx_set.add(s.upper())
                for r in rx_set:
                    rx_counts[r] += 1

        top_reactions = [
            (rx, round(float((cnt / cl_count) * 100.0), 1))
            for rx, cnt in rx_counts.most_common(top_n_reactions)
        ]

        label_name = f"Cluster {cl_id}" if cl_id != -1 else "Noise (Outliers)"
        profiles[int(cl_id)] = ClusterProfile(
            patient_count=cl_count,
            percentage=percentage,
            mean_age=mean_age,
            female_pct=female_pct,
            hospitalization_rate=hosp_rate,
            hosp_delta=hosp_delta,
            top_drugs=top_drugs,
            top_reactions=top_reactions,
            cluster_id=int(cl_id),
            label_name=label_name,
            is_noise=(cl_id == -1),
        )

    return profiles


__all__ = ["compute_odds_ratio", "ClusterProfile", "profile_clusters"]
