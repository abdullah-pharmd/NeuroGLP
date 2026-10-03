"""Cluster KPI computations and analytics for NeuroGLP dashboard."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd


def compute_cluster_kpis(df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
    """Compute epidemiological and clinical KPIs for a specific cluster versus the background cohort.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'cluster_label', 'patientonsetage', 'patientsex', and 'seriousnesshospitalization'.
    cluster_id : int
        The cluster ID to evaluate (-1 for noise, or 0, 1, 2, ...).

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'patient_count', 'cohort_pct', 'mean_age', 'female_pct',
        'hospitalization_rate', and 'hosp_delta'.
    """
    total_cohort = len(df)
    if total_cohort == 0:
        return {
            "patient_count": 0,
            "cohort_pct": 0.0,
            "mean_age": 0.0,
            "female_pct": 0.0,
            "hospitalization_rate": 0.0,
            "hosp_delta": 0.0,
        }

    cohort_hosp_rate = (
        (df["seriousnesshospitalization"] == 1).mean() * 100.0
        if "seriousnesshospitalization" in df.columns
        else 0.0
    )

    cluster_df = df[df["cluster_label"] == cluster_id]
    patient_count = len(cluster_df)
    cohort_pct = (patient_count / total_cohort) * 100.0

    if patient_count == 0:
        return {
            "patient_count": 0,
            "cohort_pct": 0.0,
            "mean_age": 0.0,
            "female_pct": 0.0,
            "hospitalization_rate": 0.0,
            "hosp_delta": 0.0,
        }

    mean_age = (
        float(cluster_df["patientonsetage"].mean())
        if "patientonsetage" in cluster_df.columns
        else 0.0
    )

    female_pct = (
        float((cluster_df["patientsex"] == 2).mean() * 100.0)
        if "patientsex" in cluster_df.columns
        else 0.0
    )

    hosp_rate = (
        float((cluster_df["seriousnesshospitalization"] == 1).mean() * 100.0)
        if "seriousnesshospitalization" in cluster_df.columns
        else 0.0
    )

    hosp_delta = hosp_rate - cohort_hosp_rate

    return {
        "patient_count": patient_count,
        "cohort_pct": cohort_pct,
        "mean_age": mean_age,
        "female_pct": female_pct,
        "hospitalization_rate": hosp_rate,
        "hosp_delta": hosp_delta,
    }
