"""Unit and integration tests for Milestone 4: Streamlit 3D Dashboard & Risk Engine.

Tests Plotly Express 3D scatter figure generation, cluster deep-dive calculations,
and the polypharmacy clinical risk rules engine.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
import pytest

try:
    from neuroglp.app.plots import render_3d_phenotype_scatter
    from neuroglp.app.risk_rules import evaluate_polypharmacy_risks, POLYPHARMACY_RULES
    from neuroglp.app.dashboard import compute_cluster_kpis
except ImportError:
    render_3d_phenotype_scatter = None
    evaluate_polypharmacy_risks = None
    POLYPHARMACY_RULES = None
    compute_cluster_kpis = None

# =====================================================================
# Fixture & Data Contract Tests (Always Active)
# =====================================================================

def test_mock_clustered_df_app_contract(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies synthetic clustered dataframe has required 3D coordinates and cluster labels."""
    assert "umap_x" in synthetic_clustered_df.columns
    assert "umap_y" in synthetic_clustered_df.columns
    assert "umap_z" in synthetic_clustered_df.columns
    assert "cluster_label" in synthetic_clustered_df.columns
    assert len(synthetic_clustered_df) == 80


@pytest.fixture(autouse=True)
def _check_app_implemented(request: pytest.FixtureRequest) -> None:
    if request.node.name.startswith(("test_fixture", "test_mock", "test_sample", "test_synthetic")):
        return
    if render_3d_phenotype_scatter is None or evaluate_polypharmacy_risks is None:
        pytest.skip("neuroglp.app is not yet implemented")


# =====================================================================
# Tier 1 & 2: Plotly 3D Scatter Rendering Tests
# =====================================================================

def test_render_3d_phenotype_scatter_figure_type(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies render_3d_phenotype_scatter returns a valid Plotly Figure with scatter3d trace."""
    fig = render_3d_phenotype_scatter(synthetic_clustered_df)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # Check trace type is 3D scatter
    trace_types = [trace.type for trace in fig.data]
    assert "scatter3d" in trace_types


def test_render_3d_phenotype_scatter_hover_and_axes(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies 3D scene axes and hover data configuration."""
    fig = render_3d_phenotype_scatter(synthetic_clustered_df)
    
    # Assert scene axis titles
    layout = fig.layout
    assert "scene" in layout
    assert "xaxis" in layout.scene
    assert "yaxis" in layout.scene
    assert "zaxis" in layout.scene

    # Verify points count matches input dataframe
    total_points = sum(len(trace.x) for trace in fig.data)
    assert total_points == len(synthetic_clustered_df)


# =====================================================================
# Tier 1 & 2: Cluster Deep-Dive KPI Metrics Tests
# =====================================================================

def test_compute_cluster_kpis_calculation(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies cluster-level vs background cohort KPI delta calculations."""
    kpis_cl0 = compute_cluster_kpis(synthetic_clustered_df, cluster_id=0)
    
    assert isinstance(kpis_cl0, dict)
    assert "patient_count" in kpis_cl0
    assert "cohort_pct" in kpis_cl0
    assert "mean_age" in kpis_cl0
    assert "female_pct" in kpis_cl0
    assert "hospitalization_rate" in kpis_cl0
    assert "hosp_delta" in kpis_cl0

    # Cluster 0 has 25 patients
    assert kpis_cl0["patient_count"] == 25
    assert kpis_cl0["cohort_pct"] == pytest.approx(31.25, rel=0.1)


def test_compute_cluster_kpis_noise_cluster(synthetic_clustered_df: pd.DataFrame) -> None:
    """Verifies KPI computation works cleanly for unassigned noise (-1) cluster."""
    kpis_noise = compute_cluster_kpis(synthetic_clustered_df, cluster_id=-1)
    
    assert kpis_noise["patient_count"] == 5
    assert kpis_noise["cohort_pct"] == pytest.approx(6.25, rel=0.1)


# =====================================================================
# Tier 1 & 2: Polypharmacy Risk Rules Engine Tests
# =====================================================================

def test_polypharmacy_rule_ssri_snri_alert() -> None:
    """Rule 1: GLP-1 + SSRI/SNRI prevalence >= 20% triggers psychiatric decompensation warning."""
    top_drugs = [
        ("sertraline", 25.0, 4.5),
        ("escitalopram", 22.0, 3.8),
        ("aspirin", 5.0, 1.0)
    ]
    alerts = evaluate_polypharmacy_risks(top_drugs)
    alert_ids = [a["rule_id"] for a in alerts]
    
    assert "SSRI_SNRI" in alert_ids or any("SSRI" in a["title"] for a in alerts)
    ssri_alert = next(a for a in alerts if "SSRI" in a.get("rule_id", "") or "SSRI" in a["title"])
    assert "gastric emptying" in ssri_alert["mechanism"].lower() or "antidepressant" in ssri_alert["mechanism"].lower()


def test_polypharmacy_rule_oral_contraceptives_alert() -> None:
    """Rule 2: GLP-1 + Oral Contraceptives prevalence >= 10% triggers efficacy reduction warning."""
    top_drugs = [
        ("ethinyl estradiol", 15.0, 8.0),
        ("levonorgestrel", 12.0, 7.5),
        ("ibuprofen", 4.0, 1.0)
    ]
    alerts = evaluate_polypharmacy_risks(top_drugs)
    alert_ids = [a["rule_id"] for a in alerts]
    
    assert "CONTRACEPTIVE" in alert_ids or any("Contraceptive" in a["title"] for a in alerts)
    contraceptive_alert = next(a for a in alerts if "CONTRACEPTIVE" in a.get("rule_id", "") or "Contraceptive" in a["title"])
    assert "ovulation" in contraceptive_alert["mechanism"].lower() or "contraceptive" in contraceptive_alert["mechanism"].lower()


def test_polypharmacy_rule_antidiabetic_hypoglycemia_alert() -> None:
    """Rule 3: GLP-1 + Antidiabetics / Insulin prevalence >= 25% triggers neuroglycopenic crisis warning."""
    top_drugs = [
        ("metformin", 40.0, 3.5),
        ("insulin glargine", 28.0, 5.0)
    ]
    alerts = evaluate_polypharmacy_risks(top_drugs)
    assert any("ANTIDIABETIC" in a.get("rule_id", "") or "Hypoglycemi" in a["title"] for a in alerts)


def test_polypharmacy_rule_bupropion_cns_stimulants_alert() -> None:
    """Rule 4: GLP-1 + Bupropion / CNS Stimulants prevalence >= 8% triggers seizure / agitation warning."""
    top_drugs = [
        ("bupropion", 12.0, 4.0),
        ("atorvastatin", 10.0, 1.2)
    ]
    alerts = evaluate_polypharmacy_risks(top_drugs)
    assert any("BUPROPION" in a.get("rule_id", "") or "Seizure" in a["title"] or "Stimulant" in a["title"] for a in alerts)


def test_polypharmacy_rules_negative_case() -> None:
    """Verifies clusters without high-risk co-medications trigger zero false-positive warnings."""
    benign_drugs = [
        ("vitamin c", 15.0, 1.1),
        ("saline nasal spray", 12.0, 1.0),
        ("artificial tears", 8.0, 0.9)
    ]
    alerts = evaluate_polypharmacy_risks(benign_drugs)
    assert len(alerts) == 0
