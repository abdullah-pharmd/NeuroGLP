"""Vigi-Pheno: Interactive Streamlit Dashboard for GLP-1 Psychiatric Adverse Event Phenotyping.

Clinical Pharmacovigilance & Polypharmacy Research Platform.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from neuroglp.data.parser import flatten_reports
from neuroglp.features.builder import build_feature_matrix
from neuroglp.models.pipeline import fit_phenotyping_pipeline
from neuroglp.app.plots import render_3d_phenotype_scatter
from neuroglp.app.dashboard import compute_cluster_kpis
from neuroglp.app.risk_rules import evaluate_polypharmacy_risks
from neuroglp.reporting.abstract_generator import generate_abstract

st.set_page_config(
    page_title="Vigi-Pheno | GLP-1 Psychiatric Adverse Event Phenotyper",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 Vigi-Pheno: Topological Phenotyping of GLP-1 Adverse Events")
st.caption(
    "Unsupervised Manifold Learning (UMAP) & Density-Based Clustering (HDBSCAN) on Real-World Pharmacovigilance Data | "
    "Clinical Pharmacovigilance Research Platform"
)


@st.cache_data
def load_and_process_data():
    """Load cached benchmark dataset and run the phenotyping pipeline."""
    from neuroglp.data.dataset import load_cached_or_benchmark_dataset
    
    records = load_cached_or_benchmark_dataset()
    df_flattened = flatten_reports(records)
    feature_matrix, feature_names, clean_df = build_feature_matrix(df_flattened, min_freq=0.005)
    
    res = fit_phenotyping_pipeline(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        clean_df=clean_df,
        n_components=3,
        random_state=42
    )
    
    # Generate abstract text in memory
    abstract_text = generate_abstract(res.cluster_stats)
    
    return res.clustered_df, res.cluster_stats, abstract_text, len(records)


with st.spinner("Executing high-dimensional manifold embedding and clustering..."):
    clustered_df, cluster_stats, abstract_text, total_raw_count = load_and_process_data()

# -------------------------------------------------------------
# Top Metric Bar
# -------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
active_clusters = [cid for cid in cluster_stats.keys() if cid != -1]
noise_stats = cluster_stats.get(-1, {})
total_pts = len(clustered_df)

with m_col1:
    st.metric("Curated Adverse Events", f"{total_pts:,}")
with m_col2:
    st.metric("Discovered Phenotypes", len(active_clusters))
with m_col3:
    st.metric("Unassigned Noise", f"{noise_stats.get('percentage', 0.0):.1f}%")
with m_col4:
    total_alerts = sum(len(evaluate_polypharmacy_risks(stats.get("top_drugs", []))) for stats in cluster_stats.values())
    st.metric("Polypharmacy Alerts", total_alerts)

st.markdown("---")

# -------------------------------------------------------------
# Tabs Interface
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 3D Topological Manifold",
    "🔬 Phenotype Deep-Dive",
    "⚠️ Polypharmacy Risk Engine",
    "📄 Research Abstract"
])

# -------------------------------------------------------------
# Tab 1: 3D Visualization
# -------------------------------------------------------------
with tab1:
    st.subheader("3D High-Dimensional Manifold Projection (UMAP Jaccard Metric)")
    st.markdown(
        "Each point represents an individual patient adverse event report projected from a sparse co-medication presence-absence space. "
        "Colors delineate distinct sub-phenotypes detected via density-based clustering."
    )
    fig_3d = render_3d_phenotype_scatter(clustered_df)
    st.plotly_chart(fig_3d, use_container_width=True)

# -------------------------------------------------------------
# Tab 2: Phenotype Deep-Dive
# -------------------------------------------------------------
with tab2:
    st.subheader("Clinical & Epidemiological Phenotype Profiles")
    selected_cluster = st.selectbox(
        "Select Cluster to Inspect:",
        options=sorted(list(cluster_stats.keys())),
        format_func=lambda c: f"Noise / Boundary (-1)" if c == -1 else f"Phenotype {c}"
    )

    kpis = compute_cluster_kpis(clustered_df, selected_cluster)
    c_stats = cluster_stats.get(selected_cluster, {})

    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        st.metric("Patients in Cluster", f"{kpis['patient_count']:,} ({kpis['cohort_pct']:.1f}%)")
    with k_col2:
        st.metric("Mean Onset Age", f"{kpis['mean_age']:.1f} yrs")
    with k_col3:
        st.metric("Female Representation", f"{kpis['female_pct']:.1f}%")
    with k_col4:
        delta_str = f"{kpis['hosp_delta']:+.1f}% vs cohort"
        st.metric("Hospitalization Rate", f"{kpis['hospitalization_rate']:.1f}%", delta=delta_str)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.write("#### Enriched Concomitant Medications")
        top_drugs = c_stats.get("top_drugs", [])
        if top_drugs:
            drug_df = pd.DataFrame(top_drugs, columns=["Drug Entity", "Prevalence (%)", "Odds Ratio / Enrichment"])
            st.dataframe(drug_df.head(10), use_container_width=True)
        else:
            st.info("No dominant co-medication enrichment in this cluster.")

    with p_col2:
        st.write("#### Enriched MedDRA Reactions")
        top_rx = c_stats.get("top_reactions", [])
        if top_rx:
            rx_df = pd.DataFrame(top_rx, columns=["MedDRA Preferred Term", "Prevalence (%)"])
            st.dataframe(rx_df.head(10), use_container_width=True)
        else:
            st.info("No reaction enrichment available.")

# -------------------------------------------------------------
# Tab 3: Polypharmacy Risk Engine
# -------------------------------------------------------------
with tab3:
    st.subheader("Clinical Pharmacology & Interaction Heuristics")
    st.markdown(
        "The risk engine systematically checks each cluster's medication signature against evidence-based pharmacokinetic "
        "and pharmacodynamic vulnerability rules."
    )

    for cid in sorted(list(cluster_stats.keys())):
        c_name = "Noise (-1)" if cid == -1 else f"Phenotype {cid}"
        c_drugs = cluster_stats[cid].get("top_drugs", [])
        alerts = evaluate_polypharmacy_risks(c_drugs)

        with st.expander(f"**{c_name}** ({len(alerts)} Clinical Risk Alerts Triggered)", expanded=(len(alerts) > 0)):
            if alerts:
                for a in alerts:
                    severity_color = "red" if a.get("severity") in ["CRITICAL", "HIGH"] else "orange"
                    st.markdown(f"### :{severity_color}[{a['title']}]")
                    st.write(f"**Clinical Mechanism:** {a['mechanism']}")
                    st.write(f"**Recommended Action:** :blue[{a['action']}]")
                    st.caption(f"Triggered by: {', '.join([d[0] for d in a.get('matched_drugs', [])])}")
                    st.divider()
            else:
                st.success("No high-risk polypharmacy heuristics triggered for this cluster.")

# -------------------------------------------------------------
# Tab 4: Research Abstract
# -------------------------------------------------------------
with tab4:
    st.subheader("Clinical Research Abstract Draft")
    st.download_button(
        label="📥 Download ABSTRACT.md",
        data=abstract_text,
        file_name="ABSTRACT.md",
        mime="text/markdown"
    )
    st.markdown(abstract_text)

