"""Plotly 3D scatter visualization for NeuroGLP clusters."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_3d_phenotype_scatter(df: pd.DataFrame) -> go.Figure:
    """Render an interactive 3D scatter plot of patient embeddings color-coded by HDBSCAN cluster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'umap_x', 'umap_y', 'umap_z', and 'cluster_label' columns.

    Returns
    -------
    go.Figure
        Plotly Figure configured with a 3D scatter trace and dark modern styling.
    """
    plot_df = df.copy()
    
    # Format cluster label for legend
    plot_df["Cluster"] = plot_df["cluster_label"].apply(
        lambda c: "Noise / Unassigned (-1)" if c == -1 else f"Phenotype {c}"
    )

    hover_cols = [c for c in ["safetyreportid", "patientonsetage", "patientsex"] if c in plot_df.columns]

    fig = px.scatter_3d(
        plot_df,
        x="umap_x",
        y="umap_y",
        z="umap_z",
        color="Cluster",
        hover_data=hover_cols,
        title="Vigi-Pheno: 3D Topological Phenotyping of GLP-1 Psychiatric Adverse Events",
        opacity=0.85,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="UMAP Component 1 (Topology)"),
            yaxis=dict(title="UMAP Component 2 (Dispersion)"),
            zaxis=dict(title="UMAP Component 3 (Enrichment)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig
