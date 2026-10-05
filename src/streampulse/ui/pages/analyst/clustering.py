# spotify-streampulse/src/streampulse/ui/pages/analyst/clustering.py
"""
Unsupervised Clustering: K-Means and Agglomerative Hierarchical Clustering.
R-ML-23..26: Elbow, Silhouette, Dendrogram analysis, and cluster profiling.
"""
from __future__ import annotations
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_CLUSTERING
from streampulse.ml.clustering import compute_clustering_agreement
from streampulse.ui.components.cache import (
    evaluate_kmeans_k_range,
    train_agglomerative,
    train_kmeans,
)
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_CLUSTERING)
def render():
    page_header(
        title="Audience Segmentation & Clustering",
        subtitle="Unsupervised grouping via K-Means and Agglomerative Hierarchical Clustering",
        concept="Unsupervised Learning",
        icon="🧮",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data.")
        return

    st.info("💡 **Feature Engineering Notice (Rule R-ML-23):** Due to strong collinearity (> 0.90), `Duration_Minutes` is dropped from clustering feature vectors to prevent distance distortion, isolating independent behavioral dimensions.")

    tab_eval, tab_km, tab_agg, tab_agree = st.tabs([
        "📈 Elbow & Silhouette Optimization",
        "🎯 K-Means Segmentation",
        "🌲 Agglomerative Clustering",
        "🤝 Algorithm Agreement (ARI)",
    ])

    with tab_eval:
        st.markdown("#### Optimal Cluster Count (k) Evaluation")
        k_eval = evaluate_kmeans_k_range(df)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=k_eval["k_values"], y=k_eval["inertias"], mode="lines+markers", name="Elbow (Inertia)", yaxis="y1"))
        fig_eval.add_trace(go.Scatter(x=k_eval["k_values"], y=k_eval["silhouettes"], mode="lines+markers", name="Silhouette Score", yaxis="y2"))

        fig_eval.update_layout(
            title="K-Means Elbow & Silhouette Across k ∈ [2, 10]",
            xaxis=dict(title="Number of Clusters (k)"),
            yaxis=dict(title="Inertia (Sum of Squared Distances)", side="left"),
            yaxis2=dict(title="Silhouette Coefficient", overlaying="y", side="right"),
            legend=dict(x=0.8, y=0.9),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_eval, use_container_width=True)

    with tab_km:
        k_choice = st.slider("Select k for K-Means", min_value=2, max_value=6, value=3)
        km_res = train_kmeans(df, k=k_choice)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Inertia", f"{km_res.metrics['inertia']:,}")
        with c2:
            st.metric("Silhouette Score", f"{km_res.metrics['silhouette']:.4f}")

        st.markdown("#### Cluster Profiles & Behavioral Archetypes")
        st.dataframe(pd.DataFrame(km_res.profile), use_container_width=True, hide_index=True)

    with tab_agg:
        linkage_method = st.selectbox("Linkage Method", options=["ward", "complete", "average"])
        k_agg = st.slider("Select Clusters for Agglomerative", min_value=2, max_value=6, value=3)
        agg_res, _ = train_agglomerative(df, k=k_agg, linkage_method=linkage_method)

        st.metric("Agglomerative Silhouette Score", f"{agg_res.metrics['silhouette']:.4f}")
        st.markdown("#### Agglomerative Cluster Profiles")
        st.dataframe(pd.DataFrame(agg_res.profile), use_container_width=True, hide_index=True)

    with tab_agree:
        st.markdown("#### Partition Agreement Metric")
        km_3 = train_kmeans(df, k=3)
        agg_3, _ = train_agglomerative(df, k=3, linkage_method="ward")
        ari = compute_clustering_agreement(km_3.cluster_labels, agg_3.cluster_labels)

        st.metric("Adjusted Rand Index (ARI) at k=3", f"{ari:.4f}")
        st.markdown(
            """
            - **Interpretation:** The Adjusted Rand Index (ARI) measures similarity between two cluster assignments independently of label permutations.
            - Score of **1.0** indicates identical clusterings; score of **0.0** indicates random labeling.
            """
        )
