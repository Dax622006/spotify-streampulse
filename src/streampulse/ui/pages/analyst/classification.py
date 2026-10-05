# spotify-streampulse/src/streampulse/ui/pages/analyst/classification.py
"""
Classification Modeling: Decision Tree and Gaussian Naive Bayes for Subscription Tier Prediction.
R-ML-07, R-ML-12..18: Stratified evaluation, ROC-AUC, Confusion Matrix, Feature Importance.
"""
from __future__ import annotations
import plotly.express as px
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_CLASSIFICATION
from streampulse.ui.components.cache import train_decision_tree, train_naive_bayes
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_CLASSIFICATION)
def render():
    page_header(
        title="Subscription Tier Classification",
        subtitle="Predict Free vs Premium user tier using Decision Tree and Gaussian Naive Bayes",
        concept="Supervised Classification",
        icon="🎯",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data.")
        return

    depth = st.slider("Decision Tree Max Depth", min_value=2, max_value=10, value=5)
    dt_res = train_decision_tree(df, max_depth=depth)
    nb_res = train_naive_bayes(df)

    tab_cmp, tab_dt, tab_nb = st.tabs([
        "⚖️ Model Benchmark Comparison",
        "🌳 Decision Tree Classifier",
        "⚡ Gaussian Naive Bayes",
    ])

    with tab_cmp:
        st.markdown("#### Head-to-Head Performance (Test Set)")
        comp_df = pd.DataFrame([
            {
                "Algorithm": "Decision Tree",
                "Accuracy": f"{dt_res.metrics['accuracy']:.4f}",
                "Precision": f"{dt_res.metrics['precision']:.4f}",
                "Recall": f"{dt_res.metrics['recall']:.4f}",
                "F1-Score": f"{dt_res.metrics['f1']:.4f}",
                "ROC-AUC": f"{dt_res.metrics['roc_auc']:.4f}",
                "5-Fold CV Acc": f"{dt_res.metrics['cv_accuracy_mean']:.4f}",
            },
            {
                "Algorithm": "Gaussian Naive Bayes",
                "Accuracy": f"{nb_res.metrics['accuracy']:.4f}",
                "Precision": f"{nb_res.metrics['precision']:.4f}",
                "Recall": f"{nb_res.metrics['recall']:.4f}",
                "F1-Score": f"{nb_res.metrics['f1']:.4f}",
                "ROC-AUC": f"{nb_res.metrics['roc_auc']:.4f}",
                "5-Fold CV Acc": f"{nb_res.metrics['cv_accuracy_mean']:.4f}",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with tab_dt:
        st.markdown("#### Decision Tree Evaluation")
        d1, d2 = st.columns([1, 1])
        with d1:
            fig_cm_dt = px.imshow(
                dt_res.confusion_matrix,
                x=dt_res.classes,
                y=dt_res.classes,
                text_auto=True,
                title="Decision Tree Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale="Greens",
            )
            st.plotly_chart(fig_cm_dt, use_container_width=True)

        with d2:
            st.markdown("#### Feature Importance")
            if dt_res.feature_importances:
                df_fi = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in sorted(dt_res.feature_importances.items(), key=lambda x: x[1], reverse=True)
                    if v > 0
                ])
                fig_fi = px.bar(df_fi, x="Importance", y="Feature", orientation="h", title="Significant Features", color="Importance")
                st.plotly_chart(fig_fi, use_container_width=True)

    with tab_nb:
        st.markdown("#### Naive Bayes Evaluation")
        fig_cm_nb = px.imshow(
            nb_res.confusion_matrix,
            x=nb_res.classes,
            y=nb_res.classes,
            text_auto=True,
            title="Gaussian Naive Bayes Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm_nb, use_container_width=True)
