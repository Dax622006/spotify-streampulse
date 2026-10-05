# spotify-streampulse/src/streampulse/ui/pages/analyst/regression.py
"""
Regression Analysis: Simple (SLR) and Multiple Linear Regression (MLR).
R-ML-01..06, R-ML-11: Evaluation, VIF diagnostics, and residual distributions.
"""
from __future__ import annotations
import plotly.express as px
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_REGRESSION
from streampulse.ml.registry import registry
from streampulse.ui.components.cache import train_mlr, train_slr
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_REGRESSION)
def render():
    page_header(
        title="Revenue Regression Modeling",
        subtitle="Predict platform Revenue_USD via Simple (SLR) and Multiple Linear Regression (MLR)",
        concept="Supervised Regression",
        icon="📉",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data.")
        return

    # Train / Retrieve results
    slr_res = train_slr(df)
    mlr_res = train_mlr(df)

    tab_slr, tab_mlr, tab_diag = st.tabs([
        "📊 Simple Linear Regression (SLR)",
        "🌐 Multiple Linear Regression (MLR)",
        "🔬 Residuals & Collinearity (VIF)",
    ])

    with tab_slr:
        st.markdown("#### Simple Linear Regression: `Revenue_USD ~ Streams`")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Test R² Score", f"{slr_res.metrics['r2_test']:.4f}")
        with s2:
            st.metric("RMSE", f"${slr_res.metrics['rmse_test']:.2f}")
        with s3:
            st.metric("MAE", f"${slr_res.metrics['mae_test']:.2f}")
        with s4:
            st.metric("5-Fold CV R²", f"{slr_res.metrics['cv_r2_mean']:.4f}")

        st.markdown(f"**Regression Formula:** `Revenue_USD = {slr_res.intercept:.4f} + {slr_res.coefficients['Streams']:.4f} * Streams`")

        # Scatter with line
        fig_slr = px.scatter(
            x=slr_res.residuals["y_true"],
            y=slr_res.residuals["y_pred"],
            labels={"x": "Actual Revenue ($)", "y": "Predicted Revenue ($)"},
            title="SLR: Actual vs Predicted Revenue (Test Set)",
            opacity=0.7,
        )
        fig_slr.add_shape(type="line", line=dict(dash="dash", color="red"), x0=0, x1=7, y0=0, y1=7)
        st.plotly_chart(fig_slr, use_container_width=True)

    with tab_mlr:
        st.markdown("#### Multiple Linear Regression: `Revenue ~ Streams + Duration + Categoricals`")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Test R² Score", f"{mlr_res.metrics['r2_test']:.4f}")
        with m2:
            st.metric("Adjusted R²", f"{mlr_res.metrics['adj_r2_test']:.4f}")
        with m3:
            st.metric("RMSE", f"${mlr_res.metrics['rmse_test']:.2f}")
        with m4:
            st.metric("5-Fold CV R²", f"{mlr_res.metrics['cv_r2_mean']:.4f}")

        st.markdown("#### Model Coefficients")
        df_coef = pd.DataFrame([
            {"Feature": k, "Coefficient": v} for k, v in mlr_res.coefficients.items()
        ])
        st.dataframe(df_coef, use_container_width=True, hide_index=True)

    with tab_diag:
        st.markdown("#### Variance Inflation Factor (VIF) - Collinearity Audit")
        if mlr_res.vif:
            df_vif = pd.DataFrame([
                {"Numeric Feature": k, "VIF Score": v, "Status": "High Collinearity (>5)" if v > 5 else "Acceptable"}
                for k, v in mlr_res.vif.items()
            ])
            st.dataframe(df_vif, use_container_width=True, hide_index=True)
            st.caption("Per R-ML-11: High VIF between Streams and Duration_Minutes is expected and transparently documented.")

        st.markdown("#### Residual Diagnostics")
        res_df = pd.DataFrame({"Residual": mlr_res.residuals["res"]})
        fig_res = px.histogram(res_df, x="Residual", nbins=25, title="MLR Residual Error Distribution", color_discrete_sequence=["#3b82f6"])
        st.plotly_chart(fig_res, use_container_width=True)
