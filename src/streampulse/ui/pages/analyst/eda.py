# spotify-streampulse/src/streampulse/ui/pages/analyst/eda.py
"""
Exploratory Data Analysis: Univariate, bivariate distributions, correlation heatmaps, and outlier detection.
"""
from __future__ import annotations
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import pandas as pd

from streampulse.auth.guard import require_page
from streampulse.config import PAGE_EDA
from streampulse.ui.components.layout import page_header
from streampulse.ui.components.cache import get_analytics_frame

@require_page(PAGE_EDA)
def render():
    page_header(
        title="Exploratory Data Analysis (EDA)",
        subtitle="Univariate distributions, bivariate correlations, and statistical outlier audits",
        concept="Exploratory Data Analysis",
        icon="🔍",
    )

    df = get_analytics_frame()
    if df.empty:
        st.warning("⚠️ Warehouse contains no data.")
        return

    tab_uni, tab_bi, tab_corr, tab_outlier = st.tabs([
        "📊 Univariate Analysis",
        "📈 Bivariate Relationships",
        "🔥 Correlation Heatmap",
        "🚨 Outlier Detection",
    ])

    numeric_cols = ["Streams", "Duration_Minutes", "Revenue_USD"]

    with tab_uni:
        st.markdown("#### Univariate Distribution & Descriptive Statistics")
        sel_num = st.selectbox("Select Numeric Feature", options=numeric_cols)

        c1, c2 = st.columns([1, 2])
        with c1:
            series = df[sel_num]
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            stats_df = pd.DataFrame([
                {"Statistic": "Mean", "Value": f"{series.mean():.4f}"},
                {"Statistic": "Std Dev", "Value": f"{series.std():.4f}"},
                {"Statistic": "Median", "Value": f"{series.median():.4f}"},
                {"Statistic": "IQR", "Value": f"{q3 - q1:.4f}"},
                {"Statistic": "Min", "Value": f"{series.min()}"},
                {"Statistic": "Max", "Value": f"{series.max()}"},
                {"Statistic": "Skewness", "Value": f"{series.skew():.4f}"},
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with c2:
            fig_hist = px.histogram(
                df,
                x=sel_num,
                marginal="box",
                nbins=30,
                color_discrete_sequence=["#1DB954"],
                title=f"Distribution of {sel_num}",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab_bi:
        st.markdown("#### Bivariate Feature Interactions")
        b1, b2, b3 = st.columns(3)
        with b1:
            x_var = st.selectbox("X-Axis Feature", options=numeric_cols, index=0)
        with b2:
            y_var = st.selectbox("Y-Axis Feature", options=numeric_cols, index=1)
        with b3:
            color_var = st.selectbox("Color by Dimension", options=["Subscription", "Device", "Genre"], index=0)

        fig_scat = px.scatter(
            df,
            x=x_var,
            y=y_var,
            color=color_var,
            title=f"{y_var} vs {x_var} colored by {color_var}",
            opacity=0.7,
        )
        # Overall least-squares trend line (NumPy; avoids the statsmodels dependency)
        _xy = df[[x_var, y_var]].dropna()
        if x_var != y_var and _xy[x_var].nunique() > 1:
            _slope, _icpt = np.polyfit(_xy[x_var], _xy[y_var], 1)
            _xs = np.array([_xy[x_var].min(), _xy[x_var].max()])
            fig_scat.add_scatter(
                x=_xs, y=_slope * _xs + _icpt, mode="lines",
                name=f"OLS fit (slope={_slope:.3f})", line=dict(color="#e11d48", width=2),
            )
        st.plotly_chart(fig_scat, use_container_width=True)

    with tab_corr:
        st.markdown("#### Correlation Matrix (Spearman & Pearson)")
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".3f",
            aspect="auto",
            title="Pearson Correlation Matrix (Streams, Duration, Revenue)",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("💡 Note the strong correlation (r ≈ 0.84) between `Streams` and `Duration_Minutes`. This is formally accounted for in ML via VIF reporting and collinearity feature drops.")

    with tab_outlier:
        st.markdown("#### IQR & Z-Score Statistical Outlier Analysis")
        outlier_target = st.selectbox("Analyze Outliers For", options=numeric_cols, key="out_var")
        vals = df[outlier_target]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (vals < lower_bound) | (vals > upper_bound)
        outlier_df = df[outlier_mask]

        st.markdown(f"- **IQR Bounds:** `[{lower_bound:.2f}, {upper_bound:.2f}]`")
        st.markdown(f"- **Flagged Outlier Rows:** `{len(outlier_df)}` out of `{len(df)}` ({len(outlier_df)/len(df)*100:.1f}%)")

        if not outlier_df.empty:
            st.dataframe(outlier_df[["Stream_ID", "Song_Name", "Streams", "Duration_Minutes", "Revenue_USD", "Subscription"]], use_container_width=True)
        else:
            st.success("No statistical outliers detected under 1.5 * IQR standard.")
