# spotify-streampulse/app.py
"""
Spotify StreamPulse: Enterprise Data Warehousing & Predictive Mining Platform.
Single-process Streamlit Application with RBAC gating and auto-bootstrap.
"""
from __future__ import annotations
import streamlit as st

from streampulse.ui.components.layout import apply_theme, page_icon

# Configure page layout first
st.set_page_config(
    page_title="Spotify StreamPulse",
    page_icon=page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

from streampulse.auth.guard import check_page_access
from streampulse.config import (
    PAGE_ADMIN_DASHBOARD,
    PAGE_ANALYST_DASHBOARD,
    PAGE_ASSOCIATION,
    PAGE_CLASSIFICATION,
    PAGE_CLUSTERING,
    PAGE_DATA_QUALITY,
    PAGE_DATA_WAREHOUSE,
    PAGE_EDA,
    PAGE_MODEL_STATUS,
    PAGE_OLAP,
    PAGE_REGRESSION,
    PAGE_STAR_SCHEMA,
    PAGE_SYSTEM_LOGS,
    PAGE_WHATIF,
)
from streampulse.config import MODEL_REGISTRY_FILE, MODELS_DIR
from streampulse.logging_setup import setup_logging
from streampulse.orchestration import run_all
from streampulse.ui.components.error_boundary import safe_render
from streampulse.ui.navigation import render_sidebar
from streampulse.ui.pages.login import render_login_page
from streampulse.warehouse.repository import warehouse_exists

# Import all 14 page render functions
from streampulse.ui.pages.admin.admin_dashboard import render as render_admin_dashboard
from streampulse.ui.pages.admin.data_warehouse import render as render_data_warehouse
from streampulse.ui.pages.admin.star_schema import render as render_star_schema
from streampulse.ui.pages.admin.data_quality import render as render_data_quality
from streampulse.ui.pages.admin.system_logs import render as render_system_logs
from streampulse.ui.pages.admin.model_status import render as render_model_status

from streampulse.ui.pages.analyst.analyst_dashboard import render as render_analyst_dashboard
from streampulse.ui.pages.analyst.olap import render as render_olap
from streampulse.ui.pages.analyst.eda import render as render_eda
from streampulse.ui.pages.analyst.regression import render as render_regression
from streampulse.ui.pages.analyst.classification import render as render_classification
from streampulse.ui.pages.analyst.clustering import render as render_clustering
from streampulse.ui.pages.analyst.association import render as render_association
from streampulse.ui.pages.analyst.whatif import render as render_whatif

# Page router mapping
PAGE_ROUTER = {
    PAGE_ADMIN_DASHBOARD: render_admin_dashboard,
    PAGE_DATA_WAREHOUSE: render_data_warehouse,
    PAGE_STAR_SCHEMA: render_star_schema,
    PAGE_DATA_QUALITY: render_data_quality,
    PAGE_SYSTEM_LOGS: render_system_logs,
    PAGE_MODEL_STATUS: render_model_status,
    PAGE_ANALYST_DASHBOARD: render_analyst_dashboard,
    PAGE_OLAP: render_olap,
    PAGE_EDA: render_eda,
    PAGE_REGRESSION: render_regression,
    PAGE_CLASSIFICATION: render_classification,
    PAGE_CLUSTERING: render_clustering,
    PAGE_ASSOCIATION: render_association,
    PAGE_WHATIF: render_whatif,
}


REQUIRED_MODELS = ("slr", "mlr", "decision_tree", "naive_bayes", "kmeans", "agglomerative")


def models_ready() -> bool:
    """True if the model registry and all six trained model files exist."""
    return MODEL_REGISTRY_FILE.exists() and all(
        (MODELS_DIR / f"{m}.joblib").exists() for m in REQUIRED_MODELS
    )


def bootstrap():
    """Auto-bootstrap pipeline on initial boot if warehouse is missing (T16.02)."""
    setup_logging()
    if st.session_state.get("_warehouse_ready"):
        return
    if not warehouse_exists() or not models_ready():
        with st.spinner("🚀 Initializing Spotify StreamPulse Data Warehouse & ML Models..."):
            run_all()
    st.session_state["_warehouse_ready"] = True


def main():
    bootstrap()

    # Authentication Gate
    is_authenticated = st.session_state.get("authenticated", False)
    if not is_authenticated:
        apply_theme(panel=False)
        render_login_page()
        return

    apply_theme(panel=True)

    # Render Navigation & Active Page
    active_page_id = render_sidebar()
    page_renderer = PAGE_ROUTER.get(active_page_id)

    if page_renderer:
        safe_render(page_renderer, active_page_id)
    else:
        st.error(f"Page '{active_page_id}' not found.")


if __name__ == "__main__":
    main()
