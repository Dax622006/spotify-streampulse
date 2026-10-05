# spotify-streampulse/src/streampulse/orchestration.py
"""
Pipeline orchestration — one-command end-to-end build.
T16.01: Checksum check -> Validation -> ETL -> Warehouse build -> Train & register 6 models.
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from streampulse.config import (
    ETL_REPORT_FILE,
    REPORTS_DIR,
    SOURCE_CHECKSUM_FILE,
    VALIDATION_REPORT_FILE,
    WAREHOUSE_REPORT_FILE,
)
from streampulse.etl.pipeline import run_etl
from streampulse.ingestion.loader import load_or_compare_checksum, read_source
from streampulse.ingestion.validator import run_validation
from streampulse.logging_setup import get_logger, log_event
from streampulse.ml.classification import train_decision_tree, train_naive_bayes
from streampulse.ml.clustering import train_agglomerative, train_kmeans
from streampulse.ml.regression import train_mlr, train_slr
from streampulse.ml.registry import registry
from streampulse.warehouse.builder import build_warehouse
from streampulse.warehouse.repository import get_analytics_frame

logger = get_logger(__name__)


def run_all(force: bool = False) -> dict[str, Any]:
    """Execute complete end-to-end data warehousing and model training pipeline."""
    start_time = time.monotonic()
    run_timestamp = datetime.now(timezone.utc).isoformat()
    log_event(logger, logging.INFO, "PIPELINE_START", "Starting end-to-end StreamPulse build")

    # Step 1: Read source & check checksum
    df_raw, current_sha = read_source()
    stored_sha, is_match = load_or_compare_checksum(current_sha)

    # Step 2: Validate source
    val_report = run_validation(df_raw, current_sha)
    if val_report.overall_status != "PASS":
        msg = f"Validation failed with {val_report.summary['failed']} critical failures. Pipeline aborted."
        log_event(logger, logging.ERROR, "PIPELINE_ERROR", msg)
        raise RuntimeError(msg)

    # Step 3: Run ETL
    etl_result = run_etl(df_raw, current_sha)

    # Step 4: Build SQLite Warehouse
    content_hash = build_warehouse(etl_result)

    # Step 5: Train & Register 6 ML Models
    df_analytics = get_analytics_frame()

    # 5.1 Simple Linear Regression
    slr_res = train_slr(df_analytics)
    registry.register(
        model_id="slr",
        model_name="Simple Linear Regression",
        model_type="regression",
        model_obj=slr_res.pipeline,
        features=slr_res.features,
        target=slr_res.target,
        metrics=slr_res.metrics,
        params={"model": "LinearRegression", "features": "Streams"},
        source_sha256=current_sha,
    )

    # 5.2 Multiple Linear Regression
    mlr_res = train_mlr(df_analytics)
    registry.register(
        model_id="mlr",
        model_name="Multiple Linear Regression",
        model_type="regression",
        model_obj=mlr_res.pipeline,
        features=mlr_res.features,
        target=mlr_res.target,
        metrics=mlr_res.metrics,
        params={"model": "LinearRegression", "scaling": "StandardScaler", "encoding": "OneHotEncoder(drop='first')"},
        source_sha256=current_sha,
    )

    # 5.3 Decision Tree Classifier
    dt_res = train_decision_tree(df_analytics, max_depth=5)
    registry.register(
        model_id="decision_tree",
        model_name="Decision Tree Classifier",
        model_type="classification",
        model_obj=dt_res.pipeline,
        features=dt_res.features,
        target=dt_res.target,
        metrics=dt_res.metrics,
        params={"max_depth": 5, "criterion": "gini", "random_state": 42},
        source_sha256=current_sha,
    )

    # 5.4 Gaussian Naive Bayes Classifier
    nb_res = train_naive_bayes(df_analytics)
    registry.register(
        model_id="naive_bayes",
        model_name="Gaussian Naive Bayes Classifier",
        model_type="classification",
        model_obj=nb_res.pipeline,
        features=nb_res.features,
        target=nb_res.target,
        metrics=nb_res.metrics,
        params={"model": "GaussianNB", "scaling": "StandardScaler"},
        source_sha256=current_sha,
    )

    # 5.5 K-Means Clustering
    km_res = train_kmeans(df_analytics, k=3)
    registry.register(
        model_id="kmeans",
        model_name="K-Means Clustering",
        model_type="clustering",
        model_obj={"model": km_res.model_artifact, "scaler": km_res.scaler, "features": km_res.features_used},
        features=km_res.features_used,
        target=None,
        metrics=km_res.metrics,
        params={"k": 3, "n_init": 10, "dropped": km_res.dropped_features},
        source_sha256=current_sha,
    )

    # 5.6 Agglomerative Hierarchical Clustering
    agg_res, _ = train_agglomerative(df_analytics, k=3, linkage_method="ward")
    registry.register(
        model_id="agglomerative",
        model_name="Agglomerative Hierarchical Clustering",
        model_type="clustering",
        model_obj={"model": agg_res.model_artifact, "scaler": agg_res.scaler, "features": agg_res.features_used},
        features=agg_res.features_used,
        target=None,
        metrics=agg_res.metrics,
        params={"k": 3, "linkage": "ward", "dropped": agg_res.dropped_features},
        source_sha256=current_sha,
    )

    total_duration = time.monotonic() - start_time
    summary = {
        "status": "SUCCESS",
        "run_timestamp": run_timestamp,
        "duration_seconds": round(total_duration, 2),
        "source_sha256": current_sha,
        "warehouse_content_hash": content_hash,
        "models_registered": ["slr", "mlr", "decision_tree", "naive_bayes", "kmeans", "agglomerative"],
    }

    log_event(logger, logging.INFO, "PIPELINE_COMPLETE", f"Pipeline completed in {total_duration:.2f}s", **summary)
    return summary
