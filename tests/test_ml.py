# spotify-streampulse/tests/test_ml.py
"""
Unit tests for Machine Learning models: Regression, Classification, Clustering, and Association.
"""
from __future__ import annotations
import pytest

from streampulse.ml.association import run_apriori_pipeline
from streampulse.ml.classification import train_decision_tree, train_naive_bayes
from streampulse.ml.clustering import (
    compute_clustering_agreement,
    train_agglomerative,
    train_kmeans,
)
from streampulse.ml.regression import train_mlr, train_slr
from streampulse.warehouse.repository import get_analytics_frame


@pytest.fixture(scope="module")
def df_analytics():
    return get_analytics_frame()


def test_slr_training(df_analytics):
    res = train_slr(df_analytics)
    assert res.metrics["r2_test"] > 0.0
    assert "Streams" in res.coefficients
    assert res.metrics["rmse_test"] < 3.0


def test_mlr_training(df_analytics):
    res = train_mlr(df_analytics)
    assert res.metrics["r2_test"] > 0.0
    assert "Streams" in res.vif
    assert "Duration_Minutes" in res.vif
    # VIF must be reported
    assert res.vif["Streams"] > 1.0


def test_classification_decision_tree(df_analytics):
    res = train_decision_tree(df_analytics, max_depth=5)
    assert res.metrics["accuracy"] >= 0.50
    assert "Free" in res.classes
    assert "Premium" in res.classes
    assert len(res.confusion_matrix) == 2


def test_classification_naive_bayes(df_analytics):
    res = train_naive_bayes(df_analytics)
    assert res.metrics["accuracy"] >= 0.40
    assert len(res.confusion_matrix) == 2


def test_kmeans_clustering(df_analytics):
    res = train_kmeans(df_analytics, k=3)
    assert res.metrics["k"] == 3
    assert res.metrics["silhouette"] > 0.0
    assert len(res.profile) == 3
    assert len(res.cluster_labels) == 100  # 100 user profiles per R-ML-23


def test_agglomerative_clustering(df_analytics):
    res, Z = train_agglomerative(df_analytics, k=3, linkage_method="ward")
    assert res.metrics["k"] == 3
    assert len(res.profile) == 3
    assert Z.shape[0] == 99  # 100 user profiles - 1 per T13.01 and R-ML-23


def test_clustering_agreement(df_analytics):
    km = train_kmeans(df_analytics, k=3)
    agg, _ = train_agglomerative(df_analytics, k=3, linkage_method="ward")
    ari = compute_clustering_agreement(km.cluster_labels, agg.cluster_labels)
    assert -1.0 <= ari <= 1.0


def test_apriori_association(df_analytics):
    res = run_apriori_pipeline(df_analytics, min_support=0.30, min_confidence=0.50)
    assert res.summary["num_transactions"] == 100
    assert res.summary["num_items"] == 10
    assert not res.itemsets.empty
