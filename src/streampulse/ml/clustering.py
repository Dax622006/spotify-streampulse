# spotify-streampulse/src/streampulse/ml/clustering.py
"""
Clustering models: K-Means and Agglomerative Hierarchical Clustering.
R-ML-23: Feature selection drops collinear features (corr > 0.90).
R-ML-24: StandardScaler applied before distance-based clustering.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from streampulse.config import (
    CLUSTER_CORR_THRESHOLD,
    KMEANS_K_RANGE,
    KMEANS_N_INIT,
    RANDOM_STATE,
)
from streampulse.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ClusteringResult:
    model_id: str
    features_used: list[str]
    dropped_features: list[str]
    scaler: StandardScaler
    k: int
    cluster_labels: list[int]
    metrics: dict[str, Any]
    profile: list[dict[str, Any]]
    model_artifact: Any


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate analytics frame into 100 user profiles per T13.01 and R-ML-23.

    Features:
    - record_count
    - mean_streams
    - mean_duration
    - premium_share
    - distinct_genres
    - total_revenue
    """
    profiles = df.groupby("User_ID").agg(
        record_count=("Stream_ID", "count"),
        mean_streams=("Streams", "mean"),
        mean_duration=("Duration_Minutes", "mean"),
        premium_share=("Subscription", lambda s: float((s == "Premium").mean())),
        distinct_genres=("Genre", "nunique"),
        total_revenue=("Revenue_USD", "sum"),
    ).reset_index()
    return profiles


def prepare_clustering_features(
    df: pd.DataFrame, corr_threshold: float = CLUSTER_CORR_THRESHOLD
) -> tuple[np.ndarray, list[str], list[str], StandardScaler, pd.DataFrame]:
    """Prepare user profiles, apply correlation filter (|r| > threshold), and scale features per R-ML-23."""
    profiles = build_user_profiles(df)
    candidate_features = [
        "record_count",
        "mean_streams",
        "mean_duration",
        "premium_share",
        "distinct_genres",
        "total_revenue",
    ]

    corr = profiles[candidate_features].corr()
    retained: list[str] = []
    dropped: list[str] = []

    for f in candidate_features:
        # Check if f has |r| > corr_threshold with any already retained feature
        is_collinear = False
        for r in retained:
            r_val = abs(corr.loc[f, r])
            if r_val > corr_threshold:
                is_collinear = True
                dropped.append(f)
                logger.info(f"Dropped feature '{f}' from clustering (|r|={r_val:.4f} > {corr_threshold} with '{r}')")
                break
        if not is_collinear:
            retained.append(f)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles[retained].values)
    return X_scaled, retained, dropped, scaler, profiles


def evaluate_kmeans_k_range(df: pd.DataFrame) -> dict[str, list]:
    """Compute Elbow (inertia) and Silhouette across k=2..10."""
    X_scaled, _, _, _, _ = prepare_clustering_features(df)
    k_vals = list(range(KMEANS_K_RANGE[0], KMEANS_K_RANGE[1] + 1))
    inertias = []
    silhouettes = []

    for k in k_vals:
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))

    return {
        "k_values": k_vals,
        "inertias": [round(i, 2) for i in inertias],
        "silhouettes": [round(s, 4) for s in silhouettes],
    }


def train_kmeans(df: pd.DataFrame, k: int = 3) -> ClusteringResult:
    """Train K-Means clustering and profile clusters on 100 user profiles."""
    X_scaled, features_used, dropped, scaler, profiles = prepare_clustering_features(df)

    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels))

    # Build cluster profile table from user profiles
    df_prof = profiles.copy()
    df_prof["Cluster"] = labels
    cluster_profiles = []
    for c in sorted(df_prof["Cluster"].unique()):
        sub = df_prof[df_prof["Cluster"] == c]
        cluster_profiles.append({
            "cluster_id": int(c),
            "size": len(sub),
            "pct": round(len(sub) / len(df_prof) * 100, 1),
            "mean_streams": round(float(sub["mean_streams"].mean()), 2),
            "mean_duration": round(float(sub["mean_duration"].mean()), 2),
            "mean_revenue": round(float(sub["total_revenue"].mean()), 2),
            "premium_ratio": round(float(sub["premium_share"].mean()), 3),
            "distinct_genres": round(float(sub["distinct_genres"].mean()), 2),
        })

    metrics = {
        "k": k,
        "inertia": round(float(km.inertia_), 2),
        "silhouette": round(sil, 4),
    }

    return ClusteringResult(
        model_id="kmeans",
        features_used=features_used,
        dropped_features=dropped,
        scaler=scaler,
        k=k,
        cluster_labels=labels.tolist(),
        metrics=metrics,
        profile=cluster_profiles,
        model_artifact=km,
    )


def train_agglomerative(df: pd.DataFrame, k: int = 3, linkage_method: str = "ward") -> tuple[ClusteringResult, np.ndarray]:
    """Train Agglomerative clustering and return linkage matrix on 100 user profiles."""
    X_scaled, features_used, dropped, scaler, profiles = prepare_clustering_features(df)

    # Compute linkage matrix
    Z = linkage(X_scaled, method=linkage_method)

    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = agg.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels))

    df_prof = profiles.copy()
    df_prof["Cluster"] = labels
    cluster_profiles = []
    for c in sorted(df_prof["Cluster"].unique()):
        sub = df_prof[df_prof["Cluster"] == c]
        cluster_profiles.append({
            "cluster_id": int(c),
            "size": len(sub),
            "pct": round(len(sub) / len(df_prof) * 100, 1),
            "mean_streams": round(float(sub["mean_streams"].mean()), 2),
            "mean_duration": round(float(sub["mean_duration"].mean()), 2),
            "mean_revenue": round(float(sub["total_revenue"].mean()), 2),
            "premium_ratio": round(float(sub["premium_share"].mean()), 3),
            "distinct_genres": round(float(sub["distinct_genres"].mean()), 2),
        })

    metrics = {
        "k": k,
        "linkage": linkage_method,
        "silhouette": round(sil, 4),
    }

    res = ClusteringResult(
        model_id="agglomerative",
        features_used=features_used,
        dropped_features=dropped,
        scaler=scaler,
        k=k,
        cluster_labels=labels.tolist(),
        metrics=metrics,
        profile=cluster_profiles,
        model_artifact=agg,
    )
    return res, Z


def compute_clustering_agreement(kmeans_labels: list[int], agg_labels: list[int]) -> float:
    """Compute Adjusted Rand Index (ARI) between two clustering assignments."""
    return round(float(adjusted_rand_score(kmeans_labels, agg_labels)), 4)
