# spotify-streampulse/src/streampulse/ml/regression.py
"""
Regression models: SLR and MLR.
R-ML-01: 80/20 train/test split with random_state=42.
R-ML-02: SLR feature is Streams; MLR features Streams, Duration_Minutes, Device, Genre, Country.
R-ML-04: No target leakage (Subscription forbidden as feature).
R-ML-11: VIF calculated and reported for MLR.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from streampulse.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    MLR_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    REGRESSION_TARGET,
    SLR_FEATURES,
    TEST_SIZE,
)
from streampulse.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionResult:
    model_id: str
    pipeline: Pipeline
    features: list[str]
    target: str
    metrics: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    residuals: dict[str, Any]
    vif: dict[str, float] | None


def compute_vif(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    """Calculate Variance Inflation Factor (VIF) for numeric features."""
    from scipy import stats
    vif_dict = {}
    X = df[numeric_cols].dropna().values
    if X.shape[1] < 2:
        return {numeric_cols[0]: 1.0}

    # Add constant
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
    for i, col in enumerate(numeric_cols):
        y_i = X_with_const[:, i + 1]
        X_not_i = np.delete(X_with_const, i + 1, axis=1)
        # OLS fit
        beta, _, _, _ = np.linalg.lstsq(X_not_i, y_i, rcond=None)
        y_hat = X_not_i @ beta
        r2 = r2_score(y_i, y_hat)
        vif = 1.0 / (1.0 - r2) if r2 < 0.99999 else 9999.0
        vif_dict[col] = round(float(vif), 2)
    return vif_dict


def train_slr(df: pd.DataFrame) -> RegressionResult:
    """Train Simple Linear Regression (Revenue ~ Streams)."""
    X = df[SLR_FEATURES].copy()
    y = df[REGRESSION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("regressor", LinearRegression())
    ])
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
    )

    residuals = y_test.values - y_pred_test
    reg = pipeline.named_steps["regressor"]
    coeffs = {"Streams": round(float(reg.coef_[0]), 4)}

    metrics = {
        "r2_train": round(float(r2_train), 4),
        "r2_test": round(float(r2_test), 4),
        "rmse_test": round(rmse_test, 4),
        "mae_test": round(mae_test, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "residual_mean": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4),
    }

    return RegressionResult(
        model_id="slr",
        pipeline=pipeline,
        features=SLR_FEATURES,
        target=REGRESSION_TARGET,
        metrics=metrics,
        coefficients=coeffs,
        intercept=round(float(reg.intercept_), 4),
        residuals={"y_true": y_test.tolist(), "y_pred": y_pred_test.tolist(), "res": residuals.tolist()},
        vif=None,
    )


def train_mlr(df: pd.DataFrame) -> RegressionResult:
    """Train Multiple Linear Regression (Revenue ~ Streams + Duration + Device + Genre + Country)."""
    X = df[MLR_FEATURES].copy()
    y = df[REGRESSION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    n = len(y_test)
    p = X.shape[1]
    adj_r2 = 1.0 - (1.0 - r2_test) * (n - 1) / (n - p - 1)

    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), scoring="r2"
    )

    residuals = y_test.values - y_pred_test

    # Feature names from transformer
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    reg = pipeline.named_steps["regressor"]
    coeffs = {name: round(float(c), 4) for name, c in zip(all_feature_names, reg.coef_)}

    # VIF on numeric features
    vif = compute_vif(df, NUMERIC_FEATURES)

    metrics = {
        "r2_train": round(float(r2_train), 4),
        "r2_test": round(float(r2_test), 4),
        "adj_r2_test": round(float(adj_r2), 4),
        "rmse_test": round(rmse_test, 4),
        "mae_test": round(mae_test, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "residual_mean": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4),
    }

    return RegressionResult(
        model_id="mlr",
        pipeline=pipeline,
        features=MLR_FEATURES,
        target=REGRESSION_TARGET,
        metrics=metrics,
        coefficients=coeffs,
        intercept=round(float(reg.intercept_), 4),
        residuals={"y_true": y_test.tolist(), "y_pred": y_pred_test.tolist(), "res": residuals.tolist()},
        vif=vif,
    )
