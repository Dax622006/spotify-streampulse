# spotify-streampulse/src/streampulse/ml/classification.py
"""
Classification models: Decision Tree and Gaussian Naive Bayes.
R-ML-01, R-ML-07: Stratified 80/20 train/test split with random_state=42.
R-ML-04: Target leakage forbidden (Revenue_USD never used as feature).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from streampulse.config import (
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    CLS_FEATURES,
    CV_FOLDS,
    NUMERIC_FEATURES,
    PREMIUM_CLASS,
    RANDOM_STATE,
    TEST_SIZE,
)
from streampulse.logging_setup import get_logger

logger = get_logger(__name__)


class DenseTransformer(TransformerMixin):
    """Convert sparse matrix to dense array for GaussianNB."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


@dataclass
class ClassificationResult:
    model_id: str
    pipeline: Pipeline
    features: list[str]
    target: str
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    classes: list[str]
    feature_importances: dict[str, float] | None


def train_decision_tree(df: pd.DataFrame, max_depth: int = 5) -> ClassificationResult:
    """Train Decision Tree Classifier for Subscription tier prediction."""
    X = df[CLS_FEATURES].copy()
    y = df[CLASSIFICATION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Metrics
    classes = pipeline.classes_.tolist()
    pos_idx = classes.index(PREMIUM_CLASS) if PREMIUM_CLASS in classes else 1
    y_test_bin = (y_test == PREMIUM_CLASS).astype(int)
    y_pred_bin = (y_pred == PREMIUM_CLASS).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(y_test_bin, y_proba[:, pos_idx])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    # Feature importances
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_names = NUMERIC_FEATURES + cat_names
    dt = pipeline.named_steps["classifier"]
    fi = {name: round(float(imp), 4) for name, imp in zip(all_names, dt.feature_importances_)}

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
    }

    return ClassificationResult(
        model_id="decision_tree",
        pipeline=pipeline,
        features=CLS_FEATURES,
        target=CLASSIFICATION_TARGET,
        metrics=metrics,
        confusion_matrix=cm,
        classes=classes,
        feature_importances=fi,
    )


def train_naive_bayes(df: pd.DataFrame) -> ClassificationResult:
    """Train Gaussian Naive Bayes Classifier for Subscription tier prediction."""
    X = df[CLS_FEATURES].copy()
    y = df[CLASSIFICATION_TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB()),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    classes = pipeline.classes_.tolist()
    pos_idx = classes.index(PREMIUM_CLASS) if PREMIUM_CLASS in classes else 1
    y_test_bin = (y_test == PREMIUM_CLASS).astype(int)
    y_pred_bin = (y_pred == PREMIUM_CLASS).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_test_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(y_test_bin, y_proba[:, pos_idx])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
    }

    return ClassificationResult(
        model_id="naive_bayes",
        pipeline=pipeline,
        features=CLS_FEATURES,
        target=CLASSIFICATION_TARGET,
        metrics=metrics,
        confusion_matrix=cm,
        classes=classes,
        feature_importances=None,
    )
