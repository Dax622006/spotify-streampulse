# spotify-streampulse/tests/conftest.py
"""Shared test setup: make sure the warehouse and trained models exist.

On a fresh clone (generated files are git-ignored) the tests would otherwise fail
because they read the warehouse and saved models. This builds them once if missing.
"""
import pytest

from streampulse.config import MODEL_REGISTRY_FILE, MODELS_DIR
from streampulse.warehouse.repository import warehouse_exists

_MODELS = ("slr", "mlr", "decision_tree", "naive_bayes", "kmeans", "agglomerative")


@pytest.fixture(scope="session", autouse=True)
def ensure_pipeline_built():
    models_ok = MODEL_REGISTRY_FILE.exists() and all(
        (MODELS_DIR / f"{m}.joblib").exists() for m in _MODELS
    )
    if not (warehouse_exists() and models_ok):
        from streampulse.orchestration import run_all

        run_all()
