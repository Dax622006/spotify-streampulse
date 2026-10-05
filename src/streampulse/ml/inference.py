# spotify-streampulse/src/streampulse/ml/inference.py
"""
What-If scenario inference service.
R-WI-01: Load pre-fitted models from registry; NEVER refit.
R-WI-02: Strictly read-only; leaves warehouse and source files untouched.
R-WI-03, R-WI-04: Validate input ranges and categories; non-blocking warning for ratio.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from streampulse.config import WHATIF_FEATURES
from streampulse.ml.registry import registry
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class InputSpec:
    streams_min: int = 1
    streams_max: int = 20
    duration_min: int = 2
    duration_max: int = 100
    devices: list[str] = None
    genres: list[str] = None
    countries: list[str] = None


@dataclass
class WhatIfPrediction:
    slr_revenue: float
    mlr_revenue: float
    dt_subscription: str
    dt_proba: dict[str, float]
    nb_subscription: str
    nb_proba: dict[str, float]
    warnings: list[str]
    model_caveats: dict[str, Any]


def validate_whatif_input(
    streams: int,
    duration: int,
    device: str,
    genre: str,
    country: str,
    spec: InputSpec,
) -> tuple[bool, list[str], list[str]]:
    """Validate user inputs for simulation.

    Returns:
        (is_valid, errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Streams check
    if not (spec.streams_min <= streams <= spec.streams_max):
        errors.append(f"Streams ({streams}) must be between {spec.streams_min} and {spec.streams_max}.")

    # Duration check
    if not (spec.duration_min <= duration <= spec.duration_max):
        errors.append(f"Duration ({duration}) must be between {spec.duration_min} and {spec.duration_max} minutes.")

    # Category checks
    if spec.devices and device not in spec.devices:
        errors.append(f"Invalid device: '{device}'.")
    if spec.genres and genre not in spec.genres:
        errors.append(f"Invalid genre: '{genre}'.")
    if spec.countries and country not in spec.countries:
        errors.append(f"Invalid country: '{country}'.")

    # Ratio check (Duration / Streams in {2, 3, 4, 5} in source dataset)
    if streams > 0:
        ratio = duration / streams
        if ratio < 2.0 or ratio > 5.0 or (duration % streams != 0):
            warnings.append(
                f"Notice: Duration-to-Streams ratio ({ratio:.1f}) is outside the observed historical training ratio (exact integers 2, 3, 4, or 5)."
            )

    return len(errors) == 0, errors, warnings


def predict_scenario(
    streams: int,
    duration: int,
    device: str,
    genre: str,
    country: str,
) -> WhatIfPrediction:
    """Run scenario inference using pre-trained pipelines from ModelRegistry."""
    # Build single-row DataFrame
    input_df = pd.DataFrame([{
        "Streams": streams,
        "Duration_Minutes": duration,
        "Device": device,
        "Genre": genre,
        "Country": country,
    }])

    # 1. SLR Prediction
    slr_pipeline = registry.load_model("slr")
    slr_pred = float(slr_pipeline.predict(input_df[["Streams"]])[0]) if slr_pipeline else 0.0

    # 2. MLR Prediction
    mlr_pipeline = registry.load_model("mlr")
    mlr_pred = float(mlr_pipeline.predict(input_df[WHATIF_FEATURES])[0]) if mlr_pipeline else 0.0

    # 3. Decision Tree Prediction
    dt_pipeline = registry.load_model("decision_tree")
    dt_sub = "Unknown"
    dt_proba = {}
    if dt_pipeline:
        dt_sub = str(dt_pipeline.predict(input_df[WHATIF_FEATURES])[0])
        probas = dt_pipeline.predict_proba(input_df[WHATIF_FEATURES])[0]
        dt_proba = {cls: round(float(p), 3) for cls, p in zip(dt_pipeline.classes_, probas)}

    # 4. Naive Bayes Prediction
    nb_pipeline = registry.load_model("naive_bayes")
    nb_sub = "Unknown"
    nb_proba = {}
    if nb_pipeline:
        nb_sub = str(nb_pipeline.predict(input_df[WHATIF_FEATURES])[0])
        probas = nb_pipeline.predict_proba(input_df[WHATIF_FEATURES])[0]
        nb_proba = {cls: round(float(p), 3) for cls, p in zip(nb_pipeline.classes_, probas)}

    # Retrieve baseline model caveats
    caveats = {}
    for mid in ["slr", "mlr", "decision_tree", "naive_bayes"]:
        meta = registry.get_metadata(mid)
        if meta:
            caveats[mid] = meta.metrics

    # Check ratio warning
    ratio = duration / streams if streams > 0 else 0
    warnings = []
    if ratio < 2.0 or ratio > 5.0:
        warnings.append(f"Duration-to-Streams ratio ({ratio:.1f}) diverges from typical training domain (2–5x).")

    log_event(
        logger,
        logging.INFO,
        "WHATIF_INFERENCE",
        "Executed What-If scenario inference",
        scenario={"streams": streams, "duration": duration, "device": device, "genre": genre, "country": country},
    )

    return WhatIfPrediction(
        slr_revenue=round(max(0.0, slr_pred), 2),
        mlr_revenue=round(max(0.0, mlr_pred), 2),
        dt_subscription=dt_sub,
        dt_proba=dt_proba,
        nb_subscription=nb_sub,
        nb_proba=nb_proba,
        warnings=warnings,
        model_caveats=caveats,
    )
