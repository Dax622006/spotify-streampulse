# spotify-streampulse/src/streampulse/etl/cleaning.py
"""
ETL Step 1 — Data cleansing and type standardisation.
R-ETL-06: Date must be parsed explicitly; never coerce to NaT silently.
R-DS-05: Log every transformation; no silent mutations.
Design.md §2.1 T01–T03.
"""
from __future__ import annotations
import logging

import pandas as pd

from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


def clean_and_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise types on the raw staging frame.

    Transformations applied (logged per R-DS-05):
    - T02: Strip whitespace from all text columns (already done in loader, but defence in depth).
    - T03: Date → datetime64[ns] with strict %Y-%m-%d format.
    - T03: Streams → int64.
    - T03: Duration_Minutes → int64.
    - T03: Revenue_USD → float64 rounded to 2 decimals.

    Args:
        df: Raw frame from ingestion (Date as str, numerics may be str or numeric).

    Returns:
        Cleaned, typed copy.

    Raises:
        ValueError: If Date parsing fails on any row (R-ETL-06).
    """
    out = df.copy()

    # Strip whitespace (belt-and-suspenders)
    str_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        before = out[col].str.strip()
        changed = (before != out[col]).sum()
        if changed > 0:
            log_event(logger, logging.WARNING, "PIPELINE_STEP",
                      f"Trimmed whitespace in '{col}'", details={"rows_changed": int(changed)})
        out[col] = before

    # Parse Date
    parsed_dates = pd.to_datetime(out["Date"], format="%Y-%m-%d", errors="coerce")
    bad_dates = int(parsed_dates.isna().sum())
    if bad_dates > 0:
        raise ValueError(
            f"R-ETL-06: {bad_dates} Date values could not be parsed as YYYY-MM-DD. "
            "Validation should have caught this — aborting ETL."
        )
    out["Date"] = parsed_dates
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              "Date column parsed to datetime64", details={"format": "%Y-%m-%d"})

    # Cast Streams
    out["Streams"] = pd.to_numeric(out["Streams"], errors="raise").astype("int64")
    log_event(logger, logging.INFO, "PIPELINE_STEP", "Streams cast to int64")

    # Cast Duration_Minutes
    out["Duration_Minutes"] = pd.to_numeric(out["Duration_Minutes"], errors="raise").astype("int64")
    log_event(logger, logging.INFO, "PIPELINE_STEP", "Duration_Minutes cast to int64")

    # Cast Revenue_USD
    out["Revenue_USD"] = (
        pd.to_numeric(out["Revenue_USD"], errors="raise")
        .astype("float64")
        .round(2)
    )
    log_event(logger, logging.INFO, "PIPELINE_STEP", "Revenue_USD cast to float64 (rounded 2dp)")

    return out
