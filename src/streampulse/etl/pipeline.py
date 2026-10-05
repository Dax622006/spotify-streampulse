# spotify-streampulse/src/streampulse/etl/pipeline.py
"""
ETL orchestrator — runs T01–T13 in order, reconciles, writes report.
R-ETL-03: Reconciliation is mandatory.
R-ETL-05: Transactional loading (built into warehouse/builder.py).
Design.md §2.1 T04–T13.
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from streampulse.config import (
    ETL_REPORT_FILE,
    EXPECTED_REVENUE_SUM,
    EXPECTED_ROWS,
    EXPECTED_STREAM_SUM,
    REPORTS_DIR,
)
from streampulse.etl.cleaning import clean_and_type
from streampulse.etl.dimensions import (
    build_dim_country,
    build_dim_device,
    build_dim_genre,
    build_dim_subscription,
    build_dim_time_with_observed,
    build_dim_track,
    build_dim_user,
)
from streampulse.etl.fact import build_fact_streams
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class StepRecord:
    name: str
    rows_in: int
    rows_out: int
    duration_s: float


@dataclass
class EtlResult:
    """All dimension and fact DataFrames plus a step-by-step report."""
    dim_user: pd.DataFrame
    dim_track: pd.DataFrame
    dim_genre: pd.DataFrame
    dim_device: pd.DataFrame
    dim_country: pd.DataFrame
    dim_subscription: pd.DataFrame
    dim_time: pd.DataFrame
    fact_streams: pd.DataFrame
    stg_raw: pd.DataFrame          # for staging load
    stg_clean: pd.DataFrame        # typed clean frame
    steps: list[StepRecord]
    source_sha256: str
    sum_streams: int
    sum_duration: int
    sum_revenue: float
    started_at: str
    finished_at: str
    duration_s: float


def run_etl(df_raw: pd.DataFrame, source_sha: str) -> EtlResult:
    """
    Execute the complete ETL pipeline.

    Args:
        df_raw: Raw DataFrame from ingestion (Date as str, all columns).
        source_sha: SHA-256 of the source file.

    Returns:
        EtlResult with all dimension/fact frames and the audit report.

    Raises:
        ValueError: On reconciliation failure or unresolved FK (R-ETL-03/04).
    """
    from datetime import timezone
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    steps: list[StepRecord] = []
    log_event(logger, logging.INFO, "PIPELINE_START", "ETL pipeline started")

    # T01: staging raw
    stg_raw = df_raw.copy()
    steps.append(StepRecord("stg_raw", 0, len(stg_raw), 0.0))

    # T01–T03: Clean + type
    t0 = time.monotonic()
    df_clean = clean_and_type(df_raw)
    steps.append(StepRecord("clean_and_type", len(df_raw), len(df_clean), time.monotonic() - t0))

    # T05: quarantined IDs stay in stg_clean for lineage, not in dims/fact
    stg_clean = df_clean.copy()

    # T06: Dim_User
    t0 = time.monotonic()
    dim_user = build_dim_user(df_clean)
    steps.append(StepRecord("dim_user", len(df_clean), len(dim_user), time.monotonic() - t0))

    # T07: Dim_Track
    t0 = time.monotonic()
    dim_track = build_dim_track(df_clean)
    steps.append(StepRecord("dim_track", len(df_clean), len(dim_track), time.monotonic() - t0))

    # T08: Simple flat dims
    t0 = time.monotonic()
    dim_genre = build_dim_genre(df_clean)
    dim_device = build_dim_device(df_clean)
    dim_country = build_dim_country(df_clean)
    dim_subscription = build_dim_subscription(df_clean)
    steps.append(StepRecord(
        "flat_dims", len(df_clean),
        len(dim_genre) + len(dim_device) + len(dim_country) + len(dim_subscription),
        time.monotonic() - t0
    ))

    # T09: Dim_Time (complete 2025 calendar)
    t0 = time.monotonic()
    dim_time = build_dim_time_with_observed(df_clean)
    steps.append(StepRecord("dim_time", 0, len(dim_time), time.monotonic() - t0))

    # T10: Fact_Streams
    t0 = time.monotonic()
    fact_streams = build_fact_streams(
        df_clean, dim_user, dim_track, dim_genre,
        dim_device, dim_country, dim_subscription, dim_time,
    )
    steps.append(StepRecord("fact_streams", len(df_clean), len(fact_streams), time.monotonic() - t0))

    # T11: Referential integrity reconciliation (R-ETL-03)
    t0 = time.monotonic()
    _reconcile(df_raw, fact_streams)
    steps.append(StepRecord("reconciliation", len(fact_streams), len(fact_streams), time.monotonic() - t0))

    finished_at = datetime.now(timezone.utc).isoformat()
    total_s = time.monotonic() - started

    result = EtlResult(
        dim_user=dim_user,
        dim_track=dim_track,
        dim_genre=dim_genre,
        dim_device=dim_device,
        dim_country=dim_country,
        dim_subscription=dim_subscription,
        dim_time=dim_time,
        fact_streams=fact_streams,
        stg_raw=stg_raw,
        stg_clean=stg_clean,
        steps=steps,
        source_sha256=source_sha,
        sum_streams=int(fact_streams["Streams"].sum()),
        sum_duration=int(fact_streams["Duration_Minutes"].sum()),
        sum_revenue=round(float(fact_streams["Revenue_USD"].sum()), 2),
        started_at=started_at,
        finished_at=finished_at,
        duration_s=round(total_s, 3),
    )
    _write_etl_report(result)
    log_event(logger, logging.INFO, "PIPELINE_END",
              f"ETL completed in {total_s:.2f}s",
              details={"rows": len(fact_streams), "duration_s": round(total_s, 3)})
    return result


def _reconcile(df_raw: pd.DataFrame, fact: pd.DataFrame) -> None:
    """R-ETL-03: Fact row count = source rows; Σ measures equal source."""
    # Row count
    if len(fact) != len(df_raw):
        raise ValueError(
            f"R-ETL-03 reconciliation FAIL: fact rows {len(fact)} ≠ source rows {len(df_raw)}"
        )

    # Σ Streams
    src_streams = int(pd.to_numeric(df_raw["Streams"], errors="raise").sum())
    fact_streams = int(fact["Streams"].sum())
    if src_streams != fact_streams:
        raise ValueError(
            f"R-ETL-03 reconciliation FAIL: Σ Streams fact {fact_streams} ≠ source {src_streams}"
        )

    # Σ Duration_Minutes
    src_dur = int(pd.to_numeric(df_raw["Duration_Minutes"], errors="raise").sum())
    fact_dur = int(fact["Duration_Minutes"].sum())
    if src_dur != fact_dur:
        raise ValueError(
            f"R-ETL-03 reconciliation FAIL: Σ Duration fact {fact_dur} ≠ source {src_dur}"
        )

    # Σ Revenue_USD (floating tolerance 1e-6)
    src_rev = float(pd.to_numeric(df_raw["Revenue_USD"], errors="raise").sum())
    fact_rev = float(fact["Revenue_USD"].sum())
    if abs(src_rev - fact_rev) > 1e-6:
        raise ValueError(
            f"R-ETL-03 reconciliation FAIL: Σ Revenue fact {fact_rev:.6f} ≠ source {src_rev:.6f}"
        )

    log_event(logger, logging.INFO, "PIPELINE_STEP",
              "Reconciliation PASSED",
              details={
                  "rows": len(fact),
                  "streams": fact_streams,
                  "duration": fact_dur,
                  "revenue": round(fact_rev, 2),
              })


def _write_etl_report(result: EtlResult) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "duration_s": result.duration_s,
        "source_sha256": result.source_sha256,
        "sum_streams": result.sum_streams,
        "sum_duration": result.sum_duration,
        "sum_revenue": result.sum_revenue,
        "table_row_counts": {
            "Dim_User": len(result.dim_user),
            "Dim_Track": len(result.dim_track),
            "Dim_Genre": len(result.dim_genre),
            "Dim_Device": len(result.dim_device),
            "Dim_Country": len(result.dim_country),
            "Dim_Subscription": len(result.dim_subscription),
            "Dim_Time": len(result.dim_time),
            "Fact_Streams": len(result.fact_streams),
        },
        "steps": [
            {"name": s.name, "rows_in": s.rows_in, "rows_out": s.rows_out,
             "duration_s": s.duration_s}
            for s in result.steps
        ],
    }
    ETL_REPORT_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_etl_report() -> dict | None:
    """Load the persisted ETL report, or None if not found."""
    if not ETL_REPORT_FILE.exists():
        return None
    return json.loads(ETL_REPORT_FILE.read_text(encoding="utf-8"))
