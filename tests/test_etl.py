# spotify-streampulse/tests/test_etl.py
"""
Tests for ETL Pipeline (T04.07).
"""
from __future__ import annotations
import pytest

from streampulse.config import (
    DIM_ROW_COUNTS,
    EXPECTED_DURATION_SUM,
    EXPECTED_REVENUE_SUM,
    EXPECTED_ROWS,
    EXPECTED_STREAM_SUM,
)
from streampulse.etl.pipeline import run_etl
from streampulse.ingestion.loader import read_source


def test_full_etl_pipeline():
    df_raw, sha = read_source()
    etl_result = run_etl(df_raw, sha)

    # Check row counts
    assert len(etl_result.dim_user) == DIM_ROW_COUNTS["Dim_User"]
    assert len(etl_result.dim_track) == DIM_ROW_COUNTS["Dim_Track"]
    assert len(etl_result.dim_genre) == DIM_ROW_COUNTS["Dim_Genre"]
    assert len(etl_result.dim_device) == DIM_ROW_COUNTS["Dim_Device"]
    assert len(etl_result.dim_country) == DIM_ROW_COUNTS["Dim_Country"]
    assert len(etl_result.dim_subscription) == DIM_ROW_COUNTS["Dim_Subscription"]
    assert len(etl_result.dim_time) == DIM_ROW_COUNTS["Dim_Time"]
    assert len(etl_result.fact_streams) == DIM_ROW_COUNTS["Fact_Streams"]

    # Check measure sums (R-ETL-03)
    assert etl_result.sum_streams == EXPECTED_STREAM_SUM
    assert etl_result.sum_duration == EXPECTED_DURATION_SUM
    assert abs(etl_result.sum_revenue - EXPECTED_REVENUE_SUM) < 1e-6

    # Verify no nulls in Fact_Streams FKs
    assert etl_result.fact_streams.isnull().sum().sum() == 0
