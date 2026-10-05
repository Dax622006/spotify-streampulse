# spotify-streampulse/tests/test_warehouse.py
"""
Tests for Data Warehouse & Repository (T05.06).
"""
from __future__ import annotations
import pytest

from streampulse.config import DIM_ROW_COUNTS, EXPECTED_ROWS
from streampulse.etl.pipeline import run_etl
from streampulse.ingestion.loader import read_source
from streampulse.warehouse.builder import build_warehouse
from streampulse.warehouse.repository import (
    get_analytics_frame,
    get_table_row_counts,
    olap_aggregate,
    warehouse_exists,
)


@pytest.fixture(scope="module")
def setup_warehouse():
    df_raw, sha = read_source()
    etl_result = run_etl(df_raw, sha)
    content_hash = build_warehouse(etl_result)
    return content_hash


def test_warehouse_built_and_integrity(setup_warehouse):
    assert warehouse_exists() is True
    counts = get_table_row_counts()
    for table, expected in DIM_ROW_COUNTS.items():
        assert counts[table] == expected, f"{table} count {counts[table]} != {expected}"


def test_analytics_frame(setup_warehouse):
    df = get_analytics_frame()
    assert len(df) == EXPECTED_ROWS
    # Assert all joined columns are populated
    assert df["User_Name"].isna().sum() == 0
    assert df["Song_Name"].isna().sum() == 0
    assert df["Artist_Name"].isna().sum() == 0
    assert df["Album_Name"].isna().sum() == 0
    assert df["Genre"].isna().sum() == 0
    assert df["Device"].isna().sum() == 0
    assert df["Country"].isna().sum() == 0
    assert df["Subscription"].isna().sum() == 0
    assert df["Date"].isna().sum() == 0


def test_olap_aggregation(setup_warehouse):
    # Slice & Dice aggregation
    agg = olap_aggregate(
        rows=["Genre"],
        cols=["Subscription"],
        measure="Streams",
        agg="SUM",
    )
    assert not agg.empty
    assert "Genre" in agg.columns
    assert "Free" in agg.columns
    assert "Premium" in agg.columns
    # Total sum across genres must equal total streams 10344
    total_streams = agg["Free"].sum() + agg["Premium"].sum()
    assert total_streams == 10344
