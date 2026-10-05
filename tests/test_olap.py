# spotify-streampulse/tests/test_olap.py
"""
Unit tests for OLAP operations: Slice, Dice, Roll-up, Drill-down, and Pivot.
"""
from __future__ import annotations
import pytest

from streampulse.warehouse.repository import olap_aggregate


def test_olap_slice():
    # Slice on Country='USA'
    res = olap_aggregate(
        rows=["Genre"],
        cols=[],
        measure="Streams",
        agg="SUM",
        filters={"Country": ["USA"]},
    )
    assert not res.empty
    assert "Genre" in res.columns
    assert "Streams" in res.columns
    # Ensure streams > 0
    assert res["Streams"].sum() > 0


def test_olap_dice():
    # Dice on Genre in ['Pop', 'Rock'] and Device in ['Mobile']
    res = olap_aggregate(
        rows=["Genre"],
        cols=["Device"],
        measure="Revenue_USD",
        agg="SUM",
        filters={"Genre": ["Pop", "Rock"], "Device": ["Mobile"]},
    )
    assert not res.empty
    assert "Genre" in res.columns
    assert "Mobile" in res.columns
    assert len(res) <= 2


def test_olap_hierarchy_rollup_drilldown():
    # Roll-up from Month to Quarter to Year
    month_res = olap_aggregate(rows=["Month_Name"], cols=[], measure="Streams", agg="SUM")
    quarter_res = olap_aggregate(rows=["Quarter"], cols=[], measure="Streams", agg="SUM")
    year_res = olap_aggregate(rows=["Year"], cols=[], measure="Streams", agg="SUM")

    # Sum of streams must be invariant across all levels
    assert month_res["Streams"].sum() == 10344
    assert quarter_res["Streams"].sum() == 10344
    assert year_res["Streams"].sum() == 10344
    assert len(quarter_res) == 4
    assert len(year_res) == 1


def test_olap_pivot():
    # Pivot Genre (rows) x Subscription (columns)
    piv = olap_aggregate(
        rows=["Genre"],
        cols=["Subscription"],
        measure="Streams",
        agg="COUNT",
    )
    assert "Genre" in piv.columns
    assert "Free" in piv.columns
    assert "Premium" in piv.columns
    # Total count = 1000
    assert (piv["Free"].sum() + piv["Premium"].sum()) == 1000
