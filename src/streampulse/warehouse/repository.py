# spotify-streampulse/src/streampulse/warehouse/repository.py
"""
Repository — all warehouse read access goes through here.
R-DW-07: Only this module reads from the SQLite warehouse.
R-APP-01: No upward imports (only config, logging_setup).
Design.md §2.4: OLAP cube and query patterns.
"""
from __future__ import annotations
import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Generator

import pandas as pd

from streampulse.config import WAREHOUSE_FILE
from streampulse.logging_setup import get_logger

logger = get_logger(__name__)


@contextmanager
def _connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a read-only-intent SQLite connection with FK checks on."""
    conn = sqlite3.connect(f"file:{WAREHOUSE_FILE}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def _q(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame."""
    with _connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Dimension tables (full)
# ---------------------------------------------------------------------------

def get_dim_user() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_User ORDER BY User_SK")

def get_dim_track() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_Track ORDER BY Track_SK")

def get_dim_genre() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_Genre ORDER BY Genre_SK")

def get_dim_device() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_Device ORDER BY Device_SK")

def get_dim_country() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_Country ORDER BY Country_SK")

def get_dim_subscription() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_Subscription ORDER BY Subscription_SK")

def get_dim_time() -> pd.DataFrame:
    return _q("SELECT * FROM Dim_Time ORDER BY Date_SK")

def get_fact_streams() -> pd.DataFrame:
    return _q("SELECT * FROM Fact_Streams ORDER BY Stream_ID")


# ---------------------------------------------------------------------------
# Staging tables
# ---------------------------------------------------------------------------

def get_stg_raw() -> pd.DataFrame:
    return _q("SELECT * FROM Stg_Raw ORDER BY Stream_ID")

def get_stg_clean() -> pd.DataFrame:
    return _q("SELECT * FROM Stg_Clean ORDER BY Stream_ID")


# ---------------------------------------------------------------------------
# Flat denormalised analytics view
# ---------------------------------------------------------------------------

ANALYTICS_SQL = """
SELECT
    f.Stream_ID,
    u.User_ID, u.User_Name,
    tr.Song_Name, tr.Artist_Name, tr.Album_Name,
    g.Genre_Name  AS Genre,
    d.Device_Name AS Device,
    c.Country_Name AS Country,
    s.Subscription_Name AS Subscription,
    t.Full_Date AS Date,
    t.Year, t.Quarter, t.Month_Number, t.Month_Name,
    t.Day_Name, t.Is_Weekend,
    f.Streams, f.Duration_Minutes, f.Revenue_USD
FROM Fact_Streams f
JOIN Dim_User u       ON f.User_SK = u.User_SK
JOIN Dim_Track tr     ON f.Track_SK = tr.Track_SK
JOIN Dim_Genre g      ON f.Genre_SK = g.Genre_SK
JOIN Dim_Device d     ON f.Device_SK = d.Device_SK
JOIN Dim_Country c    ON f.Country_SK = c.Country_SK
JOIN Dim_Subscription s ON f.Subscription_SK = s.Subscription_SK
JOIN Dim_Time t       ON f.Date_SK = t.Date_SK
ORDER BY f.Stream_ID
"""

def get_analytics_frame() -> pd.DataFrame:
    """Full denormalised frame for EDA, ML feature extraction etc."""
    return _q(ANALYTICS_SQL)


# ---------------------------------------------------------------------------
# OLAP helpers
# ---------------------------------------------------------------------------

def get_pipeline_log() -> pd.DataFrame:
    return _q("SELECT * FROM Pipeline_Log ORDER BY Run_ID DESC")


def olap_aggregate(
    rows: list[str],
    cols: list[str],
    measure: str,
    agg: str = "SUM",
    filters: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Flexible OLAP aggregation.

    Args:
        rows: Grouping dimensions for the row axis.
        cols: Grouping dimensions for the column axis (pivoted if non-empty).
        measure: Measure column name (Streams / Duration_Minutes / Revenue_USD / Record_Count).
        agg: Aggregation function (SUM / AVG / COUNT / MIN / MAX).
        filters: Dict of {dimension_col: [allowed_values]}.

    Returns:
        Pivoted DataFrame (rows × cols) or flat table if cols is empty.
    """
    # Build the denormalised frame
    df = get_analytics_frame()

    # Apply filters
    if filters:
        for col, vals in filters.items():
            if col in df.columns and vals:
                df = df[df[col].isin(vals)]

    if df.empty:
        return pd.DataFrame()

    # Measure
    if measure == "Record_Count":
        df["_measure"] = 1
        measure_col = "_measure"
        agg_func = "sum"
    else:
        measure_col = measure
        agg_func = agg.lower()

    if cols:
        pivot = df.pivot_table(
            index=rows if rows else None,
            columns=cols,
            values=measure_col,
            aggfunc=agg_func,
            fill_value=0,
            observed=True,
        )
        pivot.columns = [str(c) for c in pivot.columns]
        pivot = pivot.reset_index()
        return pivot
    else:
        aggregated = df.groupby(rows, observed=True, sort=True)[measure_col].agg(agg_func).reset_index()
        aggregated.rename(columns={measure_col: measure}, inplace=True)
        return aggregated


def get_table_row_counts() -> dict[str, int]:
    """Return {table_name: row_count} for all warehouse tables."""
    tables = [
        "Stg_Raw", "Stg_Clean",
        "Dim_User", "Dim_Track", "Dim_Genre", "Dim_Device",
        "Dim_Country", "Dim_Subscription", "Dim_Time", "Fact_Streams",
    ]
    counts: dict[str, int] = {}
    with _connection() as conn:
        for t in tables:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()
            counts[t] = row["n"] if row else 0
    return counts


def warehouse_exists() -> bool:
    """Return True if the warehouse file exists and has the Fact_Streams table."""
    if not WAREHOUSE_FILE.exists():
        return False
    try:
        with _connection() as conn:
            conn.execute("SELECT 1 FROM Fact_Streams LIMIT 1")
        return True
    except Exception:
        return False
