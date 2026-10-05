# spotify-streampulse/src/streampulse/warehouse/builder.py
"""
Warehouse builder — creates/populates the SQLite star-schema.
R-DW-06: FK constraints declared (PRAGMA foreign_keys = ON).
R-ETL-05: Transactional load — commit atomically or roll back.
R-DW-07: No view/table created outside schema definitions here.
Design.md §2.3: Table DDL specifications.
"""
from __future__ import annotations
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from streampulse.config import (
    WAREHOUSE_DIR,
    WAREHOUSE_FILE,
    WAREHOUSE_REPORT_FILE,
    REPORTS_DIR,
)
from streampulse.etl.pipeline import EtlResult
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL_STATEMENTS = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Stg_Raw (
    Stream_ID TEXT PRIMARY KEY,
    User_ID TEXT NOT NULL,
    User_Name TEXT NOT NULL,
    Song_ID TEXT NOT NULL,
    Song_Name TEXT NOT NULL,
    Artist_ID TEXT NOT NULL,
    Artist_Name TEXT NOT NULL,
    Album_ID TEXT NOT NULL,
    Album_Name TEXT NOT NULL,
    Genre TEXT NOT NULL,
    Country TEXT NOT NULL,
    Device TEXT NOT NULL,
    Subscription TEXT NOT NULL,
    Date TEXT NOT NULL,
    Streams INTEGER NOT NULL,
    Duration_Minutes INTEGER NOT NULL,
    Revenue_USD REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS Stg_Clean (
    Stream_ID TEXT PRIMARY KEY,
    User_ID TEXT NOT NULL,
    Song_Name TEXT NOT NULL,
    Genre TEXT NOT NULL,
    Country TEXT NOT NULL,
    Device TEXT NOT NULL,
    Subscription TEXT NOT NULL,
    Date TEXT NOT NULL,
    Streams INTEGER NOT NULL,
    Duration_Minutes INTEGER NOT NULL,
    Revenue_USD REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_User (
    User_SK INTEGER PRIMARY KEY,
    User_ID TEXT NOT NULL UNIQUE,
    User_Name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Track (
    Track_SK INTEGER PRIMARY KEY,
    Song_Name TEXT NOT NULL UNIQUE,
    Artist_Name TEXT NOT NULL,
    Album_Name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Genre (
    Genre_SK INTEGER PRIMARY KEY,
    Genre_Name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Dim_Device (
    Device_SK INTEGER PRIMARY KEY,
    Device_Name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Dim_Country (
    Country_SK INTEGER PRIMARY KEY,
    Country_Name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Dim_Subscription (
    Subscription_SK INTEGER PRIMARY KEY,
    Subscription_Name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Dim_Time (
    Date_SK INTEGER PRIMARY KEY,
    Full_Date TEXT NOT NULL UNIQUE,
    Year INTEGER NOT NULL,
    Quarter INTEGER NOT NULL,
    Month_Number INTEGER NOT NULL,
    Month_Name TEXT NOT NULL,
    ISO_Week INTEGER NOT NULL,
    Day_Of_Month INTEGER NOT NULL,
    Day_Name TEXT NOT NULL,
    Is_Weekend INTEGER NOT NULL,
    Is_Observed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS Fact_Streams (
    Stream_ID TEXT PRIMARY KEY,
    User_SK INTEGER NOT NULL REFERENCES Dim_User(User_SK),
    Track_SK INTEGER NOT NULL REFERENCES Dim_Track(Track_SK),
    Genre_SK INTEGER NOT NULL REFERENCES Dim_Genre(Genre_SK),
    Device_SK INTEGER NOT NULL REFERENCES Dim_Device(Device_SK),
    Country_SK INTEGER NOT NULL REFERENCES Dim_Country(Country_SK),
    Subscription_SK INTEGER NOT NULL REFERENCES Dim_Subscription(Subscription_SK),
    Date_SK INTEGER NOT NULL REFERENCES Dim_Time(Date_SK),
    Streams INTEGER NOT NULL,
    Duration_Minutes INTEGER NOT NULL,
    Revenue_USD REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS Pipeline_Log (
    Run_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Run_At TEXT NOT NULL,
    Source_SHA256 TEXT NOT NULL,
    Status TEXT NOT NULL,
    Duration_S REAL,
    Notes TEXT
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def warehouse_content_hash(conn: sqlite3.Connection) -> str:
    """Compute a stable hash over all data in the warehouse (for immutability checks)."""
    tables = [
        "Dim_User", "Dim_Track", "Dim_Genre", "Dim_Device",
        "Dim_Country", "Dim_Subscription", "Dim_Time", "Fact_Streams",
    ]
    h = hashlib.sha256()
    for t in tables:
        df = pd.read_sql(f"SELECT * FROM {t} ORDER BY 1", conn)
        h.update(t.encode())
        h.update(df.to_csv(index=False).encode())
    return h.hexdigest()


def _df_to_sqlite(df: pd.DataFrame, table: str, conn: sqlite3.Connection) -> None:
    """Write DataFrame replacing table contents (within an open transaction)."""
    # Convert date/datetime columns to str for SQLite
    df_out = df.copy()
    for col in df_out.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df_out[col] = df_out[col].dt.strftime("%Y-%m-%d")
    for col in df_out.columns:
        if hasattr(df_out[col], "dt") and hasattr(df_out[col].dt, "date"):
            try:
                df_out[col] = df_out[col].astype(str)
            except Exception:
                pass
    # Full_Date may be a date object
    if "Full_Date" in df_out.columns:
        df_out["Full_Date"] = df_out["Full_Date"].astype(str)
    df_out.to_sql(table, conn, if_exists="replace", index=False)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_warehouse(etl: EtlResult) -> str:
    """
    (Re)create the SQLite warehouse from an EtlResult.

    Executes DDL, then loads all tables inside a single transaction.
    Returns the warehouse content hash post-load.

    Raises:
        Exception: rolls back and re-raises on any error (R-ETL-05).
    """
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import timezone
    run_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    log_event(logger, logging.INFO, "WAREHOUSE_BUILD_START", "Building SQLite warehouse")

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(WAREHOUSE_FILE, timeout=30)
        conn.executescript("PRAGMA foreign_keys = OFF;")

        # --- DDL ---
        for stmt in DDL_STATEMENTS.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()

        # --- Transactional data load (R-ETL-05) ---
        with conn:  # auto-commit or roll back
            conn.execute("PRAGMA foreign_keys = ON;")
            tables_dfs = [
                ("Stg_Raw", etl.stg_raw),
                ("Stg_Clean", etl.stg_clean[[
                    "Stream_ID", "User_ID", "Song_Name", "Genre", "Country",
                    "Device", "Subscription", "Date", "Streams", "Duration_Minutes",
                    "Revenue_USD",
                ]]),
                ("Dim_User", etl.dim_user),
                ("Dim_Track", etl.dim_track),
                ("Dim_Genre", etl.dim_genre),
                ("Dim_Device", etl.dim_device),
                ("Dim_Country", etl.dim_country),
                ("Dim_Subscription", etl.dim_subscription),
                ("Dim_Time", etl.dim_time),
                ("Fact_Streams", etl.fact_streams),
            ]
            for table_name, df in tables_dfs:
                _df_to_sqlite(df, table_name, conn)
                log_event(logger, logging.DEBUG, "WAREHOUSE_TABLE_LOADED",
                          f"Loaded {table_name}: {len(df)} rows",
                          details={"table": table_name, "rows": len(df)})

        # --- Post-load integrity check ---
        _verify_fk_integrity(conn)
        content_hash = warehouse_content_hash(conn)
        duration = time.monotonic() - t0

        # --- Log pipeline run ---
        conn.execute(
            "INSERT INTO Pipeline_Log (Run_At, Source_SHA256, Status, Duration_S, Notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_at, etl.source_sha256, "SUCCESS", round(duration, 3), None),
        )
        conn.commit()

        _write_warehouse_report(etl, content_hash, duration)
        log_event(logger, logging.INFO, "WAREHOUSE_BUILD_END",
                  f"Warehouse built in {duration:.2f}s",
                  details={"content_hash": content_hash[:16], "duration_s": round(duration, 3)})
        return content_hash

    except Exception as exc:
        if conn:
            conn.rollback()
            try:
                conn.execute(
                    "INSERT INTO Pipeline_Log (Run_At, Source_SHA256, Status, Duration_S, Notes) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_at, etl.source_sha256, "FAIL", None, str(exc)[:500]),
                )
                conn.commit()
            except Exception:
                pass
        log_event(logger, logging.ERROR, "WAREHOUSE_BUILD_ERROR",
                  f"Warehouse build failed: {exc}")
        raise
    finally:
        if conn:
            conn.close()


def _verify_fk_integrity(conn: sqlite3.Connection) -> None:
    """R-DW-06: Verify FK integrity with PRAGMA foreign_key_check."""
    rows = conn.execute("PRAGMA foreign_key_check").fetchall()
    if rows:
        raise ValueError(
            f"FK integrity check failed after load — {len(rows)} violations: {rows[:3]}"
        )
    log_event(logger, logging.INFO, "PIPELINE_STEP", "FK integrity check PASSED")


def _write_warehouse_report(etl: EtlResult, content_hash: str, duration_s: float) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import timezone
    data = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": etl.source_sha256,
        "content_hash": content_hash,
        "duration_s": round(duration_s, 3),
        "table_row_counts": {
            "Stg_Raw": len(etl.stg_raw),
            "Stg_Clean": len(etl.stg_clean),
            "Dim_User": len(etl.dim_user),
            "Dim_Track": len(etl.dim_track),
            "Dim_Genre": len(etl.dim_genre),
            "Dim_Device": len(etl.dim_device),
            "Dim_Country": len(etl.dim_country),
            "Dim_Subscription": len(etl.dim_subscription),
            "Dim_Time": len(etl.dim_time),
            "Fact_Streams": len(etl.fact_streams),
        },
    }
    WAREHOUSE_REPORT_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_warehouse_report() -> dict | None:
    if not WAREHOUSE_REPORT_FILE.exists():
        return None
    return json.loads(WAREHOUSE_REPORT_FILE.read_text(encoding="utf-8"))
