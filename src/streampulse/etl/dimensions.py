# spotify-streampulse/src/streampulse/etl/dimensions.py
"""
ETL Steps T06–T09 — Extract dimension members and assign surrogate keys.
R-DW-03: Integer surrogate keys; Dim_Time uses YYYYMMDD smart key.
R-DW-04: Natural keys unique in each dimension.
R-ETL-02: Deterministic — same input → identical keys.
Design.md §2.3 Table specifications.
"""
from __future__ import annotations
import logging
from datetime import date

import pandas as pd

from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Surrogate key helpers
# ---------------------------------------------------------------------------

def _assign_sk(df: pd.DataFrame, natural_key_cols: list[str], sk_name: str,
               sort_cols: list[str] | None = None) -> pd.DataFrame:
    """Assign deterministic integer surrogate keys 1..N sorted by natural key.

    R-ETL-02: sort_cols determines ordering; defaults to natural_key_cols.
    """
    sort_by = sort_cols if sort_cols is not None else natural_key_cols
    out = df.sort_values(sort_by, key=lambda s: s.str.lower() if s.dtype == object else s)
    out = out.reset_index(drop=True)
    out[sk_name] = range(1, len(out) + 1)
    return out


# ---------------------------------------------------------------------------
# T06 — Dim_User
# ---------------------------------------------------------------------------

def build_dim_user(df: pd.DataFrame) -> pd.DataFrame:
    """Extract distinct (User_ID, User_Name) pairs and assign User_SK.

    Returns DataFrame: User_SK (PK), User_ID (natural key), User_Name.
    """
    dim = df[["User_ID", "User_Name"]].drop_duplicates(subset=["User_ID"]).copy()
    # Verify 1:1 User_ID → User_Name
    fd_check = df.groupby("User_ID")["User_Name"].nunique()
    if fd_check.max() > 1:
        raise ValueError("ETL T06: User_ID → User_Name is not functional. Validation should have caught this.")
    dim = _assign_sk(dim, ["User_ID"], "User_SK", sort_cols=["User_ID"])
    dim = dim[["User_SK", "User_ID", "User_Name"]]
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_User built: {len(dim)} rows", details={"rows": len(dim)})
    return dim.reset_index(drop=True)


# ---------------------------------------------------------------------------
# T07 — Dim_Track
# ---------------------------------------------------------------------------

def build_dim_track(df: pd.DataFrame) -> pd.DataFrame:
    """Extract distinct (Song_Name, Artist_Name, Album_Name) triples and assign Track_SK.

    Natural key: Song_Name (determines Artist_Name and Album_Name — verified).
    Source Song_ID/Artist_ID/Album_ID are NOT loaded here (R-DS-08).
    """
    dim = df[["Song_Name", "Artist_Name", "Album_Name"]].drop_duplicates(subset=["Song_Name"]).copy()
    dim = _assign_sk(dim, ["Song_Name"], "Track_SK", sort_cols=["Song_Name"])
    dim = dim[["Track_SK", "Song_Name", "Artist_Name", "Album_Name"]]
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_Track built: {len(dim)} rows", details={"rows": len(dim)})
    return dim.reset_index(drop=True)


# ---------------------------------------------------------------------------
# T08 — Simple flat dimensions
# ---------------------------------------------------------------------------

def build_dim_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct Genre values sorted alphabetically, SK=1..N."""
    dim = pd.DataFrame({"Genre_Name": sorted(df["Genre"].unique())})
    dim["Genre_SK"] = range(1, len(dim) + 1)
    dim = dim[["Genre_SK", "Genre_Name"]]
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_Genre built: {len(dim)} rows", details={"members": dim["Genre_Name"].tolist()})
    return dim.reset_index(drop=True)


def build_dim_device(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct Device values sorted alphabetically, SK=1..N."""
    dim = pd.DataFrame({"Device_Name": sorted(df["Device"].unique())})
    dim["Device_SK"] = range(1, len(dim) + 1)
    dim = dim[["Device_SK", "Device_Name"]]
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_Device built: {len(dim)} rows", details={"members": dim["Device_Name"].tolist()})
    return dim.reset_index(drop=True)


def build_dim_country(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct Country values sorted alphabetically, SK=1..N."""
    dim = pd.DataFrame({"Country_Name": sorted(df["Country"].unique())})
    dim["Country_SK"] = range(1, len(dim) + 1)
    dim = dim[["Country_SK", "Country_Name"]]
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_Country built: {len(dim)} rows",
              details={"members": dim["Country_Name"].tolist()})
    return dim.reset_index(drop=True)


def build_dim_subscription(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct Subscription values (Free=1, Premium=2 by sort)."""
    members = sorted(df["Subscription"].unique())
    dim = pd.DataFrame({"Subscription_Name": members})
    dim["Subscription_SK"] = range(1, len(dim) + 1)
    dim = dim[["Subscription_SK", "Subscription_Name"]]
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_Subscription built: {len(dim)} rows",
              details={"members": dim["Subscription_Name"].tolist()})
    return dim.reset_index(drop=True)


# ---------------------------------------------------------------------------
# T09 — Dim_Time (complete 2025 calendar)
# ---------------------------------------------------------------------------

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_dim_time(min_date: date | None = None, max_date: date | None = None) -> pd.DataFrame:
    """Generate a complete calendar year 2025 (365 rows).

    D-12: Dim_Time is a full 2025 calendar with Is_Observed flag.
    Smart key: Date_SK = YYYYMMDD integer (R-DW-03 documented exception).
    """
    year = 2025
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    dates = pd.date_range(start, end, freq="D")

    # Observed dates from the source (used for Is_Observed flag)
    observed: set[str] = set()
    # (populated by the caller via source dates; here we build the full calendar)

    records = []
    for d in dates:
        records.append({
            "Date_SK": int(d.strftime("%Y%m%d")),
            "Full_Date": d.date(),
            "Year": d.year,
            "Quarter": d.quarter,
            "Month_Number": d.month,
            "Month_Name": MONTH_NAMES[d.month - 1],
            "ISO_Week": d.isocalendar().week,
            "Day_Of_Month": d.day,
            "Day_Name": DAY_NAMES[d.dayofweek],
            "Is_Weekend": int(d.dayofweek >= 5),
        })

    dim = pd.DataFrame(records)
    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Dim_Time built: {len(dim)} rows (full 2025 calendar)")
    return dim


def build_dim_time_with_observed(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Build Dim_Time and mark Is_Observed from the actual source dates."""
    dim = build_dim_time()
    source_date_sks = set(
        df_clean["Date"].dt.strftime("%Y%m%d").astype(int).unique()
    )
    dim["Is_Observed"] = dim["Date_SK"].isin(source_date_sks).astype(int)
    return dim
