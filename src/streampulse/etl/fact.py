# spotify-streampulse/src/streampulse/etl/fact.py
"""
ETL Step T10 — Build Fact_Streams.
R-ETL-04: Unresolved FK → hard error, never a default "Unknown" member.
R-DW-02: Grain = one row per Stream_ID; Stream_ID is a degenerate dimension.
R-DW-05: Fact holds FKs and additive measures only.
"""
from __future__ import annotations
import logging

import pandas as pd

from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


def build_fact_streams(
    df_clean: pd.DataFrame,
    dim_user: pd.DataFrame,
    dim_track: pd.DataFrame,
    dim_genre: pd.DataFrame,
    dim_device: pd.DataFrame,
    dim_country: pd.DataFrame,
    dim_subscription: pd.DataFrame,
    dim_time: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve each source row to dimension surrogate keys and produce Fact_Streams.

    Args:
        df_clean: Typed and cleaned staging frame.
        dim_*: Dimension DataFrames produced by etl.dimensions.

    Returns:
        Fact DataFrame with Stream_ID, seven FKs, and three measures.

    Raises:
        ValueError: If any FK cannot be resolved (R-ETL-04).
    """
    fact = df_clean[["Stream_ID", "User_ID", "Song_Name", "Genre", "Device",
                      "Country", "Subscription", "Streams", "Duration_Minutes",
                      "Revenue_USD"]].copy()

    # Compute Date_SK from parsed Date column
    fact["Date_SK"] = fact.index.map(lambda _: None)  # placeholder
    fact["Date_SK"] = df_clean["Date"].dt.strftime("%Y%m%d").astype(int)

    # --- Join User_SK ---
    user_map = dim_user.set_index("User_ID")["User_SK"]
    fact["User_SK"] = fact["User_ID"].map(user_map)
    _assert_no_nulls(fact, "User_SK", "User_ID")

    # --- Join Track_SK ---
    track_map = dim_track.set_index("Song_Name")["Track_SK"]
    fact["Track_SK"] = fact["Song_Name"].map(track_map)
    _assert_no_nulls(fact, "Track_SK", "Song_Name")

    # --- Join Genre_SK ---
    genre_map = dim_genre.set_index("Genre_Name")["Genre_SK"]
    fact["Genre_SK"] = fact["Genre"].map(genre_map)
    _assert_no_nulls(fact, "Genre_SK", "Genre")

    # --- Join Device_SK ---
    device_map = dim_device.set_index("Device_Name")["Device_SK"]
    fact["Device_SK"] = fact["Device"].map(device_map)
    _assert_no_nulls(fact, "Device_SK", "Device")

    # --- Join Country_SK ---
    country_map = dim_country.set_index("Country_Name")["Country_SK"]
    fact["Country_SK"] = fact["Country"].map(country_map)
    _assert_no_nulls(fact, "Country_SK", "Country")

    # --- Join Subscription_SK ---
    sub_map = dim_subscription.set_index("Subscription_Name")["Subscription_SK"]
    fact["Subscription_SK"] = fact["Subscription"].map(sub_map)
    _assert_no_nulls(fact, "Subscription_SK", "Subscription")

    # --- Join Date_SK ---
    valid_date_sks = set(dim_time["Date_SK"].values)
    bad_dates = ~fact["Date_SK"].isin(valid_date_sks)
    if bad_dates.sum() > 0:
        bad_vals = fact.loc[bad_dates, "Date_SK"].unique().tolist()
        raise ValueError(
            f"R-ETL-04: {bad_dates.sum()} fact rows have Date_SK not in Dim_Time: {bad_vals[:5]}"
        )

    # Select final columns
    fact_out = fact[[
        "Stream_ID", "User_SK", "Track_SK", "Genre_SK", "Device_SK",
        "Country_SK", "Subscription_SK", "Date_SK",
        "Streams", "Duration_Minutes", "Revenue_USD",
    ]].copy()

    log_event(logger, logging.INFO, "PIPELINE_STEP",
              f"Fact_Streams built: {len(fact_out)} rows",
              details={"null_fks": int(fact_out.isnull().sum().sum())})
    return fact_out.reset_index(drop=True)


def _assert_no_nulls(fact: pd.DataFrame, sk_col: str, source_col: str) -> None:
    """R-ETL-04: Any unresolved FK is a hard error."""
    bad = fact[sk_col].isna().sum()
    if bad > 0:
        unresolved = fact.loc[fact[sk_col].isna(), source_col].unique().tolist()
        raise ValueError(
            f"R-ETL-04: {bad} fact rows have unresolved FK '{sk_col}' "
            f"(source column '{source_col}'). Unresolved values: {unresolved[:5]}"
        )
