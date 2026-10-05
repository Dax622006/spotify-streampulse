# spotify-streampulse/src/streampulse/ingestion/loader.py
"""
Source workbook loader — the ONLY module that reads the Excel file.
R-DS-02: No other module reads the Excel file.
R-DS-01: The workbook is permanently READ-ONLY.
"""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from streampulse.config import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    SOURCE_CHECKSUM_FILE,
    SOURCE_FILE,
    SOURCE_SHEET,
)
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file without loading it into pandas.

    R-DS-03: Record and compare at pipeline start.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_compare_checksum(current_sha: str) -> Tuple[str, bool]:
    """Load stored checksum or save current one on first run.

    Returns:
        (stored_sha, is_match) — is_match is True if the file is unchanged.
    """
    SOURCE_CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)

    if SOURCE_CHECKSUM_FILE.exists():
        stored = json.loads(SOURCE_CHECKSUM_FILE.read_text(encoding="utf-8"))
        stored_sha: str = stored.get("sha256", "")
        is_match = stored_sha == current_sha
        if not is_match:
            log_event(
                logger,
                logging.WARNING,
                "SOURCE_CHECKSUM_MISMATCH",
                "Source file checksum changed — rebuilding pipeline",
                details={"stored": stored_sha, "current": current_sha},
            )
        else:
            log_event(
                logger,
                logging.INFO,
                "SOURCE_CHECKSUM_OK",
                "Source checksum verified",
            )
        return stored_sha, is_match
    else:
        # First run: record checksum
        SOURCE_CHECKSUM_FILE.write_text(
            json.dumps({"sha256": current_sha, "file": str(SOURCE_FILE)}, indent=2),
            encoding="utf-8",
        )
        log_event(
            logger,
            logging.INFO,
            "SOURCE_CHECKSUM_OK",
            "Source checksum recorded for first time",
            details={"sha256": current_sha},
        )
        return current_sha, True


def read_source() -> Tuple[pd.DataFrame, str]:
    """Read the source workbook read-only and return (DataFrame, sha256).

    - Opens with openpyxl engine (read-only mode).
    - Date column kept as raw string (dtype=str handled per column).
    - Returns a copy so callers cannot mutate the loaded frame.
    R-DS-02, R-DS-05.
    """
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    current_sha = sha256_of(SOURCE_FILE)

    # Read with explicit dtypes: IDs and Date as str, numerics auto
    dtype_map: dict[str, object] = {
        "Stream_ID": str,
        "User_ID": str,
        "User_Name": str,
        "Song_ID": str,
        "Song_Name": str,
        "Artist_ID": str,
        "Artist_Name": str,
        "Album_ID": str,
        "Album_Name": str,
        "Genre": str,
        "Country": str,
        "Device": str,
        "Subscription": str,
        "Date": str,
    }
    df = pd.read_excel(
        SOURCE_FILE,
        sheet_name=SOURCE_SHEET,
        engine="openpyxl",
        dtype=dtype_map,
    )

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    log_event(
        logger,
        logging.INFO,
        "SOURCE_LOADED",
        f"Loaded source: {df.shape[0]} rows × {df.shape[1]} cols",
        details={"sha256": current_sha},
    )
    return df.copy(), current_sha
