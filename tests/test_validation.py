# spotify-streampulse/tests/test_validation.py
"""
Tests for Data Ingestion and Validation (T03.08).
"""
from __future__ import annotations
import pytest

from streampulse.config import SOURCE_FILE, EXPECTED_ROWS, EXPECTED_COLUMNS
from streampulse.ingestion.loader import read_source, sha256_of
from streampulse.ingestion.validator import run_validation


def test_source_checksum():
    sha = sha256_of(SOURCE_FILE)
    assert sha == "0a8228be9fbc05b278d70924f7efbcfe9c52622b39aa2bd2762de2e0b4712145"


def test_source_loader():
    df, sha = read_source()
    assert len(df) == EXPECTED_ROWS
    assert list(df.columns) == EXPECTED_COLUMNS
    assert sha == "0a8228be9fbc05b278d70924f7efbcfe9c52622b39aa2bd2762de2e0b4712145"


def test_validation_report_on_clean_data():
    df, sha = read_source()
    report = run_validation(df, sha)

    assert report.overall_status == "PASS"
    assert report.summary["failed"] == 0
    assert report.summary["passed"] >= 12

    # Check key integrity warnings exist (C1, C2)
    warning_checks = [c.id for c in report.checks if c.status == "WARN"]
    assert "V-K03" in warning_checks  # Song_ID is not a valid key
    assert "V-K04" in warning_checks  # Artist_ID is not a valid key
    assert "V-K05" in warning_checks  # Album_ID is not a valid key

    # Check business rules pass
    revenue_check = next(c for c in report.checks if c.id == "V-B01")
    assert revenue_check.status == "PASS"

    duration_check = next(c for c in report.checks if c.id == "V-B03")
    assert duration_check.status == "PASS"
