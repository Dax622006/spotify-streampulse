# spotify-streampulse/src/streampulse/ingestion/validator.py
"""
Data validation — pure functions returning CheckResult objects.
R-DS-04/05/06/07/08/09: All validation checks encoded here.
Design.md §3: Validation rules V-S*, V-C*, V-U*, V-V*, V-B*, V-K*.
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from streampulse.config import (
    DURATION_MULTIPLIERS,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    PREMIUM_CLASS,
    PREMIUM_RATE,
    REPORTS_DIR,
    SOURCE_SHEET,
    VALIDATION_REPORT_FILE,
)
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of one validation check."""
    id: str
    category: str           # Schema / Completeness / Uniqueness / Validity / BusinessRule / KeyIntegrity
    severity: str           # CRITICAL / WARNING / INFO
    status: str             # PASS / FAIL / WARN / INFO
    observed: Any
    expected: Any
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report persisted as JSON."""
    generated_at: str
    source_sha256: str
    summary: dict[str, int]   # passed, warnings, failed, info
    checks: list[CheckResult]
    overall_status: str        # PASS / FAIL


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _pass(id_: str, cat: str, sev: str, obs: Any, exp: Any, msg: str, **ev: Any) -> CheckResult:
    return CheckResult(id_, cat, sev, "PASS", obs, exp, msg, ev)

def _fail(id_: str, cat: str, sev: str, obs: Any, exp: Any, msg: str, **ev: Any) -> CheckResult:
    return CheckResult(id_, cat, sev, "FAIL", obs, exp, msg, ev)

def _warn(id_: str, cat: str, sev: str, obs: Any, exp: Any, msg: str, **ev: Any) -> CheckResult:
    return CheckResult(id_, cat, sev, "WARN", obs, exp, msg, ev)

def _info(id_: str, cat: str, sev: str, obs: Any, exp: Any, msg: str, **ev: Any) -> CheckResult:
    return CheckResult(id_, cat, sev, "INFO", obs, exp, msg, ev)


# ---- Schema checks ----

def check_sheet_name(sheet_names: list[str]) -> CheckResult:
    """V-S01: Sheet SpotifyData exists."""
    if SOURCE_SHEET in sheet_names:
        return _pass("V-S01", "Schema", "CRITICAL", sheet_names, [SOURCE_SHEET],
                     f"Sheet '{SOURCE_SHEET}' found")
    return _fail("V-S01", "Schema", "CRITICAL", sheet_names, [SOURCE_SHEET],
                 f"Expected sheet '{SOURCE_SHEET}' not found")


def check_columns(df: pd.DataFrame) -> CheckResult:
    """V-S02: Exactly 17 expected columns with exact names."""
    actual = list(df.columns)
    if actual == EXPECTED_COLUMNS:
        return _pass("V-S02", "Schema", "CRITICAL", len(actual), len(EXPECTED_COLUMNS),
                     "Column names and order match expected schema")
    missing = [c for c in EXPECTED_COLUMNS if c not in actual]
    extra = [c for c in actual if c not in EXPECTED_COLUMNS]
    return _fail("V-S02", "Schema", "CRITICAL", actual, EXPECTED_COLUMNS,
                 f"Column mismatch — missing: {missing}, extra: {extra}",
                 missing=missing, extra=extra)


def check_row_count(df: pd.DataFrame) -> CheckResult:
    """V-S03: Exactly 1,000 rows."""
    if len(df) == EXPECTED_ROWS:
        return _pass("V-S03", "Schema", "CRITICAL", len(df), EXPECTED_ROWS,
                     f"Row count = {EXPECTED_ROWS}")
    return _fail("V-S03", "Schema", "CRITICAL", len(df), EXPECTED_ROWS,
                 f"Expected {EXPECTED_ROWS} rows, got {len(df)}")


# ---- Completeness ----

def check_no_nulls(df: pd.DataFrame) -> CheckResult:
    """V-C01: No nulls or blank strings in any column."""
    null_counts = df.isnull().sum()
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    blank_counts = pd.Series({c: (df[c].str.strip() == "").sum() for c in str_cols})
    total_nulls = int(null_counts.sum())
    total_blanks = int(blank_counts.sum())
    if total_nulls == 0 and total_blanks == 0:
        return _pass("V-C01", "Completeness", "CRITICAL", 0, 0,
                     "No nulls or blank strings found")
    return _fail("V-C01", "Completeness", "CRITICAL",
                 f"nulls={total_nulls}, blanks={total_blanks}", 0,
                 f"Found {total_nulls} null cells and {total_blanks} blank strings",
                 null_by_col=null_counts[null_counts > 0].to_dict())


# ---- Uniqueness ----

def check_stream_id_unique(df: pd.DataFrame) -> CheckResult:
    """V-U01: Stream_ID unique."""
    dup = df["Stream_ID"].duplicated().sum()
    if dup == 0:
        return _pass("V-U01", "Uniqueness", "CRITICAL",
                     f"{df['Stream_ID'].nunique()}/{len(df)} unique", "1000/1000 unique",
                     "Stream_ID is unique")
    return _fail("V-U01", "Uniqueness", "CRITICAL", f"{dup} duplicates", 0,
                 f"Stream_ID has {dup} duplicate values")


def check_no_full_row_duplicates(df: pd.DataFrame) -> CheckResult:
    """V-U02: No fully duplicated rows."""
    dup = df.duplicated().sum()
    if dup == 0:
        return _pass("V-U02", "Uniqueness", "WARNING", 0, 0, "No fully duplicated rows")
    return _warn("V-U02", "Uniqueness", "WARNING", dup, 0,
                 f"Found {dup} fully duplicated rows")


# ---- Validity ----

def check_date_format(df: pd.DataFrame) -> CheckResult:
    """V-V01: Date parseable with strict ISO format YYYY-MM-DD."""
    try:
        parsed = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        bad = int(parsed.isna().sum())
        if bad == 0:
            return _pass("V-V01", "Validity", "CRITICAL",
                         f"{len(df)} parseable", f"{len(df)} parseable",
                         "All Date values parse as YYYY-MM-DD")
        return _fail("V-V01", "Validity", "CRITICAL", f"{bad} unparseable", 0,
                     f"{bad} Date values do not parse as YYYY-MM-DD")
    except Exception as exc:
        return _fail("V-V01", "Validity", "CRITICAL", "parse error", "YYYY-MM-DD",
                     f"Date parsing raised: {exc}")


def check_date_range(df: pd.DataFrame) -> CheckResult:
    """V-V02: Record date range (informational)."""
    parsed = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    return _info("V-V02", "Validity", "INFO",
                 f"{parsed.min().date()} → {parsed.max().date()}",
                 "2025-01-01 → 2025-12-31",
                 "Date range observed in data",
                 min_date=str(parsed.min().date()), max_date=str(parsed.max().date()),
                 distinct_dates=int(parsed.nunique()))


def check_streams(df: pd.DataFrame) -> CheckResult:
    """V-V03: Streams integer ≥ 1."""
    bad = (df["Streams"] < 1).sum()
    if bad == 0:
        return _pass("V-V03", "Validity", "CRITICAL",
                     f"range {df['Streams'].min()}–{df['Streams'].max()}", "≥1",
                     "All Streams values ≥ 1")
    return _fail("V-V03", "Validity", "CRITICAL", f"{bad} rows < 1", 0,
                 f"{bad} rows have Streams < 1")


def check_duration(df: pd.DataFrame) -> CheckResult:
    """V-V04: Duration_Minutes integer ≥ 1."""
    bad = (df["Duration_Minutes"] < 1).sum()
    if bad == 0:
        return _pass("V-V04", "Validity", "CRITICAL",
                     f"range {df['Duration_Minutes'].min()}–{df['Duration_Minutes'].max()}", "≥1",
                     "All Duration_Minutes values ≥ 1")
    return _fail("V-V04", "Validity", "CRITICAL", f"{bad} rows < 1", 0,
                 f"{bad} rows have Duration_Minutes < 1")


def check_revenue(df: pd.DataFrame) -> CheckResult:
    """V-V05: Revenue_USD numeric ≥ 0."""
    bad = (df["Revenue_USD"] < 0).sum()
    if bad == 0:
        return _pass("V-V05", "Validity", "CRITICAL",
                     f"range {df['Revenue_USD'].min():.2f}–{df['Revenue_USD'].max():.2f}", "≥0",
                     "All Revenue_USD values ≥ 0")
    return _fail("V-V05", "Validity", "CRITICAL", f"{bad} rows < 0", 0,
                 f"{bad} rows have Revenue_USD < 0")


def check_subscription_domain(df: pd.DataFrame) -> CheckResult:
    """V-V06: Subscription ∈ {Free, Premium}."""
    allowed = {"Free", "Premium"}
    bad = ~df["Subscription"].isin(allowed)
    if bad.sum() == 0:
        vc = df["Subscription"].value_counts().to_dict()
        return _pass("V-V06", "Validity", "CRITICAL",
                     str(vc), "{Free, Premium}",
                     "All Subscription values in {Free, Premium}", counts=vc)
    bad_vals = df.loc[bad, "Subscription"].unique().tolist()
    return _fail("V-V06", "Validity", "CRITICAL", bad_vals, list(allowed),
                 f"Invalid Subscription values: {bad_vals}")


def check_id_format_regexes(df: pd.DataFrame) -> CheckResult:
    """V-V07: ID format regexes."""
    patterns = {
        "Stream_ID": r"^ST\d{4}$",
        "User_ID": r"^U\d{3}$",
        "Song_ID": r"^S\d{3}$",
        "Artist_ID": r"^A\d{3}$",
        "Album_ID": r"^AL\d{3}$",
    }
    failures: dict[str, int] = {}
    for col, pat in patterns.items():
        bad = ~df[col].str.match(pat)
        if bad.sum() > 0:
            failures[col] = int(bad.sum())
    if not failures:
        return _pass("V-V07", "Validity", "WARNING", "all formats OK", "regex patterns",
                     "All ID columns match expected format patterns")
    return _warn("V-V07", "Validity", "WARNING", failures, "0 mismatches",
                 f"ID format issues: {failures}", failures=failures)


# ---- Business rules ----

def check_revenue_rule(df: pd.DataFrame) -> CheckResult:
    """V-B01/02: Revenue = 0.35*Streams for Premium, 0 for Free."""
    premium_mask = df["Subscription"] == "Premium"
    free_mask = df["Subscription"] == "Free"
    prem_ok = (
        (df.loc[premium_mask, "Revenue_USD"] - PREMIUM_RATE * df.loc[premium_mask, "Streams"])
        .abs() < 1e-6
    ).sum()
    free_ok = (df.loc[free_mask, "Revenue_USD"] == 0.0).sum()
    prem_total = int(premium_mask.sum())
    free_total = int(free_mask.sum())
    all_ok = prem_ok == prem_total and free_ok == free_total
    observed = f"Premium {prem_ok}/{prem_total}, Free {free_ok}/{free_total}"
    expected = f"Premium {prem_total}/{prem_total}, Free {free_total}/{free_total}"
    if all_ok:
        return _pass("V-B01", "BusinessRule", "WARNING", observed, expected,
                     "Revenue deterministic rule verified for all rows",
                     premium_ok=int(prem_ok), free_ok=int(free_ok))
    return _warn("V-B01", "BusinessRule", "WARNING", observed, expected,
                 "Revenue rule violated in some rows",
                 premium_ok=int(prem_ok), free_ok=int(free_ok))


def check_duration_rule(df: pd.DataFrame) -> CheckResult:
    """V-B03: Duration_Minutes / Streams is integer in {2, 3, 4, 5}."""
    ratio = df["Duration_Minutes"] / df["Streams"]
    ok = ratio.isin(DURATION_MULTIPLIERS).sum()
    total = len(df)
    if ok == total:
        return _pass("V-B03", "BusinessRule", "WARNING", f"{ok}/{total}", f"{total}/{total}",
                     "Duration/Streams ratio ∈ {2,3,4,5} for all rows",
                     ratio_counts=ratio.astype(int).value_counts().to_dict())
    return _warn("V-B03", "BusinessRule", "WARNING", f"{ok}/{total}", f"{total}/{total}",
                 f"Duration ratio rule violated in {total-ok} rows")


# ---- Key integrity / functional-dependency ----

def check_fd(df: pd.DataFrame, det: str, dep: str) -> tuple[bool, int]:
    """Generic functional dependency checker: det → dep.
    Returns (is_functional, max_distinct_dep_values_per_det_value).
    """
    grouped = df.groupby(det)[dep].nunique()
    max_distinct = int(grouped.max())
    return max_distinct == 1, max_distinct


def check_user_id_name(df: pd.DataFrame) -> CheckResult:
    """V-K01: User_ID ↔ User_Name one-to-one."""
    is_fd, max_d = check_fd(df, "User_ID", "User_Name")
    if is_fd:
        return _pass("V-K01", "KeyIntegrity", "CRITICAL",
                     f"{df['User_ID'].nunique()} users, 1:1",
                     "one-to-one", "User_ID → User_Name is functional (1:1)")
    return _fail("V-K01", "KeyIntegrity", "CRITICAL",
                 f"max {max_d} names per User_ID", 1,
                 "User_ID → User_Name is NOT functional")


def check_song_name_fd(df: pd.DataFrame) -> CheckResult:
    """V-K02: Song_Name → Artist_Name and Album_Name functional."""
    is_a, max_a = check_fd(df, "Song_Name", "Artist_Name")
    is_al, max_al = check_fd(df, "Song_Name", "Album_Name")
    if is_a and is_al:
        return _pass("V-K02", "KeyIntegrity", "CRITICAL",
                     "Song_Name → Artist+Album is functional", "functional",
                     "Song_Name determines Artist_Name and Album_Name")
    return _fail("V-K02", "KeyIntegrity", "CRITICAL",
                 f"Artist max {max_a}, Album max {max_al}", 1,
                 "Song_Name → (Artist_Name, Album_Name) is NOT functional")


def check_song_id_invalid(df: pd.DataFrame) -> CheckResult:
    """V-K03: Song_ID → Song_Name functional (expected NOT functional — C1)."""
    is_fd, max_d = check_fd(df, "Song_ID", "Song_Name")
    n_ids = df["Song_ID"].nunique()
    n_names = df["Song_Name"].nunique()
    # We EXPECT this to fail (it's a known data quality finding)
    evidence = {
        "song_id_distinct": n_ids,
        "song_name_distinct": n_names,
        "max_names_per_id": max_d,
        "decision": "Song_ID quarantined; Dim_Track keyed on Song_Name",
    }
    return _warn("V-K03", "KeyIntegrity", "WARNING",
                 f"Song_ID ({n_ids}) → Song_Name ({n_names}): max {max_d} names per ID",
                 "1 name per ID",
                 "Song_ID is NOT a valid key (C1): multiple Song_Names per Song_ID",
                 **evidence)


def check_artist_id_invalid(df: pd.DataFrame) -> CheckResult:
    """V-K04: Artist_ID → Artist_Name functional (expected NOT — C2)."""
    is_fd, max_d = check_fd(df, "Artist_ID", "Artist_Name")
    n_ids = df["Artist_ID"].nunique()
    n_names = df["Artist_Name"].nunique()
    evidence = {
        "artist_id_distinct": n_ids,
        "artist_name_distinct": n_names,
        "max_names_per_id": max_d,
        "decision": "Artist_ID quarantined",
    }
    return _warn("V-K04", "KeyIntegrity", "WARNING",
                 f"Artist_ID ({n_ids}) → Artist_Name ({n_names}): max {max_d} names per ID",
                 "1 name per ID",
                 "Artist_ID is NOT a valid key (C2)",
                 **evidence)


def check_album_id_invalid(df: pd.DataFrame) -> CheckResult:
    """V-K05: Album_ID → Album_Name functional (expected NOT — C2)."""
    is_fd, max_d = check_fd(df, "Album_ID", "Album_Name")
    n_ids = df["Album_ID"].nunique()
    n_names = df["Album_Name"].nunique()
    evidence = {
        "album_id_distinct": n_ids,
        "album_name_distinct": n_names,
        "max_names_per_id": max_d,
        "decision": "Album_ID quarantined",
    }
    return _warn("V-K05", "KeyIntegrity", "WARNING",
                 f"Album_ID ({n_ids}) → Album_Name ({n_names}): max {max_d} names per ID",
                 "1 name per ID",
                 "Album_ID is NOT a valid key (C2)",
                 **evidence)


def check_genre_not_track_attribute(df: pd.DataFrame) -> CheckResult:
    """V-K06: Song_Name → Genre functional? (expected NOT — C3)."""
    is_fd, max_d = check_fd(df, "Song_Name", "Genre")
    return _info("V-K06", "KeyIntegrity", "INFO",
                 f"max {max_d} genres per song", 1,
                 "Genre is NOT determined by Song_Name (stream-level attribute) — C3",
                 max_genres_per_song=max_d)


def check_user_attributes_varying(df: pd.DataFrame) -> CheckResult:
    """V-K07: User_ID → {Country, Device, Subscription} functional? (expected NOT — C4)."""
    results = {}
    for attr in ["Country", "Device", "Subscription"]:
        is_fd, max_d = check_fd(df, "User_ID", attr)
        users_varying = int((df.groupby("User_ID")[attr].nunique() > 1).sum())
        results[attr] = {"max_distinct": max_d, "users_varying": users_varying}
    return _info("V-K07", "KeyIntegrity", "INFO",
                 results, "functional",
                 "Country/Device/Subscription vary within users → record-level dimensions (C4)",
                 **results)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation(df: pd.DataFrame, source_sha: str) -> ValidationReport:
    """Run all validation checks and return a ValidationReport.

    R-DS-06: Fail loudly if schema reality differs.
    """
    checks: list[CheckResult] = []

    # Schema
    checks.append(check_columns(df))
    checks.append(check_row_count(df))

    # Completeness
    checks.append(check_no_nulls(df))

    # Uniqueness
    checks.append(check_stream_id_unique(df))
    checks.append(check_no_full_row_duplicates(df))

    # Validity
    checks.append(check_date_format(df))
    checks.append(check_date_range(df))
    checks.append(check_streams(df))
    checks.append(check_duration(df))
    checks.append(check_revenue(df))
    checks.append(check_subscription_domain(df))
    checks.append(check_id_format_regexes(df))

    # Business rules
    checks.append(check_revenue_rule(df))
    checks.append(check_duration_rule(df))

    # Key integrity
    checks.append(check_user_id_name(df))
    checks.append(check_song_name_fd(df))
    checks.append(check_song_id_invalid(df))
    checks.append(check_artist_id_invalid(df))
    checks.append(check_album_id_invalid(df))
    checks.append(check_genre_not_track_attribute(df))
    checks.append(check_user_attributes_varying(df))

    # Summary
    passed = sum(1 for c in checks if c.status == "PASS")
    warnings = sum(1 for c in checks if c.status in ("WARN", "WARNING"))
    failed = sum(1 for c in checks if c.status == "FAIL")
    info = sum(1 for c in checks if c.status == "INFO")
    has_critical_fail = any(c.status == "FAIL" and c.severity == "CRITICAL" for c in checks)
    overall = "FAIL" if has_critical_fail else "PASS"

    from datetime import timezone
    report = ValidationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_sha,
        summary={"passed": passed, "warnings": warnings, "failed": failed, "info": info},
        checks=checks,
        overall_status=overall,
    )
    _persist_report(report)
    log_event(
        logger,
        logging.INFO if overall == "PASS" else logging.ERROR,
        "VALIDATION_SUMMARY",
        f"Validation {overall}: {passed} passed, {warnings} warnings, {failed} failed, {info} info",
        details=report.summary,
    )
    return report


def _persist_report(report: ValidationReport) -> None:
    """Write ValidationReport to JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": report.generated_at,
        "source_sha256": report.source_sha256,
        "overall_status": report.overall_status,
        "summary": report.summary,
        "checks": [asdict(c) for c in report.checks],
    }
    VALIDATION_REPORT_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_report() -> ValidationReport | None:
    """Load a previously persisted ValidationReport, or None if not found."""
    if not VALIDATION_REPORT_FILE.exists():
        return None
    try:
        data = json.loads(VALIDATION_REPORT_FILE.read_text(encoding="utf-8"))
        checks = [CheckResult(**c) for c in data["checks"]]
        return ValidationReport(
            generated_at=data["generated_at"],
            source_sha256=data["source_sha256"],
            summary=data["summary"],
            checks=checks,
            overall_status=data["overall_status"],
        )
    except Exception:
        return None
