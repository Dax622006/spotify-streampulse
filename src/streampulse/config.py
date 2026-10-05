# spotify-streampulse/src/streampulse/config.py
"""
Central configuration — single source of truth for all constants.
R-APP-06: No literals for paths/seeds/column names may appear elsewhere.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
# When installed with "pip install -e ." the package lives under src/.
# When run from the project root the calculation still works.
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent          # streampulse/
_PKG_ROOT = _SRC_DIR.parent           # src/
PROJECT_ROOT = _PKG_ROOT.parent        # spotify-streampulse/

# Data
RAW_DIR = PROJECT_ROOT / "data" / "raw"
WAREHOUSE_DIR = PROJECT_ROOT / "data" / "warehouse"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = ARTIFACTS_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

SOURCE_FILE = RAW_DIR / "spotify_analysis_dataset_updated.xlsx"
WAREHOUSE_FILE = WAREHOUSE_DIR / "streampulse_dw.sqlite"
BUILD_LOCK_FILE = WAREHOUSE_DIR / ".build.lock"

VALIDATION_REPORT_FILE = REPORTS_DIR / "validation_report.json"
ETL_REPORT_FILE = REPORTS_DIR / "etl_report.json"
SOURCE_CHECKSUM_FILE = REPORTS_DIR / "source_checksum.json"
WAREHOUSE_REPORT_FILE = REPORTS_DIR / "warehouse_report.json"
MODEL_REGISTRY_FILE = MODELS_DIR / "model_registry.json"
LOG_FILE = LOGS_DIR / "streampulse.log"
USERS_FILE = CONFIG_DIR / "users.json"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
SOURCE_SHEET = "SpotifyData"
EXPECTED_ROWS = 1_000
EXPECTED_COLUMNS: list[str] = [
    "Stream_ID", "User_ID", "User_Name",
    "Song_ID", "Song_Name", "Artist_ID", "Artist_Name",
    "Album_ID", "Album_Name",
    "Genre", "Country", "Device", "Subscription",
    "Date", "Streams", "Duration_Minutes", "Revenue_USD",
]

# Verified sums (for reconciliation validation at runtime only — NOT for hard-coding in UI)
EXPECTED_STREAM_SUM = 10_344          # used as acceptance assertion
EXPECTED_DURATION_SUM = 36_181
EXPECTED_REVENUE_SUM = 1_856.40

# ---------------------------------------------------------------------------
# Dimension expected row counts (for integrity check)
# ---------------------------------------------------------------------------
DIM_ROW_COUNTS: dict[str, int] = {
    "Dim_User": 100,
    "Dim_Track": 10,
    "Dim_Genre": 10,
    "Dim_Device": 4,
    "Dim_Country": 8,
    "Dim_Subscription": 2,
    "Dim_Time": 365,
    "Fact_Streams": 1_000,
}

# ---------------------------------------------------------------------------
# ML parameters  (R-ML-10, R-ML-07)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
PREMIUM_CLASS: str = "Premium"
FREE_CLASS: str = "Free"

# Feature registry  (R-ML-02)
SLR_FEATURES: list[str] = ["Streams"]
MLR_FEATURES: list[str] = ["Streams", "Duration_Minutes", "Device", "Genre", "Country"]
CLS_FEATURES: list[str] = ["Streams", "Duration_Minutes", "Device", "Genre", "Country"]
WHATIF_FEATURES: list[str] = CLS_FEATURES  # same set
NUMERIC_FEATURES: list[str] = ["Streams", "Duration_Minutes"]
CATEGORICAL_FEATURES: list[str] = ["Device", "Genre", "Country"]
REGRESSION_TARGET: str = "Revenue_USD"
CLASSIFICATION_TARGET: str = "Subscription"

# Forbidden columns (R-ML-03)
FORBIDDEN_FEATURES: list[str] = [
    "Stream_ID", "User_ID", "User_Name",
    "Song_ID", "Song_Name", "Artist_ID", "Artist_Name",
    "Album_ID", "Album_Name",
    "Date", "Revenue_USD",
]
FORBIDDEN_FOR_CLASSIFICATION: list[str] = FORBIDDEN_FEATURES  # Revenue already there
FORBIDDEN_FOR_REGRESSION: list[str] = [
    "Stream_ID", "User_ID", "User_Name",
    "Song_ID", "Song_Name", "Artist_ID", "Artist_Name",
    "Album_ID", "Album_Name",
    "Date", "Subscription",
]

# Revenue deterministic rule
PREMIUM_RATE: float = 0.35  # Revenue_USD = Streams * 0.35 for Premium
DURATION_MULTIPLIERS: set[int] = {2, 3, 4, 5}

# ---------------------------------------------------------------------------
# Clustering parameters  (R-ML-23, R-ML-24)
# ---------------------------------------------------------------------------
CLUSTER_CORR_THRESHOLD: float = 0.90
KMEANS_K_RANGE: tuple[int, int] = (2, 10)
KMEANS_N_INIT: int = 10

# ---------------------------------------------------------------------------
# Apriori defaults  (R-ML-27)
# ---------------------------------------------------------------------------
APRIORI_MIN_SUPPORT: float = 0.40
APRIORI_MIN_CONFIDENCE: float = 0.60
APRIORI_MAX_LEN: int = 3
APRIORI_MIN_LIFT: float = 1.0

# ---------------------------------------------------------------------------
# Auth / RBAC
# ---------------------------------------------------------------------------
ADMIN_ROLE: str = "ADMIN"
ANALYST_ROLE: str = "DATA_ANALYST"
LOCKOUT_FAILURES: int = 5
LOCKOUT_SECONDS: int = 30
PBKDF2_ITERATIONS: int = 260_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_MAX_BYTES: int = 1_024 * 1_024   # 1 MB
LOG_BACKUP_COUNT: int = 5

# ---------------------------------------------------------------------------
# Page IDs  (R-RBAC-02)
# ---------------------------------------------------------------------------
PAGE_ADMIN_DASHBOARD = "admin_dashboard"
PAGE_DATA_WAREHOUSE = "data_warehouse"
PAGE_STAR_SCHEMA = "star_schema"
PAGE_DATA_QUALITY = "data_quality"
PAGE_SYSTEM_LOGS = "system_logs"
PAGE_MODEL_STATUS = "model_status"

PAGE_ANALYST_DASHBOARD = "analyst_dashboard"
PAGE_OLAP = "olap"
PAGE_EDA = "eda"
PAGE_REGRESSION = "regression"
PAGE_CLASSIFICATION = "classification"
PAGE_CLUSTERING = "clustering"
PAGE_ASSOCIATION = "association"
PAGE_WHATIF = "whatif"

# ---------------------------------------------------------------------------
# Hierarchy definitions  (R-DW-09)
# ---------------------------------------------------------------------------
TIME_HIERARCHY: list[str] = ["Year", "Quarter", "Month", "Date"]   # coarse → fine
TRACK_HIERARCHY: list[str] = ["Artist", "Album", "Song"]            # coarse → fine

# Map from Time hierarchy level name → cube column name
TIME_LEVEL_COLUMNS: dict[str, str] = {
    "Year": "Year",
    "Quarter": "Quarter",
    "Month": "Month_Name",
    "Date": "Full_Date",
}
# Map from Track hierarchy level name → cube column name
TRACK_LEVEL_COLUMNS: dict[str, str] = {
    "Artist": "Artist_Name",
    "Album": "Album_Name",
    "Song": "Song_Name",
}

# All available measures for OLAP
ADDITIVE_MEASURES: list[str] = ["Streams", "Duration_Minutes", "Revenue_USD", "Record_Count"]
