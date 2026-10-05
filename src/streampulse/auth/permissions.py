# spotify-streampulse/src/streampulse/auth/permissions.py
"""
Single source of authorization truth.
R-RBAC-01: Exactly two roles: ADMIN and DATA_ANALYST.
R-RBAC-02: Page IDs are role-specific.
AC-AUTH-2: No page ID is shared between roles.
"""
from __future__ import annotations
from typing import Final

from streampulse.config import (
    ADMIN_ROLE,
    ANALYST_ROLE,
    PAGE_ADMIN_DASHBOARD,
    PAGE_DATA_WAREHOUSE,
    PAGE_STAR_SCHEMA,
    PAGE_DATA_QUALITY,
    PAGE_SYSTEM_LOGS,
    PAGE_MODEL_STATUS,
    PAGE_ANALYST_DASHBOARD,
    PAGE_OLAP,
    PAGE_EDA,
    PAGE_REGRESSION,
    PAGE_CLASSIFICATION,
    PAGE_CLUSTERING,
    PAGE_ASSOCIATION,
    PAGE_WHATIF,
)

ADMIN_PAGES: Final[list[str]] = [
    PAGE_ADMIN_DASHBOARD,
    PAGE_DATA_WAREHOUSE,
    PAGE_STAR_SCHEMA,
    PAGE_DATA_QUALITY,
    PAGE_SYSTEM_LOGS,
    PAGE_MODEL_STATUS,
]

ANALYST_PAGES: Final[list[str]] = [
    PAGE_ANALYST_DASHBOARD,
    PAGE_OLAP,
    PAGE_EDA,
    PAGE_REGRESSION,
    PAGE_CLASSIFICATION,
    PAGE_CLUSTERING,
    PAGE_ASSOCIATION,
    PAGE_WHATIF,
]

ROLE_PAGES: Final[dict[str, list[str]]] = {
    ADMIN_ROLE: ADMIN_PAGES,
    ANALYST_ROLE: ANALYST_PAGES,
}

# Page labels for navigation display
PAGE_LABELS: Final[dict[str, str]] = {
    PAGE_ADMIN_DASHBOARD: "Dashboard",
    PAGE_DATA_WAREHOUSE: "Data Warehouse",
    PAGE_STAR_SCHEMA: "Star Schema",
    PAGE_DATA_QUALITY: "Data Quality Monitor",
    PAGE_SYSTEM_LOGS: "System Logs",
    PAGE_MODEL_STATUS: "Model Status",
    PAGE_ANALYST_DASHBOARD: "Dashboard",
    PAGE_OLAP: "OLAP Analysis",
    PAGE_EDA: "Exploratory Data Analysis",
    PAGE_REGRESSION: "Regression Analysis",
    PAGE_CLASSIFICATION: "Classification Models",
    PAGE_CLUSTERING: "Clustering Analysis",
    PAGE_ASSOCIATION: "Association Mining",
    PAGE_WHATIF: "What-If Simulator",
}

PAGE_ICONS: Final[dict[str, str]] = {
    PAGE_ADMIN_DASHBOARD: "📊",
    PAGE_DATA_WAREHOUSE: "🏛️",
    PAGE_STAR_SCHEMA: "⭐",
    PAGE_DATA_QUALITY: "🛡️",
    PAGE_SYSTEM_LOGS: "📜",
    PAGE_MODEL_STATUS: "🤖",
    PAGE_ANALYST_DASHBOARD: "📈",
    PAGE_OLAP: "🧊",
    PAGE_EDA: "🔍",
    PAGE_REGRESSION: "📉",
    PAGE_CLASSIFICATION: "🎯",
    PAGE_CLUSTERING: "🧮",
    PAGE_ASSOCIATION: "🛒",
    PAGE_WHATIF: "🔮",
}


def is_page_permitted(role: str | None, page_id: str) -> bool:
    """Return True if role has access to page_id."""
    if not role:
        return False
    return page_id in ROLE_PAGES.get(role, [])


def get_pages_for_role(role: str | None) -> list[str]:
    """Return permitted page IDs for role in order."""
    if not role:
        return []
    return list(ROLE_PAGES.get(role, []))


def get_default_page_for_role(role: str | None) -> str | None:
    """Return the first landing page for the role."""
    pages = get_pages_for_role(role)
    return pages[0] if pages else None
