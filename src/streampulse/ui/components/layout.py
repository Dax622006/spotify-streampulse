# spotify-streampulse/src/streampulse/ui/components/layout.py
"""
Consistent layout headers and badges per T18.01.
"""
from __future__ import annotations
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOGO_FILE = ASSETS_DIR / "spotify_logo.png"
BACKGROUND_FILE = ASSETS_DIR / "background.jpg"


@lru_cache(maxsize=None)
def _data_uri(path: str, mime: str) -> str:
    """Base64 data-URI for a local asset ('' if the file is missing)."""
    f = Path(path)
    if not f.exists():
        return ""
    return f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode("ascii")


def logo_html(height_px: int = 28) -> str:
    """Inline <img> of the Spotify logo (falls back to nothing if the file is missing)."""
    uri = _data_uri(str(LOGO_FILE), "image/png")
    if not uri:
        return ""
    return (
        f'<img src="{uri}" alt="Spotify" '
        f'style="height:{height_px}px;width:{height_px}px;vertical-align:middle;margin-right:10px;">'
    )


def page_icon():
    """Browser-tab icon: the Spotify logo (PIL image) or an emoji fallback."""
    try:
        from PIL import Image
        return Image.open(LOGO_FILE)
    except Exception:
        return "🎵"


def apply_theme(panel: bool = True) -> None:
    """Apply the green gradient background to the whole app.

    panel=True  -> content sits on a translucent dark panel (readable charts/tables).
    panel=False -> no big panel (used by the login page, which styles its own card).
    """
    bg = _data_uri(str(BACKGROUND_FILE), "image/jpeg")
    bg_rule = f'url("{bg}") center / cover no-repeat' if bg else "#0b3d2e"
    panel_css = """
    [data-testid="stMainBlockContainer"], .main .block-container {
        background: rgba(7, 24, 18, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding: 2rem 2.2rem 2.5rem 2.2rem;
    }
    """ if panel else """
    [data-testid="stColumn"]:nth-child(2), [data-testid="column"]:nth-child(2) {
        background: rgba(7, 24, 18, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 16px;
        padding: 0.5rem 1.6rem 1.6rem 1.6rem;
    }
    """
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {bg_rule}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: transparent; }}
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(5, 18, 13, 0.66);
        }}
        {panel_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, concept: str, icon: str = "🎵") -> None:
    """Standardized page header used by all 14 pages."""
    col1, col2 = st.columns([4, 1])
    with col1:
        if icon == "🎵":
            st.markdown(f"<h2>{logo_html(34)}{title}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"## {icon} {title}")
        st.markdown(f"<p style='color: #94a3b8; font-size: 14px; margin-top: -10px;'>{subtitle}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 5px;">
                <span style="background: rgba(29, 185, 84, 0.15); color: #1DB954; border: 1px solid #1DB954; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 12px;">
                    {concept}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()
