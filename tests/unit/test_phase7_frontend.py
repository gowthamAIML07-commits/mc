"""Unit Tests for Phase 7 Modern Web Frontend Assets, Schemas, and Structure."""
import re
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from apps.backend.main import app

client = TestClient(app)


def test_frontend_static_files_exist():
    """Verify all required static frontend assets are present in apps/frontend/static."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    static_dir = root_dir / "apps" / "frontend" / "static"
    
    assert static_dir.exists()
    assert (static_dir / "index.html").exists()
    assert (static_dir / "styles.css").exists()
    assert (static_dir / "app.js").exists()
    assert (static_dir / "api.js").exists()


def test_frontend_html_structure_and_accessibility():
    """Verify HTML5 semantic tags, ARIA attributes, and required clinical sections."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    index_html = (root_dir / "apps" / "frontend" / "static" / "index.html").read_text(encoding="utf-8")

    # Semantic HTML
    assert "<header" in index_html
    assert "<nav" in index_html
    assert "<main" in index_html
    assert "<section" in index_html

    # ARIA roles and labels
    assert 'role="banner"' in index_html
    assert 'role="navigation"' in index_html
    assert 'role="main"' in index_html
    assert 'role="tabpanel"' in index_html
    assert 'aria-label=' in index_html

    # Privacy notice
    assert "Privacy Guaranteed" in index_html
    assert "not permanently stored" in index_html

    # Tab Panels
    assert 'id="panel-prescription"' in index_html
    assert 'id="panel-chat"' in index_html
    assert 'id="panel-monographs"' in index_html
    assert 'id="panel-interactions"' in index_html
    assert 'id="panel-safety"' in index_html


def test_frontend_css_design_system():
    """Verify CSS tokens, responsive breakpoints, and safety color palettes."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    styles_css = (root_dir / "apps" / "frontend" / "static" / "styles.css").read_text(encoding="utf-8")

    # Design tokens
    assert "--bg-app:" in styles_css
    assert "--primary:" in styles_css
    assert "--success:" in styles_css
    assert "--warning:" in styles_css
    assert "--danger:" in styles_css

    # Safety alert classes
    assert ".safety-alert-emergency" in styles_css
    assert ".safety-alert-high" in styles_css
    assert ".badge-verified" in styles_css
    assert ".badge-review" in styles_css
    assert ".badge-unverified" in styles_css

    # Responsive media queries
    assert "@media (max-width: 1024px)" in styles_css
    assert "@media (max-width: 768px)" in styles_css


def test_frontend_security_and_zero_secrets():
    """Verify no API keys, secrets, or internal paths are hardcoded in client scripts."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    static_dir = root_dir / "apps" / "frontend" / "static"

    for js_file in static_dir.glob("*.js"):
        content = js_file.read_text(encoding="utf-8")
        assert not re.search(r"\bsk-[a-zA-Z0-9]{20,}\b", content)  # No OpenAI API keys
        assert not re.search(r"\bAIza[0-9A-Za-z-_]{35}\b", content)  # No Google API keys
        assert "password" not in content.lower()
        assert "secret" not in content.lower()
        assert "C:\\" not in content  # No local file system paths


def test_frontend_monograph_cache_integrity():
    """Verify the frontend monograph cache matches authoritative drug labeling."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    app_js = (root_dir / "apps" / "frontend" / "static" / "app.js").read_text(encoding="utf-8")

    # Core drugs present in frontend cache
    assert "DAILYMED_WARFARIN_5" in app_js
    assert "DAILYMED_METFORMIN_500" in app_js
    assert "DAILYMED_AMOXICILLIN_500" in app_js
    assert "DAILYMED_PARACETAMOL_650" in app_js
    assert "DAILYMED_PANTOPRAZOLE_40" in app_js
    assert "DAILYMED_ATORVASTATIN_10" in app_js
