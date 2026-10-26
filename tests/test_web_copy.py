from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main

ROOT = Path(__file__).resolve().parents[1]
FOOTER = "&copy; 2026 Haoming Koo"


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def test_every_pipeline_check_has_plain_language_copy():
    """Users never see raw codes: each code the pipeline can emit needs an entry in the page's CHECKS map."""
    codes = set(re.findall(r'code="([A-Z_]+)"', (ROOT / "app/services/pipeline.py").read_text()))
    html = (ROOT / "web/index.html").read_text()
    checks_block = html.split("const CHECKS = {", 1)[1].split("};", 1)[0]
    mapped = set(re.findall(r"^\s*([A-Z_]+):", checks_block, flags=re.M))

    assert codes, "pipeline codes not found"
    assert codes - mapped == set()


def test_check_codes_are_not_rendered_as_text():
    html = (ROOT / "web/index.html").read_text()
    script = html.split("<script>", 1)[1]
    # Codes may only be attached as a data attribute, never as visible text.
    assert "textContent = check.code" not in script and "textContent = it.code" not in script
    assert "Error Code" not in html


@pytest.mark.parametrize("path", ["/", "/privacy", "/terms"])
def test_pages_share_chrome(client, path):
    html = client.get(path).text

    assert '/static/site.css' in html
    assert '/static/favicon.svg' in html
    assert FOOTER in html and "privacy@kooexperience.com" in html and "kooexperience.com/" in html
    assert "—" not in html  # no em dashes in copy


def test_unknown_page_gets_styled_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert 'href="/"' in response.text and "https://kooexperience.com/" in response.text
    assert FOOTER in response.text


def test_unknown_api_path_stays_json(client):
    response = client.get("/api/no-such-endpoint")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"


@pytest.mark.parametrize("path", ["/static/site.css", "/static/favicon.svg", "/static/fonts/fonts.css", "/static/fonts/plex-sans-latin-400.woff2", "/static/media/studio-poster.jpg"])
def test_static_assets_served(client, path):
    assert client.get(path).status_code == 200


def test_tour_video_supports_range_requests(client):
    response = client.get("/static/media/studio.mp4", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) == 100
