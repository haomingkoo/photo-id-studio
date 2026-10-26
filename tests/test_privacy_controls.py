from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import logging
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import starlette.formparsers as formparsers

from app import main

FILENAME = "john_tan_passport.jpg"
CLIENT_IP = "testclient"  # TestClient's request.client.host


def _jpeg(width: int = 800, height: int = 1000) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 200, 200)).save(buf, "JPEG")
    return buf.getvalue()


def _large_png() -> bytes:
    """Random noise PNG, well over Starlette's default 1MB spool threshold."""
    rng = np.random.default_rng(0)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (1000, 1000, 3), dtype=np.uint8)).save(buf, "PNG")
    data = buf.getvalue()
    assert len(data) > 2 * 1024 * 1024
    return data


def _post(client: TestClient, data: bytes, *, consent: str | None = "true", filename: str = FILENAME):
    form = {"country_code": "SG"}
    if consent is not None:
        form["consent"] = consent
    return client.post("/api/analyze", files={"photo": (filename, data, "image/jpeg")}, data=form)


@pytest.fixture(autouse=True)
def fresh_rate_limiter(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMITER", main._IpRateLimiter(rate_per_min=60, burst=50, daily_limit=500))


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture
def spooled_files(monkeypatch) -> list[tempfile.SpooledTemporaryFile]:
    created: list[tempfile.SpooledTemporaryFile] = []

    class Recording(tempfile.SpooledTemporaryFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", Recording)
    return created


# Consent

def test_analyze_without_consent_returns_400(client):
    response = _post(client, _jpeg(), consent=None)

    assert response.status_code == 400
    assert "Consent is required" in response.json()["detail"]


@pytest.mark.parametrize("value", ["", "false", "no", "0"])
def test_analyze_with_falsy_consent_returns_400(client, value):
    response = _post(client, _jpeg(), consent=value)

    assert response.status_code == 400


def test_analyze_with_consent_returns_report(client):
    response = _post(client, _jpeg())

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["country_code"] == "SG"
    assert body["original_image_base64"]


# MIME

@pytest.mark.parametrize(
    "payload",
    [b"GIF89a" + b"\x00" * 64, b"%PDF-1.7\n" + b"\x00" * 64, b"<svg xmlns='http://www.w3.org/2000/svg'/>"],
)
def test_analyze_rejects_disallowed_type_even_with_image_content_type(client, payload):
    response = _post(client, payload)

    assert response.status_code == 415


def test_analyze_accepts_png_sent_as_octet_stream(client):
    buf = io.BytesIO()
    Image.new("RGB", (800, 1000), (200, 200, 200)).save(buf, "PNG")

    response = client.post(
        "/api/analyze",
        files={"photo": ("photo.png", buf.getvalue(), "application/octet-stream")},
        data={"consent": "true"},
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\xff\xd8\xff\xe0rest", "jpeg"),
        (b"\x89PNG\r\n\x1a\nrest", "png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "webp"),
        (b"\x00\x00\x00\x18ftypheic\x00\x00", "heif"),
        (b"\x00\x00\x00\x18ftypmif1\x00\x00", "heif"),
        (b"\x00\x00\x00\x18ftypavif\x00\x00", None),
        (b"BM\x00\x00", None),
        (b"", None),
    ],
)
def test_sniff_image_type(data, expected):
    assert main._sniff_image_type(data) == expected


# Temp files

def test_large_upload_never_spools_to_disk(client, spooled_files):
    response = _post(client, _large_png())

    assert response.status_code == 200
    assert spooled_files
    assert all(not f._rolled for f in spooled_files)
    assert all(f.closed for f in spooled_files)


def test_large_upload_never_spools_to_disk_on_decode_error(client, spooled_files):
    corrupt_jpeg = b"\xff\xd8\xff" + b"\x00" * (3 * 1024 * 1024)

    response = _post(client, corrupt_jpeg)

    assert response.status_code == 400
    assert all(not f._rolled for f in spooled_files)
    assert all(f.closed for f in spooled_files)


def test_large_upload_never_spools_to_disk_on_internal_error(client, spooled_files, monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("pipeline crashed")

    monkeypatch.setattr(main.pipeline, "analyze", boom)

    response = _post(client, _large_png())

    assert response.status_code == 500
    assert all(not f._rolled for f in spooled_files)
    assert all(f.closed for f in spooled_files)


def test_large_upload_without_consent_never_spools_to_disk(client, spooled_files):
    response = _post(client, _large_png(), consent=None)

    assert response.status_code == 400
    assert all(not f._rolled for f in spooled_files)
    assert all(f.closed for f in spooled_files)


def test_request_over_limit_rejected_before_parsing(client, spooled_files, monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 1024)

    response = _post(client, _jpeg())

    assert response.status_code == 413
    assert spooled_files == []


# Logs

def _assert_no_identifiers(caplog) -> None:
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert FILENAME not in text
    assert CLIENT_IP not in text


def test_internal_error_log_has_no_filename_or_ip(client, caplog, monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("pipeline crashed")

    monkeypatch.setattr(main.pipeline, "analyze", boom)
    caplog.set_level(logging.DEBUG)

    response = _post(client, _jpeg())

    assert response.status_code == 500
    assert any("analyze_failed" in r.getMessage() for r in caplog.records)
    _assert_no_identifiers(caplog)


def test_metrics_log_has_no_filename_or_ip(client, caplog, monkeypatch):
    monkeypatch.setattr(main, "LOG_ANALYZE_METRICS", True)
    caplog.set_level(logging.DEBUG)

    response = _post(client, _jpeg())

    assert response.status_code == 200
    assert any("analyze_metrics" in r.getMessage() for r in caplog.records)
    _assert_no_identifiers(caplog)


def test_rate_limited_log_has_no_ip(client, caplog, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMITER", main._IpRateLimiter(rate_per_min=1, burst=1, daily_limit=500))
    caplog.set_level(logging.DEBUG)

    _post(client, _jpeg())
    response = _post(client, _jpeg())

    assert response.status_code == 429
    assert any("analyze_rejected" in r.getMessage() for r in caplog.records)
    _assert_no_identifiers(caplog)


def test_uvicorn_access_log_disabled(monkeypatch):
    import run

    captured = {}
    monkeypatch.setattr(run.uvicorn, "run", lambda *a, **kw: captured.update(kw))

    run.main()

    assert captured["access_log"] is False


# Rate limiter retention

def test_rate_limiter_drops_ips_from_previous_days():
    limiter = main._IpRateLimiter(rate_per_min=10, burst=10, daily_limit=10)
    limiter.check("198.51.100.1")
    limiter._states["198.51.100.1"].day_ordinal -= 1
    limiter._pruned_day_ordinal -= 1

    limiter.check("198.51.100.2")

    assert set(limiter._states) == {"198.51.100.2"}


# Status, health and legal pages

def test_public_status_hides_request_details(client):
    _post(client, _jpeg())

    body = client.get("/api/status").json()

    assert body["latest_run"] is None
    assert set(body["errors"]) == {"failed_runs_7d"}
    assert "elapsed_ms" not in str(body)
    assert "runtime" not in body["pipeline"]


def test_admin_status_requires_auth(client, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_API_KEY", "test-key")

    assert client.get("/api/admin/status").status_code == 401
    authed = client.get("/api/admin/status", headers={"X-API-Key": "test-key"})
    assert authed.status_code == 200
    assert "latest_run" in authed.json()


def test_health_is_public(client):
    assert client.get("/api/health").json()["ok"] is True


@pytest.mark.parametrize("path", ["/privacy", "/terms"])
def test_legal_pages_served(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert "privacy@kooexperience.com" in response.text
    assert "LAWYER REVIEW" not in response.text and "owner to confirm" not in response.text
    assert "privacy@kooexperience.com" in response.text


def test_index_has_required_consent_and_legal_links(client):
    html = client.get("/").text

    assert 'id="consent"' in html and "required" in html.split('id="consent"')[1].split(">")[0]
    assert 'href="/privacy"' in html
    assert 'href="/terms"' in html


def test_privacy_endpoint_copy_matches_behaviour(client):
    body = client.get("/api/privacy").json()

    assert "never written to disk" in body["retention_policy"]
    assert body["privacy_url"] == "/privacy"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_hidden_by_default(client, path):
    assert client.get(path).status_code == 404


def test_api_docs_served_when_enabled():
    code = "from app import main; assert main.app.openapi_url == '/openapi.json' and main.app.docs_url == '/docs'"
    env = {**os.environ, "PHOTO_API_ENABLE_DOCS": "1"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_csp_allows_index_inline_script(client):
    response = client.get("/")
    csp = response.headers["content-security-policy"]
    scripts = re.findall(r"<script>(.*?)</script>", response.text, flags=re.S)

    assert scripts
    for body in scripts:
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
    for directive in ("frame-ancestors 'none'", "object-src 'none'", "base-uri 'self'", "form-action 'self'", "media-src 'self'"):
        assert directive in csp
    assert "https://static.cloudflareinsights.com" in csp
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
