from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import math
import os
import resource
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import load_country_settings
from app.schemas import AnalyzeResponse, CountryInfo
from app.services.pipeline import PhotoCompliancePipeline

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
LOG = logging.getLogger("photo_id_studio.api")


def _split_csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _flag_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_proc_status_kb(key: str) -> int | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{key}:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _read_cgroup_bytes(paths: list[str]) -> int | None:
    for candidate in paths:
        p = Path(candidate)
        if not p.exists():
            continue
        try:
            raw = p.read_text(encoding="utf-8").strip()
            if not raw or raw == "max":
                return None
            value = int(raw)
            if value <= 0:
                return None
            return value
        except Exception:
            continue
    return None


def _maxrss_bytes() -> int | None:
    try:
        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if usage <= 0:
        return None
    if sys.platform == "darwin":
        return usage
    return usage * 1024


def _bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024.0 * 1024.0), 2)


def _memory_snapshot() -> dict[str, float | None]:
    vmrss_kb = _parse_proc_status_kb("VmRSS")
    vmhwm_kb = _parse_proc_status_kb("VmHWM")
    cgroup_usage = _read_cgroup_bytes(
        ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"]
    )
    cgroup_limit = _read_cgroup_bytes(
        ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
    )
    return {
        "rss_mb": _bytes_to_mb(vmrss_kb * 1024 if vmrss_kb is not None else None),
        "hwm_mb": _bytes_to_mb(vmhwm_kb * 1024 if vmhwm_kb is not None else None),
        "maxrss_mb": _bytes_to_mb(_maxrss_bytes()),
        "cgroup_usage_mb": _bytes_to_mb(cgroup_usage),
        "cgroup_limit_mb": _bytes_to_mb(cgroup_limit),
    }


ALLOWED_ORIGINS = _split_csv_env("PHOTO_API_ALLOWED_ORIGINS")
TRUSTED_PROXY_IPS = set(_split_csv_env("PHOTO_API_TRUSTED_PROXY_IPS"))
LOG_ANALYZE_METRICS = _flag_env("PHOTO_API_LOG_ANALYZE_METRICS", False)

app = FastAPI(title="Photo ID Compliance Studio", version=__version__)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

pipeline = PhotoCompliancePipeline()


@dataclass
class _BucketState:
    tokens: float
    last_refill_mono: float
    day_ordinal: int
    daily_count: int = 0


@dataclass
class _RateDecision:
    allowed: bool
    reason: str | None
    retry_after: int | None
    minute_remaining: int
    daily_remaining: int


class _IpRateLimiter:
    def __init__(self, rate_per_min: int, burst: int, daily_limit: int):
        self._rate_per_min = max(1, rate_per_min)
        self._burst = max(1, burst)
        self._daily_limit = max(1, daily_limit)
        self._tokens_per_second = self._rate_per_min / 60.0
        self._states: dict[str, _BucketState] = {}
        self._lock = threading.Lock()

    @property
    def rate_per_min(self) -> int:
        return self._rate_per_min

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    @staticmethod
    def _seconds_until_utc_midnight(now_utc: dt.datetime) -> int:
        tomorrow = (now_utc + dt.timedelta(days=1)).date()
        midnight = dt.datetime.combine(tomorrow, dt.time.min, tzinfo=dt.timezone.utc)
        return max(1, int((midnight - now_utc).total_seconds()))

    def check(self, ip: str) -> _RateDecision:
        now_mono = time.monotonic()
        now_utc = dt.datetime.now(dt.timezone.utc)
        day_ordinal = now_utc.date().toordinal()

        with self._lock:
            state = self._states.get(ip)
            if state is None:
                state = _BucketState(
                    tokens=float(self._burst),
                    last_refill_mono=now_mono,
                    day_ordinal=day_ordinal,
                )
                self._states[ip] = state

            elapsed = max(0.0, now_mono - state.last_refill_mono)
            if elapsed > 0:
                state.tokens = min(float(self._burst), state.tokens + elapsed * self._tokens_per_second)
                state.last_refill_mono = now_mono

            if state.day_ordinal != day_ordinal:
                state.day_ordinal = day_ordinal
                state.daily_count = 0

            if state.daily_count >= self._daily_limit:
                return _RateDecision(
                    allowed=False,
                    reason="daily_limit",
                    retry_after=self._seconds_until_utc_midnight(now_utc),
                    minute_remaining=max(0, int(math.floor(state.tokens))),
                    daily_remaining=0,
                )

            if state.tokens < 1.0:
                needed = 1.0 - state.tokens
                retry_after = max(1, int(math.ceil(needed / self._tokens_per_second)))
                return _RateDecision(
                    allowed=False,
                    reason="rate_limit",
                    retry_after=retry_after,
                    minute_remaining=0,
                    daily_remaining=max(0, self._daily_limit - state.daily_count),
                )

            state.tokens -= 1.0
            state.daily_count += 1
            return _RateDecision(
                allowed=True,
                reason=None,
                retry_after=None,
                minute_remaining=max(0, int(math.floor(state.tokens))),
                daily_remaining=max(0, self._daily_limit - state.daily_count),
            )


RATE_PER_MIN = max(1, int(os.getenv("PHOTO_API_RATE_PER_MIN", "10")))
RATE_BURST = max(1, int(os.getenv("PHOTO_API_BURST", "20")))
DAILY_LIMIT = max(1, int(os.getenv("PHOTO_API_DAILY_LIMIT", "200")))
MAX_INFLIGHT_ANALYZE = max(1, int(os.getenv("PHOTO_API_MAX_INFLIGHT", "3")))
MAX_UPLOAD_MB = max(1.0, float(os.getenv("PHOTO_API_MAX_UPLOAD_MB", "20")))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
UPLOAD_READ_CHUNK_BYTES = max(64 * 1024, int(os.getenv("PHOTO_API_UPLOAD_CHUNK_BYTES", str(1024 * 1024))))
TRUST_X_FORWARDED_FOR = os.getenv("PHOTO_API_TRUST_XFF", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

RATE_LIMITER = _IpRateLimiter(rate_per_min=RATE_PER_MIN, burst=RATE_BURST, daily_limit=DAILY_LIMIT)
INFLIGHT_ANALYZE_GUARD = threading.BoundedSemaphore(MAX_INFLIGHT_ANALYZE)


def _resolve_client_ip(request: Request) -> str:
    if TRUST_X_FORWARDED_FOR and _request_came_through_trusted_proxy(request):
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            forwarded_ip = _parse_forwarded_ip(xff)
            if forwarded_ip:
                return forwarded_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _request_came_through_trusted_proxy(request: Request) -> bool:
    if request.client is None or not request.client.host:
        return False
    if not TRUSTED_PROXY_IPS:
        return True
    return request.client.host in TRUSTED_PROXY_IPS


def _parse_forwarded_ip(xff_header: str) -> str | None:
    first = xff_header.split(",")[0].strip()
    if not first:
        return None
    try:
        return str(ipaddress.ip_address(first))
    except ValueError:
        return None


def _set_limit_headers(response: Response | JSONResponse, decision: _RateDecision) -> None:
    response.headers["X-RateLimit-Limit-Minute"] = str(RATE_LIMITER.rate_per_min)
    response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, decision.minute_remaining))
    response.headers["X-RateLimit-Limit-Day"] = str(RATE_LIMITER.daily_limit)
    response.headers["X-RateLimit-Remaining-Day"] = str(max(0, decision.daily_remaining))
    response.headers["X-InFlight-Limit"] = str(MAX_INFLIGHT_ANALYZE)


def _content_length_exceeds_limit(request: Request, max_bytes: int) -> bool:
    header = request.headers.get("content-length")
    if not header:
        return False
    try:
        return int(header) > max_bytes
    except ValueError:
        return False


async def _read_upload_with_limit(photo: UploadFile, max_bytes: int, chunk_size: int) -> bytes:
    chunks = bytearray()
    total_bytes = 0
    while True:
        chunk = await photo.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file is too large. Hard limit is {MAX_UPLOAD_MB:.0f}MB.",
            )
        chunks.extend(chunk)
    return bytes(chunks)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "analyze_limits": {
            "rate_per_min": RATE_PER_MIN,
            "burst": RATE_BURST,
            "daily_limit": DAILY_LIMIT,
            "max_inflight": MAX_INFLIGHT_ANALYZE,
            "trust_x_forwarded_for": TRUST_X_FORWARDED_FOR,
            "log_analyze_metrics": LOG_ANALYZE_METRICS,
        },
    }


@app.get("/api/countries", response_model=list[CountryInfo])
def countries() -> list[CountryInfo]:
    settings = load_country_settings()
    payload: list[CountryInfo] = []
    for code, profile in settings.countries.items():
        payload.append(
            CountryInfo(
                code=code,
                name=profile.name,
                output_width_px=profile.output_width_px,
                output_height_px=profile.output_height_px,
                max_file_size_mb=profile.max_file_size_mb,
                allowed_extensions=profile.allowed_extensions,
            )
        )
    return payload


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    response: Response,
    photo: UploadFile = File(...),
    country_code: str = Form("SG"),
    mode: str = Form("assist"),
    beauty_mode: str = Form("none"),
) -> AnalyzeResponse:
    client_ip = _resolve_client_ip(request)
    decision = RATE_LIMITER.check(client_ip)
    if not decision.allowed:
        retry_after = str(decision.retry_after or 1)
        payload = JSONResponse(
            status_code=429,
            content={
                "detail": "Too many analyze requests. Please retry later.",
                "reason": decision.reason,
            },
            headers={"Retry-After": retry_after},
        )
        _set_limit_headers(payload, decision)
        LOG.warning(
            "analyze_rejected ip=%s reason=%s retry_after=%s",
            client_ip,
            decision.reason,
            retry_after,
        )
        return payload

    if not INFLIGHT_ANALYZE_GUARD.acquire(blocking=False):
        payload = JSONResponse(
            status_code=429,
            content={"detail": "Server is busy processing other analyze requests. Please retry shortly."},
            headers={"Retry-After": "5"},
        )
        _set_limit_headers(payload, decision)
        LOG.warning("analyze_rejected ip=%s reason=max_inflight", client_ip)
        return payload

    _set_limit_headers(response, decision)
    started_mono = time.perf_counter()
    mem_before = _memory_snapshot() if LOG_ANALYZE_METRICS else {}
    upload_bytes = 0
    report = None
    processed_b64 = None
    original_b64 = None
    comparison_no_corr_b64 = None
    comparison_color_corr_b64 = None
    outcome = "ok"
    status_code = 200
    error_kind = None

    try:
        if _content_length_exceeds_limit(request, MAX_UPLOAD_BYTES):
            raise HTTPException(
                status_code=413,
                detail=f"Request body is too large. Hard limit is {MAX_UPLOAD_MB:.0f}MB.",
            )

        if photo.content_type is None or not photo.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload a valid image file.")

        file_bytes = await _read_upload_with_limit(
            photo,
            max_bytes=MAX_UPLOAD_BYTES,
            chunk_size=UPLOAD_READ_CHUNK_BYTES,
        )
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        upload_bytes = len(file_bytes)

        report, processed_b64, original_b64, comparison_no_corr_b64, comparison_color_corr_b64 = pipeline.analyze(
            file_bytes=file_bytes,
            filename=photo.filename or "upload.jpg",
            country_code=country_code,
            mode=mode,
            beauty_mode=beauty_mode,
        )
    except HTTPException as exc:
        outcome = "http_error"
        status_code = exc.status_code
        error_kind = "HTTPException"
        raise
    except ValueError as exc:
        outcome = "value_error"
        status_code = 400
        error_kind = exc.__class__.__name__
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        outcome = "internal_error"
        status_code = 500
        error_kind = exc.__class__.__name__
        LOG.exception("analyze_failed ip=%s filename=%s", client_ip, photo.filename or "upload.jpg")
        return JSONResponse(status_code=500, content={"error": "analysis_failed", "detail": "Internal error"})
    finally:
        if LOG_ANALYZE_METRICS:
            mem_after = _memory_snapshot()
            elapsed_ms = int((time.perf_counter() - started_mono) * 1000)
            rss_before = mem_before.get("rss_mb")
            rss_after = mem_after.get("rss_mb")
            rss_delta = None
            if isinstance(rss_before, float) and isinstance(rss_after, float):
                rss_delta = round(rss_after - rss_before, 2)
            LOG.info(
                "analyze_metrics ip=%s status=%s outcome=%s error_kind=%s country=%s mode=%s beauty_mode=%s "
                "upload_bytes=%s original_b64_bytes=%s processed_b64_bytes=%s cmp_no_corr_b64_bytes=%s "
                "cmp_color_corr_b64_bytes=%s overall_status=%s segmentation_backend=%s elapsed_ms=%s "
                "rss_before_mb=%s rss_after_mb=%s rss_delta_mb=%s hwm_after_mb=%s maxrss_after_mb=%s "
                "cgroup_usage_mb=%s cgroup_limit_mb=%s",
                client_ip,
                status_code,
                outcome,
                error_kind,
                country_code,
                mode,
                beauty_mode,
                upload_bytes,
                len(original_b64 or ""),
                len(processed_b64 or ""),
                len(comparison_no_corr_b64 or ""),
                len(comparison_color_corr_b64 or ""),
                getattr(report, "overall_status", None),
                getattr(report, "segmentation_backend", None),
                elapsed_ms,
                rss_before,
                rss_after,
                rss_delta,
                mem_after.get("hwm_mb"),
                mem_after.get("maxrss_mb"),
                mem_after.get("cgroup_usage_mb"),
                mem_after.get("cgroup_limit_mb"),
            )
        await photo.close()
        INFLIGHT_ANALYZE_GUARD.release()

    return AnalyzeResponse(
        report=report,
        original_image_base64=original_b64,
        original_image_mime="image/jpeg",
        processed_image_base64=processed_b64,
        processed_image_mime="image/jpeg",
        comparison_no_correction_base64=comparison_no_corr_b64,
        comparison_color_correction_base64=comparison_color_corr_b64,
        comparison_image_mime="image/jpeg",
    )


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)
