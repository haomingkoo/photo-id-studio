from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


CheckStatus = Literal["pass", "fail", "warn", "manual"]
OverallStatus = Literal["pass", "review", "fail"]


class CheckResult(BaseModel):
    code: str
    status: CheckStatus
    message: str
    action: str
    score: float | None = None
    details: dict[str, Any] | None = None


class CropBox(BaseModel):
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int


class AnalysisReport(BaseModel):
    country_code: str
    country_name: str
    mode: str
    segmentation_backend: str | None = None
    width: int
    height: int
    file_size_bytes: int
    captured_at: datetime | None = None
    overall_status: OverallStatus
    checks: list[CheckResult] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    crop_box: CropBox | None = None


class AnalyzeResponse(BaseModel):
    report: AnalysisReport
    original_image_base64: str | None = None
    original_image_mime: str | None = None
    processed_image_base64: str | None = None
    processed_image_mime: str | None = None
    comparison_no_correction_base64: str | None = None
    comparison_color_correction_base64: str | None = None
    comparison_image_mime: str | None = None


class CountryInfo(BaseModel):
    code: str
    name: str
    output_width_px: int
    output_height_px: int
    max_file_size_mb: int
    allowed_extensions: list[str]
