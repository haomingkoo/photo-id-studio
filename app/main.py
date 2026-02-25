from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_country_settings
from app.schemas import AnalyzeResponse, CountryInfo
from app.services.pipeline import PhotoCompliancePipeline

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Photo ID Compliance Studio", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PhotoCompliancePipeline()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    photo: UploadFile = File(...),
    country_code: str = Form("SG"),
    mode: str = Form("assist"),
    beauty_mode: str = Form("none"),
    segmentation_backend: str = Form("mediapipe"),
    shadow_mode: str = Form("balanced"),
) -> AnalyzeResponse:
    if photo.content_type is None or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    file_bytes = await photo.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        report, processed_b64, original_b64, comparison_no_corr_b64, comparison_color_corr_b64 = pipeline.analyze(
            file_bytes=file_bytes,
            filename=photo.filename or "upload.jpg",
            country_code=country_code,
            mode=mode,
            beauty_mode=beauty_mode,
            segmentation_backend=segmentation_backend,
            shadow_mode=shadow_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        return JSONResponse(status_code=500, content={"error": "analysis_failed", "detail": str(exc)})

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
