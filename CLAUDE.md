# Photo ID Compliance Studio

Passport/ID photo checker: uploads a photo, validates it against a country's
official ID-photo rules (face position, background, lighting, blur), and
returns a compliant auto-cropped output. Deployed at studio.kooexperience.com.

## Tech Stack

- Backend: FastAPI + Uvicorn (single `app/main.py`, no DB)
- CV/ML: MediaPipe FaceMesh (landmarks + fallback segmentation), `rembg`
  (primary background segmentation), OpenCV, Pillow/`pillow-heif` (HEIC)
- Frontend: single static `web/index.html` (vanilla JS), served by FastAPI
  `StaticFiles` + a `/` route — no build step
- Config: `app/config/countries.yaml` (per-country crop/quality rules,
  loaded via Pydantic `CountrySettings`)
- Deploy: Docker (`Dockerfile`) on Railway (`railway.json`, healthcheck
  `/api/health`); `Procfile` for `python run.py`

## Commands

```bash
# setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run (dev, auto-reload)
uvicorn app.main:app --reload --port 8020
# then open http://127.0.0.1:8020  (docs at /docs)

# run (prod-style, reads $PORT/$HOST)
python run.py
```

No test suite exists yet — `tests/` is present but empty, and there is no
pytest/pyproject config in the repo. Do not invent a `pytest` command.

## Architecture

- `app/main.py` — FastAPI app: CORS, security headers, IP rate limiting
  (token bucket + daily cap), in-flight request guard, admin auth
  (`X-API-Key`, brute-force lockout), telemetry (`_AnalyzeTelemetry`), and
  routes. Mounts `web/` as static and serves `web/index.html` at `/`.
- `app/services/pipeline.py` — `PhotoCompliancePipeline.analyze()`: the core
  flow — decode/EXIF-normalize, MediaPipe face landmark extraction, person
  segmentation (rembg primary, MediaPipe fallback, with idle-unload for
  rembg), crop-box solving + eye-line/vertical rebalancing, assist-mode
  background whitening, optional beautify color correction, then builds the
  `AnalysisReport`.
- `app/services/image_ops.py` — stateless CV helpers: image decode, mask
  refinement (GrabCut/guided filter), blur/lighting/text-overlay scoring,
  face color correction, edge-artifact suppression, JPEG/base64 encoding.
- `app/config.py` / `app/config/countries.yaml` — per-country profiles
  (SG, US, UK, CA, AU, IN): output dimensions, face/eye size thresholds,
  pose/lighting/background tolerances.
- `app/schemas.py` — Pydantic models: `AnalysisReport`, `CheckResult`,
  `CropBox`, `AnalyzeResponse`, `CountryInfo`.

## Key Endpoints

- `POST /api/analyze` — multipart `photo` + `country_code`/`mode`/
  `beauty_mode`/`mirror_mode` → `AnalyzeResponse` (report + image previews)
- `GET /api/countries`, `GET /api/health`, `GET /api/privacy`, `GET /api/status`
- `GET /api/admin/*` (status/usage/pipeline/report/logs) — require
  `PHOTO_API_ADMIN_API_KEY` via `X-API-Key`; used by koo-control-center

## Notes

- All processing is in-memory per request; nothing is persisted to disk/DB.
- Env vars configure rate limits, upload size caps, CORS origins, rembg
  behavior, and admin auth — see `README.md` for the full list.
