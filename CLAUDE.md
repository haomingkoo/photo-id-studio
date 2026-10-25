# CLAUDE.md — photo-id-studio

FastAPI app that checks and crops passport/ID photos. Live at studio.kooexperience.com.

## Run and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload --port 8020   # dev
python run.py                               # prod-style, reads HOST/PORT
python -m pytest -q tests                   # what CI (.github/workflows/tests.yml) runs
```

- CI needs `libgl1` and `libglib2.0-0` for OpenCV/MediaPipe; install them locally if imports fail.
- `tests/conftest.py` sets `PHOTO_API_ENABLE_REMBG=0` so tests skip the ~170MB rembg model. Keep it that way.

## Deploy

- Railway project `artistic-vibrancy` auto-deploys every push to `main` (no watch paths, so docs-only merges redeploy too).
- Built from `Dockerfile` via `railway.json`; healthcheck `/api/health`.
- The rembg `u2net_human_seg` model is baked into the image. Don't add runtime model downloads.
- `run.py` pins uvicorn to `workers=1`. rembg is memory-heavy and unloads after idle (`PHOTO_API_REMBG_IDLE_UNLOAD_SEC`), so don't raise workers casually.

## Rules

- Photos stay in memory. Never write uploads to disk, temp files or logs.
- Logs must not include client IPs or filenames. Uvicorn access logging stays off.
- `POST /api/analyze` requires the `consent` form field server-side (400 without it). Don't drop that check.
- `/api/admin/*` needs `PHOTO_API_ADMIN_API_KEY` in the `X-API-Key` header. These endpoints serve the control-center contract (`PHOTO_API_CONTRACT`).
- Per-country crop and quality rules live in `app/config/countries.yaml`. Change rules there, not in code.
- Dependencies are pinned to the exact versions running in production. Bump them on purpose, never as a side effect.
- Release commits bump `VERSION` and add an entry to `CHANGELOG.md`.
