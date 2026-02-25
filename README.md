# Photo ID Compliance Studio

Photo upload app with:

1. Face/landmark detection (MediaPipe FaceMesh)
2. Person segmentation for background checks (MediaPipe Selfie Segmentation)
3. Rule-based compliance engine with clear error codes and actions
4. Auto-crop to country profile dimensions (default: Singapore)
5. Processed output image (crop + straighten + optional assistive cleanup)

## Features

1. Upload bad/uncentered photos and get structured pass/fail checks.
2. Returns specific error codes like `NO_FACE_DETECTED`, `FACE_TOO_SMALL`, `CROP_OUT_OF_BOUNDS`.
3. Shows what to do next for each failed/warned check.
4. Output image is generated only when crop requirements can be satisfied.
5. UI style is aligned to the `kooexperience.com` dark theme.
6. Optional beautify mode supports color correction only (no geometry edits).

## Why MediaPipe

MediaPipe was selected as the default CV backend for this app because it gives the best implementation trade-off for a local-first product:

1. Fast landmark inference on CPU (important for laptop/server deployments without GPU).
2. Stable, dense facial landmarks suitable for rule checks (eye visibility, roll, pitch/yaw heuristics, mouth-closed checks).
3. Built-in person segmentation model for fast background checks and assistive cleanup mode.
4. No external API dependency for core detection logic (lower cost and better privacy posture).

## Segmentation Strategy

The app uses a single segmentation backend:

1. `MediaPipe`:
   - CPU-friendly and stable for server deployment.
   - Works with OpenCV refinement and edge guards for cleaner shoulders/hair boundaries.
   - No external ONNX model file path required.

### Why not GenAI replacement

1. Compliance workflows require identity fidelity and deterministic behavior.
2. Generative edits can hallucinate edges or alter identity cues.
3. This system keeps transformations non-generative and bounded to explainable CV operations.

### Alternatives considered

1. OpenCV Haar cascades + custom landmarks:
   - Pros: very lightweight, easy to deploy.
   - Cons: lower robustness for pose/lighting; not enough landmark fidelity for strict photo-rule checks.
2. dlib/face_recognition-style pipelines:
   - Pros: good landmarks on clean inputs.
   - Cons: slower setup/runtime on some systems; less integrated segmentation path.
3. Heavy DL stacks (RetinaFace/YOLO + segmentation nets):
   - Pros: highest ceiling on difficult edge cases.
   - Cons: larger models, higher latency/cost, more ops complexity than needed for this use case.

## Benchmarking

To answer "is this accurate and fast enough?", benchmark three dimensions:

1. Latency:
   - `analyze_ms_p50`, `analyze_ms_p95`
2. Reliability:
   - success rate (`PASS` + `REVIEW`)
   - failure distribution by error code (`NO_FACE_DETECTED`, `CROP_OUT_OF_BOUNDS`, etc.)
3. Quality consistency:
   - blur and lighting score distributions before/after assist mode

### Recommended test sets

1. "Good" set: photos that should pass.
2. "Common failure" set: bad lighting, off-center face, tilted head, blurry photos.
3. "Edge" set: glasses glare, strong backlight, low-resolution inputs.

Target at least 100 images total for a useful baseline.

### How to run a quick local benchmark

1. Start server:

```bash
python -m uvicorn app.main:app --reload --port 8020
```

2. Run batch requests against `/api/analyze` using your dataset (for example with a simple script or Postman collection runner).
3. Capture:
   - request latency per image
   - `overall_status`
   - all check codes and statuses

### Benchmark report template

Document this in your project wiki or release notes:

| Metric | Value | Notes |
|---|---:|---|
| Dataset size | - | Good + failure + edge mix |
| `analyze_ms_p50` | - | CPU only |
| `analyze_ms_p95` | - | CPU only |
| Pass/Review/Fail | - / - / - | Overall outcome distribution |
| Top fail code | - | Highest frequency fail reason |
| Top warn code | - | Highest frequency warning reason |

### Fair comparison guidance

When benchmarking MediaPipe vs another stack, keep these fixed:

1. Same input files and country profile.
2. Same hardware and CPU load conditions.
3. Same acceptance thresholds in rule checks.

## Deploy Optimization Profile

For low-cost deployment, use these environment variables:

1. `MAX_PROCESSING_LONG_SIDE=1920`
2. `MAX_PROCESSING_MEGAPIXELS=4.0`
3. `MAX_PREVIEW_LONG_SIDE=1400`

What this does:

1. Caps inference resolution to reduce CPU/RAM usage.
2. Keeps source metadata/resolution checks intact.
3. Keeps final country output dimensions unchanged.

## Deploy Online (Railway)

This repo is now deployment-ready with:

1. `Dockerfile`
2. `Procfile`
3. `run.py` (reads `PORT` safely)

### Railway steps

1. Push this folder to a GitHub repo.
2. In Railway, create a new service from that repo.
3. Deploy with Dockerfile (auto-detected).
4. Set environment variables:
   - `MAX_PROCESSING_LONG_SIDE=1920`
   - `MAX_PROCESSING_MEGAPIXELS=4.0`
   - `MAX_PREVIEW_LONG_SIDE=1400`
5. Healthcheck path:
   - `/api/health`

### Suggested instance size

1. Recommended: `1 vCPU / 2 GB RAM` for MediaPipe-only deployment.

## Run Locally

```bash
cd /Users/koohaoming/dev/photo-id-studio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8020
```

Open:

- `http://127.0.0.1:8020`
- API docs: `http://127.0.0.1:8020/docs`

## API

### `POST /api/analyze`

Multipart form fields:

1. `photo` (image file)
2. `country_code` (default `SG`)
3. `mode` (`strict` or `assist`)

Returns:

1. `report` with checks/error codes/actions
2. `processed_image_base64` when crop succeeds

### `GET /api/countries`

Returns available country profiles from `app/config/countries.yaml`.

## Notes

1. Current country profile: Singapore (`SG`) with output `400x514`.
2. Add more country specs by extending `app/config/countries.yaml`.
3. Some checks (for example whether an image was edited externally) are heuristic and should be reviewed before final submission workflows.
