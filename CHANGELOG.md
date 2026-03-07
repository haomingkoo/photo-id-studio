# Changelog

All notable changes to this project will be documented in this file.

## [0.3.1] - 2026-03-07

### Added
1. Mirror handling request mode (`auto`, `unmirror`, `keep`) for `/api/analyze`.
2. Mirror-result checks surfaced in report output (`MIRROR_CORRECTION_APPLIED`, `MIRROR_ADJUSTMENT_APPLIED`).
3. Frontend control for selfie mirror handling plus mirror-status summary chip.

### Changed
1. Decode pipeline now supports deterministic pre-analysis unmirror correction.
2. README includes iPhone mirrored selfie troubleshooting guidance.

## [0.3.0] - 2026-03-07

### Added
1. Control-center contract/admin endpoints for status, usage, pipeline, report, and logs.
2. Public transparency endpoint: `GET /api/privacy`.
3. Frontend privacy/legal notice with live repo/deploy metadata.
4. rembg runtime diagnostics surfaced in status/pipeline payloads.

### Changed
1. rembg memory lifecycle now supports idle auto-unload via `PHOTO_API_REMBG_IDLE_UNLOAD_SEC`.
2. API responses under `/api/*` now include no-store and hardening headers.
3. README updated with privacy policy, legal-risk controls, and control-center integration details.

### Security
1. Admin auth includes IP-based failure window/block controls.
2. Deployment guidance emphasizes secret-only key handling (no commits/log leakage).

## [0.2.0] - 2026-03-07

### Added
1. Request-level analyze memory telemetry logging (`PHOTO_API_LOG_ANALYZE_METRICS`).
2. Segmentation backend RAM controls:
   - `PHOTO_API_ENABLE_REMBG`
   - `PHOTO_API_REMBG_LAZY_LOAD`
3. Stricter assist white-background compositing toggle (`PHOTO_API_BG_STRICT_WHITE`).
4. Canonical versioning files: `VERSION` and `CHANGELOG.md`.

### Changed
1. FastAPI app version now reads from `VERSION` (single source of truth).
2. README deployment docs now include memory-focused env settings and diagnostics guidance.
