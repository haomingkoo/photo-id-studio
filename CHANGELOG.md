# Changelog

All notable changes to this project will be documented in this file.

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
