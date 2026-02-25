from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import os

import cv2
import numpy as np

from app.config import load_country_settings
from app.schemas import AnalysisReport, CheckResult, CropBox
from app.services.image_ops import (
    DecodedImage,
    brightness_stats,
    detect_text_overlay_score,
    decode_image_bytes,
    encode_jpeg_base64,
    enhance_image,
    even_lighting_score,
    laplacian_blur_score,
    suppress_edge_artifacts,
)

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None



EAR_LEFT_IDX = [33, 160, 158, 133, 153, 144]
EAR_RIGHT_IDX = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CENTER_IDX = [33, 133, 159, 145]
RIGHT_EYE_CENTER_IDX = [362, 263, 386, 374]
NOSE_TIP_IDX = 1
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14


@dataclass
class FaceGeometry:
    landmarks_px: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    nose: np.ndarray
    mouth_top: np.ndarray
    mouth_bottom: np.ndarray
    bbox: tuple[int, int, int, int]


class PhotoCompliancePipeline:
    def __init__(self) -> None:
        self.face_mesh = None
        self.segmentation_mediapipe = None
        if mp is not None:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=2,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
            except Exception:
                self.face_mesh = None
            try:
                self.segmentation_mediapipe = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            except Exception:
                self.segmentation_mediapipe = None

    def _resize_long_side(self, bgr_image: np.ndarray, max_long_side: int) -> tuple[np.ndarray, float]:
        h, w = bgr_image.shape[:2]
        longest = max(h, w)
        if max_long_side <= 0 or longest <= max_long_side:
            return bgr_image, 1.0
        scale = float(max_long_side) / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(
            bgr_image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        return resized, scale

    def _downscale_for_processing(self, bgr_image: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
        h, w = bgr_image.shape[:2]
        max_long_side = max(512, int(os.getenv("MAX_PROCESSING_LONG_SIDE", "1920")))
        max_megapixels = max(1.0, float(os.getenv("MAX_PROCESSING_MEGAPIXELS", "4.0")))
        src_mp = float(h * w) / 1_000_000.0

        longest = max(h, w)
        scale_long = min(1.0, float(max_long_side) / max(float(longest), 1.0))
        scale_mp = min(1.0, float(np.sqrt(max_megapixels / max(src_mp, 1e-6))))
        scale = min(scale_long, scale_mp)

        if scale >= 0.999:
            return bgr_image, 1.0, {
                "max_long_side": max_long_side,
                "max_megapixels": max_megapixels,
                "source_megapixels": round(src_mp, 2),
            }

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale, {
            "max_long_side": max_long_side,
            "max_megapixels": max_megapixels,
            "source_megapixels": round(src_mp, 2),
            "scaled_megapixels": round((new_h * new_w) / 1_000_000.0, 2),
        }

    def _check(
        self,
        code: str,
        status: str,
        message: str,
        action: str,
        score: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> CheckResult:
        return CheckResult(code=code, status=status, message=message, action=action, score=score, details=details)

    def _extract_face_geometry(self, bgr_image: np.ndarray) -> list[FaceGeometry]:
        if self.face_mesh is None:
            return []

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []

        h, w = bgr_image.shape[:2]
        output: list[FaceGeometry] = []
        for face in result.multi_face_landmarks:
            landmarks = np.array([(pt.x * w, pt.y * h) for pt in face.landmark], dtype=np.float32)
            x0 = int(np.clip(np.min(landmarks[:, 0]), 0, w - 1))
            y0 = int(np.clip(np.min(landmarks[:, 1]), 0, h - 1))
            x1 = int(np.clip(np.max(landmarks[:, 0]), 1, w))
            y1 = int(np.clip(np.max(landmarks[:, 1]), 1, h))

            output.append(
                FaceGeometry(
                    landmarks_px=landmarks,
                    left_eye=np.mean(landmarks[LEFT_EYE_CENTER_IDX], axis=0),
                    right_eye=np.mean(landmarks[RIGHT_EYE_CENTER_IDX], axis=0),
                    nose=landmarks[NOSE_TIP_IDX],
                    mouth_top=landmarks[MOUTH_TOP_IDX],
                    mouth_bottom=landmarks[MOUTH_BOTTOM_IDX],
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                )
            )
        return output

    def _eye_aspect_ratio(self, landmarks: np.ndarray, idx: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = max(np.linalg.norm(p1 - p4), 1e-6)
        return float(vertical / (2.0 * horizontal))

    def _segment_person_mediapipe(self, bgr_image: np.ndarray) -> np.ndarray | None:
        if self.segmentation_mediapipe is None:
            return None
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self.segmentation_mediapipe.process(rgb)
        if result.segmentation_mask is None:
            return None
        return result.segmentation_mask.astype(np.float32)

    def _segment_person(
        self,
        bgr_image: np.ndarray,
        backend: str = "mediapipe",
    ) -> tuple[np.ndarray | None, str, str | None]:
        requested = (backend or "mediapipe").strip().lower()
        note = None
        if requested not in {"mediapipe", "auto"}:
            note = "Only MediaPipe backend is enabled in this build."
        elif requested == "auto":
            note = "Auto mode maps to MediaPipe in this build."

        mask = self._segment_person_mediapipe(bgr_image)
        if mask is not None:
            return mask, "mediapipe", note
        return None, "none", "MediaPipe segmentation unavailable."

    def _compute_crop(
        self,
        image_shape: tuple[int, int, int],
        geometry: FaceGeometry,
        eye_distance: float,
        profile: Any,
    ) -> tuple[CropBox | None, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None, bool]:
        h, w = image_shape[:2]
        if eye_distance <= 0:
            return None, None, None, False

        output_w = float(profile.output_width_px)
        output_h = float(profile.output_height_px)
        aspect = output_h / max(output_w, 1.0)

        x, y, bw, bh = geometry.bbox
        if bw <= 0 or bh <= 0:
            return None, None, None, False

        base_crop_w = eye_distance / max(float(profile.eye_distance_fraction_of_width), 1e-6)
        eye_mid = (geometry.left_eye + geometry.right_eye) / 2.0
        eye_from_top = float(profile.eye_height_fraction_of_height) + float(getattr(profile, "extra_headroom_fraction", 0.0))
        eye_from_top = float(np.clip(eye_from_top, 0.32, 0.62))
        torso_bias = float(np.clip(getattr(profile, "extra_torso_fraction", 0.05), 0.0, 0.18))
        hair_margin_fraction = float(np.clip(getattr(profile, "hair_margin_fraction", 0.24), 0.10, 0.40))

        # Guarantee enough frame around head so we do not clip top/sides.
        side_margin = float(bw) * 0.22
        top_margin = float(bh) * hair_margin_fraction
        bottom_margin = float(bh) * (0.70 + torso_bias * 1.25)
        req_w = float(bw) + side_margin * 2.0
        req_h = float(bh) + top_margin + bottom_margin
        req_w_from_h = req_h / max(aspect, 1e-6)

        crop_w = int(round(max(base_crop_w, req_w, req_w_from_h)))
        crop_h = int(round(crop_w * aspect))
        if crop_w <= 0 or crop_h <= 0:
            return None, None, None, False

        if crop_w > w or crop_h > h:
            desired_left = int(round(float(eye_mid[0] - (crop_w / 2.0))))
            desired_top = int(round(float(eye_mid[1] - (crop_h * eye_from_top))))
            desired_box = (desired_left, desired_top, desired_left + crop_w, desired_top + crop_h)
            return None, None, desired_box, False

        bbox_center_x = float(x + (bw * 0.5))
        target_center_x = float((eye_mid[0] * 0.9) + (bbox_center_x * 0.1))
        desired_left = float(target_center_x - (crop_w / 2.0))
        desired_top = float(eye_mid[1] - (crop_h * eye_from_top))
        desired_top += float(crop_h) * torso_bias

        # Clamp crop so bbox + margins stays inside final crop.
        left_min = float(x + bw + side_margin - crop_w)
        left_max = float(x - side_margin)
        if left_min <= left_max:
            desired_left = float(np.clip(desired_left, left_min, left_max))

        top_min = float(y + bh + bottom_margin - crop_h)
        top_max = float(y - top_margin)
        if top_min <= top_max:
            desired_top = float(np.clip(desired_top, top_min, top_max))

        desired_left = float(np.clip(desired_left, 0.0, float(w - crop_w)))
        desired_top = float(np.clip(desired_top, 0.0, float(h - crop_h)))

        desired_left_i = int(round(desired_left))
        desired_top_i = int(round(desired_top))
        desired_box = (desired_left_i, desired_top_i, desired_left_i + crop_w, desired_top_i + crop_h)

        crop_left = int(np.clip(desired_left_i, 0, w - crop_w))
        crop_top = int(np.clip(desired_top_i, 0, h - crop_h))
        crop_right = crop_left + crop_w
        crop_bottom = crop_top + crop_h
        adjusted = crop_left != desired_left_i or crop_top != desired_top_i

        return (
            CropBox(
                left=crop_left,
                top=crop_top,
                right=crop_right,
                bottom=crop_bottom,
                width=crop_right - crop_left,
                height=crop_bottom - crop_top,
            ),
            (crop_left, crop_top, crop_right, crop_bottom),
            desired_box,
            adjusted,
        )

    def analyze(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        country_code: str,
        mode: str,
        beauty_mode: str = "none",
        segmentation_backend: str = "mediapipe",
        shadow_mode: str = "balanced",
    ) -> tuple[AnalysisReport, str | None, str | None, str | None, str | None]:
        settings = load_country_settings()
        requested_code = (country_code or settings.default_country).strip().upper()
        resolved_code = requested_code if requested_code in settings.countries else settings.default_country
        profile = settings.countries[resolved_code]
        checks: list[CheckResult] = []
        extension = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()

        if extension and extension not in profile.allowed_extensions:
            checks.append(
                self._check(
                    code="FILE_FORMAT_UNSUPPORTED",
                    status="fail",
                    message=f"Unsupported format '.{extension}'.",
                    action=f"Upload one of: {', '.join(profile.allowed_extensions)}.",
                )
            )
        else:
            checks.append(
                self._check(
                    code="FILE_FORMAT_OK",
                    status="pass",
                    message="File format accepted.",
                    action="No action needed.",
                )
            )

        file_size_bytes = len(file_bytes)
        max_bytes = int(profile.max_file_size_mb * 1024 * 1024)
        if file_size_bytes > max_bytes:
            checks.append(
                self._check(
                    code="FILE_SIZE_TOO_LARGE",
                    status="fail",
                    message=f"File size {file_size_bytes / (1024*1024):.2f}MB exceeds {profile.max_file_size_mb}MB.",
                    action="Compress the image or export at lower quality before upload.",
                )
            )

        decoded: DecodedImage = decode_image_bytes(file_bytes, extension=extension)
        image = decoded.bgr
        metadata = decoded.metadata
        orig_h, orig_w = image.shape[:2]

        preview_max_long = max(640, int(os.getenv("MAX_PREVIEW_LONG_SIDE", "1400")))
        preview_image, _ = self._resize_long_side(image, preview_max_long)
        original_preview_b64 = encode_jpeg_base64(preview_image, quality=88)

        image, processing_scale, processing_limits = self._downscale_for_processing(image)
        h, w = image.shape[:2]
        if processing_scale < 0.999:
            checks.append(
                self._check(
                    code="PROCESSING_DOWNSCALED",
                    status="pass",
                    message=f"Processing image downscaled to {w}x{h} for performance.",
                    action="No action needed.",
                    details={
                        "source_width": orig_w,
                        "source_height": orig_h,
                        "processing_width": w,
                        "processing_height": h,
                        "scale": round(processing_scale, 4),
                        **processing_limits,
                    },
                )
            )

        if orig_w < profile.min_input_width_px or orig_h < profile.min_input_height_px:
            checks.append(
                self._check(
                    code="INPUT_RESOLUTION_TOO_LOW",
                    status="fail",
                    message=f"Input resolution {orig_w}x{orig_h} is below minimum {profile.min_input_width_px}x{profile.min_input_height_px}.",
                    action="Retake with higher resolution (prefer rear camera and good distance).",
                )
            )
        else:
            checks.append(
                self._check(
                    code="INPUT_RESOLUTION_OK",
                    status="pass",
                    message="Input resolution is sufficient.",
                    action="No action needed.",
                    details={
                        "source_width": orig_w,
                        "source_height": orig_h,
                        "processing_width": w,
                        "processing_height": h,
                    },
                )
            )

        text_overlay = detect_text_overlay_score(image)
        text_score = float(text_overlay.get("score", 0.0))
        if text_score >= 0.68:
            checks.append(
                self._check(
                    code="TEXT_OR_WATERMARK_DETECTED",
                    status="fail",
                    message="Visible text or watermark detected in the photo.",
                    action="Use an original photo without logos, watermark text, or overlays.",
                    score=text_score,
                    details=text_overlay,
                )
            )
        elif text_score >= 0.45:
            checks.append(
                self._check(
                    code="POSSIBLE_TEXT_OVERLAY",
                    status="warn",
                    message="Possible text overlay detected.",
                    action="Ensure the background has no visible text, logos, or watermark artifacts.",
                    score=text_score,
                    details=text_overlay,
                )
            )
        else:
            checks.append(
                self._check(
                    code="TEXT_OVERLAY_OK",
                    status="pass",
                    message="No obvious text or watermark overlays detected.",
                    action="No action needed.",
                    score=text_score,
                    details=text_overlay,
                )
            )

        raw_orientation = metadata.get("raw_orientation")
        mirrored_orientation = bool(metadata.get("mirrored_orientation"))
        if mirrored_orientation:
            checks.append(
                self._check(
                    code="POSSIBLE_MIRRORED_IMAGE",
                    status="fail",
                    message="EXIF indicates mirrored orientation.",
                    action="Retake with standard camera mode (avoid mirrored selfie output).",
                    details={"raw_orientation": raw_orientation},
                )
            )

        captured_at = metadata.get("captured_at")
        if isinstance(captured_at, datetime):
            age_days = (datetime.now(timezone.utc) - captured_at.replace(tzinfo=timezone.utc)).days
            if age_days > profile.max_age_days:
                checks.append(
                    self._check(
                        code="PHOTO_TOO_OLD",
                        status="fail",
                        message=f"Photo metadata indicates ~{age_days} days old.",
                        action=f"Use a recent photo taken within {profile.max_age_days} days.",
                        details={"captured_at": captured_at.isoformat()},
                    )
                )
            else:
                checks.append(
                    self._check(
                        code="PHOTO_RECENCY_OK",
                        status="pass",
                        message="Photo date appears recent.",
                        action="No action needed.",
                        details={"captured_at": captured_at.isoformat()},
                    )
                )
        else:
            checks.append(
                self._check(
                    code="PHOTO_RECENCY_UNKNOWN",
                    status="manual",
                    message="Capture date metadata not found.",
                    action=f"Confirm the photo was taken within {profile.max_age_days} days.",
                )
            )

        if self.face_mesh is None:
            checks.append(
                self._check(
                    code="DETECTION_BACKEND_UNAVAILABLE",
                    status="fail",
                    message="Mediapipe backend unavailable.",
                    action="Install required dependencies and restart the server.",
                )
            )
            return (
                self._finalize_report(resolved_code, profile, mode, w, h, file_size_bytes, captured_at, checks, None),
                None,
                original_preview_b64,
                None,
                None,
            )

        faces = self._extract_face_geometry(image)
        if len(faces) == 0:
            checks.append(
                self._check(
                    code="NO_FACE_DETECTED",
                    status="fail",
                    message="No face detected in the image.",
                    action="Use a clear, front-facing photo with full head and shoulders visible.",
                    details={"detected_faces": 0},
                )
            )
            return (
                self._finalize_report(resolved_code, profile, mode, w, h, file_size_bytes, captured_at, checks, None),
                None,
                original_preview_b64,
                None,
                None,
            )
        if len(faces) > 1:
            checks.append(
                self._check(
                    code="MULTIPLE_FACES_DETECTED",
                    status="fail",
                    message=f"Detected {len(faces)} faces. Exactly one person is required.",
                    action="Retake with only one person fully visible in the frame.",
                    details={"detected_faces": len(faces)},
                )
            )
            return (
                self._finalize_report(resolved_code, profile, mode, w, h, file_size_bytes, captured_at, checks, None),
                None,
                original_preview_b64,
                None,
                None,
            )

        face = faces[0]
        x, y, fw, fh = face.bbox
        inter_eye = float(np.linalg.norm(face.left_eye - face.right_eye))

        if inter_eye < profile.min_eye_distance_px:
            checks.append(
                self._check(
                    code="FACE_TOO_SMALL",
                    status="fail",
                    message=f"Inter-eye distance {inter_eye:.1f}px is too small.",
                    action="Move closer to camera so facial details are sharper.",
                    score=inter_eye,
                )
            )
        else:
            checks.append(
                self._check(
                    code="FACE_SCALE_OK",
                    status="pass",
                    message="Face scale appears sufficient.",
                    action="No action needed.",
                    score=inter_eye,
                )
            )

        roll_deg = float(np.degrees(np.arctan2(face.right_eye[1] - face.left_eye[1], face.right_eye[0] - face.left_eye[0])))
        eye_mid = (face.left_eye + face.right_eye) / 2.0
        yaw_ratio = float(abs((face.nose[0] - eye_mid[0]) / max(inter_eye / 2.0, 1e-6)))
        mouth_mid = (face.mouth_top + face.mouth_bottom) / 2.0
        denom = max(float(mouth_mid[1] - eye_mid[1]), 1e-6)
        pitch_ratio = float(abs(((face.nose[1] - eye_mid[1]) / denom) - 0.50))

        if abs(roll_deg) > profile.max_roll_degrees:
            checks.append(
                self._check(
                    code="HEAD_ROLL_TOO_HIGH",
                    status="fail",
                    message=f"Head tilt {roll_deg:.1f}° exceeds {profile.max_roll_degrees}°.",
                    action="Keep your head straight and camera level.",
                    score=roll_deg,
                )
            )
        else:
            checks.append(
                self._check(
                    code="HEAD_ROLL_OK",
                    status="pass",
                    message="Head roll is within tolerance.",
                    action="No action needed.",
                    score=roll_deg,
                )
            )

        if yaw_ratio > profile.max_yaw_ratio:
            checks.append(
                self._check(
                    code="HEAD_YAW_TOO_HIGH",
                    status="fail",
                    message=f"Face is not fully frontal (yaw ratio {yaw_ratio:.2f}).",
                    action="Face the camera directly with both eyes equally visible.",
                    score=yaw_ratio,
                )
            )
        else:
            checks.append(
                self._check(
                    code="HEAD_YAW_OK",
                    status="pass",
                    message="Frontal face orientation looks good.",
                    action="No action needed.",
                    score=yaw_ratio,
                )
            )

        if pitch_ratio > profile.max_pitch_ratio:
            checks.append(
                self._check(
                    code="HEAD_PITCH_TOO_HIGH",
                    status="fail",
                    message=f"Head pitch appears off-center (ratio {pitch_ratio:.2f}).",
                    action="Keep chin level and eyes looking directly at camera.",
                    score=pitch_ratio,
                )
            )
        else:
            checks.append(
                self._check(
                    code="HEAD_PITCH_OK",
                    status="pass",
                    message="Head pitch is within tolerance.",
                    action="No action needed.",
                    score=pitch_ratio,
                )
            )

        ear_left = self._eye_aspect_ratio(face.landmarks_px, EAR_LEFT_IDX)
        ear_right = self._eye_aspect_ratio(face.landmarks_px, EAR_RIGHT_IDX)
        ear_min = min(ear_left, ear_right)
        if ear_min < 0.17:
            checks.append(
                self._check(
                    code="EYES_NOT_CLEARLY_VISIBLE",
                    status="fail",
                    message="Eyes appear partially closed or obscured.",
                    action="Retake with both eyes fully open and clear lenses.",
                    score=ear_min,
                )
            )
        else:
            checks.append(
                self._check(
                    code="EYES_VISIBLE_OK",
                    status="pass",
                    message="Eyes are clearly visible.",
                    action="No action needed.",
                    score=ear_min,
                )
            )

        mouth_gap_ratio = float(np.linalg.norm(face.mouth_top - face.mouth_bottom) / max(inter_eye, 1e-6))
        if mouth_gap_ratio > 0.085:
            checks.append(
                self._check(
                    code="MOUTH_NOT_CLOSED",
                    status="fail",
                    message="Mouth appears open.",
                    action="Maintain a neutral expression with mouth closed.",
                    score=mouth_gap_ratio,
                )
            )
        else:
            checks.append(
                self._check(
                    code="MOUTH_CLOSED_OK",
                    status="pass",
                    message="Neutral mouth expression detected.",
                    action="No action needed.",
                    score=mouth_gap_ratio,
                )
            )

        face_margin = int(round(max(fw, fh) * 0.18))
        roi_left = max(0, x - face_margin)
        roi_top = max(0, y - face_margin)
        roi_right = min(w, x + fw + face_margin)
        roi_bottom = min(h, y + fh + face_margin)
        blur_roi = image[roi_top:roi_bottom, roi_left:roi_right]
        blur_score = laplacian_blur_score(blur_roi if blur_roi.size > 0 else image)
        blur_fail_threshold = float(profile.min_blur_score) * 0.55
        if blur_score < blur_fail_threshold:
            checks.append(
                self._check(
                    code="IMAGE_BLUR_TOO_HIGH",
                    status="fail",
                    message=f"Image sharpness score {blur_score:.1f} is too low.",
                    action="Retake with steady hand, better focus, and stronger light.",
                    score=blur_score,
                    details={"roi": [roi_left, roi_top, roi_right, roi_bottom]},
                )
            )
        elif blur_score < profile.min_blur_score:
            checks.append(
                self._check(
                    code="IMAGE_BLUR_BORDERLINE",
                    status="warn",
                    message=f"Image sharpness score {blur_score:.1f} is borderline.",
                    action="Use stronger light and steady framing for a sharper photo.",
                    score=blur_score,
                    details={
                        "roi": [roi_left, roi_top, roi_right, roi_bottom],
                        "warn_threshold": profile.min_blur_score,
                        "fail_threshold": round(blur_fail_threshold, 1),
                    },
                )
            )
        else:
            checks.append(
                self._check(
                    code="IMAGE_SHARPNESS_OK",
                    status="pass",
                    message="Image sharpness is acceptable.",
                    action="No action needed.",
                    score=blur_score,
                    details={"roi": [roi_left, roi_top, roi_right, roi_bottom]},
                )
            )

        lighting = even_lighting_score(image)
        if lighting < profile.min_even_lighting_score:
            checks.append(
                self._check(
                    code="LIGHTING_NOT_EVEN",
                    status="warn",
                    message=f"Lighting uniformity score {lighting:.2f} is low.",
                    action="Use front-facing soft light and avoid side shadows.",
                    score=lighting,
                )
            )
        else:
            checks.append(
                self._check(
                    code="LIGHTING_OK",
                    status="pass",
                    message="Lighting appears even.",
                    action="No action needed.",
                    score=lighting,
                )
            )

        segment_mask, segmentation_used, segmentation_note = self._segment_person(image, backend=segmentation_backend)
        checks.append(
            self._check(
                code="SEGMENTATION_BACKEND_USED",
                status="pass" if segmentation_used != "none" else "manual",
                message=f"Segmentation backend used: {segmentation_used}.",
                action="No action needed." if segmentation_used != "none" else "Install/enable a segmentation backend.",
                details={"requested_backend": segmentation_backend, "used_backend": segmentation_used},
            )
        )
        if segmentation_note:
            checks.append(
                self._check(
                    code="SEGMENTATION_BACKEND_FALLBACK",
                    status="warn" if segmentation_used != "none" else "manual",
                    message=segmentation_note,
                    action="If needed, select another backend from the UI.",
                    details={"requested_backend": segmentation_backend, "used_backend": segmentation_used},
                )
            )

        if segment_mask is not None:
            person = segment_mask > 0.5
            bg = ~person
            if int(bg.sum()) > 400:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                bg_v = hsv[:, :, 2][bg]
                bg_s = hsv[:, :, 1][bg]
                bg_brightness = float(bg_v.mean())
                bg_saturation = float(bg_s.mean())
                bg_std = float(bg_v.std())

                if (
                    bg_brightness < profile.min_background_brightness
                    or bg_saturation > profile.max_background_saturation
                    or bg_std > profile.max_background_stddev
                ):
                    checks.append(
                        self._check(
                            code="BACKGROUND_NOT_COMPLIANT",
                            status="warn",
                            message="Background may not be plain bright white.",
                            action="Use plain white wall and even lighting behind you.",
                            details={
                                "bg_brightness": round(bg_brightness, 2),
                                "bg_saturation": round(bg_saturation, 2),
                                "bg_std": round(bg_std, 2),
                            },
                        )
                    )
                else:
                    checks.append(
                        self._check(
                            code="BACKGROUND_OK",
                            status="pass",
                            message="Background appears bright and uniform.",
                            action="No action needed.",
                        )
                    )

            ys, xs = np.where(person)
            if len(ys) > 0:
                top_person = int(ys.min())
                bottom_person = int(ys.max())
                face_bottom = y + fh
                hair_ok = top_person > 1
                shoulder_ok = bottom_person > int(h * 0.78) and face_bottom < bottom_person
                if hair_ok and shoulder_ok:
                    checks.append(
                        self._check(
                            code="HEAD_SHOULDERS_VISIBILITY_OK",
                            status="pass",
                            message="Head and shoulders look visible.",
                            action="No action needed.",
                        )
                    )
                else:
                    checks.append(
                        self._check(
                            code="HEAD_SHOULDERS_PARTIAL",
                            status="warn",
                            message="Hair or shoulders may be partially cropped in source image.",
                            action="Retake with more space above hair and include shoulders.",
                        )
                    )
        else:
            checks.append(
                self._check(
                    code="SEGMENTATION_UNAVAILABLE",
                    status="manual",
                    message="Background segmentation check unavailable.",
                    action="Verify background is plain white and shoulders are visible.",
                )
            )

        raw_stats = brightness_stats(image)
        if raw_stats["brightness_mean"] < 70 or raw_stats["brightness_mean"] > 245:
            checks.append(
                self._check(
                    code="EXPOSURE_EXTREME",
                    status="warn",
                    message="Photo exposure may be too dark or too bright.",
                    action="Retake in balanced indoor lighting without harsh backlight.",
                    details={k: round(v, 2) for k, v in raw_stats.items()},
                )
            )

        crop_box, crop_raw, desired_box, crop_adjusted = self._compute_crop(image.shape, face, inter_eye, profile)
        processed_b64 = None
        comparison_no_correction_b64 = None
        comparison_color_correction_b64 = None
        if crop_box is None or crop_raw is None:
            checks.append(
                self._check(
                    code="CROP_OUT_OF_BOUNDS",
                    status="fail",
                    message="Cannot crop to required dimensions with current framing.",
                    action="Retake farther from edges so full crop fits naturally.",
                    details={"suggested_box": desired_box},
                )
            )
        else:
            left, top, right, bottom = crop_raw
            crop_w = max(1, right - left)
            crop_h = max(1, bottom - top)

            # Multi-pass horizontal recentering on the provisional crop.
            # This keeps framing stable even when bbox/hair asymmetry biases the first estimate.
            recenter_total_shift_x = 0
            recenter_iterations = 0
            recenter_last_err_x = None
            for _ in range(3):
                probe = image[top:bottom, left:right]
                probe_faces = self._extract_face_geometry(probe) if probe.size > 0 else []
                if not probe_faces:
                    break
                probe_face = max(probe_faces, key=lambda item: item.bbox[2] * item.bbox[3])
                probe_eye_mid = (probe_face.left_eye + probe_face.right_eye) / 2.0
                probe_bbox_cx = float(probe_face.bbox[0] + (probe_face.bbox[2] * 0.5))
                # Use mostly eye midpoint but blend in bbox center for stronger visual centering.
                probe_target_cx = float((probe_eye_mid[0] * 0.7) + (probe_bbox_cx * 0.3))
                center_err_x = float(probe_target_cx - (crop_w * 0.5))
                recenter_last_err_x = center_err_x
                if abs(center_err_x) < 1.5:
                    break

                max_shift = int(round(crop_w * 0.16))
                shift_x = int(np.clip(int(round(center_err_x * 0.92)), -max_shift, max_shift))
                if abs(shift_x) < 1:
                    break

                new_left = int(np.clip(left + shift_x, 0, w - crop_w))
                applied_shift = new_left - left
                if applied_shift == 0:
                    break
                left = new_left
                right = left + crop_w
                recenter_total_shift_x += applied_shift
                recenter_iterations += 1

            if recenter_iterations > 0:
                crop_raw = (left, top, right, bottom)
                crop_box = CropBox(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    width=crop_w,
                    height=crop_h,
                )
                crop_adjusted = True
                checks.append(
                    self._check(
                        code="CROP_RECENTERED",
                        status="pass",
                        message=f"Crop horizontally recentered by {recenter_total_shift_x}px.",
                        action="No action needed.",
                        details={
                            "iterations": recenter_iterations,
                            "shift_x_px": recenter_total_shift_x,
                            "final_center_error_x_px": round(float(recenter_last_err_x or 0.0), 2),
                        },
                    )
                )

            if crop_adjusted:
                checks.append(
                    self._check(
                        code="CROP_ADJUSTED_TO_FRAME",
                        status="warn",
                        message="Crop was adjusted to keep full head/hair inside frame.",
                        action="Step back slightly when taking photo for ideal framing.",
                        details={"desired_box": desired_box, "applied_box": crop_raw},
                    )
                )
            cropped = image[top:bottom, left:right]

            # Straighten mild roll while preserving white corners.
            if abs(roll_deg) > 0.3:
                ch, cw = cropped.shape[:2]
                matrix = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), -roll_deg, 1.0)
                cropped = cv2.warpAffine(
                    cropped,
                    matrix,
                    (cw, ch),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

            output_size = (int(profile.output_width_px), int(profile.output_height_px))
            processed = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
            crop_w = max(1, right - left)
            crop_h = max(1, bottom - top)
            fh_scaled = fh * output_size[1] / crop_h
            # Face mesh bbox often excludes full hairline/crown; compensate to estimate head height.
            estimated_head_h = fh_scaled * 1.10
            min_face_h_out_fail = float(min(profile.min_face_height_px, int(output_size[1] * 0.48)))
            min_face_h_out_warn = float(min(profile.min_face_height_px, int(output_size[1] * 0.56)))
            max_face_h_out = float(output_size[1]) * 0.84
            if estimated_head_h < min_face_h_out_fail:
                checks.append(
                    self._check(
                        code="FACE_HEIGHT_TOO_SMALL",
                        status="fail",
                        message=(
                            f"Estimated head height in output ({estimated_head_h:.1f}px) "
                            f"is below minimum {min_face_h_out_fail:.0f}px."
                        ),
                        action="Move closer to camera before taking the photo.",
                        score=estimated_head_h,
                        details={"min_head_height_output_px": min_face_h_out_fail, "raw_face_height_px": fh_scaled},
                    )
                )
            elif estimated_head_h < min_face_h_out_warn:
                checks.append(
                    self._check(
                        code="FACE_HEIGHT_BORDERLINE",
                        status="warn",
                        message=(
                            f"Estimated head height in output ({estimated_head_h:.1f}px) "
                            f"is borderline for ideal framing."
                        ),
                        action="Move slightly closer for a stronger head-and-shoulders frame.",
                        score=estimated_head_h,
                        details={"warn_head_height_output_px": min_face_h_out_warn, "raw_face_height_px": fh_scaled},
                    )
                )
            else:
                checks.append(
                    self._check(
                        code="FACE_HEIGHT_OK",
                        status="pass",
                        message="Face height in output is within minimum target.",
                        action="No action needed.",
                        score=estimated_head_h,
                    )
                )
            if estimated_head_h > max_face_h_out:
                checks.append(
                    self._check(
                        code="FACE_HEIGHT_TOO_LARGE",
                        status="warn",
                        message=f"Estimated head height in output ({estimated_head_h:.1f}px) is too large for ideal ICA framing.",
                        action="Retake slightly farther from camera to include more shoulders/chest.",
                        score=estimated_head_h,
                        details={"max_face_height_output_px": round(max_face_h_out, 1)},
                    )
                )

            if mode.lower() in {"assist", "enhanced", "enhance"}:
                mask_for_processed = None
                if segment_mask is not None:
                    mask_crop = segment_mask[top:bottom, left:right]
                    if mask_crop.size > 0:
                        mask_for_processed = cv2.resize(mask_crop, output_size, interpolation=cv2.INTER_LINEAR)
                fx = (x - left) * output_size[0] / crop_w
                fy = (y - top) * output_size[1] / crop_h
                fw_scaled = fw * output_size[0] / crop_w
                fh_scaled = fh * output_size[1] / crop_h
                face_box_processed = (
                    int(round(fx)),
                    int(round(fy)),
                    int(round(fw_scaled)),
                    int(round(fh_scaled)),
                )
                processed_base = enhance_image(
                    processed,
                    person_mask=mask_for_processed,
                    background_whitening=True,
                    beauty_mode="none",
                    face_box=face_box_processed,
                    shadow_mode=shadow_mode,
                )
                processed_base = suppress_edge_artifacts(processed_base, border=2)
                processed = processed_base

                beauty_mode_norm = (beauty_mode or "").lower()
                if beauty_mode_norm in {"color", "tone", "color_correction", "soft", "soft_light", "natural"}:
                    apply_mode = "soft" if beauty_mode_norm in {"soft", "soft_light", "natural"} else "color"
                    processed_beauty = enhance_image(
                        processed_base.copy(),
                        person_mask=mask_for_processed,
                        background_whitening=False,
                        beauty_mode=apply_mode,
                        face_box=face_box_processed,
                        shadow_mode=shadow_mode,
                    )
                    processed_beauty = suppress_edge_artifacts(processed_beauty, border=2)
                    processed = processed_beauty
                    comparison_no_correction_b64 = encode_jpeg_base64(processed_base)
                    comparison_color_correction_b64 = encode_jpeg_base64(processed_beauty)

            processed = suppress_edge_artifacts(processed, border=2)
            processed_b64 = encode_jpeg_base64(processed)
            checks.append(
                self._check(
                    code="CROP_OK",
                    status="pass",
                    message="Photo cropped to target dimensions successfully.",
                    action="No action needed.",
                    details={"output_width": profile.output_width_px, "output_height": profile.output_height_px},
                )
            )

        report = self._finalize_report(
            resolved_code,
            profile,
            mode,
            orig_w,
            orig_h,
            file_size_bytes,
            captured_at,
            checks,
            crop_box,
            segmentation_backend=segmentation_used,
        )
        return (
            report,
            processed_b64,
            original_preview_b64,
            comparison_no_correction_b64,
            comparison_color_correction_b64,
        )

    def _finalize_report(
        self,
        country_code: str,
        profile: Any,
        mode: str,
        width: int,
        height: int,
        file_size_bytes: int,
        captured_at: datetime | None,
        checks: list[CheckResult],
        crop_box: CropBox | None,
        segmentation_backend: str | None = None,
    ) -> AnalysisReport:
        has_fail = any(check.status == "fail" for check in checks)
        has_non_pass = any(check.status in {"warn", "manual"} for check in checks)

        if has_fail:
            overall = "fail"
        elif has_non_pass:
            overall = "review"
        else:
            overall = "pass"

        error_codes = [check.code for check in checks if check.status == "fail"]
        actions: list[str] = []
        seen: set[str] = set()
        for check in checks:
            if check.status in {"fail", "warn", "manual"} and check.action not in seen:
                actions.append(check.action)
                seen.add(check.action)

        return AnalysisReport(
            country_code=country_code,
            country_name=profile.name,
            mode=mode,
            segmentation_backend=segmentation_backend,
            width=width,
            height=height,
            file_size_bytes=file_size_bytes,
            captured_at=captured_at,
            overall_status=overall,
            checks=checks,
            error_codes=error_codes,
            actions=actions,
            crop_box=crop_box,
        )
