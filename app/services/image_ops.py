from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import os
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
try:
    from pillow_heif import register_heif_opener
except Exception:  # pragma: no cover
    register_heif_opener = None

if register_heif_opener is not None:
    try:
        register_heif_opener()
    except Exception:
        pass

EXIF_TAG_MAP = {v: k for k, v in ExifTags.TAGS.items()}
ORIENTATION_TAG = EXIF_TAG_MAP.get("Orientation", 274)
DATETIME_ORIGINAL_TAG = EXIF_TAG_MAP.get("DateTimeOriginal", 36867)
DATETIME_TAG = EXIF_TAG_MAP.get("DateTime", 306)
MIRRORED_ORIENTATION_VALUES = {2, 4, 5, 7}
HEIC_EXTENSIONS = {"heic", "heif"}
MAX_DECODE_MEGAPIXELS = max(1.0, float(os.getenv("PHOTO_API_MAX_DECODE_MEGAPIXELS", "36")))
MAX_DECODE_PIXELS = int(MAX_DECODE_MEGAPIXELS * 1_000_000)
Image.MAX_IMAGE_PIXELS = MAX_DECODE_PIXELS


@dataclass
class DecodedImage:
    bgr: np.ndarray
    metadata: dict[str, Any]


def _enforce_decode_pixel_limit(width: int, height: int) -> None:
    total_pixels = int(width) * int(height)
    if total_pixels > MAX_DECODE_PIXELS:
        raise ValueError(
            "Image resolution is too large to process safely. "
            f"Maximum decode limit is {MAX_DECODE_MEGAPIXELS:.0f} megapixels."
        )



def _parse_exif_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None



def decode_image_bytes(file_bytes: bytes, extension: str | None = None) -> DecodedImage:
    metadata: dict[str, Any] = {
        "captured_at": None,
        "raw_orientation": None,
        "mirrored_orientation": False,
    }

    try:
        pil_img = Image.open(BytesIO(file_bytes))
        _enforce_decode_pixel_limit(*pil_img.size)
        exif = pil_img.getexif() if hasattr(pil_img, "getexif") else None

        if exif:
            raw_orientation = exif.get(ORIENTATION_TAG)
            metadata["raw_orientation"] = raw_orientation
            metadata["mirrored_orientation"] = raw_orientation in MIRRORED_ORIENTATION_VALUES

            captured_at = _parse_exif_datetime(exif.get(DATETIME_ORIGINAL_TAG))
            if captured_at is None:
                captured_at = _parse_exif_datetime(exif.get(DATETIME_TAG))
            metadata["captured_at"] = captured_at

        # Applies EXIF orientation, including mirrored modes.
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        rgb = np.asarray(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return DecodedImage(bgr=bgr, metadata=metadata)
    except Image.DecompressionBombError as exc:
        raise ValueError(
            "Image resolution is too large to process safely. "
            f"Maximum decode limit is {MAX_DECODE_MEGAPIXELS:.0f} megapixels."
        ) from exc
    except ValueError:
        raise
    except Exception:
        pass

    array = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        ext = (extension or "").lower()
        if ext in HEIC_EXTENSIONS and register_heif_opener is None:
            raise ValueError(
                "Unable to decode HEIC/HEIF image. Missing HEIC codec support. Install 'pillow-heif' and retry."
            )
        raise ValueError("Unable to decode image. Please upload a valid JPG/PNG/HEIC image.")
    _enforce_decode_pixel_limit(bgr.shape[1], bgr.shape[0])
    return DecodedImage(bgr=bgr, metadata=metadata)



def encode_jpeg_base64(bgr_image: np.ndarray, quality: int = 95) -> str:
    ok, encoded = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode processed image")
    import base64

    return base64.b64encode(encoded.tobytes()).decode("ascii")



def laplacian_blur_score(bgr_image: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())



def even_lighting_score(bgr_image: np.ndarray) -> float:
    yuv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0]
    h, w = y.shape
    regions = [
        y[: h // 2, : w // 2],
        y[: h // 2, w // 2 :],
        y[h // 2 :, : w // 2],
        y[h // 2 :, w // 2 :],
    ]
    means = np.array([region.mean() for region in regions], dtype=np.float32)
    spread = float(means.std())
    global_std = float(y.std())
    # Normalize to 0..1 where 1 means balanced brightness and low contrast variance.
    return float(max(0.0, min(1.0, 1.0 - (spread / 80.0) - (global_std / 250.0))))



def brightness_stats(bgr_image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return {
        "brightness_mean": float(hsv[:, :, 2].mean()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "brightness_std": float(hsv[:, :, 2].std()),
    }


def detect_text_overlay_score(bgr_image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 120 or w < 120:
        return {"score": 0.0, "regions": 0.0, "bottom_regions": 0.0, "max_width_ratio": 0.0}

    # Emphasize stroke-like transitions.
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions: list[tuple[int, int, int, int]] = []
    max_width_ratio = 0.0
    for contour in contours:
        x, y, bw_box, bh_box = cv2.boundingRect(contour)
        if bw_box < int(0.08 * w) or bh_box < 8:
            continue
        if bw_box > int(0.95 * w) or bh_box > int(0.14 * h):
            continue
        aspect = float(bw_box) / max(float(bh_box), 1.0)
        if aspect < 1.6:
            continue

        roi = bw[y : y + bh_box, x : x + bw_box]
        fill = float((roi > 0).mean())
        if fill < 0.06 or fill > 0.82:
            continue

        edge_roi = cv2.Canny(gray[y : y + bh_box, x : x + bw_box], 55, 150)
        edge_density = float((edge_roi > 0).mean())
        if edge_density < 0.04:
            continue

        text_regions.append((x, y, bw_box, bh_box))
        max_width_ratio = max(max_width_ratio, float(bw_box) / float(w))

    regions = float(len(text_regions))
    bottom_regions = float(sum(1 for (_, y, _, _) in text_regions if y > int(h * 0.60)))
    score = min(1.0, 0.17 * regions + 0.30 * bottom_regions + 0.55 * max_width_ratio)
    return {
        "score": float(score),
        "regions": regions,
        "bottom_regions": bottom_regions,
        "max_width_ratio": float(max_width_ratio),
    }


def _refine_person_mask(person_mask: np.ndarray) -> np.ndarray:
    mask = np.clip(person_mask.astype(np.float32), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.1)
    mask_u8 = (mask * 255.0).astype(np.uint8)
    mask_u8 = cv2.bilateralFilter(mask_u8, d=7, sigmaColor=36, sigmaSpace=7)
    mask = mask_u8.astype(np.float32) / 255.0

    # Fill tiny interior holes while preserving hair-edge softness.
    solid = (mask > 0.62).astype(np.uint8)
    if int(solid.sum()) > 0:
        num_labels, component_map, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            solid = (component_map == largest).astype(np.uint8)

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_mid = np.ones((5, 5), np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel_mid, iterations=1)
    # Keep border conservative to avoid preserving background shadow as foreground halo.
    # (no dilation here on purpose)
    solid = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 0.9)

    core = cv2.GaussianBlur((mask > 0.86).astype(np.float32), (0, 0), 0.8)
    refined = np.clip(mask * 0.95 + solid * 0.05, 0.0, 1.0)
    return np.clip(np.maximum(refined, core * 0.97), 0.0, 1.0)


def _refine_mask_with_grabcut(bgr_image: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    h, w = soft_mask.shape[:2]
    if h < 80 or w < 80:
        return soft_mask

    try:
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[soft_mask < 0.10] = cv2.GC_BGD
        gc_mask[(soft_mask >= 0.10) & (soft_mask < 0.60)] = cv2.GC_PR_BGD
        gc_mask[(soft_mask >= 0.60) & (soft_mask < 0.92)] = cv2.GC_PR_FGD
        gc_mask[soft_mask >= 0.92] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            bgr_image,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )

        refined_fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            1.0,
            0.0,
        ).astype(np.float32)
        refined_fg = cv2.GaussianBlur(refined_fg, (0, 0), 1.1)
        blended = np.clip(soft_mask * 0.50 + refined_fg * 0.50, 0.0, 1.0)
        return blended
    except Exception:
        return soft_mask


def _guided_refine_mask(bgr_image: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    try:
        if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
            return soft_mask
        guide = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        src = np.clip(soft_mask.astype(np.float32), 0.0, 1.0)
        refined = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=6, eps=1e-4)
        return np.clip(refined.astype(np.float32), 0.0, 1.0)
    except Exception:
        return soft_mask


def _subject_border_touch_ratio(mask: np.ndarray, threshold: float = 0.35) -> float:
    h, w = mask.shape[:2]
    edge = max(2, int(round(min(h, w) * 0.01)))
    border_zone = np.zeros((h, w), dtype=np.uint8)
    border_zone[:edge, :] = 1
    border_zone[-edge:, :] = 1
    border_zone[:, :edge] = 1
    border_zone[:, -edge:] = 1
    border_pixels = int(border_zone.sum())
    if border_pixels <= 0:
        return 0.0
    person = mask > float(threshold)
    touching = int((person & (border_zone > 0)).sum())
    return float(touching) / float(border_pixels)


def _build_border_touch_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    person_solid = (mask > 0.50).astype(np.uint8)
    if int(person_solid.sum()) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    min_area = max(24, int(0.00035 * h * w))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_solid, connectivity=8)
    border_touch = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        touches_border = x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h
        if touches_border:
            border_touch[labels == label] = 1

    if int(border_touch.sum()) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    ks = max(5, int(round(min(h, w) * 0.028)))
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    border_touch = cv2.dilate(border_touch, kernel, iterations=1).astype(np.float32)
    sigma = max(1.2, min(h, w) * 0.006)
    border_touch = cv2.GaussianBlur(border_touch, (0, 0), sigma)
    return np.clip(border_touch, 0.0, 1.0)


def _estimate_background_color(bgr_image: np.ndarray, bg_alpha: np.ndarray) -> np.ndarray:
    confident = bg_alpha > 0.93
    if int(confident.sum()) < 64:
        confident = bg_alpha > 0.82
    if int(confident.sum()) < 64:
        return np.array([245.0, 245.0, 245.0], dtype=np.float32)
    pixels = bgr_image[confident].astype(np.float32)
    return np.percentile(pixels, 50, axis=0).astype(np.float32)


def _build_face_mask(shape: tuple[int, int], face_box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = face_box
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    bw = int(np.clip(bw, 1, w - x))
    bh = int(np.clip(bh, 1, h - y))

    cx = int(round(x + bw * 0.5))
    cy = int(round(y + bh * 0.52))
    axes = (max(8, int(round(bw * 0.62))), max(10, int(round(bh * 0.78))))

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
    sigma = max(1.8, min(h, w) * 0.012)
    return cv2.GaussianBlur(mask, (0, 0), sigma)


def _apply_face_color_correction(
    bgr_image: np.ndarray,
    face_box: tuple[int, int, int, int],
    strength: float = 0.28,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 0.45))
    if strength <= 1e-6:
        return bgr_image

    h, w = bgr_image.shape[:2]
    face_mask = _build_face_mask((h, w), face_box)

    # Keep correction constrained to likely skin area in face ROI.
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 178, 133)).astype(np.float32) / 255.0
    skin = cv2.GaussianBlur(skin, (0, 0), 1.0)
    roi = np.clip(face_mask * skin, 0.0, 1.0)
    weight = float(roi.sum())
    if weight < 10.0:
        return bgr_image

    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l_mean = float((l * roi).sum() / weight)
    a_mean = float((a * roi).sum() / weight)
    b_mean = float((b * roi).sum() / weight)

    # Subtle luminance lift and gentle cast neutralization while preserving natural skin chroma.
    l_delta = np.clip(136.0 - l_mean, -8.0, 8.0)
    a_delta = np.clip(128.0 - a_mean, -5.0, 5.0)
    b_delta = np.clip(128.0 - b_mean, -3.0, 3.0)

    l = l + roi * l_delta * (0.72 * strength)
    a = a + roi * a_delta * (0.36 * strength)
    b = b + roi * b_delta * (0.34 * strength)

    corrected = cv2.cvtColor(cv2.merge((np.clip(l, 0, 255), np.clip(a, 0, 255), np.clip(b, 0, 255))).astype(np.uint8), cv2.COLOR_LAB2BGR)
    hsv_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_src = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_floor = hsv_src[:, :, 1] * 0.94
    hsv_corr[:, :, 1] = np.maximum(hsv_corr[:, :, 1], sat_floor * roi + hsv_corr[:, :, 1] * (1.0 - roi))
    src_v_mean = float((hsv_src[:, :, 2] * roi).sum() / weight)
    corr_v_mean = float((hsv_corr[:, :, 2] * roi).sum() / weight)
    v_lift = np.clip(src_v_mean - corr_v_mean, 0.0, 9.0)
    if v_lift > 0.5:
        hsv_corr[:, :, 2] = np.clip(hsv_corr[:, :, 2] + roi * (v_lift * 0.9), 0.0, 255.0)
    corrected = cv2.cvtColor(np.clip(hsv_corr, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    roi3 = np.repeat(roi[:, :, None], 3, axis=2)
    blend = np.clip(roi3 * 0.92, 0.0, 1.0)
    return (bgr_image.astype(np.float32) * (1.0 - blend) + corrected.astype(np.float32) * blend).clip(0, 255).astype(np.uint8)


def _apply_face_soft_light(
    bgr_image: np.ndarray,
    face_box: tuple[int, int, int, int],
    strength: float = 0.18,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 0.35))
    if strength <= 1e-6:
        return bgr_image

    h, w = bgr_image.shape[:2]
    face_mask = _build_face_mask((h, w), face_box)
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 178, 133)).astype(np.float32) / 255.0
    skin = cv2.GaussianBlur(skin, (0, 0), 1.2)
    roi = np.clip(face_mask * skin, 0.0, 1.0)
    if float(roi.sum()) < 10.0:
        return bgr_image

    smoothed = cv2.bilateralFilter(bgr_image, d=7, sigmaColor=22, sigmaSpace=7).astype(np.float32)
    base = bgr_image.astype(np.float32)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + roi * (7.0 * strength * 3.0), 0.0, 255.0)
    lit = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    roi3 = np.repeat(roi[:, :, None], 3, axis=2)
    blend = np.clip(roi3 * strength * 1.8, 0.0, 0.52)
    out = base * (1.0 - blend) + smoothed * (blend * 0.58) + lit * (blend * 0.42)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)



def enhance_image(
    bgr_image: np.ndarray,
    person_mask: np.ndarray | None = None,
    background_whitening: bool = True,
    beauty_mode: str = "none",
    face_box: tuple[int, int, int, int] | None = None,
    shadow_mode: str = "balanced",
    seg_backend: str = "mediapipe",
) -> np.ndarray:
    # Local contrast enhancement blended with original to boost face clarity without over-processing.
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(10, 10))
    l_enhanced = clahe.apply(l)
    l_blended = cv2.addWeighted(l_enhanced, 0.34, l, 0.66, 0)
    enhanced = cv2.cvtColor(cv2.merge((l_blended, a, b)), cv2.COLOR_LAB2BGR)

    # Gentle denoise for grainy mobile images.
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 1, 1, 7, 21)
    if person_mask is not None:
        # Preserve natural facial/clothing texture; avoid waxy or rough over-processing.
        pm = np.repeat(np.clip(person_mask.astype(np.float32), 0.0, 1.0)[:, :, None], 3, axis=2)
        denoised = (
            bgr_image.astype(np.float32) * pm * 0.74
            + denoised.astype(np.float32) * pm * 0.26
            + denoised.astype(np.float32) * (1.0 - pm)
        ).clip(0, 255).astype(np.uint8)

    # Preserve face brightness so "assist" mode doesn't darken skin tones.
    hsv_src = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_new = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV).astype(np.float32)
    src_v = hsv_src[:, :, 2]
    new_v = hsv_new[:, :, 2]
    if person_mask is not None:
        mask = np.clip(person_mask.astype(np.float32), 0.0, 1.0)
        weight = float(mask.sum())
        if weight > 1.0:
            src_mean = float((src_v * mask).sum() / weight)
            new_mean = float((new_v * mask).sum() / weight)
        else:
            src_mean = float(src_v.mean())
            new_mean = float(new_v.mean())
    else:
        src_mean = float(src_v.mean())
        new_mean = float(new_v.mean())
    lift = max(0.0, src_mean - new_mean)
    if lift > 2.0:
        denoised = cv2.convertScaleAbs(denoised, alpha=1.0, beta=min(18.0, lift))

    if person_mask is not None and background_whitening:
        is_rembg = (seg_backend or "").lower() == "rembg"
        # Edge-aware, gradual background cleanup to avoid fake cutout artifacts.
        # rembg already produces sharp, clean masks — skip heavy smoothing that would
        # introduce blur and halo artifacts at the subject silhouette.
        if is_rembg:
            mask = np.clip(person_mask.astype(np.float32), 0.0, 1.0)
            solid = (mask > 0.50).astype(np.uint8)
            if int(solid.sum()) > 0:
                kernel_small = np.ones((3, 3), np.uint8)
                kernel_mid = np.ones((5, 5), np.uint8)
                solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel_small, iterations=1)
                solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel_mid, iterations=1)
            core_fill = (mask > 0.86).astype(np.float32)
            mask = np.clip(np.maximum(
                mask * 0.98 + cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 0.6) * 0.02,
                core_fill * 0.97,
            ), 0.0, 1.0)
        else:
            mask = _refine_person_mask(person_mask)
        # GrabCut can over-shrink clothes when the subject already touches frame borders.
        # Skip for rembg: its neural mask already handles complex backgrounds better than GrabCut.
        border_touch_ratio = _subject_border_touch_ratio(mask, threshold=0.34)
        if border_touch_ratio < 0.10 and not is_rembg:
            mask = _refine_mask_with_grabcut(denoised, mask)
        mask = _guided_refine_mask(denoised, mask)

        # Contract uncertain matte fringe so soft wall shadows are less likely to be
        # preserved as pseudo-foreground around the silhouette.
        # rembg masks are precise — use minimal contraction to avoid clipping fine edge details.
        if is_rembg:
            contract = 0.04 if border_touch_ratio < 0.06 else 0.02
            scale = 0.94 if border_touch_ratio < 0.06 else 0.96
        else:
            contract = 0.22 if border_touch_ratio < 0.06 else 0.17
            scale = 0.70 if border_touch_ratio < 0.06 else 0.76
        subject_core = (mask > 0.86).astype(np.float32)
        subject_core = cv2.GaussianBlur(subject_core, (0, 0), 0.8)
        mask = np.clip((mask - contract) / scale, 0.0, 1.0)
        mask = np.clip(np.maximum(mask, subject_core * 0.985), 0.0, 1.0)
        bg_alpha = np.clip(1.0 - mask, 0.0, 1.0)

        # Keep background edits active near the silhouette; otherwise wall shadow is preserved.
        bg_edit_gate = np.ones_like(bg_alpha, dtype=np.float32)

        # Protect only true border-touching subject components (not the full frame edge band).
        border_subject = _build_border_touch_map(mask)
        bg_edit_gate = np.clip(bg_edit_gate * (1.0 - border_subject * 0.35), 0.70, 1.0)

        # Restrict heavy whitening/shadow edits to confident background only.
        bg_core = np.clip((bg_alpha - 0.07) / 0.93, 0.0, 1.0)
        fg = denoised.astype(np.float32)
        bg_strength = np.power(bg_core, 1.42) * bg_edit_gate
        bg_color = _estimate_background_color(fg, bg_alpha)
        bg_color_3 = np.full_like(fg, bg_color[None, None, :], dtype=np.float32)
        bg_confident = bg_alpha > 0.90
        if int(bg_confident.sum()) > 50:
            bg_pixels = fg[bg_confident]
            bg_std_mean = float(np.mean(np.std(bg_pixels, axis=0)))
            hsv_bg_src = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            bg_v_mean_src = float(hsv_bg_src[:, :, 2][bg_confident].mean())
            bg_s_mean_src = float(hsv_bg_src[:, :, 1][bg_confident].mean())
            bg_v_std_src = float(hsv_bg_src[:, :, 2][bg_confident].std())
        else:
            bg_std_mean = 999.0
            bg_v_mean_src = 0.0
            bg_s_mean_src = 255.0
            bg_v_std_src = 999.0
        is_flat_bg = bg_std_mean < 23.0
        is_clean_bg = bg_v_mean_src > 222.0 and bg_s_mean_src < 20.0 and bg_v_std_src < 18.0
        if is_clean_bg:
            bg_strength = bg_strength * 0.44

        shadow_mode_norm = (shadow_mode or "balanced").strip().lower()
        aggressive_shadow = shadow_mode_norm in {"aggressive", "strong", "high"}
        shadow_gain = 1.60 if aggressive_shadow else 1.10

        # Reduce color spill around hair and neckline by decontaminating transition pixels.
        alpha3 = np.repeat(mask[:, :, None], 3, axis=2)
        edge_band = np.clip(1.0 - np.abs(mask - 0.5) * 2.2, 0.0, 1.0)
        edge_band *= ((mask > 0.07) & (mask < 0.93)).astype(np.float32)
        denom = np.maximum(alpha3, 0.20)
        fg_unmixed = np.clip((fg - (1.0 - alpha3) * bg_color_3) / denom, 0.0, 255.0)
        # Avoid aggressive unmixed reconstruction in very low-alpha transition pixels.
        edge_mix = np.repeat((edge_band * 0.10)[:, :, None], 3, axis=2)
        alpha_gate = np.repeat((mask > 0.34)[:, :, None].astype(np.float32), 3, axis=2)
        border_subject_3 = np.repeat(border_subject[:, :, None], 3, axis=2)
        edge_mix = edge_mix * alpha_gate * (1.0 - border_subject_3 * 0.70)
        fg = fg * (1.0 - edge_mix) + fg_unmixed * edge_mix

        # High-gradient transition edges (shoulders/hairline) should resist background fill.
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        grad_ref = max(8.0, float(np.percentile(grad, 95)))
        grad_norm = np.clip(grad / grad_ref, 0.0, 1.0)
        transition = ((mask > 0.06) & (mask < 0.94)).astype(np.float32)
        edge_guard = np.clip(grad_norm * transition * 1.45, 0.0, 1.0)

        hsv = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Nudge background toward white and flatten wall shadows.
        v = v + (bg_strength * (22.0 if is_clean_bg else 38.0) * shadow_gain)
        bg_ref = float(np.percentile(v[bg_confident], 85)) if int(bg_confident.sum()) > 40 else 225.0
        shadow_lift = np.clip(bg_ref - v, 0.0, 150.0)
        v = v + (shadow_lift * bg_strength * (0.34 if is_clean_bg else 0.54) * shadow_gain)
        s = s * (1.0 - (bg_strength * (0.40 if is_clean_bg else 0.60)))

        if aggressive_shadow:
            # Additional low-frequency background equalization for stronger wall shadow cleanup.
            v_blur = cv2.GaussianBlur(v, (0, 0), 18.0)
            ref_hi = float(np.percentile(v[bg_confident], 92)) if int(bg_confident.sum()) > 40 else 238.0
            flatten_lift = np.clip(ref_hi - v_blur, 0.0, 90.0)
            v = np.clip(v + flatten_lift * bg_alpha * bg_edit_gate * 0.58, 0.0, 255.0)
            s = s * (1.0 - (bg_alpha * bg_edit_gate * 0.14))

        hsv_bg = cv2.merge((h, np.clip(s, 0, 255), np.clip(v, 0, 255))).astype(np.uint8)
        bg_adjusted = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR).astype(np.float32)
        white_target = np.full_like(bg_adjusted, 252.0)
        bg_dist = np.linalg.norm(fg - bg_color_3, axis=2)
        bg_like = np.clip(1.0 - (bg_dist / 68.0), 0.0, 1.0)
        white_mix = np.clip(
            np.power(bg_alpha, 1.04) * (0.70 + (0.14 if aggressive_shadow else 0.0))
            + (bg_like * bg_alpha * (0.22 + (0.06 if aggressive_shadow else 0.0))),
            0.0,
            0.98,
        )

        # Do not bleach saturated/textured transition regions (often clothes or hairline details).
        sat_norm = cv2.cvtColor(fg.astype(np.uint8), cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
        edge_zone = np.clip(1.0 - np.abs(mask - 0.5) * 2.6, 0.0, 1.0)
        protect = np.clip((sat_norm - 0.30) * 1.6, 0.0, 1.0) * edge_zone
        white_mix = np.clip(white_mix * (1.0 - protect * 0.75), 0.0, 0.98)
        white_mix = np.clip(white_mix * (1.0 - edge_guard * 0.24), 0.0, 0.98)
        white_mix = np.clip(white_mix * (1.0 - border_subject * 0.30), 0.0, 0.98)
        white_mix = np.clip(white_mix * bg_edit_gate, 0.0, 0.98)

        if is_flat_bg and (not is_clean_bg):
            # Flat wall: push closer to clean white and flatten cast/shadows more.
            flat_like = np.clip(1.0 - (bg_dist / 52.0), 0.0, 1.0) * bg_alpha * bg_edit_gate
            white_mix = np.clip(white_mix + flat_like * (0.28 if not aggressive_shadow else 0.42), 0.0, 0.998)
            if int(bg_confident.sum()) > 40:
                v_ref_hi = float(np.percentile(v[bg_confident], 90))
            else:
                v_ref_hi = 230.0
            extra_shadow = np.clip(v_ref_hi - v, 0.0, 160.0)
            v = np.clip(v + (extra_shadow * flat_like * (0.62 if not aggressive_shadow else 0.88)), 0.0, 255.0)
            s = s * (1.0 - (flat_like * 0.34))
            hsv_bg = cv2.merge((h, np.clip(s, 0, 255), np.clip(v, 0, 255))).astype(np.uint8)
            bg_adjusted = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR).astype(np.float32)

        white_mix_3 = np.repeat(np.clip(white_mix[:, :, None], 0.0, 1.0), 3, axis=2)
        bg_adjusted = bg_adjusted * (1.0 - white_mix_3) + white_target * white_mix_3

        # Hard-override: force confident background pixels to pure white regardless of original color.
        # This ensures any non-white background (tan, gray, colored wall, etc.) is fully replaced.
        # The ramp preserves soft blending at the subject edge (bg_alpha < 0.55 = transition zone).
        hard_bg_3 = np.repeat(np.clip((bg_alpha - 0.55) / 0.30, 0.0, 1.0)[:, :, None], 3, axis=2)
        bg_adjusted = bg_adjusted * (1.0 - hard_bg_3) + white_target * hard_bg_3

        # Flatten the immediate background-side ring so a residual shadow does not read as halo.
        bg_ring = np.clip(cv2.GaussianBlur(subject_core, (0, 0), 1.8) - subject_core, 0.0, 1.0)
        bg_ring = np.clip(bg_ring * bg_alpha, 0.0, 1.0)
        bg_ring_3 = np.repeat((bg_ring * 0.65)[:, :, None], 3, axis=2)
        bg_adjusted = bg_adjusted * (1.0 - bg_ring_3) + white_target * bg_ring_3

        # Narrow subject blend keeps natural edges but avoids preserving a dark ring.
        # rembg produces sharp masks — use a tighter blend range and smaller blur so the
        # subject edge is crisp rather than soft/glowing.
        if is_rembg:
            mask_keep = np.clip((mask - 0.30) / 0.55, 0.0, 1.0)
        else:
            mask_keep = np.clip((mask - 0.46) / 0.44, 0.0, 1.0)
        mask_keep = np.clip(np.maximum(mask_keep, subject_core * 0.995), 0.0, 1.0)
        edge_soft = cv2.GaussianBlur(mask_keep, (0, 0), 0.45 if is_rembg else 0.95)
        edge_soft_3 = np.repeat(edge_soft[:, :, None], 3, axis=2)
        denoised = (fg * edge_soft_3 + bg_adjusted * (1.0 - edge_soft_3)).clip(0, 255).astype(np.uint8)

    mode = (beauty_mode or "").lower()
    if face_box is not None and mode in {"color", "tone", "color_correction"}:
        denoised = _apply_face_color_correction(denoised, face_box=face_box, strength=0.16)
    elif face_box is not None and mode in {"soft", "soft_light", "natural"}:
        denoised = _apply_face_soft_light(denoised, face_box=face_box, strength=0.18)

    return denoised



def safe_crop(image: np.ndarray, left: int, top: int, right: int, bottom: int) -> np.ndarray:
    return image[max(0, top) : max(top, bottom), max(0, left) : max(left, right)]


def suppress_edge_artifacts(bgr_image: np.ndarray, border: int = 2) -> np.ndarray:
    border = int(max(1, border))
    h, w = bgr_image.shape[:2]
    if h <= (border * 2 + 2) or w <= (border * 2 + 2):
        return bgr_image

    out = bgr_image.copy()
    # Always stabilize the outermost ring. This removes 1px seam lines that can appear
    # after padded whitening/trim steps without changing interior content.
    out[0, :] = out[1, :]
    out[h - 1, :] = out[h - 2, :]
    out[:, 0] = out[:, 1]
    out[:, w - 1] = out[:, w - 2]

    def _should_fix(edge_strip: np.ndarray, inner_strip: np.ndarray) -> bool:
        edge_f = edge_strip.astype(np.float32)
        inner_f = inner_strip.astype(np.float32)
        edge_luma = cv2.cvtColor(edge_f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        inner_luma = cv2.cvtColor(inner_f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        edge_mean = float(edge_luma.mean())
        inner_mean = float(inner_luma.mean())
        edge_std = float(edge_luma.std())
        inner_std = float(inner_luma.std())
        color_delta = float(np.mean(np.abs(edge_f.mean(axis=(0, 1)) - inner_f.mean(axis=(0, 1)))))
        row_edge = edge_luma.mean(axis=1)
        row_inner = inner_luma.mean(axis=1)
        row_delta = np.abs(row_edge - row_inner)
        high_delta_ratio = float((row_delta > 26.0).mean())
        extreme_ratio = float(((row_edge < 6.0) | (row_edge > 249.0)).mean())
        row_band_std = float(row_edge.std())

        is_extreme_band = edge_mean < 4.0 or edge_mean > 251.0
        is_flat_band = edge_std < max(2.0, inner_std * 0.35 + 1.0)
        has_strong_jump = abs(edge_mean - inner_mean) > 32.0 or color_delta > 28.0
        # Fix only persistent full-edge strip artifacts; do not touch natural content
        # where clothes/hair legitimately reach the border.
        is_consistent_strip = row_band_std < 7.0 and (high_delta_ratio > 0.88 or extreme_ratio > 0.88)
        return is_flat_band and is_consistent_strip and (is_extreme_band or has_strong_jump)

    top_edge = out[:border, :]
    top_inner = out[border : border * 2, :]
    if _should_fix(top_edge, top_inner):
        out[:border, :] = top_inner

    bottom_edge = out[h - border :, :]
    bottom_inner = out[h - border * 2 : h - border, :]
    if _should_fix(bottom_edge, bottom_inner):
        out[h - border :, :] = bottom_inner

    left_edge = out[:, :border]
    left_inner = out[:, border : border * 2]
    if _should_fix(left_edge, left_inner):
        out[:, :border] = left_inner

    right_edge = out[:, w - border :]
    right_inner = out[:, w - border * 2 : w - border]
    if _should_fix(right_edge, right_inner):
        out[:, w - border :] = right_inner

    return out
