from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class TTATransform:
    name: str
    kind: str
    value: float = 0.0

def _min_max_normalize(image: np.ndarray) -> np.ndarray:
    valid = np.isfinite(image)
    if not np.any(valid):
        return np.zeros_like(image, dtype=np.float32)

    value_min = float(np.percentile(image[valid], 2.0))
    value_max = float(np.percentile(image[valid], 98.0))
    if value_max - value_min < 1e-6:
        value_min = float(np.min(image[valid]))
        value_max = float(np.max(image[valid]))

    normalized = (image - value_min) / (value_max - value_min + 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid] = 0.0
    return normalized.astype(np.float32)


def default_recommended_tta_transforms(
    include_identity: bool = True,
    include_hflip: bool = True,
    shift_ratios: Sequence[float] = (-0.03, 0.03),
    zoom_factors: Sequence[float] = (0.95, 1.05),
    gaussian_noise_stds: Sequence[float] = (),
    brightness_factors: Sequence[float] = (),
    color_jitter_strengths: Sequence[float] = (),
) -> list[TTATransform]:
    transforms: list[TTATransform] = []

    if include_identity:
        transforms.append(TTATransform(name="identity", kind="identity"))
    if include_hflip:
        transforms.append(TTATransform(name="hflip", kind="hflip"))

    for ratio in shift_ratios:
        sign = "pos" if ratio >= 0 else "neg"
        transforms.append(
            TTATransform(
                name=f"xshift_{sign}_{abs(int(round(ratio * 100))):02d}",
                kind="xshift",
                value=float(ratio),
            )
        )

    for factor in zoom_factors:
        transforms.append(
            TTATransform(
                name=f"zoom_{int(round(factor * 100)):03d}",
                kind="zoom",
                value=float(factor),
            )
        )

    for std in gaussian_noise_stds:
        transforms.append(
            TTATransform(
                name=f"gaussian_noise_{int(round(std * 1000)):03d}",
                kind="gaussian_noise",
                value=float(std),
            )
        )

    for factor in brightness_factors:
        transforms.append(
            TTATransform(
                name=f"brightness_{int(round(factor * 100)):03d}",
                kind="brightness",
                value=float(factor),
            )
        )

    for strength in color_jitter_strengths:
        magnitude = abs(float(strength))
        transforms.append(
            TTATransform(
                name=f"color_jitter_pos_{int(round(magnitude * 100)):03d}",
                kind="color_jitter",
                value=magnitude,
            )
        )
        transforms.append(
            TTATransform(
                name=f"color_jitter_neg_{int(round(magnitude * 100)):03d}",
                kind="color_jitter",
                value=-magnitude,
            )
        )

    return transforms


def build_tta_inference_batch(
    images: Sequence[Image.Image | np.ndarray],
    transforms: Sequence[TTATransform] | None = None,
) -> tuple[list[Image.Image], list[TTATransform]]:
    transforms = list(transforms or default_recommended_tta_transforms())
    infer_list: list[Image.Image] = []

    for image in images:
        for transform in transforms:
            infer_list.append(apply_tta_transform(image, transform))

    return infer_list, transforms


def invert_tta_depth_predictions(
    predictions: Sequence[dict | np.ndarray],
    transforms: Sequence[TTATransform],
    num_original_images: int,
) -> list[list[np.ndarray]]:
    transforms = list(transforms)
    expected = num_original_images * len(transforms)
    if len(predictions) != expected:
        raise ValueError(
            f"Expected {expected} predictions for {num_original_images} images and "
            f"{len(transforms)} transforms, got {len(predictions)}"
        )

    inverse_depths: list[list[np.ndarray]] = []
    for image_idx in range(num_original_images):
        start = image_idx * len(transforms)
        grouped: list[np.ndarray] = []
        for offset, transform in enumerate(transforms):
            pred = predictions[start + offset]
            depth = np.asarray(pred["depth"] if isinstance(pred, dict) else pred)
            grouped.append(invert_tta_depth_transform(depth, transform))
        inverse_depths.append(grouped)

    return inverse_depths


def compute_tta_uncertainty(
    inverse_depths: Sequence[np.ndarray],
    reduction: str = "mean",
) -> tuple[np.ndarray, float]:
    if not inverse_depths:
        raise ValueError("inverse_depths must not be empty")

    stacked = np.stack([np.asarray(depth, dtype=np.float32) for depth in inverse_depths], axis=0)
    variance_map = np.var(stacked, axis=0)
    variance_map = _min_max_normalize(variance_map)
    
    if reduction == "mean":
        uncertainty = float(np.mean(variance_map))
    elif reduction == "p90":
        uncertainty = float(np.percentile(variance_map, 90.0))
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return variance_map, uncertainty


def apply_tta_transform(
    image: Image.Image | np.ndarray,
    transform: TTATransform,
) -> Image.Image:
    image_np = _to_numpy_image(image)
    transformed = _apply_numpy_transform(image_np, transform, inverse=False, is_depth=False)
    return Image.fromarray(transformed)


def invert_tta_depth_transform(
    depth: np.ndarray,
    transform: TTATransform,
) -> np.ndarray:
    depth_np = np.asarray(depth, dtype=np.float32)
    return _apply_numpy_transform(depth_np, transform, inverse=True, is_depth=True)


def _to_numpy_image(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape [H, W, 3], got {image.shape}")

    return image


def _apply_numpy_transform(
    array: np.ndarray,
    transform: TTATransform,
    *,
    inverse: bool,
    is_depth: bool,
) -> np.ndarray:
    kind = transform.kind

    if kind == "identity":
        return array.copy()
    if kind == "hflip":
        return np.ascontiguousarray(np.flip(array, axis=1))
    if kind == "xshift":
        ratio = -transform.value if inverse else transform.value
        return _translate_x(array, ratio=ratio, is_depth=is_depth)
    if kind == "zoom":
        factor = (1.0 / transform.value) if inverse else transform.value
        return _zoom_center(array, scale=factor, is_depth=is_depth)
    if kind == "gaussian_noise":
        if inverse or is_depth:
            return array.copy()
        return _add_gaussian_noise(array, std=transform.value)
    if kind == "brightness":
        if inverse or is_depth:
            return array.copy()
        return _adjust_brightness(array, factor=transform.value)
    if kind == "color_jitter":
        if inverse or is_depth:
            return array.copy()
        return _apply_color_jitter(array, strength=transform.value)

    raise ValueError(f"Unsupported TTA transform kind: {kind}")


def _translate_x(array: np.ndarray, ratio: float, *, is_depth: bool) -> np.ndarray:
    h, w = array.shape[:2]
    tx = float(ratio) * float(w)

    matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
    interpolation = cv2.INTER_LINEAR if is_depth else cv2.INTER_CUBIC
    border_mode = cv2.BORDER_REFLECT_101

    shifted = cv2.warpAffine(
        array,
        matrix,
        (w, h),
        flags=interpolation,
        borderMode=border_mode,
    )
    return shifted.astype(array.dtype, copy=False)


def _zoom_center(array: np.ndarray, scale: float, *, is_depth: bool) -> np.ndarray:
    h, w = array.shape[:2]
    interpolation = cv2.INTER_LINEAR if is_depth else cv2.INTER_CUBIC

    if abs(scale - 1.0) < 1e-6:
        return array.copy()

    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(array, (new_w, new_h), interpolation=interpolation)

    if scale >= 1.0:
        start_x = max((new_w - w) // 2, 0)
        start_y = max((new_h - h) // 2, 0)
        cropped = resized[start_y : start_y + h, start_x : start_x + w]
        if cropped.shape[:2] != (h, w):
            cropped = cv2.resize(cropped, (w, h), interpolation=interpolation)
        return cropped.astype(array.dtype, copy=False)

    pad_w = max(w - new_w, 0)
    pad_h = max(h - new_h, 0)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if padded.shape[:2] != (h, w):
        padded = cv2.resize(padded, (w, h), interpolation=interpolation)
    return padded.astype(array.dtype, copy=False)


def _add_gaussian_noise(array: np.ndarray, std: float) -> np.ndarray:
    image = array.astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=max(float(std), 0.0), size=image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0.0, 1.0)
    return (noisy * 255.0).round().astype(np.uint8)


def _adjust_brightness(array: np.ndarray, factor: float) -> np.ndarray:
    image = array.astype(np.float32) / 255.0
    adjusted = np.clip(image * max(float(factor), 0.0), 0.0, 1.0)
    return (adjusted * 255.0).round().astype(np.uint8)


def _apply_color_jitter(array: np.ndarray, strength: float) -> np.ndarray:
    magnitude = abs(float(strength))
    direction = 1.0 if strength >= 0.0 else -1.0

    contrast_factor = max(0.1, 1.0 + direction * 0.35 * magnitude)
    saturation_factor = max(0.1, 1.0 + direction * 0.45 * magnitude)

    image = Image.fromarray(array)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    return np.asarray(image, dtype=np.uint8)

def select_eval_prediction(
    inverse_depths: list[np.ndarray],
    transforms: list[TTATransform],
    prediction_mode: str,
) -> np.ndarray:
    if prediction_mode == "mean":
        return np.mean(np.stack(inverse_depths, axis=0), axis=0)

    for idx, transform in enumerate(transforms):
        if transform.kind == "identity":
            return inverse_depths[idx]

    raise ValueError('prediction_mode="identity" requires an identity transform in the TTA set.')


__all__ = [
    "TTATransform",
    "apply_tta_transform",
    "build_tta_inference_batch",
    "compute_tta_uncertainty",
    "default_recommended_tta_transforms",
    "invert_tta_depth_predictions",
    "invert_tta_depth_transform",
    "select_eval_prediction",
]
