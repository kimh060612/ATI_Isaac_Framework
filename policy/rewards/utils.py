from l3_perception_layer import TTATransform
from typing import Sequence
import numpy as np
import cv2
from PIL import Image

DEPTH_AUG_FUNCS = [
    [
        Image.FLIP_LEFT_RIGHT
    ],
    [
        Image.FLIP_TOP_BOTTOM
    ],
    [
        Image.FLIP_TOP_BOTTOM, 
        Image.FLIP_LEFT_RIGHT
    ],
]

def _to_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    return image

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    image = _to_hwc(image).astype(np.float32)
    if image.ndim == 2:
        gray = image
    else:
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.max() > 1.0:
            image = image / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray

def min_max_normalize(image: np.ndarray) -> np.ndarray:
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

def directional_gradient_score(image: np.ndarray, eps: float = 1e-8) -> float:
    """
    Directional-gradient-based motion blur score.
    Returns a scalar where a larger motion blur generally leads to a smaller value.

    Idea:
    - Compute Sobel gradients gx, gy
    - Measure total gradient energy
    - Penalize anisotropy between x/y gradient energies
    - Strong motion blur often causes directional imbalance and reduced sharpness

    Returns:
        score (float): larger is sharper / less motion-blurred
    """
    gray = _to_gray_float(image)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    ex = float(np.mean(gx ** 2))
    ey = float(np.mean(gy ** 2))

    total_energy = ex + ey
    anisotropy = abs(ex - ey) / (total_energy + eps)

    # high total gradient energy is good
    # large anisotropy is suspicious for directional motion blur
    score = total_energy * (1.0 - anisotropy)

    return float(score)


def fft_motion_blur_score(
    image: np.ndarray,
    r_min_ratio: float = 0.15,
    r_max_ratio: float = 0.45,
    num_angles: int = 180,
    eps: float = 1e-8,
) -> float:
    """
    FFT-based motion blur score.
    Returns a scalar where a larger motion blur generally leads to a smaller value.

    Idea:
    - Compute FFT magnitude spectrum
    - Focus on mid/high-frequency annulus
    - Aggregate energy by angle
    - Motion blur tends to:
        (1) reduce high-frequency energy
        (2) create stronger directional anisotropy
    - Final score = high-frequency energy / directional anisotropy penalty

    Args:
        image: input image, grayscale or BGR
        r_min_ratio: inner radius ratio for annulus
        r_max_ratio: outer radius ratio for annulus
        num_angles: number of angular bins

    Returns:
        score (float): larger is sharper / less motion-blurred
    """
    gray = _to_gray_float(image)
    h, w = gray.shape

    # Windowing to reduce FFT boundary artifacts
    win_y = np.hanning(h)
    win_x = np.hanning(w)
    window = np.outer(win_y, win_x)
    gray_win = gray * window

    # FFT magnitude
    fft = np.fft.fftshift(np.fft.fft2(gray_win))
    mag = np.abs(fft)

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    x = xx - cx
    y = yy - cy

    r = np.sqrt(x**2 + y**2)
    theta = (np.arctan2(y, x) + np.pi) % np.pi   # [0, pi), orientation only

    r_max = min(h, w) / 2.0
    r_min_thr = r_min_ratio * r_max
    r_max_thr = r_max_ratio * r_max

    # annulus mask for mid/high frequencies
    mask = (r >= r_min_thr) & (r <= r_max_thr)

    if not np.any(mask):
        return 0.0

    # angular aggregation
    angle_bins = np.linspace(0, np.pi, num_angles + 1)
    angular_energy = np.zeros(num_angles, dtype=np.float64)

    theta_m = theta[mask]
    mag_m = mag[mask]

    bin_idx = np.digitize(theta_m, angle_bins) - 1
    bin_idx = np.clip(bin_idx, 0, num_angles - 1)

    for i in range(num_angles):
        vals = mag_m[bin_idx == i]
        if len(vals) > 0:
            angular_energy[i] = vals.mean()

    mean_energy = float(np.mean(angular_energy))
    std_energy = float(np.std(angular_energy))

    # coefficient of variation: larger -> more directional anisotropy
    anisotropy = std_energy / (mean_energy + eps)

    # final score: high-frequency energy kept, but penalize strong directionality
    score = mean_energy / (1.0 + anisotropy)

    return float(score)


def motion_blur_score(
    image: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """
    Combined motion blur score from:
    - directional gradient score
    - FFT-based motion blur score

    Returns one scalar value.
    Larger motion blur -> smaller score (heuristically).

    Args:
        image: input image
        alpha: weight for directional gradient score
        beta: weight for FFT score

    Returns:
        score (float)
    """
    g_score = directional_gradient_score(image)
    f_score = fft_motion_blur_score(image)

    # geometric-style fusion for robustness
    score = (max(g_score, eps) ** alpha) * (max(f_score, eps) ** beta)
    return float(score) # 

def compute_tta_uncertainty(
    inverse_depths: Sequence[np.ndarray],
    reduction: str = "mean",
) -> tuple[np.ndarray, float]:
    if not inverse_depths:
        raise ValueError("inverse_depths must not be empty")

    stacked = np.stack([np.asarray(depth, dtype=np.float32) for depth in inverse_depths], axis=0)
    variance_map = np.var(stacked, axis=0)
    variance_map = min_max_normalize(variance_map)
    
    if reduction == "mean":
        uncertainty = float(np.mean(variance_map))
    elif reduction == "p90":
        uncertainty = float(np.percentile(variance_map, 90.0))
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return variance_map, uncertainty



# def _bounded_score(value: float, scale: float) -> float:
#     value = max(float(value), 0.0)
#     scale = max(float(scale), 1e-6)
#     return value / (value + scale)

# def _normalize_float_map(image: np.ndarray) -> np.ndarray:
#     image = _to_hwc(image).astype(np.float32)
#     if image.ndim == 3:
#         if image.shape[-1] == 1:
#             image = image[..., 0]
#         else:
#             image = np.mean(image, axis=-1)
#     image = np.squeeze(image)
#     if image.ndim != 2:
#         raise ValueError(f"Expected a single-channel map, got shape {image.shape}")

#     valid = np.isfinite(image)
#     if not np.any(valid):
#         return np.zeros_like(image, dtype=np.float32)

#     value_min = float(np.percentile(image[valid], 2.0))
#     value_max = float(np.percentile(image[valid], 98.0))
#     if value_max - value_min < 1e-6:
#         value_min = float(np.min(image[valid]))
#         value_max = float(np.max(image[valid]))

#     normalized = (image - value_min) / (value_max - value_min + 1e-6)
#     normalized = np.clip(normalized, 0.0, 1.0)
#     normalized[~valid] = 0.0
#     return normalized.astype(np.float32)


# def _resize_to_match(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
#     if image.shape != target_shape:
#         image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
#     return image


# def _gradient_magnitude(image: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
#     image = np.asarray(image, dtype=np.float32)
#     if blur_ksize > 1:
#         image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

#     grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
#     grad_mag = cv2.magnitude(grad_x, grad_y)

#     valid = np.isfinite(grad_mag)
#     if not np.any(valid):
#         return np.zeros_like(image, dtype=np.float32)

#     scale = float(np.percentile(grad_mag[valid], 95.0))
#     if scale < 1e-6:
#         scale = float(np.max(grad_mag[valid]))
#     if scale < 1e-6:
#         return np.zeros_like(image, dtype=np.float32)

#     return np.clip(grad_mag / (scale + 1e-6), 0.0, 1.0)


# def _edge_mask(edge_strength: np.ndarray, percentile: float = 75.0, min_threshold: float = 0.1) -> tuple[np.ndarray, float]:
#     values = edge_strength[np.isfinite(edge_strength)]
#     values = values[values > 0.0]
#     if values.size == 0:
#         return np.zeros_like(edge_strength, dtype=bool), float(min_threshold)

#     threshold = float(np.percentile(values, percentile))
#     threshold = max(threshold, float(min_threshold))
#     return edge_strength >= threshold, threshold