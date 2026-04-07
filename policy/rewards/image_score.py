import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CompositeIQAResult:
    score: float
    cpbd_quality: float
    # brisque_quality: Optional[float]
    brightness_quality: float
    details: Dict[str, float]


def _to_uint8_bgr(image: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - grayscale [H, W]
      - RGB/BGR uint8 or float image
    Returns:
      - BGR uint8 image
    """
    img = np.asarray(image)

    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # Assume input is RGB by default; if your pipeline is already BGR, remove this line.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _brightness_quality_from_histogram(
    gray_u8: np.ndarray,
    target_mean: float = 0.50,
    mean_tolerance: float = 0.35,
    low_clip_thr: int = 5,
    high_clip_thr: int = 250,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns a 0~1 brightness/exposure quality score using:
      - mean brightness closeness to target
      - clipping penalty
      - histogram entropy

    This is a practical heuristic, not a standard benchmark metric.
    """
    gray = gray_u8.astype(np.float32) / 255.0

    mean_val = float(gray.mean())
    std_val = float(gray.std())

    low_clip_ratio = float((gray_u8 <= low_clip_thr).mean())
    high_clip_ratio = float((gray_u8 >= high_clip_thr).mean())
    clipping_ratio = low_clip_ratio + high_clip_ratio

    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    hist /= (hist.sum() + 1e-12)
    entropy = float(-(hist * np.log2(hist + 1e-12)).sum())
    entropy_norm = entropy / 8.0  # max entropy for 256 bins is 8

    mean_score = max(0.0, 1.0 - abs(mean_val - target_mean) / mean_tolerance)
    clipping_score = max(0.0, 1.0 - min(1.0, clipping_ratio / 0.20))  # 20% clipped => 0
    contrast_score = min(1.0, std_val / 0.25)  # mild reward for non-flat brightness distribution

    brightness_quality = (
        0.45 * mean_score +
        0.30 * clipping_score +
        0.15 * entropy_norm +
        0.10 * contrast_score
    )
    brightness_quality = float(np.clip(brightness_quality, 0.0, 1.0))

    details = {
        "mean_brightness": mean_val,
        "brightness_std": std_val,
        "low_clip_ratio": low_clip_ratio,
        "high_clip_ratio": high_clip_ratio,
        "clipping_ratio": clipping_ratio,
        "hist_entropy_norm": entropy_norm,
        "brightness_mean_score": mean_score,
        "brightness_clipping_score": clipping_score,
        "brightness_contrast_score": contrast_score,
    }
    return brightness_quality, details


def _compute_cpbd_quality(gray_u8: np.ndarray) -> float:
    """
    Self-contained CPBD implementation based on the public python-cpbd port.

    Input:
        gray_u8: uint8 grayscale image, shape [H, W]
    Output:
        cpbd_score in [0, 1] approximately, where higher means sharper.
    """
    if gray_u8.ndim != 2:
        raise ValueError(f"Expected grayscale image [H, W], got {gray_u8.shape}")

    image = gray_u8.astype(np.float64)

    # Constants used by python-cpbd
    THRESHOLD = 0.002
    BETA = 3.6
    BLOCK_HEIGHT = 64
    BLOCK_WIDTH = 64
    WIDTH_JNB = np.concatenate([5 * np.ones(51), 3 * np.ones(205)]).astype(np.float64)

    def _simple_thinning(strength: np.ndarray) -> np.ndarray:
        num_rows, num_cols = strength.shape
        zero_column = np.zeros((num_rows, 1), dtype=strength.dtype)
        zero_row = np.zeros((1, num_cols), dtype=strength.dtype)

        x = (
            (strength > np.c_[zero_column, strength[:, :-1]]) &
            (strength > np.c_[strength[:, 1:], zero_column])
        )
        y = (
            (strength > np.r_[zero_row, strength[:-1, :]]) &
            (strength > np.r_[strength[1:, :], zero_row])
        )
        return x | y

    def _sobel_octave_like(img: np.ndarray) -> np.ndarray:
        # skimage.filters.edges.HSOBEL_WEIGHTS equivalent:
        # [[ 1, 2, 1],
        #  [ 0, 0, 0],
        #  [-1,-2,-1]] / 4  (normalization below follows python-cpbd logic)
        h1 = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float64)
        h1 /= np.sum(np.abs(h1))

        strength2 = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=h1.T)
        strength2 = np.square(strength2)

        thresh2 = 2.0 * np.sqrt(np.mean(strength2))
        strength2[strength2 <= thresh2] = 0.0
        return _simple_thinning(strength2)

    def _marziliano_method(edges: np.ndarray, img: np.ndarray) -> np.ndarray:
        edge_widths = np.zeros(img.shape, dtype=np.float64)

        gradient_y, gradient_x = np.gradient(img)
        img_height, img_width = img.shape
        edge_angles = np.zeros(img.shape, dtype=np.float64)

        for row in range(img_height):
            for col in range(img_width):
                gx = gradient_x[row, col]
                gy = gradient_y[row, col]

                if gx != 0:
                    edge_angles[row, col] = math.atan2(gy, gx) * (180.0 / math.pi)
                elif gx == 0 and gy == 0:
                    edge_angles[row, col] = 0.0
                elif gx == 0 and gy == math.pi / 2:
                    edge_angles[row, col] = 90.0

        if not np.any(edge_angles):
            return edge_widths

        quantized_angles = 45.0 * np.round(edge_angles / 45.0)

        for row in range(1, img_height - 1):
            for col in range(1, img_width - 1):
                if edges[row, col] != 1:
                    continue

                angle = quantized_angles[row, col]

                # gradient angle = 180 or -180
                if angle == 180 or angle == -180:
                    width_left = 0
                    width_right = 0

                    for margin in range(101):
                        inner_border = (col - 1) - margin
                        outer_border = (col - 2) - margin
                        if outer_border < 0 or (img[row, outer_border] - img[row, inner_border]) <= 0:
                            break
                        width_left = margin + 1

                    for margin in range(101):
                        inner_border = (col + 1) + margin
                        outer_border = (col + 2) + margin
                        if outer_border >= img_width or (img[row, outer_border] - img[row, inner_border]) >= 0:
                            break
                        width_right = margin + 1

                    edge_widths[row, col] = width_left + width_right

                # gradient angle = 0
                elif angle == 0:
                    width_left = 0
                    width_right = 0

                    for margin in range(101):
                        inner_border = (col - 1) - margin
                        outer_border = (col - 2) - margin
                        if outer_border < 0 or (img[row, outer_border] - img[row, inner_border]) >= 0:
                            break
                        width_left = margin + 1

                    for margin in range(101):
                        inner_border = (col + 1) + margin
                        outer_border = (col + 2) + margin
                        if outer_border >= img_width or (img[row, outer_border] - img[row, inner_border]) <= 0:
                            break
                        width_right = margin + 1

                    edge_widths[row, col] = width_left + width_right

        return edge_widths

    def _is_edge_block(block: np.ndarray, threshold: float) -> bool:
        return np.count_nonzero(block) > (block.size * threshold)

    def _get_block_contrast(block: np.ndarray) -> int:
        contrast = int(np.max(block) - np.min(block))
        return max(0, min(255, contrast))

    def _calculate_sharpness_metric(
        img: np.ndarray,
        edges: np.ndarray,
        edge_widths: np.ndarray
    ) -> float:
        img_height, img_width = img.shape
        total_num_edges = 0
        hist_pblur = np.zeros(101, dtype=np.float64)

        num_blocks_vertically = int(img_height / BLOCK_HEIGHT)
        num_blocks_horizontally = int(img_width / BLOCK_WIDTH)

        for i in range(num_blocks_vertically):
            for j in range(num_blocks_horizontally):
                rows = slice(BLOCK_HEIGHT * i, BLOCK_HEIGHT * (i + 1))
                cols = slice(BLOCK_WIDTH * j, BLOCK_WIDTH * (j + 1))

                if not _is_edge_block(edges[rows, cols], THRESHOLD):
                    continue

                block_widths = edge_widths[rows, cols]

                # Mimic the package behavior
                block_widths = np.rot90(np.flipud(block_widths), 3)
                block_widths = block_widths[block_widths != 0]

                if block_widths.size == 0:
                    continue

                block_contrast = _get_block_contrast(img[rows, cols])
                block_jnb = WIDTH_JNB[block_contrast]

                prob_blur_detection = 1.0 - np.exp(-np.abs(block_widths / block_jnb) ** BETA)

                for probability in prob_blur_detection:
                    bucket = int(round(float(probability) * 100))
                    bucket = max(0, min(100, bucket))
                    hist_pblur[bucket] += 1.0
                    total_num_edges += 1

        if total_num_edges > 0:
            hist_pblur /= total_num_edges

        # Same final score as python-cpbd
        return float(np.sum(hist_pblur[:64]))

    # 1) Canny edge map for block classification
    # python-cpbd uses skimage.feature.canny(image)
    # Here we use OpenCV Canny as a dependency-light substitute.
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    canny_edges = cv2.Canny(img_u8, 100, 200) > 0

    # 2) Sobel-like thinned edge map for Marziliano width
    sobel_edges = _sobel_octave_like(image)

    # 3) Edge width calculation + CPBD pooling
    marziliano_widths = _marziliano_method(sobel_edges, image)
    cpbd_score = _calculate_sharpness_metric(image, canny_edges, marziliano_widths)

    return float(np.clip(cpbd_score, 0.0, 1.0))



# brisque_model_path: Optional[str] = None,
# brisque_range_path: Optional[str] = None,
def compute_composite_image_quality(
    image: np.ndarray,
    weights: Tuple[float, float] = (0.40, 0.60),
) -> CompositeIQAResult:
    """
    Composite score in [0, 1] from:
      - CPBD sharpness
      - brightness/clipping histogram heuristic

    Args:
        image:
            RGB, BGR, or grayscale image as numpy array.
        weights:
            (w_cpbd, w_brightness)

    Returns:
        CompositeIQAResult
    """
    w_cpbd, w_brightness = weights
    w_sum = w_cpbd + w_brightness
    if w_sum <= 0:
        raise ValueError("Weights must sum to a positive value.")

    w_cpbd /= w_sum
    w_brightness /= w_sum

    bgr_u8 = _to_uint8_bgr(image)
    gray_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY)

    cpbd_quality = _compute_cpbd_quality(gray_u8)
    brightness_quality, brightness_details = _brightness_quality_from_histogram(gray_u8)

    redistributed = w_cpbd + w_brightness
    final_score = (
        (w_cpbd / redistributed) * cpbd_quality +
        (w_brightness / redistributed) * brightness_quality
    )

    final_score = float(np.clip(final_score, 0.0, 1.0))

    details = {
        "cpbd_quality": cpbd_quality,
        "brightness_quality": brightness_quality,
        **brightness_details,
    }

    return CompositeIQAResult(
        score=final_score,
        cpbd_quality=cpbd_quality,
        brightness_quality=brightness_quality,
        details=details,
    )
    

# def _compute_brisque_quality(
#     bgr_u8: np.ndarray,
#     brisque_model_path: str,
#     brisque_range_path: str,
#     brisque_score_clip: Tuple[float, float] = (0.0, 100.0),
# ) -> Tuple[float, float]:
#     """
#     Computes raw BRISQUE and maps it heuristically to 0~1 quality.

#     Requires OpenCV contrib:
#         pip install opencv-contrib-python

#     Raw BRISQUE is lower-is-better.
#     We convert it as:
#         brisque_quality = 1 - normalized(raw_brisque)
#     """
#     if not hasattr(cv2, "quality"):
#         raise ImportError(
#             "OpenCV quality module not found. Install `opencv-contrib-python`."
#         )

#     raw = cv2.quality.QualityBRISQUE_compute(
#         bgr_u8, brisque_model_path, brisque_range_path
#     )

#     # OpenCV returns cv::Scalar-like output
#     if isinstance(raw, (tuple, list, np.ndarray)):
#         raw_brisque = float(np.array(raw).ravel()[0])
#     else:
#         raw_brisque = float(raw)

#     lo, hi = brisque_score_clip
#     raw_clamped = min(max(raw_brisque, lo), hi)
#     brisque_quality = 1.0 - (raw_clamped - lo) / max(1e-8, (hi - lo))
#     brisque_quality = float(np.clip(brisque_quality, 0.0, 1.0))

#     return raw_brisque, brisque_quality
