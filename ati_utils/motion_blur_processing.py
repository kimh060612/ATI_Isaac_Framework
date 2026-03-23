import numpy as np
import cv2


def realtime_like_motion_blur(
    image: np.ndarray,
    flow: np.ndarray,
    exposure_time_fraction: float = 0.5,
    blur_diameter_fraction: float = 0.02,
    num_samples: int = 8,
    centered: bool = True,
) -> np.ndarray:
    """
    Approximate NVIDIA Omniverse real-time post-process motion blur semantics.

    Args:
        image:
            RGB image, shape (H, W, 3), dtype uint8 or float32/float64.
        flow:
            Screen-space motion vectors in pixels per frame, shape (H, W, 2).
            flow[..., 0] = dx, flow[..., 1] = dy.
        exposure_time_fraction:
            Fraction of one frame duration to sample.
            1.0 means one frame duration.
        blur_diameter_fraction:
            Fraction of the largest screen dimension used as maximum blur diameter.
        num_samples:
            Number of samples used in the blur filter.
        centered:
            If True, sample symmetrically around the current pixel.
            If False, sample from current pixel toward motion direction only.

    Returns:
        Blurred RGB image with same dtype family as input.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"flow must have shape (H, W, 2), got {flow.shape}")

    if image.shape[:2] != flow.shape[:2]:
        raise ValueError("image and flow must have the same height/width")

    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    original_dtype = image.dtype
    img = image.astype(np.float32)

    # Normalize image to 0~1 for stable accumulation
    if img.max() > 1.0:
        img = img / 255.0

    h, w = img.shape[:2]
    max_dim = float(max(h, w))

    # Omniverse semantics:
    # - exposure_time_fraction: fraction of one frame duration
    # - blur_diameter_fraction: maximum blur diameter relative to largest screen dimension
    max_blur_diameter_px = blur_diameter_fraction * max_dim
    max_blur_radius_px = 0.5 * max_blur_diameter_px

    # Effective motion vector during the exposure interval
    eff_flow = flow.astype(np.float32) * float(exposure_time_fraction)

    # Clamp blur radius to maximum allowed by blur_diameter_fraction
    mag = np.linalg.norm(eff_flow, axis=2, keepdims=True) + 1e-8
    clamped_mag = np.minimum(mag, max_blur_radius_px)
    eff_flow = eff_flow * (clamped_mag / mag)

    # Base sampling grid
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    accum = np.zeros_like(img, dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)

    if num_samples == 1:
        return image.copy()

    if centered:
        # Symmetric line integral around current pixel: [-0.5, +0.5]
        ts = np.linspace(-0.5, 0.5, num_samples, dtype=np.float32)
    else:
        # Forward-only line integral: [0, 1]
        ts = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

    # Simple box filter along the motion direction
    for t in ts:
        sample_x = xs + eff_flow[..., 0] * t
        sample_y = ys + eff_flow[..., 1] * t

        sampled = cv2.remap(
            img,
            sample_x,
            sample_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        accum += sampled
        weight_sum += 1.0

    out = accum / np.maximum(weight_sum, 1e-8)

    # Convert back to original dtype style
    if np.issubdtype(original_dtype, np.integer):
        out = np.clip(out * 255.0, 0, 255).astype(original_dtype)
    else:
        out = out.astype(original_dtype)

    return out