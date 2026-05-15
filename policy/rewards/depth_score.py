import numpy as np
import cv2

def to_uint8_rgb(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def normalize01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

def gradient_magnitude(x):
    x = x.astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)

def edge_alignment_score(rgb, depth):
    img = to_uint8_rgb(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape[:2] != gray.shape[:2]:
        depth = cv2.resize(depth, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    # relative depth scale 문제 완화
    d = normalize01(depth)

    img_grad = normalize01(gradient_magnitude(gray))
    dep_grad = normalize01(gradient_magnitude(d))

    # depth edge 상위 영역만 본다
    dep_edge_mask = dep_grad > np.percentile(dep_grad, 85)

    if dep_edge_mask.mean() < 1e-4:
        return 0.0

    score = img_grad[dep_edge_mask].mean()
    return float(np.clip(score, 0.0, 1.0))

def edge_aware_depth_smoothness_score(rgb, depth):
    img = to_uint8_rgb(rgb)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape[:2] != gray.shape[:2]:
        depth = cv2.resize(depth, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    d = normalize01(depth)

    dx_d = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))
    dy_d = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))

    dx_i = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    dy_i = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    # image edge가 강하면 depth 변화 허용, textureless region에서는 depth 요동 penalize
    smooth_loss = (
        dx_d * np.exp(-10.0 * dx_i)
        + dy_d * np.exp(-10.0 * dy_i)
    ).mean()

    # loss를 score로 변환
    score = np.exp(-5.0 * smooth_loss)
    return float(np.clip(score, 0.0, 1.0))