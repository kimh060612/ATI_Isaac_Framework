from policy.rewards.utils import *
from l3_perception_layer import TTATransform
from PIL import Image

def reward_oracle(
    original_rgb,
    abs_rel_error,
    delta_1,
    image_weight: float = 0.1,
    depth_weight: float = 0.9,
):
    sharp_original = motion_blur_score(original_rgb) # _bounded_score( , scale=0.01)
    depth_reward = (1.0 - min(abs_rel_error, 1.0)) / 2 + delta_1 / 2
    total_reward = image_weight * sharp_original + depth_weight * depth_reward
    
    return {
        "reward": float(total_reward),
        "image_reward": float(sharp_original),
        "depth_reward": float(depth_reward),
        "uncertainty": float(np.exp(-depth_reward)),
    }

def reward_flipped_img(
    original_rgb,
    depth_original,
    depth_flipped,
    image_weight: float = 0.1,
    depth_weight: float = 0.9,
):
    sharp_original = motion_blur_score(original_rgb) # _bounded_score( , scale=0.01)
    depth_flipped = Image.fromarray(depth_flipped).transpose(Image.FLIP_LEFT_RIGHT)
    depth_flipped = np.asarray(depth_flipped)
    depth_diff = np.abs(depth_original - depth_flipped)
    depth_diff = (depth_diff - np.min(depth_diff)) / (np.max(depth_diff) - np.min(depth_diff) + 1e-6)
    depth_diff = float(np.mean(depth_diff))
    depth_confidence = float(1. / (1 + depth_diff))
    
    total_reward = image_weight * sharp_original + depth_weight * depth_confidence
    
    return {
        "reward": float(total_reward),
        "image_reward": float(sharp_original),
        "depth_reward": float(depth_confidence),
        "uncertainty": float(depth_diff),
    }

def reward_test_time_augment(
    rgb: np.ndarray,
    inverse_depths: list[np.ndarray],
    uncertainty_reduction: str,
    image_weight: float = 0.1,
    depth_weight: float = 0.9,
) -> dict:
    _, uncertainty = compute_tta_uncertainty(
        inverse_depths=inverse_depths,
        reduction=uncertainty_reduction,
    )
    image_reward = motion_blur_score(rgb) # compute_composite_image_quality(rgb).score
    # motion_blur_score(rgb)
    confidence = float(1. / (1 + uncertainty)) # Convert uncertainty to confidence (heuristic)
    total_reward = image_weight * image_reward + depth_weight * confidence

    return {
        "reward": float(total_reward),
        "image_reward": float(image_reward),
        "depth_reward": float(confidence),
        "uncertainty": float(uncertainty),
    }

def reward_classification_confidence(
    rgb_image: np.ndarray,
    confidence: float,
    image_weight: float = 0.1,
    task_weight: float = 0.9,
) -> dict:
    image_reward = motion_blur_score(rgb_image)
    total_reward = image_weight * image_reward + task_weight * confidence

    return {
        "reward": float(total_reward),
        "image_reward": float(image_reward),
        "task_reward": float(confidence),
        "uncertainty": float(1 - confidence),  # Higher confidence means lower uncertainty
    }
    
def reward_classification_oracle(
    rgb_image: np.ndarray,
    correct: float,
    image_weight: float = 0.1,
    task_weight: float = 0.9,
) -> dict:
    image_reward = motion_blur_score(rgb_image)
    task_reward = correct
    total_reward = image_weight * image_reward + task_weight * task_reward

    return {
        "reward": float(total_reward),
        "image_reward": float(image_reward),
        "task_reward": float(task_reward),
        "uncertainty": float(1 - correct),  # Higher correctness means lower uncertainty
    }   