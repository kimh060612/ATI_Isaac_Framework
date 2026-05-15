from policy.rewards.utils import *
from policy.rewards.image_score import *
from policy.rewards.depth_score import *
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
    image_reward = np.clip(ati_laplacian_score(rgb) / 800, 0.0, 1.0)
    # motion_blur_score(rgb)
    ent = grayscale_entropy(rgb)
    ent_score = target_entropy_score(ent, target=0.70, sigma=0.18)
    sat_pen = saturation_penalty(rgb)
    smooth = edge_aware_depth_smoothness_score(rgb, inverse_depths[0])
    align = edge_alignment_score(rgb, inverse_depths[0])
    confidence = float(1. / (1 + uncertainty)) # Convert uncertainty to confidence (heuristic)
    depth_reward = 0.5 * confidence + 0.25 * smooth + 0.25 * align
    image_reward = 0.4 * image_reward + 0.4 * ent_score + 0.2 * (1 - sat_pen) 
    
    total_reward = depth_weight * depth_reward + image_weight * image_reward

    return {
        "reward": float(total_reward),
        "image_reward": float(image_reward),
        "depth_reward": float(depth_reward),
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