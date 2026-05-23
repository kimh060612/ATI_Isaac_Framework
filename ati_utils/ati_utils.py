from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle

import argparse
from PIL import Image
import traceback
import wandb
import numpy as np


def initialize_wandb(
    policy_type,
    l3_model_name,
    context_len, 
    max_laps, 
    max_steps, 
    exp_name=None
):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_mde_simulation",
        name=exp_name,
        config={
            "policy_type": policy_type,
            "turn_per_lap": context_len,
            "max_laps": max_laps,
            "max_steps": max_steps,
            "l3_mde_model": l3_model_name,
        },
    )

def select_reward_function(reward_type: str):
    if reward_type == "flipped":
        return reward_flipped_img
    elif reward_type == "test_time_augment":
        return reward_test_time_augment
    elif reward_type == "oracle":
        return reward_oracle
    else:
        raise ValueError(f"Invalid reward_type: {reward_type}. Must be one of ['flipped', 'test_time_augment', 'oracle']")

def build_observation_info(
    reward_type: str, 
    rgb_image: np.ndarray, 
    pred_depths, 
    metric_info: dict = None
) -> dict:
    if reward_type == "flipped":
        return {
            "original_rgb": np.array(rgb_image),
            "depth_original": pred_depths[0],
            "depth_flipped": pred_depths[1],
            "image_weight": 0.0,
            "depth_weight": 1.0,
        }
    elif reward_type == "test_time_augment":
        return {
            "rgb": np.array(rgb_image),
            "inverse_depths": pred_depths,
            "uncertainty_reduction": "mean",
            "image_weight": 0.4,
            "depth_weight": 0.6,
        }
    elif reward_type == "oracle":
        return {
            "original_rgb": np.array(rgb_image),
            "abs_rel_error": metric_info["abs_rel"],
            "delta_1": metric_info["a1"],
            "image_weight": 0.0,
            "depth_weight": 1.0,
        }
    else:
        raise ValueError(f"Invalid reward_type: {reward_type}. Must be one of ['flipped', 'test_time_augment', 'oracle']")

def get_avg_aggregation(reward_obs: list[dict]) -> dict:
    if not reward_obs:
        return {
            "reward": 0.0,
            "image_reward": 0.0,
            "depth_reward": 0.0,
            "uncertainty": 0.0,
        }
    return {
        key: float(np.mean([
            metric[key] for metric in reward_obs
        ])) 
        for key in reward_obs[0].keys()
    }