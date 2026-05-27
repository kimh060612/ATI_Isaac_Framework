"""
Lap-level epsilon-greedy sensor-control training in Isaac Sim.

This script follows the Android ATI control protocol more closely than the
per-frame bandit runners: M simulation steps define one lap, camera settings
are held for that lap, and the policy is updated once from an aggregated lap
reward.  Context is always derived from IMU + scene light, while the scenario
trajectory is used only to drive the robot and lighting schedule.
"""

## Basic imports: Must be here.
from isaacsim import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,
    "renderer": "RaytracedLighting",
    "display_options": 3286,
}
simulation_app = SimulationApp(launch_config=CONFIG)

import argparse
import json
import os
import random
import traceback

import numpy as np
import torch
from PIL import Image

try:
    import wandb
except Exception:
    wandb = None

from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from ati_utils.log_utils import configure_isaac_sim_logging, save_synthetic_data
from ati_utils.ati_utils import *
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy import (
    L2SharedEGreedyRGBCamPolicy,
    SensorParamSpace,
    episode_bank,
    load_heuristic_memory,
    save_heuristic_memory,
)
from scene import ATIDepthScene


RANDOM_SEED = 42
VERBOSE = True
DEBUG = True


DEFAULT_SPEED_RANGES = "SLOW:0.0:0.4,NORMAL:0.9:1.1,FAST:1.5:1.7,SUPER_FAST:1.9:2.1"
DEFAULT_LIGHT_RANGES = "DARK:100:300,DIM:500:600,NORMAL:1000:1100,BRIGHT:4000:4200,SUPER_BRIGHT:9000:9200"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATI lap-level epsilon-greedy sensor control in Isaac Sim")
    parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_egreedy_oracle")
    parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
    parser.add_argument("--data_path", type=str, default="/home/kimh060612/ATI_research/dataset")
    parser.add_argument("--num_episode", type=int, default=200, help="Number of training laps repeated per context.")
    parser.add_argument("--lap_period", type=int, default=20, help="M simulation steps that define one lap.")
    parser.add_argument("--num_inference_episode", type=int, default=1, help="Number of inference laps repeated per context.")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--policy_variant", type=str, default="egreedy", choices=["egreedy"])
    parser.add_argument("--epsilon_start", type=float, default=0.5)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--initial_expected_reward", type=float, default=1.0)
    parser.add_argument("--exp_ratio", type=float, default=1.0, help="Kept for BaseCMABPolicy compatibility.")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lap_reward_top_percent", type=float, default=20.0, help="Average the top K percent of rewards in a lap.")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Simulation steps before policy learning/evaluation starts.")
    parser.add_argument("--heuristic_update_threshold", type=int, default=30)
    parser.add_argument("--heuristic_blend_alpha", type=float, default=0.35)
    parser.add_argument("--heuristic_memory_path", type=str, default=None)
    parser.add_argument(
        "--checkpoint_episode_interval",
        type=int,
        default=1,
        help="Save a checkpoint after every N completed contexts. Kept as a legacy argument name.",
    )
    parser.add_argument("--scenario_repeat_type", type=str, default="sin", choices=["sin", "step"])
    parser.add_argument("--speed_ranges", type=str, default=DEFAULT_SPEED_RANGES)
    parser.add_argument("--light_ranges", type=str, default=DEFAULT_LIGHT_RANGES)
    parser.add_argument("--angular_speed_scale", type=float, default=float(np.pi / 12), help="Scale scenario speed before robot_control.")
    parser.add_argument(
        "--imu_speed_context_scale",
        type=float,
        default=None,
        help="Scale observed gyro_magnitude into the scenario speed unit. Defaults to 1 / angular_speed_scale.",
    )
    parser.add_argument("--tie_break_random", action="store_true", help="Randomly break equal-score ties during inference.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    return parser


def parse_named_ranges(value: str) -> list[tuple[str, tuple[float, float]]]:
    ranges = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid range item '{item}'. Expected NAME:LOW:HIGH.")
        label, low, high = parts
        low_f = float(low)
        high_f = float(high)
        if high_f < low_f:
            raise ValueError(f"Invalid range '{item}': HIGH must be >= LOW.")
        ranges.append((label, (low_f, high_f)))
    if not ranges:
        raise ValueError("At least one scenario range is required.")
    return ranges


def build_context_ranges(args) -> list[dict]:
    speed_ranges = parse_named_ranges(args.speed_ranges)
    light_ranges = parse_named_ranges(args.light_ranges)
    context_ranges = []
    for speed_label, speed_range in speed_ranges:
        for light_label, light_range in light_ranges:
            context_ranges.append(
                {
                    "name": f"{speed_label}_{light_label}",
                    "speed_label": speed_label,
                    "light_label": light_label,
                    "speed_range": speed_range,
                    "light_range": light_range,
                }
            )
    return context_ranges


def midpoint_thresholds(named_ranges: list[tuple[str, tuple[float, float]]]) -> tuple[float, ...]:
    return tuple(
        float((named_ranges[idx][1][1] + named_ranges[idx + 1][1][0]) / 2.0)
        for idx in range(len(named_ranges) - 1)
    )


def build_policy_thresholds(args) -> tuple[tuple[float, ...], tuple[float, ...], dict]:
    speed_ranges = parse_named_ranges(args.speed_ranges)
    light_ranges = parse_named_ranges(args.light_ranges)
    if len(speed_ranges) > len(L2SharedEGreedyRGBCamPolicy.MOTION_STATES):
        raise ValueError(
            "EGreedyPolicy supports at most "
            f"{len(L2SharedEGreedyRGBCamPolicy.MOTION_STATES)} motion states, got {len(speed_ranges)}."
        )
    if len(light_ranges) > len(L2SharedEGreedyRGBCamPolicy.LIGHT_STATES):
        raise ValueError(
            "EGreedyPolicy supports at most "
            f"{len(L2SharedEGreedyRGBCamPolicy.LIGHT_STATES)} light states, got {len(light_ranges)}."
        )

    range_limits = {
        "motion_min": float(min(item[1][0] for item in speed_ranges)),
        "motion_max": float(max(item[1][1] for item in speed_ranges)),
        "light_min": float(min(item[1][0] for item in light_ranges)),
        "light_max": float(max(item[1][1] for item in light_ranges)),
    }
    return midpoint_thresholds(speed_ranges), midpoint_thresholds(light_ranges), range_limits


def flatten_metrics(metrics: dict | None, prefix: str) -> dict:
    if not metrics:
        return {}
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            payload[f"{prefix}/{key}"] = float(value)
    return payload


def print_device_debug(requested_device: str) -> None:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        "[Device] requested={} torch={} cuda_available={} cuda_device_count={} "
        "CUDA_VISIBLE_DEVICES={}".format(
            requested_device,
            torch.__version__,
            torch.cuda.is_available(),
            torch.cuda.device_count(),
            cuda_visible_devices,
        )
    )
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            print(f"[Device] cuda:{device_idx} name={torch.cuda.get_device_name(device_idx)}")


def get_policy_context(scene: ATIDepthScene, syn_data: dict | None = None) -> dict:
    syn_data = syn_data or {}
    context = syn_data.get("canonical_context")
    if isinstance(context, dict):
        return context
    return scene.get_policy_context(
        imu_frame=syn_data.get("imu_sensor"),
        light_meter_info=syn_data.get("light_meter"),
    )


def resolve_imu_speed_context_scale(args) -> float:
    if args.imu_speed_context_scale is not None:
        return float(args.imu_speed_context_scale)
    if abs(float(args.angular_speed_scale)) <= 1e-12:
        return 1.0
    return float(1.0 / float(args.angular_speed_scale))


def build_egreedy_context(
    raw_policy_context: dict,
    *,
    fallback_light: float,
    fallback_speed: float,
    imu_speed_context_scale: float,
) -> dict:
    gyro = float(raw_policy_context.get("gyro_magnitude", np.nan))
    angular_velocity = abs(gyro) * float(imu_speed_context_scale)
    if not np.isfinite(angular_velocity):
        angular_velocity = abs(float(fallback_speed))

    light_intensity = float(raw_policy_context.get("light_intensity", fallback_light))
    if not np.isfinite(light_intensity):
        light_intensity = float(fallback_light)

    return {
        **raw_policy_context,
        "angular_velocity": float(angular_velocity),
        "light_intensity": float(light_intensity),
        "imu_gyro_magnitude": float(0.0 if not np.isfinite(gyro) else gyro),
        "imu_speed_context_scale": float(imu_speed_context_scale),
    }


def summarize_context_samples(samples: list[dict]) -> dict:
    if not samples:
        return {
            "angular_velocity": 0.0,
            "light_intensity": 1.0,
            "acceleration_magnitude": 0.0,
            "gyro_magnitude": 0.0,
            "imu_gyro_magnitude": 0.0,
            "num_context_samples": 0.0,
        }

    numeric_keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, (int, float, np.integer, np.floating, bool))
        }
    )
    summary = {
        key: float(np.mean([float(sample.get(key, 0.0)) for sample in samples]))
        for key in numeric_keys
    }
    summary["angular_velocity"] = float(np.mean([float(sample["angular_velocity"]) for sample in samples]))
    summary["light_intensity"] = float(np.mean([float(sample["light_intensity"]) for sample in samples]))
    summary["num_context_samples"] = float(len(samples))
    return summary


def top_percent_reward_info(reward_history: list[dict], top_percent: float) -> dict:
    if not reward_history:
        return {"reward": 0.0}

    top_percent = float(np.clip(top_percent, 0.0, 100.0))
    if top_percent <= 0.0:
        top_percent = 100.0
    rewards = np.asarray([float(item.get("reward", 0.0)) for item in reward_history], dtype=np.float64)
    top_count = max(1, int(np.ceil(len(rewards) * top_percent / 100.0)))
    top_indices = np.argsort(rewards)[-top_count:]

    keys = sorted(
        {
            key
            for item in reward_history
            for key, value in item.items()
            if isinstance(value, (int, float, np.integer, np.floating, bool))
        }
    )
    payload = {
        key: float(np.mean([float(reward_history[int(idx)].get(key, 0.0)) for idx in top_indices]))
        for key in keys
    }
    payload["reward"] = float(np.mean(rewards[top_indices]))
    payload["top_percent"] = float(top_percent)
    payload["top_count"] = float(top_count)
    payload["num_lap_rewards"] = float(len(reward_history))
    return payload


def mean_numeric_metrics(metrics_history: list[dict]) -> dict:
    if not metrics_history:
        return {}
    keys = sorted(
        {
            key
            for item in metrics_history
            for key, value in item.items()
            if isinstance(value, (int, float, np.integer, np.floating, bool))
        }
    )
    return {
        key: float(np.mean([float(item.get(key, 0.0)) for item in metrics_history]))
        for key in keys
    }


def make_egreedy_policy(
    args,
    sensor_param_space: SensorParamSpace,
    reward_function,
    motion_thresholds: tuple[float, ...],
    light_thresholds: tuple[float, ...],
    range_limits: dict,
    *,
    random_seed: int,
) -> L2SharedEGreedyRGBCamPolicy:
    policy = L2SharedEGreedyRGBCamPolicy(
        sensor_names="agent_camera",
        sensor_config=sensor_param_space,
        reward_function=reward_function,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        initial_expected_reward=args.initial_expected_reward,
        motion_thresholds=motion_thresholds,
        light_thresholds=light_thresholds,
        alpha=args.exp_ratio,
        lambda_reg=args.lambda_reg,
        random_seed=random_seed,
    )
    policy.motion_min = float(range_limits["motion_min"])
    policy.motion_max = float(range_limits["motion_max"])
    policy.light_min = float(range_limits["light_min"])
    policy.light_max = float(range_limits["light_max"])
    return policy


def policy_metadata(
    args,
    motion_thresholds: tuple[float, ...],
    light_thresholds: tuple[float, ...],
    range_limits: dict,
    heuristic_memory_path: str,
) -> dict:
    return {
        "policy_type": "L2SharedEGreedyRGBCamPolicy",
        "policy_variant": args.policy_variant,
        "reward_type": args.reward_type,
        "epsilon_start": args.epsilon_start,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "learning_rate": args.learning_rate,
        "initial_expected_reward": args.initial_expected_reward,
        "alpha": args.exp_ratio,
        "lambda_reg": args.lambda_reg,
        "lap_period": args.lap_period,
        "training_laps_per_context": args.num_episode,
        "inference_laps_per_context": args.num_inference_episode,
        "lap_reward_top_percent": args.lap_reward_top_percent,
        "warmup_steps": args.warmup_steps,
        "heuristic_update_threshold": args.heuristic_update_threshold,
        "heuristic_blend_alpha": args.heuristic_blend_alpha,
        "heuristic_memory_path": heuristic_memory_path,
        "speed_ranges": args.speed_ranges,
        "light_ranges": args.light_ranges,
        "motion_thresholds": list(motion_thresholds),
        "light_thresholds": list(light_thresholds),
        "range_limits": range_limits,
        "context_source": "imu_gyro_magnitude_plus_scene_light",
        "imu_speed_context_scale": resolve_imu_speed_context_scale(args),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def egreedy_policy_payload(policy: L2SharedEGreedyRGBCamPolicy, include_history: bool = False) -> dict:
    payload = {
        "policy_type": type(policy).__name__,
        "actions": [
            {
                "action": [int(action[0]), int(action[1])],
                "description": policy.ACTION_DESCRIPTIONS.get(action, str(action)),
            }
            for action in policy.action_space
        ],
        "expected_rewards": {
            state: values.tolist()
            for state, values in sorted(policy.expected_rewards.items())
        },
        "action_counts": {
            state: counts.astype(int).tolist()
            for state, counts in sorted(policy.action_counts.items())
        },
        "epsilon": float(policy.current_epsilon),
        "epsilon_start": float(policy.epsilon_start),
        "epsilon_min": float(policy.epsilon_min),
        "epsilon_decay": float(policy.epsilon_decay),
        "learning_rate": float(policy.learning_rate),
        "initial_expected_reward": float(policy.initial_expected_reward),
        "motion_thresholds": list(policy.motion_thresholds),
        "light_thresholds": list(policy.light_thresholds),
        "motion_min": float(policy.motion_min),
        "motion_max": float(policy.motion_max),
        "light_min": float(policy.light_min),
        "light_max": float(policy.light_max),
        "stats": policy.get_stats(),
    }
    if include_history:
        payload["history"] = policy.history
    return payload


def save_policy_bundle(
    checkpoint_path: str,
    policy: L2SharedEGreedyRGBCamPolicy,
    metadata: dict,
    heuristic_offsets: dict[str, tuple[float, float]],
    state_update_counts: dict[str, int],
) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "metadata": metadata,
        "policy": egreedy_policy_payload(policy),
        "heuristic_offsets": {
            state: [float(offset[0]), float(offset[1])]
            for state, offset in sorted(heuristic_offsets.items())
        },
        "state_update_counts": {
            state: int(count)
            for state, count in sorted(state_update_counts.items())
        },
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
    print(f"[Checkpoint] Saved EGreedy policy to {checkpoint_path}")
    return checkpoint_path


def make_checkpoint(
    policy: L2SharedEGreedyRGBCamPolicy,
    data_path: str,
    checkpoint_idx: int,
    metadata: dict,
    heuristic_offsets: dict[str, tuple[float, float]],
    state_update_counts: dict[str, int],
) -> str:
    checkpoint_path = os.path.join(
        data_path,
        "checkpoints",
        f"egreedy_policy_context_{checkpoint_idx + 1:04d}.json",
    )
    return save_policy_bundle(
        checkpoint_path=checkpoint_path,
        policy=policy,
        metadata=metadata,
        heuristic_offsets=heuristic_offsets,
        state_update_counts=state_update_counts,
    )


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {"num_steps": 0, "mean_reward": 0.0, "per_scenario": {}}

    summary = {
        "num_steps": len(records),
        "mean_reward": float(np.mean([record["reward"] for record in records])),
        "per_scenario": {},
    }
    scenario_names = sorted({record["scenario_name"] for record in records})
    for scenario_name in scenario_names:
        scenario_records = [record for record in records if record["scenario_name"] == scenario_name]
        reward_keys = sorted(
            {
                key
                for record in scenario_records
                for key, value in record["reward_info"].items()
                if isinstance(value, (int, float, np.integer, np.floating, bool))
            }
        )
        metric_keys = sorted(
            {
                key
                for record in scenario_records
                for key, value in record["metric_info"].items()
                if isinstance(value, (int, float, np.integer, np.floating, bool))
            }
        )
        summary["per_scenario"][scenario_name] = {
            "num_steps": len(scenario_records),
            "mean_reward": float(np.mean([record["reward"] for record in scenario_records])),
            "reward_info": {
                key: float(np.mean([record["reward_info"][key] for record in scenario_records]))
                for key in reward_keys
            },
            "metric_info": {
                key: float(np.mean([record["metric_info"][key] for record in scenario_records]))
                for key in metric_keys
            },
        }
    return summary


def save_eval_outputs(data_path: str, records: list[dict], summary: dict) -> tuple[str, str]:
    os.makedirs(data_path, exist_ok=True)
    records_path = os.path.join(data_path, "inference_records.jsonl")
    summary_path = os.path.join(data_path, "inference_summary.json")
    with open(records_path, "w", encoding="utf-8") as records_file:
        for record in records:
            records_file.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(json_ready(summary), summary_file, indent=2, sort_keys=True)
    return records_path, summary_path


def reset_sensor_to_midpoint(scene: ATIDepthScene, sensor_param_space: SensorParamSpace) -> tuple[int, int]:
    exposure_idx = len(sensor_param_space.exposure_values) // 2
    iso_idx = len(sensor_param_space.iso_values) // 2
    scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[iso_idx],
            "shutter_time": sensor_param_space.exposure_values[exposure_idx],
        }
    )
    return exposure_idx, iso_idx


def select_and_apply_lap_action(
    *,
    scene: ATIDepthScene,
    policy: L2SharedEGreedyRGBCamPolicy,
    sensor_param_space: SensorParamSpace,
    context_summary: dict,
    heuristic_offsets: dict[str, tuple[float, float]],
    tie_break_random: bool,
    is_infer_mode: bool,
) -> tuple[dict, int, int, int, int, str]:
    base_exposure_idx, base_iso_idx, base_state = policy.calculate_base_indices(
        context_information=context_summary,
        heuristic_offsets=heuristic_offsets,
    )
    selection = policy.select_action(
        context_information={
            **context_summary,
            "exposure_idx": base_exposure_idx,
            "iso_idx": base_iso_idx,
        },
        tie_break_random=tie_break_random,
        is_infer_mode=is_infer_mode,
    )
    next_exposure_idx, next_iso_idx = policy.transition(
        base_exposure_idx,
        base_iso_idx,
        selection["chosen_action"],
    )
    scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[next_iso_idx],
            "shutter_time": sensor_param_space.exposure_values[next_exposure_idx],
        }
    )
    return selection, base_exposure_idx, base_iso_idx, next_exposure_idx, next_iso_idx, base_state


def initialize_lap_context(
    *,
    scene: ATIDepthScene,
    scenario,
    angular_speed_scale: float,
    imu_speed_context_scale: float,
) -> dict:
    env_context = scenario.step(0)
    scene.control_light_intensity(env_context["light_intensity"])
    scene.robot_control(
        time=scene.get_simulation_current_time,
        control_parameters={
            "angular_velocity": float(env_context["angular_velocity"]) * float(angular_speed_scale),
        },
    )
    syn_data = scene.step(render=False)
    raw_context = get_policy_context(scene, syn_data if isinstance(syn_data, dict) else None)
    return build_egreedy_context(
        raw_context,
        fallback_light=float(env_context["light_intensity"]),
        fallback_speed=float(env_context["angular_velocity"]),
        imu_speed_context_scale=imu_speed_context_scale,
    )


def run_warmup(
    *,
    scene: ATIDepthScene,
    context_ranges: list[dict],
    args,
) -> None:
    if args.warmup_steps <= 0 or not context_ranges:
        return
    warmup_scenario = episode_bank(
        context_ranges=context_ranges[:1],
        scenario_period=max(args.warmup_steps, 1),
        repeat_type=args.scenario_repeat_type,
        random_seed=RANDOM_SEED,
        shuffle=False,
    )[0]
    print(f"[Warmup] Running {args.warmup_steps} simulation steps before lap training.")
    for warmup_step in range(args.warmup_steps):
        if not simulation_app._app.is_running() or simulation_app.is_exiting():
            break
        env_context = warmup_scenario.step(warmup_step)
        scene.control_light_intensity(env_context["light_intensity"])
        scene.robot_control(
            time=scene.get_simulation_current_time,
            control_parameters={
                "angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
            },
        )
        scene.step(render=False)


def run_lap(
    *,
    phase: str,
    scene: ATIDepthScene,
    scenario,
    lap_idx: int,
    context_idx: int,
    context_lap_idx: int,
    global_step: int,
    args,
    sensor_param_space: SensorParamSpace,
    mde_model: L3PLayerDepthAnythingv2,
    reward_function,
    l3_mde_config: L3MDEConfig,
    imu_speed_context_scale: float,
    curr_exposure_idx: int,
    curr_iso_idx: int,
) -> tuple[dict, list[dict], dict, int, int, int, dict]:
    reward_history: list[dict] = []
    metric_history: list[dict] = []
    context_samples: list[dict] = []
    step_records: list[dict] = []
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
        "imu": [],
    }

    for lap_step in range(args.lap_period):
        if not simulation_app._app.is_running() or simulation_app.is_exiting():
            break

        env_context = scenario.step(lap_step)
        scene.control_light_intensity(env_context["light_intensity"])
        scene.robot_control(
            time=scene.get_simulation_current_time,
            control_parameters={
                "angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
            },
        )

        syn_data = scene.step(render=True)
        rgb_image = syn_data.get("rgb", None)
        gt_depth = syn_data.get(scene.get_anno("depth"), None)
        bbox_data = syn_data.get(scene.get_anno("2d_bounding_box"), None)
        imu_data = syn_data.get("imu_sensor", None)
        raw_policy_context = get_policy_context(scene, syn_data)
        policy_context = build_egreedy_context(
            raw_policy_context,
            fallback_light=float(env_context["light_intensity"]),
            fallback_speed=float(env_context["angular_velocity"]),
            imu_speed_context_scale=imu_speed_context_scale,
        )
        context_samples.append(policy_context)

        if rgb_image is None or rgb_image.size == 0:
            if VERBOSE:
                print(f"Warning: Received empty RGB image during {phase}. Skipping step reward.")
            global_step += 1
            continue
        if gt_depth is None or gt_depth.size == 0:
            if VERBOSE:
                print(f"Warning: Received empty ground-truth depth image during {phase}. Skipping step reward.")
            global_step += 1
            continue

        syn_data_cache["rgb"].append(rgb_image)
        syn_data_cache["depth"].append(gt_depth)
        syn_data_cache["bbox"].append(bbox_data)
        syn_data_cache["imu"].append(imu_data)

        pred_depths, metric_info = mde_model.predict_depth([Image.fromarray(rgb_image)], gt_depth)
        syn_data_cache["pred_depth"].append(
            pred_depths if isinstance(pred_depths, np.ndarray) else pred_depths[0]
        )
        observation_info = build_observation_info(
            reward_type=l3_mde_config.reward_type,
            rgb_image=rgb_image,
            pred_depths=pred_depths,
            metric_info=metric_info,
        )
        reward_info = reward_function(**observation_info)
        reward = float(reward_info.get("reward", 0.0))
        reward_history.append(reward_info)
        metric_history.append(metric_info)

        step_record = {
            "phase": phase,
            "lap_idx": lap_idx,
            "episode_idx": context_lap_idx,
            "context_idx": context_idx,
            "context_lap_idx": context_lap_idx,
            "lap_step": lap_step,
            "global_step": global_step,
            "scenario_name": scenario.name,
            "scenario_speed_label": scenario.speed_label,
            "scenario_light_label": scenario.light_label,
            "scenario_light_intensity": float(env_context["light_intensity"]),
            "scenario_speed": float(env_context["angular_velocity"]),
            "scenario_robot_angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
            "context_acceleration_magnitude": float(policy_context.get("acceleration_magnitude", 0.0)),
            "context_gyro_magnitude": float(policy_context.get("gyro_magnitude", 0.0)),
            "context_light_intensity": float(policy_context.get("light_intensity", 0.0)),
            "context_egreedy_angular_velocity": float(policy_context.get("angular_velocity", 0.0)),
            "exposure_idx": curr_exposure_idx,
            "iso_idx": curr_iso_idx,
            "reward": reward,
            "reward_info": reward_info,
            "metric_info": metric_info,
        }
        step_records.append(step_record)
        global_step += 1

    lap_reward_info = top_percent_reward_info(reward_history, args.lap_reward_top_percent)
    lap_metric_info = mean_numeric_metrics(metric_history)
    lap_context = summarize_context_samples(context_samples)
    lap_summary = {
        "phase": phase,
        "lap_idx": lap_idx,
        "episode_idx": context_lap_idx,
        "context_idx": context_idx,
        "context_lap_idx": context_lap_idx,
        "scenario_name": scenario.name,
        "scenario_speed_label": scenario.speed_label,
        "scenario_light_label": scenario.light_label,
        "num_valid_reward_steps": len(reward_history),
        "num_context_samples": len(context_samples),
        "reward": float(lap_reward_info.get("reward", 0.0)),
        "reward_info": lap_reward_info,
        "metric_info": lap_metric_info,
        "context_summary": lap_context,
        "exposure_idx": curr_exposure_idx,
        "iso_idx": curr_iso_idx,
    }
    return lap_summary, step_records, syn_data_cache, global_step, curr_exposure_idx, curr_iso_idx, lap_context


def main() -> None:
    args = build_parser().parse_args()
    set_deterministic(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print_device_debug(args.device)

    context_ranges = build_context_ranges(args)
    motion_thresholds, light_thresholds, range_limits = build_policy_thresholds(args)
    imu_speed_context_scale = resolve_imu_speed_context_scale(args)

    data_path = os.path.join(
        args.data_path,
        f"experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps_policy_{args.policy_variant}",
    )
    os.makedirs(data_path, exist_ok=True)
    heuristic_memory_path = args.heuristic_memory_path or os.path.join(
        data_path,
        f"heuristic_offsets_{args.exp_name}_{args.reward_type}.json",
    )

    configure_isaac_sim_logging()
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_egreedy_rendering",
        robot_config=kaya_config,
    )
    render_config.set_agent_sensor_controller("exposure_iso_controller")
    render_config.set_rendering_mode("realtime")
    render_config.set_pathtracing_param(spp=128, num_subsamples=32)
    scene = ATIDepthScene(
        simulation_app,
        config=render_config,
        physics_dt=render_config.physics_dt,
        rendering_dt=render_config.rendering_dt,
        stage_units_in_meters=render_config.stage_units_in_meters,
    )

    sensor_param_space = SensorParamSpace()
    curr_exposure_idx, curr_iso_idx = reset_sensor_to_midpoint(scene, sensor_param_space)
    run_warmup(scene=scene, context_ranges=context_ranges, args=args)

    reward_function = select_reward_function(args.reward_type)
    policy = make_egreedy_policy(
        args,
        sensor_param_space,
        reward_function,
        motion_thresholds,
        light_thresholds,
        range_limits,
        random_seed=RANDOM_SEED,
    )
    heuristic_offsets, state_update_counts = load_heuristic_memory(heuristic_memory_path)
    print(f"[LTM] Loaded {len(heuristic_offsets)} heuristic offsets from {heuristic_memory_path}")

    l3_mde_config = L3MDEConfig(
        reward_type=args.reward_type,
        model_name="depth-anything/Depth-Anything-V2-Small-hf",
        shift_ratios=[],
        zoom_factors=[],
        gaussian_noise_stds=(0.01, 0.02, 0.05),
        brightness_factors=(),
        color_jitter_strengths=(0.1, 0.15),
        disable_hflip=False,
        prediction_mode="identity",
    )
    mde_model = L3PLayerDepthAnythingv2(l3_config=l3_mde_config, device=args.device)
    print(f"[Device] mde_device={mde_model.device} mde_pipeline_device={getattr(mde_model.model, 'device', '<unknown>')}")

    train_max_laps = args.num_episode * len(context_ranges)
    infer_max_laps = args.num_inference_episode * len(context_ranges)
    max_steps = (train_max_laps + infer_max_laps) * args.lap_period
    wandb_run = None
    if not args.disable_wandb:
        wandb_run = initialize_wandb(
            policy_type=args.policy_variant,
            l3_model_name=l3_mde_config.model_name,
            context_len=args.lap_period,
            max_laps=train_max_laps + infer_max_laps,
            max_steps=max_steps,
            exp_name=args.exp_name,
        )

    global_step = 0
    train_lap_idx = 0
    inference_records: list[dict] = []

    try:
        training_scenarios = episode_bank(
            context_ranges=context_ranges,
            scenario_period=args.lap_period,
            repeat_type=args.scenario_repeat_type,
            random_seed=RANDOM_SEED,
            shuffle=True,
        )
        for context_idx, scenario in enumerate(training_scenarios):
            for context_lap_idx in range(args.num_episode):
                initial_context = initialize_lap_context(
                    scene=scene,
                    scenario=scenario,
                    angular_speed_scale=args.angular_speed_scale,
                    imu_speed_context_scale=imu_speed_context_scale,
                )
                selection, base_exposure_idx, base_iso_idx, curr_exposure_idx, curr_iso_idx, base_state = select_and_apply_lap_action(
                    scene=scene,
                    policy=policy,
                    sensor_param_space=sensor_param_space,
                    context_summary=initial_context,
                    heuristic_offsets=heuristic_offsets,
                    tie_break_random=True,
                    is_infer_mode=False,
                )

                lap_summary, step_records, syn_data_cache, global_step, curr_exposure_idx, curr_iso_idx, lap_context = run_lap(
                    phase="train",
                    scene=scene,
                    scenario=scenario,
                    lap_idx=train_lap_idx,
                    context_idx=context_idx,
                    context_lap_idx=context_lap_idx,
                    global_step=global_step,
                    args=args,
                    sensor_param_space=sensor_param_space,
                    mde_model=mde_model,
                    reward_function=reward_function,
                    l3_mde_config=l3_mde_config,
                    imu_speed_context_scale=imu_speed_context_scale,
                    curr_exposure_idx=curr_exposure_idx,
                    curr_iso_idx=curr_iso_idx,
                )

                lap_reward = float(lap_summary["reward"])
                update_info = None
                if lap_summary["num_valid_reward_steps"] > 0:
                    update_info = policy.update_parameters(
                        (str(selection["state"]), int(selection["chosen_action_idx"])),
                        lap_reward,
                    )
                    if policy.update_long_term_memory(
                        heuristic_offsets=heuristic_offsets,
                        state_update_counts=state_update_counts,
                        update_info=update_info,
                        update_threshold=args.heuristic_update_threshold,
                        blend_alpha=args.heuristic_blend_alpha,
                    ):
                        save_heuristic_memory(
                            heuristic_memory_path,
                            heuristic_offsets,
                            state_update_counts,
                        )
                        lap_summary["ltm_updated"] = True
                    else:
                        lap_summary["ltm_updated"] = False

                lap_summary.update(
                    {
                        "base_state": base_state,
                        "state": selection["state"],
                        "base_exposure_idx": base_exposure_idx,
                        "base_iso_idx": base_iso_idx,
                        "action": selection["chosen_action"],
                        "action_description": selection["chosen_description"],
                        "chosen_expected_reward": float(selection["chosen_expected_reward"]),
                        "chosen_count": int(selection["chosen_count"]),
                        "epsilon": float(selection["epsilon"]),
                        "selection_mode": selection["mode"],
                        "context_idx": context_idx,
                        "context_lap_idx": context_lap_idx,
                        "laps_per_context": args.num_episode,
                        "update_info": update_info,
                    }
                )
                policy.history.append(lap_summary)

                if VERBOSE:
                    print(
                        f"Train Lap {train_lap_idx} | Context {context_idx + 1}/{len(training_scenarios)} | "
                        f"Context Lap {context_lap_idx + 1}/{args.num_episode} | "
                        f"Scenario: {scenario.name} | State: {selection['state']} | "
                        f"Action: {selection['chosen_action']} | "
                        f"Top-{args.lap_reward_top_percent:g}% Reward: {lap_reward:.4f} | "
                        f"Epsilon: {selection['epsilon']:.4f}"
                    )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            **flatten_metrics(lap_summary["reward_info"], f"train/{scenario.name}"),
                            **flatten_metrics(lap_summary["metric_info"], f"train/{scenario.name}"),
                            **flatten_metrics(update_info, f"train/{scenario.name}"),
                            **flatten_metrics(lap_context, f"train/{scenario.name}/context"),
                            f"train/{scenario.name}/lap_reward": lap_reward,
                            f"train/{scenario.name}/base_exposure_idx": base_exposure_idx,
                            f"train/{scenario.name}/base_iso_idx": base_iso_idx,
                            f"train/{scenario.name}/exposure_idx": curr_exposure_idx,
                            f"train/{scenario.name}/iso_idx": curr_iso_idx,
                            f"train/{scenario.name}/chosen_expected_reward": float(selection["chosen_expected_reward"]),
                            f"train/{scenario.name}/chosen_count": int(selection["chosen_count"]),
                            f"train/{scenario.name}/epsilon": float(selection["epsilon"]),
                            f"train/{scenario.name}/explore_flag": float(selection["mode"] == "explore"),
                            f"train/{scenario.name}/ltm_updated": float(bool(lap_summary.get("ltm_updated", False))),
                            "global_step": global_step,
                            "context_idx": context_idx,
                            "context_lap_idx": context_lap_idx,
                            "lap_idx": train_lap_idx,
                        }
                    )

                if args.save_data and any(len(values) > 0 for values in syn_data_cache.values()):
                    save_synthetic_data(
                        DATA_PATH=data_path,
                        syn_data=syn_data_cache,
                        lap_idx=train_lap_idx,
                    )

                train_lap_idx += 1

            checkpoint_metadata = {
                **policy_metadata(args, motion_thresholds, light_thresholds, range_limits, heuristic_memory_path),
                "context_idx": context_idx,
                "scenario_name": scenario.name,
                "completed_train_laps": train_lap_idx,
                "l3_mde_config": l3_mde_config.__dict__,
            }
            if args.checkpoint_episode_interval > 0 and (context_idx + 1) % args.checkpoint_episode_interval == 0:
                make_checkpoint(
                    policy=policy,
                    data_path=data_path,
                    checkpoint_idx=context_idx,
                    metadata=checkpoint_metadata,
                    heuristic_offsets=heuristic_offsets,
                    state_update_counts=state_update_counts,
                )

        final_metadata = {
            **policy_metadata(args, motion_thresholds, light_thresholds, range_limits, heuristic_memory_path),
            "completed_contexts": len(training_scenarios),
            "completed_train_laps": train_lap_idx,
            "l3_mde_config": l3_mde_config.__dict__,
        }
        save_heuristic_memory(heuristic_memory_path, heuristic_offsets, state_update_counts)
        save_policy_bundle(
            checkpoint_path=os.path.join(data_path, "checkpoints", "egreedy_policy_final.json"),
            policy=policy,
            metadata=final_metadata,
            heuristic_offsets=heuristic_offsets,
            state_update_counts=state_update_counts,
        )

        if not args.skip_inference and args.num_inference_episode > 0:
            curr_exposure_idx, curr_iso_idx = reset_sensor_to_midpoint(scene, sensor_param_space)
            run_warmup(scene=scene, context_ranges=context_ranges, args=args)
            inference_lap_idx = 0
            inference_scenarios = episode_bank(
                context_ranges=context_ranges,
                scenario_period=args.lap_period,
                repeat_type=args.scenario_repeat_type,
                random_seed=RANDOM_SEED,
                shuffle=False,
            )
            for inference_context_idx, scenario in enumerate(inference_scenarios):
                for inference_context_lap_idx in range(args.num_inference_episode):
                    initial_context = initialize_lap_context(
                        scene=scene,
                        scenario=scenario,
                        angular_speed_scale=args.angular_speed_scale,
                        imu_speed_context_scale=imu_speed_context_scale,
                    )
                    selection, base_exposure_idx, base_iso_idx, curr_exposure_idx, curr_iso_idx, base_state = select_and_apply_lap_action(
                        scene=scene,
                        policy=policy,
                        sensor_param_space=sensor_param_space,
                        context_summary=initial_context,
                        heuristic_offsets=heuristic_offsets,
                        tie_break_random=args.tie_break_random,
                        is_infer_mode=True,
                    )

                    lap_summary, step_records, syn_data_cache, global_step, curr_exposure_idx, curr_iso_idx, lap_context = run_lap(
                        phase="inference",
                        scene=scene,
                        scenario=scenario,
                        lap_idx=inference_lap_idx,
                        context_idx=inference_context_idx,
                        context_lap_idx=inference_context_lap_idx,
                        global_step=global_step,
                        args=args,
                        sensor_param_space=sensor_param_space,
                        mde_model=mde_model,
                        reward_function=reward_function,
                        l3_mde_config=l3_mde_config,
                        imu_speed_context_scale=imu_speed_context_scale,
                        curr_exposure_idx=curr_exposure_idx,
                        curr_iso_idx=curr_iso_idx,
                    )
                    lap_summary.update(
                        {
                            "base_state": base_state,
                            "state": selection["state"],
                            "base_exposure_idx": base_exposure_idx,
                            "base_iso_idx": base_iso_idx,
                            "action": selection["chosen_action"],
                            "action_description": selection["chosen_description"],
                            "chosen_expected_reward": float(selection["chosen_expected_reward"]),
                            "chosen_count": int(selection["chosen_count"]),
                            "epsilon": float(selection["epsilon"]),
                            "selection_mode": selection["mode"],
                            "context_idx": inference_context_idx,
                            "context_lap_idx": inference_context_lap_idx,
                            "laps_per_context": args.num_inference_episode,
                        }
                    )
                    inference_records.append(lap_summary)

                    if VERBOSE:
                        print(
                            f"Infer Lap {inference_lap_idx} | Context {inference_context_idx + 1}/{len(inference_scenarios)} | "
                            f"Context Lap {inference_context_lap_idx + 1}/{args.num_inference_episode} | "
                            f"Scenario: {scenario.name} | State: {selection['state']} | "
                            f"Action: {selection['chosen_action']} | "
                            f"Top-{args.lap_reward_top_percent:g}% Reward: {float(lap_summary['reward']):.4f}"
                        )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                **flatten_metrics(lap_summary["reward_info"], f"inference/{scenario.name}"),
                                **flatten_metrics(lap_summary["metric_info"], f"inference/{scenario.name}"),
                                **flatten_metrics(lap_context, f"inference/{scenario.name}/context"),
                                f"inference/{scenario.name}/lap_reward": float(lap_summary["reward"]),
                                f"inference/{scenario.name}/base_exposure_idx": base_exposure_idx,
                                f"inference/{scenario.name}/base_iso_idx": base_iso_idx,
                                f"inference/{scenario.name}/exposure_idx": curr_exposure_idx,
                                f"inference/{scenario.name}/iso_idx": curr_iso_idx,
                                f"inference/{scenario.name}/chosen_expected_reward": float(selection["chosen_expected_reward"]),
                                f"inference/{scenario.name}/chosen_count": int(selection["chosen_count"]),
                                "global_step": global_step,
                                "inference_context_idx": inference_context_idx,
                                "inference_context_lap_idx": inference_context_lap_idx,
                                "inference_lap_idx": inference_lap_idx,
                            }
                        )

                    if args.save_data and any(len(values) > 0 for values in syn_data_cache.values()):
                        save_synthetic_data(
                            DATA_PATH=data_path,
                            syn_data=syn_data_cache,
                            lap_idx=train_lap_idx + inference_lap_idx,
                        )

                    inference_lap_idx += 1

            summary = summarize_records(inference_records)
            records_path, summary_path = save_eval_outputs(
                os.path.join(data_path, "inference"),
                inference_records,
                summary,
            )
            print(f"[Inference] mean_lap_reward={summary['mean_reward']:.4f} num_laps={summary['num_steps']}")
            print(f"[Inference] Saved records to {records_path}")
            print(f"[Inference] Saved summary to {summary_path}")

    except KeyboardInterrupt:
        print("[Main] Caught Keyboard Interrupt Command, Shutting Down...")
    except Exception as exc:
        print("[Error]", exc)
        traceback.print_exc()
    finally:
        try:
            save_heuristic_memory(heuristic_memory_path, heuristic_offsets, state_update_counts)
        except Exception:
            pass
        if wandb_run is not None:
            wandb_run.finish()
        print("[Main] Shutting down...")
        try:
            simulation_app.close()
        except Exception:
            print("")
        print("[Main] Done. If process hangs, run: pkill -f 'python.sh|kit'")


if __name__ == "__main__":
    main()
