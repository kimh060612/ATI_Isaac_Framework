"""
Scenario-wise epsilon-greedy sensor-control training in Isaac Sim.

Each coarse scenario owns an epsilon-greedy table policy during training.  At
checkpoint/evaluation time the scenario policies are consolidated into one
expected-reward table, then that consolidated policy is used for inference
without further updates.
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
)
from scene import ATIDepthScene


RANDOM_SEED = 42
VERBOSE = True
DEBUG = True


DEFAULT_SPEED_RANGES = "SLOW:0.0:0.4,NORMAL:0.9:1.1,FAST:1.5:1.7,SUPER_FAST:1.9:2.1"
DEFAULT_LIGHT_RANGES = "DARK:100:300,DIM:500:600,NORMAL:1000:1100,BRIGHT:4000:4200,SUPER_BRIGHT:9000:9200"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATI scenario-wise epsilon-greedy sensor control in Isaac Sim")
    parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_egreedy_oracle")
    parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
    parser.add_argument("--data_path", type=str, default="/home/kimh060612/ATI_research/dataset")
    parser.add_argument("--num_episode", type=int, default=200, help="Number of full scenario-bank training passes.")
    parser.add_argument("--lap_period", type=int, default=200, help="Rendered policy-training steps per scenario episode.")
    parser.add_argument("--num_inference_episode", type=int, default=1, help="Full scenario-bank inference passes after training.")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--policy_variant", type=str, default="egreedy", choices=["egreedy"])
    parser.add_argument("--epsilon_start", type=float, default=0.5)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--initial_expected_reward", type=float, default=1.0)
    parser.add_argument("--exp_ratio", type=float, default=1.0, help="Kept for BaseCMABPolicy compatibility.")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--checkpoint_episode_interval", type=int, default=1)
    parser.add_argument("--scenario_repeat_type", type=str, default="sin", choices=["sin", "step"])
    parser.add_argument("--speed_ranges", type=str, default=DEFAULT_SPEED_RANGES)
    parser.add_argument("--light_ranges", type=str, default=DEFAULT_LIGHT_RANGES)
    parser.add_argument("--angular_speed_scale", type=float, default=float(np.pi / 12), help="Scale scenario speed before robot_control.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--speed_context_source",
        type=str,
        default="scenario",
        choices=["scenario", "imu"],
        help="Use scenario angular velocity or scaled IMU gyro magnitude for EGreedy state binning.",
    )
    parser.add_argument(
        "--imu_speed_context_scale",
        type=float,
        default=None,
        help="Scale observed gyro_magnitude into the scenario speed unit. Defaults to 1 / angular_speed_scale.",
    )
    parser.add_argument("--tie_break_random", action="store_true", help="Randomly break equal-score ties during inference.")
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
    env_context: dict,
    *,
    speed_context_source: str,
    imu_speed_context_scale: float,
) -> dict:
    fallback_speed = float(env_context.get("angular_velocity", 0.0))
    if speed_context_source == "imu":
        gyro = float(raw_policy_context.get("gyro_magnitude", np.nan))
        angular_velocity = abs(gyro) * float(imu_speed_context_scale)
        if not np.isfinite(angular_velocity):
            angular_velocity = fallback_speed
    else:
        angular_velocity = fallback_speed

    light_intensity = float(raw_policy_context.get("light_intensity", env_context.get("light_intensity", 1.0)))
    if not np.isfinite(light_intensity):
        light_intensity = float(env_context.get("light_intensity", 1.0))

    return {
        **raw_policy_context,
        "angular_velocity": float(angular_velocity),
        "light_intensity": float(light_intensity),
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


def policy_metadata(args, motion_thresholds: tuple[float, ...], light_thresholds: tuple[float, ...], range_limits: dict) -> dict:
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
        "speed_ranges": args.speed_ranges,
        "light_ranges": args.light_ranges,
        "motion_thresholds": list(motion_thresholds),
        "light_thresholds": list(light_thresholds),
        "range_limits": range_limits,
        "speed_context_source": args.speed_context_source,
        "imu_speed_context_scale": resolve_imu_speed_context_scale(args),
    }


def consolidate_egreedy_policies(
    scenario_policies: dict[str, L2SharedEGreedyRGBCamPolicy],
    consolidated_policy: L2SharedEGreedyRGBCamPolicy,
) -> L2SharedEGreedyRGBCamPolicy:
    all_states = set(consolidated_policy.expected_rewards.keys())
    for policy in scenario_policies.values():
        all_states.update(policy.expected_rewards.keys())

    total_update_count = 0
    epsilons = []
    for policy in scenario_policies.values():
        total_update_count += int(policy.update_count)
        epsilons.append(float(policy.current_epsilon))

    for state in sorted(all_states):
        weighted_rewards = np.zeros(consolidated_policy.num_actions, dtype=np.float64)
        merged_counts = np.zeros(consolidated_policy.num_actions, dtype=np.int64)
        for policy in scenario_policies.values():
            policy._ensure_state(state)
            counts = policy.action_counts[state].astype(np.int64)
            rewards = policy.expected_rewards[state].astype(np.float64)
            weighted_rewards += rewards * counts
            merged_counts += counts

        rewards = np.full(
            consolidated_policy.num_actions,
            consolidated_policy.initial_expected_reward,
            dtype=np.float64,
        )
        observed_mask = merged_counts > 0
        rewards[observed_mask] = weighted_rewards[observed_mask] / merged_counts[observed_mask]
        consolidated_policy.expected_rewards[state] = rewards
        consolidated_policy.action_counts[state] = merged_counts

    consolidated_policy.update_count = total_update_count
    consolidated_policy.current_epsilon = float(np.mean(epsilons)) if epsilons else consolidated_policy.epsilon_min
    consolidated_policy.pending_update = None
    return consolidated_policy


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
    consolidated_policy: L2SharedEGreedyRGBCamPolicy,
    scenario_policies: dict[str, L2SharedEGreedyRGBCamPolicy],
    metadata: dict,
) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "metadata": metadata,
        "consolidated_policy": egreedy_policy_payload(consolidated_policy),
        "scenario_policies": {
            name: egreedy_policy_payload(policy)
            for name, policy in sorted(scenario_policies.items())
        },
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
    print(f"[Checkpoint] Saved EGreedy policy bundle to {checkpoint_path}")
    return checkpoint_path


def make_checkpoint(
    consolidated_policy: L2SharedEGreedyRGBCamPolicy,
    scenario_policies: dict[str, L2SharedEGreedyRGBCamPolicy],
    data_path: str,
    episode_idx: int,
    metadata: dict,
) -> str:
    checkpoint_path = os.path.join(
        data_path,
        "checkpoints",
        f"egreedy_policy_episode_{episode_idx + 1:04d}.json",
    )
    return save_policy_bundle(
        checkpoint_path=checkpoint_path,
        consolidated_policy=consolidated_policy,
        scenario_policies=scenario_policies,
        metadata=metadata,
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

    reward_function = select_reward_function(args.reward_type)
    scenario_policies = {
        scenario_config["name"]: make_egreedy_policy(
            args,
            sensor_param_space,
            reward_function,
            motion_thresholds,
            light_thresholds,
            range_limits,
            random_seed=RANDOM_SEED + idx,
        )
        for idx, scenario_config in enumerate(context_ranges)
    }
    consolidated_policy = make_egreedy_policy(
        args,
        sensor_param_space,
        reward_function,
        motion_thresholds,
        light_thresholds,
        range_limits,
        random_seed=RANDOM_SEED,
    )

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

    train_max_steps = args.num_episode * len(context_ranges) * args.lap_period
    infer_max_steps = args.num_inference_episode * len(context_ranges) * args.lap_period
    wandb_run = None
    if not args.disable_wandb:
        wandb_run = initialize_wandb(
            policy_type=args.policy_variant,
            l3_model_name=l3_mde_config.model_name,
            context_len=args.lap_period,
            max_laps=(args.num_episode + args.num_inference_episode) * len(context_ranges),
            max_steps=train_max_steps + infer_max_steps,
            exp_name=args.exp_name,
        )

    global_step = 0
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
        "imu": [],
    }
    inference_records = []

    try:
        for episode_idx in range(args.num_episode):
            scenario_bank = episode_bank(
                context_ranges=context_ranges,
                scenario_period=args.lap_period,
                repeat_type=args.scenario_repeat_type,
                random_seed=RANDOM_SEED,
                shuffle=True,
            )
            for scenario in scenario_bank:
                scenario_policy = scenario_policies[scenario.name]
                for episode_step in range(args.lap_period):
                    if not simulation_app._app.is_running() or simulation_app.is_exiting():
                        break

                    env_context = scenario.step(episode_step)
                    scene.control_light_intensity(env_context["light_intensity"])
                    scene.robot_control(
                        time=scene.get_simulation_current_time,
                        control_parameters={
                            "angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
                        },
                    )

                    raw_policy_context = get_policy_context(scene)
                    policy_context = build_egreedy_context(
                        raw_policy_context,
                        env_context,
                        speed_context_source=args.speed_context_source,
                        imu_speed_context_scale=imu_speed_context_scale,
                    )
                    context_information = {
                        **policy_context,
                        "exposure_idx": curr_exposure_idx,
                        "iso_idx": curr_iso_idx,
                    }
                    selection = scenario_policy.select_action(
                        context_information=context_information,
                        tie_break_random=True,
                    )
                    action = selection["chosen_action"]
                    prev_exposure_idx = curr_exposure_idx
                    prev_iso_idx = curr_iso_idx
                    next_exposure_idx, next_iso_idx = scenario_policy.transition(
                        curr_exposure_idx,
                        curr_iso_idx,
                        action,
                    )
                    scene.sensor_control(
                        control_parameters={
                            "iso": sensor_param_space.iso_values[next_iso_idx],
                            "shutter_time": sensor_param_space.exposure_values[next_exposure_idx],
                        }
                    )
                    curr_exposure_idx = next_exposure_idx
                    curr_iso_idx = next_iso_idx

                    syn_data = scene.step(render=True)
                    rgb_image = syn_data.get("rgb", None)
                    gt_depth = syn_data.get(scene.get_anno("depth"), None)
                    bbox_data = syn_data.get(scene.get_anno("2d_bounding_box"), None)
                    imu_data = syn_data.get("imu_sensor", None)
                    if rgb_image is None or rgb_image.size == 0:
                        if VERBOSE:
                            print("Warning: Received empty RGB image. Skipping update.")
                        continue
                    if gt_depth is None or gt_depth.size == 0:
                        if VERBOSE:
                            print("Warning: Received empty ground-truth depth image. Skipping update.")
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
                    update_info = scenario_policy.update_parameters(
                        (str(selection["state"]), int(selection["chosen_action_idx"])),
                        reward,
                    )
                    record = {
                        "phase": "train",
                        "episode_idx": episode_idx,
                        "episode_step": episode_step,
                        "global_step": global_step,
                        "scenario_name": scenario.name,
                        "scenario_speed_label": scenario.speed_label,
                        "scenario_light_label": scenario.light_label,
                        "scenario_light_intensity": float(env_context["light_intensity"]),
                        "scenario_speed": float(env_context["angular_velocity"]),
                        "scenario_robot_angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
                        "context_acceleration_magnitude": float(raw_policy_context.get("acceleration_magnitude", 0.0)),
                        "context_gyro_magnitude": float(raw_policy_context.get("gyro_magnitude", 0.0)),
                        "context_light_intensity": float(policy_context.get("light_intensity", 0.0)),
                        "context_egreedy_angular_velocity": float(policy_context.get("angular_velocity", 0.0)),
                        "prev_exposure_idx": prev_exposure_idx,
                        "prev_iso_idx": prev_iso_idx,
                        "exposure_idx": curr_exposure_idx,
                        "iso_idx": curr_iso_idx,
                        "state": selection["state"],
                        "action": action,
                        "chosen_score": float(selection["chosen_expected_reward"]),
                        "chosen_mean": float(selection["chosen_expected_reward"]),
                        "chosen_bonus": 0.0,
                        "chosen_count": int(selection["chosen_count"]),
                        "epsilon": float(selection["epsilon"]),
                        "selection_mode": selection["mode"],
                        "reward": reward,
                        "reward_info": reward_info,
                        "metric_info": metric_info,
                        "update_info": update_info,
                    }
                    scenario_policy.history.append(record)

                    if VERBOSE:
                        print(
                            f"Train Episode {episode_idx} Step {episode_step} | "
                            f"Scenario: {scenario.name} | "
                            f"State: {selection['state']} | "
                            f"Action: {action} | "
                            f"Reward: {reward:.4f} | "
                            f"Epsilon: {selection['epsilon']:.4f}"
                        )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                **flatten_metrics(reward_info, f"train/{scenario.name}"),
                                **flatten_metrics(metric_info, f"train/{scenario.name}"),
                                **flatten_metrics(update_info, f"train/{scenario.name}"),
                                f"train/{scenario.name}/reward": reward,
                                f"train/{scenario.name}/exposure_idx": curr_exposure_idx,
                                f"train/{scenario.name}/iso_idx": curr_iso_idx,
                                f"train/{scenario.name}/chosen_expected_reward": float(selection["chosen_expected_reward"]),
                                f"train/{scenario.name}/chosen_count": int(selection["chosen_count"]),
                                f"train/{scenario.name}/epsilon": float(selection["epsilon"]),
                                f"train/{scenario.name}/explore_flag": float(selection["mode"] == "explore"),
                                f"train/{scenario.name}/cmd_ang_vel": float(env_context["angular_velocity"]) * args.angular_speed_scale,
                                f"train/{scenario.name}/cmd_light_intensity": float(env_context["light_intensity"]),
                                f"train/{scenario.name}/egreedy_angular_velocity": float(policy_context.get("angular_velocity", 0.0)),
                                f"train/{scenario.name}/gyro_magnitude": float(raw_policy_context.get("gyro_magnitude", 0.0)),
                                f"train/{scenario.name}/light_intensity": float(policy_context.get("light_intensity", 0.0)),
                                "global_step": global_step,
                                "episode_idx": episode_idx,
                                "episode_step": episode_step,
                                f"train/{scenario.name}/scenario_step": episode_idx * args.lap_period + episode_step,
                            }
                        )

                    global_step += 1
                    if args.save_data:
                        save_synthetic_data(
                            DATA_PATH=data_path,
                            syn_data=syn_data_cache,
                            lap_idx=global_step,
                        )
                    syn_data_cache = {
                        "rgb": [],
                        "depth": [],
                        "bbox": [],
                        "pred_depth": [],
                        "imu": [],
                    }

            consolidate_egreedy_policies(scenario_policies, consolidated_policy)
            checkpoint_metadata = {
                **policy_metadata(args, motion_thresholds, light_thresholds, range_limits),
                "episode_idx": episode_idx,
                "l3_mde_config": l3_mde_config.__dict__,
                "num_scenario_policies": len(scenario_policies),
            }
            if args.checkpoint_episode_interval > 0 and (episode_idx + 1) % args.checkpoint_episode_interval == 0:
                make_checkpoint(
                    consolidated_policy=consolidated_policy,
                    scenario_policies=scenario_policies,
                    data_path=data_path,
                    episode_idx=episode_idx,
                    metadata=checkpoint_metadata,
                )

        consolidate_egreedy_policies(scenario_policies, consolidated_policy)
        final_metadata = {
            **policy_metadata(args, motion_thresholds, light_thresholds, range_limits),
            "episode_idx": args.num_episode - 1,
            "l3_mde_config": l3_mde_config.__dict__,
            "num_scenario_policies": len(scenario_policies),
        }
        save_policy_bundle(
            checkpoint_path=os.path.join(data_path, "checkpoints", "egreedy_policy_final.json"),
            consolidated_policy=consolidated_policy,
            scenario_policies=scenario_policies,
            metadata=final_metadata,
        )

        if not args.skip_inference and args.num_inference_episode > 0:
            curr_exposure_idx, curr_iso_idx = reset_sensor_to_midpoint(scene, sensor_param_space)
            for inference_episode_idx in range(args.num_inference_episode):
                scenario_bank = episode_bank(
                    context_ranges=context_ranges,
                    scenario_period=args.lap_period,
                    repeat_type=args.scenario_repeat_type,
                    random_seed=RANDOM_SEED,
                    shuffle=False,
                )
                for scenario in scenario_bank:
                    for episode_step in range(args.lap_period):
                        if not simulation_app._app.is_running() or simulation_app.is_exiting():
                            break

                        env_context = scenario.step(episode_step)
                        scene.control_light_intensity(env_context["light_intensity"])
                        scene.robot_control(
                            time=scene.get_simulation_current_time,
                            control_parameters={
                                "angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
                            },
                        )

                        raw_policy_context = get_policy_context(scene)
                        policy_context = build_egreedy_context(
                            raw_policy_context,
                            env_context,
                            speed_context_source=args.speed_context_source,
                            imu_speed_context_scale=imu_speed_context_scale,
                        )
                        context_information = {
                            **policy_context,
                            "exposure_idx": curr_exposure_idx,
                            "iso_idx": curr_iso_idx,
                        }
                        selection = consolidated_policy.select_action(
                            context_information=context_information,
                            tie_break_random=args.tie_break_random,
                            is_infer_mode=True,
                        )
                        action = selection["chosen_action"]
                        prev_exposure_idx = curr_exposure_idx
                        prev_iso_idx = curr_iso_idx
                        next_exposure_idx, next_iso_idx = consolidated_policy.transition(
                            curr_exposure_idx,
                            curr_iso_idx,
                            action,
                        )
                        scene.sensor_control(
                            control_parameters={
                                "iso": sensor_param_space.iso_values[next_iso_idx],
                                "shutter_time": sensor_param_space.exposure_values[next_exposure_idx],
                            }
                        )
                        curr_exposure_idx = next_exposure_idx
                        curr_iso_idx = next_iso_idx

                        syn_data = scene.step(render=True)
                        rgb_image = syn_data.get("rgb", None)
                        gt_depth = syn_data.get(scene.get_anno("depth"), None)
                        bbox_data = syn_data.get(scene.get_anno("2d_bounding_box"), None)
                        imu_data = syn_data.get("imu_sensor", None)
                        if rgb_image is None or rgb_image.size == 0:
                            if VERBOSE:
                                print("Warning: Received empty RGB image during inference. Skipping step.")
                            continue
                        if gt_depth is None or gt_depth.size == 0:
                            if VERBOSE:
                                print("Warning: Received empty ground-truth depth image during inference. Skipping step.")
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
                        record = {
                            "phase": "inference",
                            "episode_idx": inference_episode_idx,
                            "episode_step": episode_step,
                            "global_step": global_step,
                            "scenario_name": scenario.name,
                            "scenario_speed_label": scenario.speed_label,
                            "scenario_light_label": scenario.light_label,
                            "scenario_light_intensity": float(env_context["light_intensity"]),
                            "scenario_speed": float(env_context["angular_velocity"]),
                            "scenario_robot_angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
                            "context_acceleration_magnitude": float(raw_policy_context.get("acceleration_magnitude", 0.0)),
                            "context_gyro_magnitude": float(raw_policy_context.get("gyro_magnitude", 0.0)),
                            "context_light_intensity": float(policy_context.get("light_intensity", 0.0)),
                            "context_egreedy_angular_velocity": float(policy_context.get("angular_velocity", 0.0)),
                            "prev_exposure_idx": prev_exposure_idx,
                            "prev_iso_idx": prev_iso_idx,
                            "exposure_idx": curr_exposure_idx,
                            "iso_idx": curr_iso_idx,
                            "state": selection["state"],
                            "action": action,
                            "chosen_score": float(selection["chosen_expected_reward"]),
                            "chosen_mean": float(selection["chosen_expected_reward"]),
                            "chosen_bonus": 0.0,
                            "chosen_count": int(selection["chosen_count"]),
                            "epsilon": float(selection["epsilon"]),
                            "selection_mode": selection["mode"],
                            "reward": reward,
                            "reward_info": reward_info,
                            "metric_info": metric_info,
                        }
                        inference_records.append(record)

                        if VERBOSE:
                            print(
                                f"Infer Episode {inference_episode_idx} Step {episode_step} | "
                                f"Scenario: {scenario.name} | "
                                f"State: {selection['state']} | "
                                f"Action: {action} | "
                                f"Reward: {reward:.4f}"
                            )

                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    **flatten_metrics(reward_info, f"inference/{scenario.name}"),
                                    **flatten_metrics(metric_info, f"inference/{scenario.name}"),
                                    f"inference/{scenario.name}/reward": reward,
                                    f"inference/{scenario.name}/exposure_idx": curr_exposure_idx,
                                    f"inference/{scenario.name}/iso_idx": curr_iso_idx,
                                    f"inference/{scenario.name}/chosen_expected_reward": float(selection["chosen_expected_reward"]),
                                    f"inference/{scenario.name}/chosen_count": int(selection["chosen_count"]),
                                    f"inference/{scenario.name}/cmd_ang_vel": float(env_context["angular_velocity"]) * args.angular_speed_scale,
                                    f"inference/{scenario.name}/cmd_light_intensity": float(env_context["light_intensity"]),
                                    f"inference/{scenario.name}/egreedy_angular_velocity": float(policy_context.get("angular_velocity", 0.0)),
                                    f"inference/{scenario.name}/gyro_magnitude": float(raw_policy_context.get("gyro_magnitude", 0.0)),
                                    f"inference/{scenario.name}/light_intensity": float(policy_context.get("light_intensity", 0.0)),
                                    "global_step": global_step,
                                    "inference_episode_idx": inference_episode_idx,
                                    "episode_step": episode_step,
                                    f"inference/{scenario.name}/scenario_step": inference_episode_idx * args.lap_period + episode_step,
                                }
                            )

                        global_step += 1
                        if args.save_data:
                            save_synthetic_data(
                                DATA_PATH=data_path,
                                syn_data=syn_data_cache,
                                lap_idx=global_step,
                            )
                        syn_data_cache = {
                            "rgb": [],
                            "depth": [],
                            "bbox": [],
                            "pred_depth": [],
                            "imu": [],
                        }

            summary = summarize_records(inference_records)
            records_path, summary_path = save_eval_outputs(
                os.path.join(data_path, "inference"),
                inference_records,
                summary,
            )
            print(f"[Inference] mean_reward={summary['mean_reward']:.4f} num_steps={summary['num_steps']}")
            print(f"[Inference] Saved records to {records_path}")
            print(f"[Inference] Saved summary to {summary_path}")

    except KeyboardInterrupt:
        print("[Main] Caught Keyboard Interrupt Command, Shutting Down...")
    except Exception as exc:
        print("[Error]", exc)
        traceback.print_exc()
    finally:
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
