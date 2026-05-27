"""
Batched NeuralUCB/LinUCB sensor-control training in Isaac Sim.

Training is organized as context laps.  M simulation steps define one lap,
camera settings are held for that lap, and the policy is updated once from an
aggregated lap reward.
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
    L2BatchedNeuralUCBPolicy,
    L2NeuralLinearUCBPolicy,
    L2NeuralUCBPolicy,
    SensorParamSpace,
    episode_bank,
)
from scene import ATIDepthScene
import wandb


RANDOM_SEED = 42
VERBOSE = True
DEBUG = True


DEFAULT_SPEED_RANGES = "SLOW:0.0:0.4,NORMAL:0.9:1.1,FAST:1.5:1.7,SUPER_FAST:1.9:2.1"
DEFAULT_LIGHT_RANGES = "DARK:100:300,DIM:500:600,NORMAL:1000:1100,BRIGHT:4000:4200,SUPER_BRIGHT:9000:9200"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATI batched NeuralUCB sensor control in Isaac Sim")
    parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_neurucb_oracle")
    parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
    parser.add_argument("--data_path", type=str, default="/home/kimh060612/ATI_research/dataset")
    parser.add_argument("--num_episode", type=int, default=200, help="Number of training laps repeated per context.")
    parser.add_argument("--lap_period", type=int, default=200, help="M simulation steps that define one lap.")
    parser.add_argument("--exp_ratio", type=float, default=0.5, help="UCB exploration bonus alpha.")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--policy_variant", type=str, default="neural_ucb", choices=["neural_ucb", "neural_linear_ucb"])
    parser.add_argument("--forced_exploration_prob", type=float, default=0.05)
    parser.add_argument("--hidden_dims", type=str, default="32", help="Comma-separated MLP hidden dimensions.")
    parser.add_argument("--neural_feature_dim", type=int, default=32, help="Encoder feature size for neural_linear_ucb.")
    parser.add_argument("--replay_capacity", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--action_update_interval", type=int, default=10, help="Legacy batched-policy interval; lap-level training updates once per lap.")
    parser.add_argument("--train_every", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--network_lr", type=float, default=1e-3)
    parser.add_argument("--network_weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lap_reward_top_percent", type=float, default=20.0, help="Average the top K percent of rewards in a lap.")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Simulation steps before policy learning starts.")
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
    parser.add_argument("--max_acceleration_context", type=float, default=5.0)
    parser.add_argument("--max_gyro_context", type=float, default=3.0)
    parser.add_argument("--max_light_context", type=float, default=10000.0)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    return parser


def parse_int_tuple(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not dims:
        raise ValueError("hidden_dims must contain at least one integer.")
    if any(dim <= 0 for dim in dims):
        raise ValueError("hidden_dims must be positive integers.")
    return dims


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

def flatten_metrics(metrics: dict | None, prefix: str) -> dict:
    if not metrics:
        return {}
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            payload[f"{prefix}/{key}"] = float(value)
    return payload


def append_numeric_metrics(accumulator: dict[str, list[float]], metrics: dict | None) -> None:
    if not metrics:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            accumulator.setdefault(key, []).append(float(value))


def mean_accumulated_metrics(accumulator: dict[str, list[float]], prefix: str) -> dict:
    return {
        f"{prefix}/{key}": float(np.mean(values))
        for key, values in accumulator.items()
        if values
    }


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


def build_neuralucb_context(raw_policy_context: dict, *, fallback_light: float) -> dict:
    acceleration = float(raw_policy_context.get("acceleration_magnitude", 0.0))
    if not np.isfinite(acceleration):
        acceleration = 0.0

    gyro = float(raw_policy_context.get("gyro_magnitude", raw_policy_context.get("angular_velocity", 0.0)))
    if not np.isfinite(gyro):
        gyro = 0.0

    light_intensity = float(raw_policy_context.get("light_intensity", fallback_light))
    if not np.isfinite(light_intensity):
        light_intensity = float(fallback_light)

    return {
        **raw_policy_context,
        "acceleration_magnitude": float(acceleration),
        "gyro_magnitude": float(gyro),
        "angular_velocity": float(gyro),
        "light_intensity": float(light_intensity),
    }


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


def summarize_context_samples(samples: list[dict]) -> dict:
    if not samples:
        return {
            "acceleration_magnitude": 0.0,
            "gyro_magnitude": 0.0,
            "light_intensity": 1.0,
            "num_context_samples": 0.0,
        }
    keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, (int, float, np.integer, np.floating, bool))
        }
    )
    summary = {
        key: float(np.mean([float(sample.get(key, 0.0)) for sample in samples]))
        for key in keys
    }
    summary["num_context_samples"] = float(len(samples))
    return summary


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


def initialize_lap_context(
    *,
    scene: ATIDepthScene,
    scenario,
    args,
) -> dict:
    env_context = scenario.step(0)
    scene.control_light_intensity(env_context["light_intensity"])
    scene.robot_control(
        time=scene.get_simulation_current_time,
        control_parameters={
            "angular_velocity": float(env_context["angular_velocity"]) * args.angular_speed_scale,
        },
    )
    syn_data = scene.step(render=False)
    raw_context = get_policy_context(scene, syn_data if isinstance(syn_data, dict) else None)
    return build_neuralucb_context(
        raw_context,
        fallback_light=float(env_context["light_intensity"]),
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


def select_and_apply_lap_action(
    *,
    scene: ATIDepthScene,
    policy: L2NeuralUCBPolicy,
    sensor_param_space: SensorParamSpace,
    context_summary: dict,
    curr_exposure_idx: int,
    curr_iso_idx: int,
) -> tuple[dict, int, int]:
    selection = policy.select_action(
        context_information={
            **context_summary,
            "exposure_idx": curr_exposure_idx,
            "iso_idx": curr_iso_idx,
        },
        tie_break_random=True,
    )
    next_exposure_idx, next_iso_idx = policy.transition(
        curr_exposure_idx,
        curr_iso_idx,
        selection["chosen_action"],
    )
    scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[next_iso_idx],
            "shutter_time": sensor_param_space.exposure_values[next_exposure_idx],
        }
    )
    return selection, next_exposure_idx, next_iso_idx


def run_lap(
    *,
    scene: ATIDepthScene,
    scenario,
    lap_idx: int,
    context_idx: int,
    context_lap_idx: int,
    global_step: int,
    args,
    curr_exposure_idx: int,
    curr_iso_idx: int,
    mde_model: L3PLayerDepthAnythingv2,
    reward_function,
    l3_mde_config: L3MDEConfig,
) -> tuple[dict, dict, int]:
    reward_history: list[dict] = []
    metric_history: list[dict] = []
    context_samples: list[dict] = []
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
        policy_context = build_neuralucb_context(
            raw_policy_context,
            fallback_light=float(env_context["light_intensity"]),
        )
        context_samples.append(policy_context)

        if rgb_image is None or rgb_image.size == 0:
            if VERBOSE:
                print("Warning: Received empty RGB image during lap. Skipping step reward.")
            global_step += 1
            continue
        if gt_depth is None or gt_depth.size == 0:
            if VERBOSE:
                print("Warning: Received empty ground-truth depth image during lap. Skipping step reward.")
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
        reward_history.append(reward_info)
        metric_history.append(metric_info)
        global_step += 1

    lap_reward_info = top_percent_reward_info(reward_history, args.lap_reward_top_percent)
    lap_metric_info = mean_numeric_metrics(metric_history)
    lap_context = summarize_context_samples(context_samples)
    lap_summary = {
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
    return lap_summary, syn_data_cache, global_step


def make_checkpoint(
    policy: L2NeuralUCBPolicy,
    data_path: str,
    checkpoint_idx: int,
    metadata: dict,
) -> str:
    os.makedirs(f"{data_path}/checkpoints", exist_ok=True)
    checkpoint_path = os.path.join(
        data_path,
        f"checkpoints/neurucb_policy_context_{checkpoint_idx + 1:04d}.pt",
    )
    saved_path = policy.save(checkpoint_path, metadata=metadata)
    print(f"[Checkpoint] Saved NeuralUCB policy to {saved_path}")
    return saved_path


def main() -> None:
    args = build_parser().parse_args()
    set_deterministic(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print_device_debug(args.device)
    context_ranges = build_context_ranges(args)

    data_path = os.path.join(
        args.data_path,
        f"experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps_policy_{args.policy_variant}",
    )
    os.makedirs(data_path, exist_ok=True)

    configure_isaac_sim_logging()
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_neurucb_rendering",
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

    hidden_dims = parse_int_tuple(args.hidden_dims)
    policy_class = (
        L2BatchedNeuralUCBPolicy
        if args.policy_variant == "neural_ucb"
        else L2NeuralLinearUCBPolicy
    )
    policy_kwargs = {
        "sensor_names": "agent_camera",
        "sensor_config": sensor_param_space,
        "reward_function": select_reward_function(args.reward_type),
        "alpha": args.exp_ratio,
        "lambda_reg": args.lambda_reg,
        "random_seed": RANDOM_SEED,
        "forced_exploration_prob": args.forced_exploration_prob,
        "max_acceleration_context": args.max_acceleration_context,
        "max_gyro_context": args.max_gyro_context,
        "max_light_context": args.max_light_context,
        "hidden_dims": hidden_dims,
        "replay_capacity": args.replay_capacity,
        "batch_size": args.batch_size,
        "train_every": args.train_every,
        "gradient_steps": args.gradient_steps,
        "learning_rate": args.network_lr,
        "weight_decay": args.network_weight_decay,
        "device": args.device,
    }
    if args.policy_variant == "neural_ucb":
        policy_kwargs["update_interval"] = 1
    if args.policy_variant == "neural_linear_ucb":
        policy_kwargs["neural_feature_dim"] = args.neural_feature_dim
    policy = policy_class(**policy_kwargs)
    policy_param_device = next(policy.model.parameters()).device
    print(f"[Device] policy_device={policy.device} policy_param_device={policy_param_device}")

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
    reward_function = select_reward_function(args.reward_type)
    wandb_run = initialize_wandb(
        policy_type=args.policy_variant,
        l3_model_name=l3_mde_config.model_name,
        context_len=args.lap_period,
        max_laps=args.num_episode * len(context_ranges),
        max_steps=args.num_episode * len(context_ranges) * args.lap_period,
        exp_name=args.exp_name if not args.disable_wandb else None,
    )

    global_step = 0
    train_lap_idx = 0

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
                    args=args,
                )
                selection, curr_exposure_idx, curr_iso_idx = select_and_apply_lap_action(
                    scene=scene,
                    policy=policy,
                    sensor_param_space=sensor_param_space,
                    context_summary=initial_context,
                    curr_exposure_idx=curr_exposure_idx,
                    curr_iso_idx=curr_iso_idx,
                )

                lap_summary, syn_data_cache, global_step = run_lap(
                    scene=scene,
                    scenario=scenario,
                    lap_idx=train_lap_idx,
                    context_idx=context_idx,
                    context_lap_idx=context_lap_idx,
                    global_step=global_step,
                    args=args,
                    curr_exposure_idx=curr_exposure_idx,
                    curr_iso_idx=curr_iso_idx,
                    mde_model=mde_model,
                    reward_function=reward_function,
                    l3_mde_config=l3_mde_config,
                )

                lap_reward = float(lap_summary["reward"])
                update_info = None
                if lap_summary["num_valid_reward_steps"] > 0:
                    update_info = policy.observe(selection, lap_reward)
                elif hasattr(policy, "_active_selection"):
                    policy._active_selection = None
                    policy._active_rewards = []

                lap_summary.update(
                    {
                        "global_step": global_step,
                        "action": selection["chosen_action"],
                        "chosen_score": float(selection["chosen_score"]),
                        "chosen_mean": float(selection["chosen_mean"]),
                        "chosen_bonus": float(selection["chosen_bonus"]),
                        "reward_prediction": float(selection.get("chosen_reward_prediction", selection["chosen_mean"])),
                        "selection_mode": selection["selection_mode"],
                        "forced_explore": bool(selection.get("forced_explore", False)),
                        "batch_phase": selection.get("batch_phase"),
                        "batch_observed_steps": selection.get("batch_observed_steps"),
                        "batch_update_interval": selection.get("batch_update_interval"),
                        "active_batch_action": selection.get("active_batch_action"),
                        "batched_update": None if update_info is None else update_info.get("batched_update"),
                        "update_info": update_info,
                        "laps_per_context": args.num_episode,
                    }
                )
                policy.history.append(lap_summary)

                if VERBOSE:
                    print(
                        f"Train Lap {train_lap_idx} | Context {context_idx + 1}/{len(training_scenarios)} | "
                        f"Context Lap {context_lap_idx + 1}/{args.num_episode} | "
                        f"Scenario: {scenario.name} | "
                        f"Action: {selection['chosen_action']} | "
                        f"Top-{args.lap_reward_top_percent:g}% Reward: {lap_reward:.4f} | "
                        f"Chosen Score: {selection['chosen_score']:.4f}"
                    )

                if not args.disable_wandb:
                    wandb_run.log(
                        {
                            **flatten_metrics(lap_summary["reward_info"], f"{scenario.name}"),
                            **flatten_metrics(lap_summary["metric_info"], f"{scenario.name}"),
                            **flatten_metrics(update_info, f"{scenario.name}"),
                            **flatten_metrics(lap_summary["context_summary"], f"{scenario.name}/context"),
                            f"{scenario.name}/lap_reward": lap_reward,
                            f"{scenario.name}/exposure_idx": curr_exposure_idx,
                            f"{scenario.name}/iso_idx": curr_iso_idx,
                            f"{scenario.name}/chosen_score": float(selection["chosen_score"]),
                            f"{scenario.name}/chosen_mean": float(selection["chosen_mean"]),
                            f"{scenario.name}/chosen_bonus": float(selection["chosen_bonus"]),
                            f"{scenario.name}/reward_prediction": float(
                                selection.get("chosen_reward_prediction", selection["chosen_mean"])
                            ),
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

            policy_metadata = {
                "context_idx": context_idx,
                "scenario_name": scenario.name,
                "completed_train_laps": train_lap_idx,
                "training_laps_per_context": args.num_episode,
                "lap_period": args.lap_period,
                "lap_reward_top_percent": args.lap_reward_top_percent,
                "warmup_steps": args.warmup_steps,
                "requested_action_update_interval": args.action_update_interval,
                "context_source": "imu_plus_scene_light",
                "policy_kwargs": policy_kwargs,
                "l3_mde_config": l3_mde_config.__dict__,
            }
            if args.checkpoint_episode_interval > 0 and (context_idx + 1) % args.checkpoint_episode_interval == 0:
                make_checkpoint(
                    policy=policy,
                    data_path=data_path,
                    checkpoint_idx=context_idx,
                    metadata=policy_metadata,
                )
                

    except KeyboardInterrupt:
        print("[Main] Caught Keyboard Interrupt Command, Shutting Down...")
    except Exception as exc:
        print("[Error]", exc)
        traceback.print_exc()
    finally:
        print("[Main] Shutting down...")
        try:
            simulation_app.close()
        except Exception:
            print("")
        print("[Main] Done. If process hangs, run: pkill -f 'python.sh|kit'")


if __name__ == "__main__":
    main()
