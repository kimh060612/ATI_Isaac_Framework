"""
Evaluate a trained NeuralUCB policy in Isaac Sim without updating it.
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
from pathlib import Path

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
    L2NeuralLinearUCBPolicy,
    L2NeuralUCBPolicy,
    SensorParamSpace,
    episode_bank,
)
from scene import ATIDepthScene


RANDOM_SEED = 42
VERBOSE = True


DEFAULT_SPEED_RANGES = "SLOW:0.0:0.4,NORMAL:0.9:1.2,FAST:1.5:1.8,SUPER_FAST:1.9:2.1"
DEFAULT_LIGHT_RANGES = "DARK:100:300,DIM:500:600,NORMAL:1000:1200,BRIGHT:4000:4200,SUPER_BRIGHT:9000:9200"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained NeuralUCB policy in Isaac Sim")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_neurucb_eval")
    parser.add_argument("--reward_type", type=str, default=None, choices=["flipped", "test_time_augment", "oracle"])
    parser.add_argument("--data_path", type=str, default="/home/kimh060612/ATI_research/dataset")
    parser.add_argument("--num_episode", type=int, default=1)
    parser.add_argument("--lap_period", type=int, default=200)
    parser.add_argument("--policy_variant", type=str, default=None, choices=["neural_ucb", "neural_linear_ucb"])
    parser.add_argument("--exp_ratio", type=float, default=None, help="Override saved UCB exploration bonus alpha.")
    parser.add_argument("--lambda_reg", type=float, default=None)
    parser.add_argument("--hidden_dims", type=str, default=None)
    parser.add_argument("--neural_feature_dim", type=int, default=None)
    parser.add_argument("--replay_capacity", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_every", type=int, default=None)
    parser.add_argument("--gradient_steps", type=int, default=None)
    parser.add_argument("--network_lr", type=float, default=None)
    parser.add_argument("--network_weight_decay", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--l3_model_name", type=str, default=None)
    parser.add_argument("--scenario_repeat_type", type=str, default="sin", choices=["sin", "step"])
    parser.add_argument("--speed_ranges", type=str, default=DEFAULT_SPEED_RANGES)
    parser.add_argument("--light_ranges", type=str, default=DEFAULT_LIGHT_RANGES)
    parser.add_argument("--angular_speed_scale", type=float, default=float(np.pi / 12))
    parser.add_argument("--max_acceleration_context", type=float, default=None)
    parser.add_argument("--max_gyro_context", type=float, default=None)
    parser.add_argument("--max_light_context", type=float, default=None)
    parser.add_argument("--tie_break_random", action="store_true")
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


def load_checkpoint_metadata(checkpoint_path: str) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return dict(payload.get("metadata", {}))


def checkpoint_policy_variant(metadata: dict) -> str | None:
    policy_type = str(metadata.get("policy_type", ""))
    if policy_type == "L2NeuralLinearUCBPolicy":
        return "neural_linear_ucb"
    if policy_type == "L2NeuralUCBPolicy":
        return "neural_ucb"
    return None


def metadata_or_arg(args, metadata: dict, arg_name: str, metadata_name: str, default):
    arg_value = getattr(args, arg_name)
    if arg_value is not None:
        return arg_value
    return metadata.get(metadata_name, default)


def resolve_hidden_dims(args, metadata: dict) -> tuple[int, ...]:
    if args.hidden_dims is not None:
        return parse_int_tuple(args.hidden_dims)
    saved_dims = metadata.get("hidden_dims")
    if saved_dims:
        return tuple(int(dim) for dim in saved_dims)
    return (32,)


def resolve_reward_type(args, metadata: dict) -> str:
    if args.reward_type is not None:
        return args.reward_type
    l3_metadata = metadata.get("l3_mde_config", {})
    if isinstance(l3_metadata, dict) and l3_metadata.get("reward_type"):
        return str(l3_metadata["reward_type"])
    return "oracle"


def build_l3_config(args, metadata: dict, reward_type: str) -> L3MDEConfig:
    l3_metadata = metadata.get("l3_mde_config", {})
    if not isinstance(l3_metadata, dict):
        l3_metadata = {}

    def tuple_field(name: str, default):
        value = l3_metadata.get(name, default)
        return tuple(value) if value is not None else tuple()

    return L3MDEConfig(
        reward_type=reward_type,
        model_name=args.l3_model_name or l3_metadata.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf"),
        shift_ratios=tuple_field("shift_ratios", []),
        zoom_factors=tuple_field("zoom_factors", []),
        gaussian_noise_stds=tuple_field("gaussian_noise_stds", (0.01, 0.02, 0.05)),
        brightness_factors=tuple_field("brightness_factors", ()),
        color_jitter_strengths=tuple_field("color_jitter_strengths", (0.1, 0.15)),
        disable_hflip=bool(l3_metadata.get("disable_hflip", False)),
        prediction_mode=str(l3_metadata.get("prediction_mode", "identity")),
    )


def build_policy(args, metadata: dict, sensor_param_space: SensorParamSpace, reward_type: str):
    policy_variant = args.policy_variant or checkpoint_policy_variant(metadata) or "neural_ucb"
    policy_class = L2NeuralUCBPolicy if policy_variant == "neural_ucb" else L2NeuralLinearUCBPolicy
    policy_kwargs = {
        "sensor_names": "agent_camera",
        "sensor_config": sensor_param_space,
        "reward_function": select_reward_function(reward_type),
        "alpha": float(metadata_or_arg(args, metadata, "exp_ratio", "alpha", 0.5)),
        "lambda_reg": float(metadata_or_arg(args, metadata, "lambda_reg", "lambda_reg", 1.0)),
        "random_seed": RANDOM_SEED,
        "forced_exploration_prob": 0.0,
        "max_acceleration_context": float(
            metadata_or_arg(args, metadata, "max_acceleration_context", "max_acceleration_context", 5.0)
        ),
        "max_gyro_context": float(metadata_or_arg(args, metadata, "max_gyro_context", "max_gyro_context", 3.0)),
        "max_light_context": float(metadata_or_arg(args, metadata, "max_light_context", "max_light_context", 10000.0)),
        "hidden_dims": resolve_hidden_dims(args, metadata),
        "replay_capacity": int(metadata_or_arg(args, metadata, "replay_capacity", "replay_capacity", 5000)),
        "batch_size": int(metadata_or_arg(args, metadata, "batch_size", "batch_size", 64)),
        "train_every": int(metadata_or_arg(args, metadata, "train_every", "train_every", 1)),
        "gradient_steps": int(metadata_or_arg(args, metadata, "gradient_steps", "gradient_steps", 0)),
        "learning_rate": float(metadata_or_arg(args, metadata, "network_lr", "learning_rate", 1e-3)),
        "weight_decay": float(metadata_or_arg(args, metadata, "network_weight_decay", "weight_decay", 1e-4)),
        "device": args.device,
    }
    if policy_variant == "neural_linear_ucb":
        neural_feature_dim = args.neural_feature_dim or metadata.get("neural_feature_dim", 32)
        policy_kwargs["neural_feature_dim"] = int(neural_feature_dim)

    policy = policy_class(**policy_kwargs)
    loaded_metadata = policy.load(args.checkpoint_path)
    policy.model.eval()
    print(f"[Checkpoint] Loaded {args.checkpoint_path}")
    print(f"[Checkpoint] saved_policy_type={loaded_metadata.get('policy_type', '<unknown>')}")
    return policy, policy_variant, policy_kwargs


def get_policy_context(scene: ATIDepthScene, syn_data: dict | None = None) -> dict:
    syn_data = syn_data or {}
    context = syn_data.get("canonical_context")
    if isinstance(context, dict):
        return context
    return scene.get_policy_context(
        imu_frame=syn_data.get("imu_sensor"),
        light_meter_info=syn_data.get("light_meter"),
    )


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
        scenario_summary = {
            "num_steps": len(scenario_records),
            "mean_reward": float(np.mean([record["reward"] for record in scenario_records])),
        }
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
        scenario_summary["reward_info"] = {
            key: float(np.mean([record["reward_info"][key] for record in scenario_records]))
            for key in reward_keys
        }
        scenario_summary["metric_info"] = {
            key: float(np.mean([record["metric_info"][key] for record in scenario_records]))
            for key in metric_keys
        }
        summary["per_scenario"][scenario_name] = scenario_summary
    return summary


def save_eval_outputs(data_path: str, records: list[dict], summary: dict) -> tuple[str, str]:
    os.makedirs(data_path, exist_ok=True)
    records_path = os.path.join(data_path, "eval_records.jsonl")
    summary_path = os.path.join(data_path, "eval_summary.json")
    with open(records_path, "w", encoding="utf-8") as records_file:
        for record in records:
            records_file.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(json_ready(summary), summary_file, indent=2, sort_keys=True)
    return records_path, summary_path


def main() -> None:
    args = build_parser().parse_args()
    set_deterministic(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print_device_debug(args.device)

    checkpoint_metadata = load_checkpoint_metadata(args.checkpoint_path)
    reward_type = resolve_reward_type(args, checkpoint_metadata)
    context_ranges = build_context_ranges(args)
    checkpoint_stem = Path(args.checkpoint_path).stem
    data_path = os.path.join(
        args.data_path,
        f"evaluation_{args.exp_name}_{reward_type}_{args.lap_period}steps_{checkpoint_stem}",
    )
    os.makedirs(data_path, exist_ok=True)

    configure_isaac_sim_logging()
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_neurucb_inference",
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
    curr_exposure_idx = len(sensor_param_space.exposure_values) // 2
    curr_iso_idx = len(sensor_param_space.iso_values) // 2
    scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[curr_iso_idx],
            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
        }
    )

    policy, policy_variant, policy_kwargs = build_policy(args, checkpoint_metadata, sensor_param_space, reward_type)
    policy_param_device = next(policy.model.parameters()).device
    print(f"[Device] policy_device={policy.device} policy_param_device={policy_param_device}")

    l3_mde_config = build_l3_config(args, checkpoint_metadata, reward_type)
    mde_model = L3PLayerDepthAnythingv2(l3_config=l3_mde_config, device=args.device)
    print(f"[Device] mde_device={mde_model.device} mde_pipeline_device={getattr(mde_model.model, 'device', '<unknown>')}")
    reward_function = select_reward_function(reward_type)

    wandb_run = None
    if not args.disable_wandb:
        wandb_run = initialize_wandb(
            policy_type=f"{policy_variant}_eval",
            l3_model_name=l3_mde_config.model_name,
            context_len=args.lap_period,
            max_laps=args.num_episode * len(context_ranges),
            max_steps=args.num_episode * len(context_ranges) * args.lap_period,
            exp_name=args.exp_name,
        )

    global_step = 0
    eval_records = []
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
        "imu": [],
    }

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

                    policy_context = get_policy_context(scene)
                    context_information = {
                        **policy_context,
                        "exposure_idx": curr_exposure_idx,
                        "iso_idx": curr_iso_idx,
                    }
                    selection = policy.select_action(
                        context_information=context_information,
                        tie_break_random=args.tie_break_random,
                        force_explore=False,
                    )
                    action = selection["chosen_action"]
                    next_exposure_idx, next_iso_idx = policy.transition(
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
                            print("Warning: Received empty RGB image. Skipping step.")
                        continue
                    if gt_depth is None or gt_depth.size == 0:
                        if VERBOSE:
                            print("Warning: Received empty ground-truth depth image. Skipping step.")
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
                        "episode_idx": episode_idx,
                        "episode_step": episode_step,
                        "global_step": global_step,
                        "scenario_name": scenario.name,
                        "scenario_speed_label": scenario.speed_label,
                        "scenario_light_label": scenario.light_label,
                        "scenario_light_intensity": float(env_context["light_intensity"]),
                        "scenario_speed": float(env_context["angular_velocity"]),
                        "scenario_robot_angular_velocity": float(env_context["angular_velocity"])
                        * args.angular_speed_scale,
                        "context_acceleration_magnitude": float(policy_context.get("acceleration_magnitude", 0.0)),
                        "context_gyro_magnitude": float(policy_context.get("gyro_magnitude", 0.0)),
                        "context_light_intensity": float(policy_context.get("light_intensity", 0.0)),
                        "exposure_idx": curr_exposure_idx,
                        "iso_idx": curr_iso_idx,
                        "action": action,
                        "chosen_score": float(selection["chosen_score"]),
                        "chosen_mean": float(selection["chosen_mean"]),
                        "chosen_bonus": float(selection["chosen_bonus"]),
                        "reward_prediction": float(selection.get("chosen_reward_prediction", selection["chosen_mean"])),
                        "selection_mode": selection["selection_mode"],
                        "reward": reward,
                        "reward_info": reward_info,
                        "metric_info": metric_info,
                    }
                    eval_records.append(record)

                    if VERBOSE:
                        print(
                            f"Eval Episode {episode_idx} Step {episode_step} | "
                            f"Scenario: {scenario.name} | "
                            f"Action: {action} | "
                            f"Reward: {reward:.4f} | "
                            f"Chosen Score: {selection['chosen_score']:.4f}"
                        )

                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                **flatten_metrics(reward_info, f"{scenario.name}"),
                                **flatten_metrics(metric_info, f"{scenario.name}"),
                                f"{scenario.name}/reward": reward,
                                f"{scenario.name}/exposure_idx": curr_exposure_idx,
                                f"{scenario.name}/iso_idx": curr_iso_idx,
                                f"{scenario.name}/chosen_score": float(selection["chosen_score"]),
                                f"{scenario.name}/chosen_mean": float(selection["chosen_mean"]),
                                f"{scenario.name}/chosen_bonus": float(selection["chosen_bonus"]),
                                f"{scenario.name}/cmd_ang_vel": float(env_context["angular_velocity"])
                                * args.angular_speed_scale,
                                f"{scenario.name}/cmd_light_intensity": float(env_context["light_intensity"]),
                                f"{scenario.name}/acceleration_magnitude": float(
                                    policy_context.get("acceleration_magnitude", 0.0)
                                ),
                                f"{scenario.name}/gyro_magnitude": float(policy_context.get("gyro_magnitude", 0.0)),
                                f"{scenario.name}/light_intensity": float(policy_context.get("light_intensity", 0.0)),
                                "global_step": global_step,
                                "episode_idx": episode_idx,
                                "episode_step": episode_step,
                                f"{scenario.name}/scenario_step": episode_idx * args.lap_period + episode_step,
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

        summary = summarize_records(eval_records)
        records_path, summary_path = save_eval_outputs(data_path, eval_records, summary)
        print(f"[Eval] mean_reward={summary['mean_reward']:.4f} num_steps={summary['num_steps']}")
        print(f"[Eval] Saved records to {records_path}")
        print(f"[Eval] Saved summary to {summary_path}")
        metadata_path = os.path.join(data_path, "eval_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(
                json_ready(
                    {
                        "checkpoint_path": args.checkpoint_path,
                        "checkpoint_metadata": checkpoint_metadata,
                        "policy_variant": policy_variant,
                        "policy_kwargs": policy_kwargs,
                        "l3_mde_config": l3_mde_config.__dict__,
                    }
                ),
                metadata_file,
                indent=2,
                sort_keys=True,
            )
        print(f"[Eval] Saved metadata to {metadata_path}")

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
