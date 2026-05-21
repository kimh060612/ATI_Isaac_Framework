"""
ATI sensor-control experiment with a contextual Soft Actor-Critic policy.

The SAC state is the current controllable camera state:
    [exposure_time, iso]

The SAC context is action-independent deployment information exposed through
BaseScene:
    [imu_acceleration_magnitude, imu_gyro_magnitude, light_meter_lux]

The SAC action is continuous:
    [delta_exposure_time, delta_iso]
"""

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
from PIL import Image

try:
    import wandb
except Exception:
    wandb = None

from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from ati_utils.log_utils import configure_isaac_sim_logging, save_synthetic_data
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy import (
    L2ContextualSACRGBCamPolicy,
    SACBoxSpec,
    SensorParamSpace,
    reward_flipped_img,
    reward_oracle,
    reward_test_time_augment,
)
from robot_control import build_default_context_trajectory, build_step_context_trajectory
from scene import ATIDepthScene


RANDOM_SEED = 42
VERBOSE = True
DEBUG = True


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def select_reward_function(reward_type: str):
    if reward_type == "flipped":
        return reward_flipped_img
    if reward_type == "test_time_augment":
        return reward_test_time_augment
    if reward_type == "oracle":
        return reward_oracle
    raise ValueError(
        f"Invalid reward_type: {reward_type}. "
        "Must be one of ['flipped', 'test_time_augment', 'oracle']"
    )


def build_observation_info(
    reward_type: str,
    rgb_image: np.ndarray,
    pred_depths,
    metric_info: dict | None = None,
) -> dict:
    if reward_type == "flipped":
        return {
            "original_rgb": np.array(rgb_image),
            "depth_original": pred_depths[0],
            "depth_flipped": pred_depths[1],
            "image_weight": 0.0,
            "depth_weight": 1.0,
        }
    if reward_type == "test_time_augment":
        return {
            "rgb": np.array(rgb_image),
            "inverse_depths": pred_depths,
            "uncertainty_reduction": "mean",
            "image_weight": 0.4,
            "depth_weight": 0.6,
        }
    if reward_type == "oracle":
        if metric_info is None:
            raise ValueError("metric_info is required for oracle reward.")
        return {
            "original_rgb": np.array(rgb_image),
            "abs_rel_error": metric_info["abs_rel"],
            "delta_1": metric_info["a1"],
            "image_weight": 0.0,
            "depth_weight": 1.0,
        }
    raise ValueError(
        f"Invalid reward_type: {reward_type}. "
        "Must be one of ['flipped', 'test_time_augment', 'oracle']"
    )


def average_metrics(metrics_list: list[dict], prefix: str | None = None) -> dict:
    if not metrics_list:
        return {}
    payload = {}
    for key in metrics_list[0].keys():
        values = [
            float(metric[key])
            for metric in metrics_list
            if key in metric and np.isscalar(metric[key])
        ]
        if values:
            metric_key = f"{prefix}/{key}" if prefix else key
            payload[metric_key] = float(np.mean(values))
    return payload


def flatten_metrics(metrics: dict | None, prefix: str) -> dict:
    if not metrics:
        return {}
    payload = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            payload[f"{prefix}/{key}"] = float(value)
    return payload


def get_sensor_state(scene: ATIDepthScene, sensor_name: str = "agent_camera") -> np.ndarray:
    params = scene.get_sensor_control_params(sensor_name=sensor_name)
    return np.asarray(
        [
            float(params.get("shutter_time", 0.008)),
            float(params.get("iso", 400.0)),
        ],
        dtype=np.float32,
    )


def context_to_array(context: dict) -> np.ndarray:
    return np.asarray(
        [
            float(context.get("acceleration_magnitude", 0.0)),
            float(context.get("gyro_magnitude", 0.0)),
            float(context.get("light_intensity", 1.0)),
        ],
        dtype=np.float32,
    )


def build_robot_config(robot_name: str) -> ATIBaseRobotConfig:
    robot_config = ATIBaseRobotConfig(robot_name=robot_name)
    if robot_name == "limo":
        robot_config.set_limo_config()
    elif robot_name == "kaya":
        robot_config.set_kaya_config()
    else:
        raise ValueError("robot_name must be one of ['kaya', 'limo'].")
    return robot_config


def build_robot_command(robot_name: str, angular_velocity: float, linear_velocity: float) -> dict:
    if robot_name == "limo":
        return {
            "linear_velocity": float(linear_velocity),
            "angular_velocity": float(angular_velocity),
        }
    return {
        "linear_velocity_x": 0.0,
        "linear_velocity_y": 0.0,
        "angular_velocity": float(angular_velocity),
    }


def initialize_wandb(args, max_steps: int):
    if args.disable_wandb or wandb is None:
        return None
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=args.exp_name,
        config={
            "policy_type": "L2ContextualSACRGBCamPolicy",
            "robot_name": args.robot,
            "reward_type": args.reward_type,
            "max_laps": args.max_laps,
            "max_steps": max_steps,
            "lap_period": args.lap_period,
            "light_values": args.light_values,
            "angular_velocity_values": args.angular_velocity_values,
            "state": ["exposure_time", "iso"],
            "context": ["acceleration_magnitude", "gyro_magnitude", "light_intensity"],
            "action": ["delta_exposure_time", "delta_iso"],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ATI contextual SAC sensor control in Isaac Sim")
    parser.add_argument("--exp_name", type=str, default="ati_sac_contextual_sensor_control")
    parser.add_argument("--robot", type=str, default="kaya", choices=["kaya", "limo"])
    parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
    parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype")
    parser.add_argument("--max_laps", type=int, default=600)
    parser.add_argument("--lap_period", type=int, default=30)
    parser.add_argument("--trajectory_mode", type=str, default="smooth", choices=["smooth", "step"])
    parser.add_argument("--light_values", type=str, default="200,1000,3000,6000,9000")
    parser.add_argument("--angular_velocity_values", type=str, default="0.05,0.1,0.50")
    parser.add_argument("--linear_velocity", type=float, default=0.3)
    parser.add_argument("--max_delta_exposure", type=float, default=0.004)
    parser.add_argument("--max_delta_iso", type=float, default=400.0)
    parser.add_argument("--max_acceleration_context", type=float, default=5.0)
    parser.add_argument("--max_gyro_context", type=float, default=3.0)
    parser.add_argument("--max_light_context", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--random_steps", type=int, default=10)
    parser.add_argument("--update_after", type=int, default=64)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    set_deterministic(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    change_context_every = args.lap_period
    max_laps = args.max_laps
    max_steps = change_context_every * max_laps
    data_path = os.path.join(
        args.data_path,
        f"experiment_{args.exp_name}_{args.robot}_{args.reward_type}_{args.lap_period}steps",
    )

    configure_isaac_sim_logging()
    robot_config = build_robot_config(args.robot)
    render_config = ATIBaseConfig(
        name="ati_sac_rendering",
        robot_config=robot_config,
    )
    render_config.set_agent_sensor_controller("l1_sac_clamping_controller")
    render_config.set_rendering_mode("realtime")
    render_config.set_pathtracing_param(spp=128, num_subsamples=32)
    scene = ATIDepthScene(
        simulation_app,
        config=render_config,
        physics_dt=render_config.physics_dt,
        rendering_dt=render_config.rendering_dt,
        stage_units_in_meters=render_config.stage_units_in_meters,
    )

    light_values = parse_float_list(args.light_values)
    angular_velocity_values = parse_float_list(args.angular_velocity_values)
    if args.trajectory_mode == "step":
        trajectory = build_step_context_trajectory(
            light_values=light_values,
            speed_values=angular_velocity_values,
            light_hold_steps=args.lap_period * 10,
            speed_hold_steps=args.lap_period * 5,
            speed_phase_offset_steps=args.lap_period,
        )
    else:
        trajectory = build_default_context_trajectory(
            light_values=light_values,
            speed_values=angular_velocity_values,
            light_transition_steps=args.lap_period * 20,
            speed_transition_steps=args.lap_period * 5,
            light_hold_steps=args.lap_period,
            speed_hold_steps=args.lap_period,
            speed_phase_offset_steps=args.lap_period,
        )

    sensor_param_space = SensorParamSpace()
    initial_exposure = sensor_param_space.exposure_values[len(sensor_param_space.exposure_values) // 2]
    initial_iso = sensor_param_space.iso_values[len(sensor_param_space.iso_values) // 2]
    scene.sensor_control(
        control_parameters={
            "shutter_time": initial_exposure,
            "iso": initial_iso,
        }
    )

    sac_policy = L2ContextualSACRGBCamPolicy(
        state_spec=SACBoxSpec(
            low=(min(sensor_param_space.exposure_values), min(sensor_param_space.iso_values)),
            high=(max(sensor_param_space.exposure_values), max(sensor_param_space.iso_values)),
            log_indices=(0, 1),
        ),
        context_spec=SACBoxSpec(
            low=(0.0, 0.0, 1.0),
            high=(args.max_acceleration_context, args.max_gyro_context, args.max_light_context),
            log_indices=(2,),
        ),
        action_spec=SACBoxSpec(
            low=(-args.max_delta_exposure, -args.max_delta_iso),
            high=(args.max_delta_exposure, args.max_delta_iso),
        ),
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        random_steps=args.random_steps,
        update_after=args.update_after,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        reward_scale=args.reward_scale,
        device=args.device,
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
    mde_model = L3PLayerDepthAnythingv2(l3_config=l3_mde_config, device="cuda")
    reward_function = select_reward_function(args.reward_type)
    wandb_run = initialize_wandb(args, max_steps=max_steps)

    step = 0
    lap_idx = 0
    env_context = trajectory.value_at(0)
    scene.control_light_intensity(env_context["light_intensity"])
    scene.robot_control(
        time=scene.get_simulation_current_time,
        control_parameters=build_robot_command(
            args.robot,
            env_context["angular_velocity"],
            args.linear_velocity,
        ),
    )

    current_state = get_sensor_state(scene)
    current_context_dict = scene.get_policy_context()
    current_context = context_to_array(current_context_dict)
    action, action_info = sac_policy.select_action(current_state, current_context, explore=True)
    control_result = scene.sensor_control(
        control_parameters={
            "sac_action": {
                "delta_exposure": float(action[0]),
                "delta_iso": float(action[1]),
            },
            "max_delta_exposure": args.max_delta_exposure,
            "max_delta_iso": args.max_delta_iso,
        }
    )
    pending_transition = {
        "state": current_state,
        "context": current_context,
        "action": action,
        "action_info": action_info,
        "control_result": control_result,
    }

    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
    }
    log_reward_history = []
    log_performance_history = []
    log_context_history = []

    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            if VERBOSE:
                print(
                    f"Step: {step + 1}/{max_steps}, "
                    f"Lap: {lap_idx + 1}/{max_laps}, "
                    f"Simulation Time: {scene.get_simulation_current_time:.4f}"
                )

            syn_data = scene.step(render=True)
            scene.robot_control(
                time=scene.get_simulation_current_time,
                control_parameters=build_robot_command(
                    args.robot,
                    env_context["angular_velocity"],
                    args.linear_velocity,
                ),
            )

            rgb_image = syn_data.get("rgb", None)
            gt_depth = syn_data.get(scene.get_anno("depth"), None)
            bbox_data = syn_data.get(scene.get_anno("2d_bounding_box"), None)
            if rgb_image is None or rgb_image.size == 0:
                if VERBOSE:
                    print("Warning: Received empty RGB image. Skipping this step.")
                continue
            if gt_depth is None or gt_depth.size == 0:
                if VERBOSE:
                    print("Warning: Received empty ground-truth depth image. Skipping this step.")
                continue

            syn_data_cache["rgb"].append(rgb_image)
            syn_data_cache["depth"].append(gt_depth)
            syn_data_cache["bbox"].append(bbox_data)

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
            log_reward_history.append(reward_info)
            log_performance_history.append(metric_info)
            log_context_history.append(syn_data.get("canonical_context", {}))

            if (step + 1) % change_context_every == 0 and step > 0:
                lap_reward = float(average_metrics(log_reward_history).get("reward", 0.0))
                next_lap_idx = lap_idx + 1
                done = next_lap_idx >= max_laps
                next_state = get_sensor_state(scene)
                next_context_dict = syn_data.get("canonical_context") or scene.get_policy_context(
                    imu_frame=syn_data.get("imu_sensor"),
                    light_meter_info=syn_data.get("light_meter"),
                )
                next_context = context_to_array(next_context_dict)
                update_metrics = sac_policy.observe(
                    state=pending_transition["state"],
                    context=pending_transition["context"],
                    action=pending_transition["action"],
                    reward=lap_reward,
                    next_state=next_state,
                    next_context=next_context,
                    done=done,
                )

                payload = {
                    "scenario/env_light_intensity": float(env_context["light_intensity"]),
                    "scenario/env_angular_velocity": float(env_context["angular_velocity"]),
                    "sensor/shutter_time": float(next_state[0]),
                    "sensor/iso": float(next_state[1]),
                    "context/acceleration_magnitude": float(next_context[0]),
                    "context/gyro_magnitude": float(next_context[1]),
                    "context/light_intensity": float(next_context[2]),
                    "policy/action_delta_exposure": float(pending_transition["action"][0]),
                    "policy/action_delta_iso": float(pending_transition["action"][1]),
                    **flatten_metrics(pending_transition.get("action_info"), "policy_action"),
                    **flatten_metrics(pending_transition.get("control_result"), "l1_controller"),
                    **flatten_metrics(update_metrics, "sac_update"),
                    **average_metrics(log_reward_history, prefix="reward"),
                    **average_metrics(log_performance_history, prefix="performance"),
                    **average_metrics(log_context_history, prefix="context_avg"),
                }
                if wandb_run is not None:
                    wandb_run.log(payload, step=lap_idx, commit=True)
                elif VERBOSE:
                    print("[LapMetrics]", payload)

                if args.save_data:
                    save_synthetic_data(data_path, syn_data_cache, lap_idx)

                if done:
                    print("All laps completed. Ending simulation")
                    break

                env_context = trajectory.value_at(step + 1)
                scene.control_light_intensity(env_context["light_intensity"])
                scene.robot_control(
                    time=scene.get_simulation_current_time,
                    control_parameters=build_robot_command(
                        args.robot,
                        env_context["angular_velocity"],
                        args.linear_velocity,
                    ),
                )

                action_context_dict = scene.get_policy_context()
                action_context = context_to_array(action_context_dict)
                action, action_info = sac_policy.select_action(next_state, action_context, explore=True)
                control_result = scene.sensor_control(
                    control_parameters={
                        "sac_action": {
                            "delta_exposure": float(action[0]),
                            "delta_iso": float(action[1]),
                        },
                        "max_delta_exposure": args.max_delta_exposure,
                        "max_delta_iso": args.max_delta_iso,
                    }
                )
                pending_transition = {
                    "state": next_state,
                    "context": action_context,
                    "action": action,
                    "action_info": action_info,
                    "control_result": control_result,
                }

                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                    "pred_depth": [],
                }
                log_reward_history = []
                log_performance_history = []
                log_context_history = []
                lap_idx += 1

            if step >= max_steps - 1:
                print("All steps completed. Ending simulation")
                break
            step = scene.num_steps

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
