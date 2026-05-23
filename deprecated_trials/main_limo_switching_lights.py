"""LIMO straight-line LinUCB sensor-control experiment in Isaac Sim."""

from __future__ import annotations

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

from ati_utils.log_utils import configure_isaac_sim_logging, save_synthetic_data, get_eval_averages
from robot_control import build_default_context_trajectory, build_step_context_trajectory
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle
from policy import L2DisjointLinUCBRGBCamPolicy, SensorParamSpace
from robot_control import StraightLineLapFollower, get_limo_pose_2d

# Enable Livestream extension
from isaacsim.core.utils.extensions import enable_extension
simulation_app.set_setting("/app/window/drawMouse", True)
enable_extension("omni.services.livestream.nvcf")

import argparse
from PIL import Image
import traceback
import wandb
import numpy as np
import random


parser = argparse.ArgumentParser(description="ATI LinUCB sensor control with a LIMO straight-line lap trajectory")
parser.add_argument("--exp_name", type=str, default="atil2l3_limo_fixed_path_depthany_oracle")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype/limo_fixed_path")
parser.add_argument("--max_laps", type=int, default=600)
parser.add_argument("--lap_period", type=int, default=30, help="Number of rendered steps per policy update.")
parser.add_argument("--warmup_laps", type=int, default=1, help="Number of initial laps to skip policy updates and logging.")
parser.add_argument("--spawn_random_objs", action="store_true", help="Spawn random scene objects.")
parser.add_argument("--path_speed", type=float, default=1.5, help="Maximum straight-line forward speed in m/s.")
parser.add_argument("--path_acceleration", type=float, default=2.0, help="Straight-line acceleration in m/s^2.")
parser.add_argument("--path_deceleration", type=float, default=2.0, help="Straight-line deceleration in m/s^2.")
parser.add_argument("--endpoint_turn_speed", type=float, default=3.14, help="Maximum in-place yaw speed in rad/s for endpoint 180-degree turns.")
parser.add_argument("--endpoint_distance", type=float, default=1.5, help="Endpoint distance from the origin along the x-axis in meters.")
args = parser.parse_args()

RANDOM_SEED = 42
VERBOSE = True
DEBUG = True
POSITION_THRESHOLD = 0.04
HEADING_THRESHOLD = 0.04
ANGULAR_GAIN = 2.5
MAX_ANGULAR_VELOCITY = 1.0
FORWARD_ANGLE_THRESHOLD = float(np.pi / 3.0)
RECOVERY_SPEED = 0.8
TURN_ACCELERATION = 2.0
TURN_DECELERATION = 2.0

def initialize_wandb(context_len, max_laps, max_steps, exp_name=None):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=exp_name,
        config={
            "policy_type": "L2DisjointLinUCBRGBCamPolicy",
            "robot": "limo",
            "trajectory_type": "straight_line_lap",
            "turn_per_lap": context_len,
            "context_period_steps": context_len,
            "lap_definition": "straight_line_out_and_back",
            "warmup_laps": args.warmup_laps,
            "path_speed": args.path_speed,
            "path_acceleration": args.path_acceleration,
            "path_deceleration": args.path_deceleration,
            "endpoint_turn_speed": args.endpoint_turn_speed,
            "endpoint_distance": args.endpoint_distance,
            "position_threshold": POSITION_THRESHOLD,
            "heading_threshold": HEADING_THRESHOLD,
            "angular_gain": ANGULAR_GAIN,
            "max_angular_velocity": MAX_ANGULAR_VELOCITY,
            "forward_angle_threshold": FORWARD_ANGLE_THRESHOLD,
            "recovery_speed": RECOVERY_SPEED,
            "turn_acceleration": TURN_ACCELERATION,
            "turn_deceleration": TURN_DECELERATION,
            "max_laps": max_laps,
            "max_steps": max_steps,
            "l3_mde_model": "Depth-Anything-V2-Small-hf",
        },
    )


def select_reward_function(reward_type: str):
    if reward_type == "flipped":
        return reward_flipped_img
    if reward_type == "test_time_augment":
        return reward_test_time_augment
    if reward_type == "oracle":
        return reward_oracle
    raise ValueError(f"Invalid reward_type: {reward_type}.")


def build_observation_info(reward_type: str, rgb_image: np.ndarray, pred_depths, metric_info: dict = None) -> dict:
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
        return {
            "original_rgb": np.array(rgb_image),
            "abs_rel_error": metric_info["abs_rel"],
            "delta_1": metric_info["a1"],
            "image_weight": 0.0,
            "depth_weight": 1.0,
        }
    raise ValueError(f"Invalid reward_type: {reward_type}.")


def get_avg_aggregation(reward_obs: list[dict]) -> dict:
    if not reward_obs:
        return {
            "reward": 0.0,
            "image_reward": 0.0,
            "depth_reward": 0.0,
            "uncertainty": 0.0,
        }
    return {
        key: float(np.mean([metric[key] for metric in reward_obs]))
        for key in reward_obs[0].keys()
    }


def average_motion_history(motion_history: list[dict]) -> dict:
    if not motion_history:
        return {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
            "abs_angular_velocity": 0.0,
            "path_id": 0.0,
        }
    return {
        "linear_velocity": float(np.mean([entry["linear_velocity"] for entry in motion_history])),
        "angular_velocity": float(np.mean([entry["angular_velocity"] for entry in motion_history])),
        "abs_angular_velocity": float(np.mean([abs(entry["angular_velocity"]) for entry in motion_history])),
        "path_id": float(motion_history[-1]["path_id"]),
    }


def make_syn_data_cache() -> dict:
    return {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
        "imu": [],
    }


if __name__ == "__main__":
    CHANGE_CONTEXT_EVERY = args.lap_period
    WARMUP_LAPS = max(0, int(args.warmup_laps))
    DATA_PATH = f"{args.data_path}/experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps_4midpoints"
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS

    configure_isaac_sim_logging()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_deterministic(RANDOM_SEED)

    limo_config = ATIBaseRobotConfig(robot_name="limo")
    limo_config.set_limo_config()
    render_config = ATIBaseConfig(
        name="ati_limo_linucb_fixed_mpc_path",
        robot_config=limo_config,
    )
    render_config.set_rendering_mode("realtime")
    render_config.set_pathtracing_param(spp=128, num_subsamples=32)
    render_config.set_random_obj_spawn(args.spawn_random_objs)

    my_scene = ATIDepthScene(
        simulation_app,
        config=render_config,
        physics_dt=render_config.physics_dt,
        rendering_dt=render_config.rendering_dt,
        stage_units_in_meters=render_config.stage_units_in_meters,
    )
    context_agent_speed = [1.0, 1.0, 1.0, 1.0, 1.0]
    RAD_COEFF = np.pi / 12
    trajectory = build_step_context_trajectory(
        light_values=[6000, 100],
        speed_values=[s * RAD_COEFF for s in context_agent_speed],
        light_hold_steps=args.lap_period * 10,
        speed_hold_steps=args.lap_period,
        speed_phase_offset_steps=0,
    )

    sensor_param_space = SensorParamSpace()
    l2_policy = L2DisjointLinUCBRGBCamPolicy(
        sensor_names="agent_camera",
        sensor_config=sensor_param_space,
        reward_function=select_reward_function(args.reward_type),
        alpha=1.0,
        random_seed=RANDOM_SEED,
    )
    context = trajectory.value_at(0)
    curr_light = context["light_intensity"]
    curr_motion_context = context["angular_velocity"]
    curr_exposure_idx = len(sensor_param_space.exposure_values) // 2
    curr_iso_idx = len(sensor_param_space.iso_values) // 2

    initial_policy_result = l2_policy.step(
        context_information={
            "light_intensity": curr_light,
            "angular_velocity": curr_motion_context,
            "iso_idx": curr_iso_idx,
            "exposure_idx": curr_exposure_idx,
            "tie_break_random": True,
            "skip_update": True,
            "store_pending": WARMUP_LAPS == 0,
        },
        observations={
            "reward_info_override": get_avg_aggregation([]),
        },
    )
    curr_exposure_idx = initial_policy_result["next_exposure_idx"]
    curr_iso_idx = initial_policy_result["next_iso_idx"]
    current_lap_policy_result = initial_policy_result

    my_scene.control_light_intensity(curr_light)
    my_scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[curr_iso_idx],
            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
        }
    )

    l3_mde_config = L3MDEConfig(
        reward_type=args.reward_type,
        model_name="depth-anything/Depth-Anything-V2-Small-hf",
        shift_ratios=[],
        zoom_factors=[],
        gaussian_noise_stds=(0.01, 0.02, 0.05),
        brightness_factors=(0.8, 0.9),
        color_jitter_strengths=[],
        disable_hflip=False,
        prediction_mode="identity",
    )
    mde_model = L3PLayerDepthAnythingv2(l3_config=l3_mde_config, device="cuda")
    path_follower = StraightLineLapFollower(
        straight_speed=args.path_speed,
        linear_acceleration=args.path_acceleration,
        linear_deceleration=args.path_deceleration,
        endpoint_turn_speed=args.endpoint_turn_speed,
        angular_acceleration=TURN_ACCELERATION,
        angular_deceleration=TURN_DECELERATION,
        endpoint_distance=args.endpoint_distance,
        position_threshold=POSITION_THRESHOLD,
        heading_threshold=HEADING_THRESHOLD,
        heading_gain=ANGULAR_GAIN,
        max_heading_correction=MAX_ANGULAR_VELOCITY,
        forward_angle_threshold=FORWARD_ANGLE_THRESHOLD,
        recovery_speed=RECOVERY_SPEED,
    )

    step = 0
    lap_idx = 0
    syn_data_cache = make_syn_data_cache()
    wandb_run = initialize_wandb(
        context_len=CHANGE_CONTEXT_EVERY,
        max_laps=MAX_LAPS,
        max_steps=MAX_STEPS,
        exp_name=f"ati_limo_depthany_{l3_mde_config.reward_type}_{args.exp_name}",
    )
    log_context_history = []
    log_reward_history = []
    log_performance_history = []
    log_motion_history = []
    route_lap_count_at_lap_start = path_follower.lap_count

    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            if VERBOSE:
                print(
                    f"Step: {step + 1}/{MAX_STEPS}, Lap: {lap_idx + 1}/{MAX_LAPS}, "
                    f"Simulation Time: {my_scene.get_simulation_current_time:.4f} seconds"
                )

            pose = get_limo_pose_2d(my_scene)
            if DEBUG: print(pose)
            cmd = path_follower.step(
                pose,
                step_dt=render_config.rendering_dt,
            )
            if DEBUG: print(f"Motion Command - Linear Velocity: {cmd.linear_velocity:.3f} m/s, Angular Velocity: {cmd.angular_velocity:.3f} rad/s") 
            my_scene.robot_control(
                time=my_scene.get_simulation_current_time,
                control_parameters={
                    "linear_velocity": cmd.linear_velocity,
                    "angular_velocity": cmd.angular_velocity,
                },
            )

            syn_data = my_scene.step(render=True)
            rgb_image: np.array = syn_data.get("rgb", None)
            gt_depth: np.array = syn_data.get(my_scene.get_anno("depth"), None)
            bbox_data: np.array = syn_data.get(my_scene.get_anno("2d_bounding_box"), None)
            imu_sensor_data = syn_data.get("imu_sensor", None)
            if rgb_image is None or rgb_image.size == 0:
                if VERBOSE:
                    print("Warning: Received empty RGB image. Skipping this step.")
                continue
            if gt_depth is None or gt_depth.size == 0:
                if VERBOSE:
                    print("Warning: Received empty ground-truth depth image. Skipping this step.")
                continue
            if bbox_data is None or imu_sensor_data is None:
                if VERBOSE:
                    print("Warning: Missing bbox or IMU data. Skipping this step.")
                continue

            syn_data_cache["rgb"].append(rgb_image)
            syn_data_cache["depth"].append(gt_depth)
            syn_data_cache["bbox"].append(bbox_data)
            syn_data_cache["imu"].append(imu_sensor_data)
            log_motion_history.append({
                "linear_velocity": cmd.linear_velocity,
                "angular_velocity": cmd.angular_velocity,
                "path_id": path_follower.lap_count,
            })

            pred_depths, metric_info = mde_model.predict_depth([Image.fromarray(rgb_image)], gt_depth)
            syn_data_cache["pred_depth"].append(pred_depths if isinstance(pred_depths, np.ndarray) else pred_depths[0])
            if DEBUG:
                print(f"Depth Prediction Metrics: {metric_info}")

            observation_info = build_observation_info(
                reward_type=l3_mde_config.reward_type,
                rgb_image=rgb_image,
                pred_depths=pred_depths,
                metric_info=metric_info,
            )
            reward_info = l2_policy.reward_function(**observation_info)
            log_reward_history.append(reward_info)
            log_performance_history.append(metric_info)

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0:
                completed_light = curr_light
                completed_motion_context = curr_motion_context
                completed_iso_idx = curr_iso_idx
                completed_exposure_idx = curr_exposure_idx
                completed_policy_result = current_lap_policy_result
                motion_summary = average_motion_history(log_motion_history)
                completed_path_in_lap = path_follower.lap_count > route_lap_count_at_lap_start
                completed_lap_is_warmup = lap_idx < WARMUP_LAPS
                next_lap_idx = lap_idx + 1
                has_next_lap = next_lap_idx < MAX_LAPS
                next_step = step + 1
                next_context = trajectory.value_at(next_step)
                next_motion_context = next_context["angular_velocity"]
                next_lap_is_warmup = next_lap_idx < WARMUP_LAPS

                result = l2_policy.step(
                    context_information={
                        "light_intensity": next_context["light_intensity"],
                        "angular_velocity": next_motion_context,
                        "iso_idx": curr_iso_idx,
                        "exposure_idx": curr_exposure_idx,
                        "tie_break_random": True,
                        "skip_update": completed_lap_is_warmup,
                        "store_pending": has_next_lap and not next_lap_is_warmup,
                    },
                    observations={
                        "reward_info_override": get_avg_aggregation(log_reward_history),
                    },
                )
                update_info = result.get("update_info")
                if DEBUG:
                    print("[DEBUG]Policy Step Reward Result:", result["reward_info"])

                if not completed_lap_is_warmup and wandb_run is not None:
                    wandb_run.log(
                        {
                            "context/light_intensity": completed_light,
                            "context/agent_speed": completed_motion_context,
                            "context/observed_linear_velocity": motion_summary["linear_velocity"],
                            "context/angular_velocity": motion_summary["angular_velocity"],
                            "context/abs_angular_velocity": motion_summary["abs_angular_velocity"],
                            "context/path_id": motion_summary["path_id"],
                            "context/path_completed": float(completed_path_in_lap),
                            "context/iso_idx": completed_iso_idx,
                            "context/exposure_idx": completed_exposure_idx,
                            **({
                                "policy/action_delta_exposure": float(completed_policy_result["action"][0]),
                                "policy/action_delta_iso": float(completed_policy_result["action"][1]),
                                "policy/chosen_score": float(completed_policy_result["chosen_score"]),
                                "policy/chosen_mean": float(completed_policy_result["chosen_mean"]),
                                "policy/chosen_bonus": float(completed_policy_result["chosen_bonus"]),
                            } if completed_policy_result is not None else {}),
                            **({
                                "policy/update_reward": float(update_info["reward"]),
                                "policy/update_action_delta_exposure": float(update_info["action"][0]),
                                "policy/update_action_delta_iso": float(update_info["action"][1]),
                            } if update_info is not None else {}),
                            **({
                                "policy/next_light_intensity": float(next_context["light_intensity"]),
                                "policy/next_agent_speed": float(next_motion_context),
                                "policy/next_iso_idx": float(result["next_iso_idx"]),
                                "policy/next_exposure_idx": float(result["next_exposure_idx"]),
                            } if has_next_lap else {}),
                            **get_eval_averages(log_reward_history, key_category="reward"),
                            **get_eval_averages(log_performance_history, key_category="performance"),
                        },
                        step=lap_idx - WARMUP_LAPS,
                        commit=True,
                    )
                    save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx - WARMUP_LAPS)

                if has_next_lap:
                    curr_light = next_context["light_intensity"]
                    curr_motion_context = next_motion_context
                    curr_exposure_idx = result["next_exposure_idx"]
                    curr_iso_idx = result["next_iso_idx"]
                    current_lap_policy_result = result
                    log_context_history.append({
                        "iso_idx": curr_iso_idx,
                        "exposure_idx": curr_exposure_idx,
                    })

                    if DEBUG:
                        print("[DEBUG] Sensor Control Action Taken - Exposure Index:", curr_exposure_idx, "ISO Index:", curr_iso_idx)
                    my_scene.control_light_intensity(curr_light)
                    my_scene.sensor_control(
                        control_parameters={
                            "iso": sensor_param_space.iso_values[curr_iso_idx],
                            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
                        }
                    )

                syn_data_cache = make_syn_data_cache()
                log_reward_history = []
                log_performance_history = []
                log_motion_history = []
                lap_idx += 1
                route_lap_count_at_lap_start = path_follower.lap_count
                if VERBOSE:
                    lap_mode = "warmup" if completed_lap_is_warmup else "linucb"
                    print(
                        f"Fixed-step lap {lap_idx}/{MAX_LAPS} completed at step {step + 1} ({lap_mode}). "
                        f"Path completed: {completed_path_in_lap}. "
                        f"Light Intensity set to {curr_light}, "
                        f"LIMO avg linear speed {motion_summary['linear_velocity']:.3f}, "
                        f"avg angular speed {motion_summary['angular_velocity']:.3f}"
                    )

            if lap_idx >= MAX_LAPS:
                print("All laps completed. Ending simulation")
                break
            step = my_scene.num_steps

    except KeyboardInterrupt:
        print("[Main] Caught Keyboard Interrupt Command, Shutting Down...")
    except Exception as e:
        print("[Error]", e)
        traceback.print_exc()
    finally:
        print("[Main] Shutting down...")
        try:
            simulation_app.close()
        except Exception:
            print("")
        print("[Main] Done. If process hangs, run: pkill -f 'python.sh|kit'")

    simulation_app.close()
