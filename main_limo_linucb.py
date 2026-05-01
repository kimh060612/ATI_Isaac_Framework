"""
LIMO random-path LinUCB sensor-control experiment in Isaac Sim.

This entrypoint keeps the L2-L3 sensor-control loop from main_sensor_control.py,
but drives a LIMO robot with a ROS-free random trajectory follower inspired by
MobilityGen's RandomPathFollowingScenario.
"""

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
from robot_control import build_default_context_trajectory
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle
from policy import L2DisjointLinUCBRGBCamPolicy, SensorParamSpace
from robot_control import RandomPathFollower, get_limo_pose_2d

import argparse
from dataclasses import dataclass
from PIL import Image
import traceback
import wandb
import numpy as np
import random


parser = argparse.ArgumentParser(description="ATI LinUCB sensor control with LIMO random trajectories")
parser.add_argument("--exp_name", type=str, default="atil2l3_limo_random_path_depthany_oracle")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype/limo_random_path")
parser.add_argument("--max_laps", type=int, default=600)
parser.add_argument("--lap_period", type=int, default=30)
parser.add_argument("--path_speed", type=float, default=0.5, help="LIMO nominal forward speed in m/s")
parser.add_argument("--lookahead_distance", type=float, default=0.8, help="Pure-pursuit lookahead distance in meters")
parser.add_argument("--angular_gain", type=float, default=1.6, help="Heading-error proportional gain")
parser.add_argument("--max_angular_velocity", type=float, default=1.4, help="Yaw-rate command limit in rad/s")
parser.add_argument("--path_bounds", type=float, nargs=4, default=(-1.2, 1.8, 0.0, 1.8), metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
parser.add_argument("--waypoint_count", type=int, default=8)
parser.add_argument("--spawn_random_objs", action="store_true", help="Spawn random scene objects. The built-in path follower does not avoid them.")
args = parser.parse_args()

RANDOM_SEED = 42
VERBOSE = True
DEBUG = True

def initialize_wandb(context_len, max_laps, max_steps, exp_name=None):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=exp_name,
        config={
            "policy_type": "L2DisjointLinUCBRGBCamPolicy",
            "robot": "limo",
            "trajectory_type": "random_path_following",
            "turn_per_lap": context_len,
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
            "image_weight": 0.0,
            "depth_weight": 1.0,
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


if __name__ == "__main__":
    CHANGE_CONTEXT_EVERY = args.lap_period
    DATA_PATH = f"{args.data_path}/experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps"
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS

    configure_isaac_sim_logging()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_deterministic(RANDOM_SEED)

    limo_config = ATIBaseRobotConfig(robot_name="limo")
    limo_config.set_limo_config()
    render_config = ATIBaseConfig(
        name="ati_limo_linucb_random_path",
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

    context_agent_speed = [1.5, 2.0, 1.5, 2.0, 1.5]
    trajectory = build_default_context_trajectory(
        light_values=[1000, 1000, 1000, 1000, 1000],
        speed_values=[s * np.pi / 12 for s in context_agent_speed],
        light_transition_steps=args.lap_period * 20,
        speed_transition_steps=args.lap_period * 10,
        light_hold_steps=args.lap_period,
        speed_hold_steps=args.lap_period,
        speed_phase_offset_steps=args.lap_period,
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
    rng = np.random.default_rng(RANDOM_SEED)
    random_path_follower = RandomPathFollower(
        bounds=tuple(args.path_bounds),
        waypoint_count=args.waypoint_count,
        path_speed=args.path_speed,
        lookahead_distance=args.lookahead_distance,
        angular_gain=args.angular_gain,
        max_angular_velocity=args.max_angular_velocity,
        rng=rng,
    )

    step = 0
    lap_idx = 0
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
        "imu": [],
    }
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

    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            if VERBOSE:
                print(f"Step: {step + 1}/{MAX_STEPS}, Simulation Time: {my_scene.get_simulation_current_time:.4f} seconds")

            pose = get_limo_pose_2d(my_scene)
            cmd = random_path_follower.step(pose)
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
                "linear_velocity": np.linalg.norm(imu_sensor_data["lin_acc"][:2]) if imu_sensor_data else 0.0,
                "angular_velocity": np.linalg.norm(imu_sensor_data["ang_vel"][:2]) if imu_sensor_data else 0.0,
                "path_id": random_path_follower.path_id,
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

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0 and step > 0:
                motion_summary = average_motion_history(log_motion_history)
                curr_motion_context = max(
                    motion_summary["linear_velocity"], 
                    motion_summary["abs_angular_velocity"]
                ) 
                wandb_run.log(
                    {
                        "context/light_intensity": curr_light,
                        "context/agent_speed": motion_summary["linear_velocity"],
                        "context/angular_velocity": motion_summary["angular_velocity"],
                        "context/abs_angular_velocity": motion_summary["abs_angular_velocity"],
                        "context/path_id": motion_summary["path_id"],
                        "context/iso_idx": curr_iso_idx,
                        "context/exposure_idx": curr_exposure_idx,
                        **get_eval_averages(log_reward_history, key_category="reward"),
                        **get_eval_averages(log_performance_history, key_category="performance"),
                    },
                    step=lap_idx,
                    commit=True,
                )

                result = l2_policy.step(
                    context_information={
                        "light_intensity": curr_light,
                        "angular_velocity": curr_motion_context,
                        "iso_idx": curr_iso_idx,
                        "exposure_idx": curr_exposure_idx,
                        "tie_break_random": False,
                    },
                    observations={
                        "reward_info_override": get_avg_aggregation(log_reward_history),
                    },
                )
                if DEBUG:
                    print("[DEBUG]Policy Step Reward Result:", result["reward_info"])
                curr_exposure_idx = result["next_exposure_idx"]
                curr_iso_idx = result["next_iso_idx"]
                log_context_history.append({
                    "iso_idx": curr_iso_idx,
                    "exposure_idx": curr_exposure_idx,
                })

                if DEBUG:
                    print("[DEBUG] Sensor Control Action Taken - Exposure Index:", curr_exposure_idx, "ISO Index:", curr_iso_idx)
                my_scene.sensor_control(
                    control_parameters={
                        "iso": sensor_param_space.iso_values[curr_iso_idx],
                        "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
                    }
                )

                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)
                context = trajectory.value_at(lap_idx)
                curr_light = context["light_intensity"]
                my_scene.control_light_intensity(curr_light)

                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                    "pred_depth": [],
                    "imu": [],
                }
                log_reward_history = []
                log_performance_history = []
                log_motion_history = []
                lap_idx += 1
                if VERBOSE:
                    print(
                        f"Context changed at step {step + 1}: "
                        f"Light Intensity set to {curr_light}, "
                        f"LIMO avg linear speed {motion_summary['linear_velocity']:.3f}, "
                        f"avg angular speed {motion_summary['angular_velocity']:.3f}"
                    )

            if step >= MAX_STEPS - 1:
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
