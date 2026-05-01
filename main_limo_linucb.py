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
from robot_control import FixedWayPointFollower, Pose2D, get_limo_pose_2d

import argparse
from dataclasses import dataclass
from PIL import Image
import traceback
import wandb
import numpy as np
import random


parser = argparse.ArgumentParser(description="ATI LinUCB sensor control with LIMO fixed waypoint trajectories")
parser.add_argument("--exp_name", type=str, default="atil2l3_limo_fixed_path_depthany_oracle")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"])
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype/limo_fixed_path")
parser.add_argument("--max_laps", type=int, default=600)
parser.add_argument("--lap_period", type=int, default=30, help="Number of rendered steps per fixed lap.")
parser.add_argument("--path_speed", type=float, default=2.0, help="LIMO nominal forward speed in m/s")
parser.add_argument("--turn_path_speed", type=float, default=0.4, help="Forward speed in m/s while heading error is above forward_angle_threshold")
parser.add_argument("--max_path_speed", type=float, default=3.0, help="Maximum adaptive straight-path speed in m/s")
parser.add_argument("--max_turn_path_speed", type=float, default=1.0, help="Maximum adaptive turn speed in m/s")
parser.add_argument("--min_motion_blur_speed", type=float, default=1.5, help="Minimum straight-path speed used to encourage visible motion blur")
parser.add_argument("--lap_speed_margin", type=float, default=1.25, help="Multiplier applied to required speed for finishing one path inside one lap")
parser.add_argument("--lookahead_distance", type=float, default=0.8, help="Pure-pursuit lookahead distance in meters")
parser.add_argument("--angular_gain", type=float, default=2.5, help="Heading-error proportional gain")
parser.add_argument("--max_angular_velocity", type=float, default=1.0, help="Yaw-rate command limit in rad/s")
parser.add_argument("--forward_angle_threshold", type=float, default=float(np.pi / 3.0), help="Heading-error threshold in radians for using full path_speed")
parser.add_argument("--stop_distance_threshold", type=float, default=0.35, help="Distance in meters from path end that completes the current waypoint lap")
parser.add_argument("--path_bounds", type=float, nargs=4, default=(-1.2, 1.8, 0.0, 1.2), metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
parser.add_argument("--path_bounds_margin", type=float, default=0.1, help="Inset margin used for waypoint sampling and command-level bounds guarding")
parser.add_argument("--boundary_turn_gain", type=float, default=2.5, help="Heading correction gain used when the robot approaches path bounds")
parser.add_argument("--boundary_recovery_speed", type=float, default=0.2, help="Reserved recovery speed parameter for path-boundary control")
parser.add_argument("--boundary_spin_velocity", type=float, default=0.8, help="In-place yaw velocity in rad/s used at path bounds or after early path completion")
parser.add_argument("--max_lap_path_length", type=float, default=None, help="Maximum sampled path length per lap in meters; defaults to the feasible length from max_path_speed and lap_period")
parser.add_argument("--waypoint_count", type=int, default=4, help="Deprecated; fixed MPC path always uses the four edge midpoints of path_bounds.")
parser.add_argument("--fixed_path_error", type=float, default=0.08, help="Path error threshold that makes the fixed follower slow down for recovery")
parser.add_argument("--fixed_recovery_speed", type=float, default=0.8, help="Linear speed used when fixed follower is correcting path error")
parser.add_argument("--fixed_smoothing_passes", type=int, default=0, help="Chaikin smoothing passes for the fixed waypoint path")
parser.add_argument("--mpc_horizon_steps", type=int, default=12)
parser.add_argument("--mpc_angular_samples", type=int, default=9)
parser.add_argument("--mpc_path_error_weight", type=float, default=80.0)
parser.add_argument("--mpc_heading_error_weight", type=float, default=8.0)
parser.add_argument("--mpc_progress_weight", type=float, default=2.0)
parser.add_argument("--mpc_speed_weight", type=float, default=1.0)
parser.add_argument("--warmup_laps", type=int, default=1, help="Number of initial waypoint laps to skip policy updates and logging.")
parser.add_argument("--spawn_random_objs", action="store_true", help="Spawn random scene objects. The built-in path follower does not avoid them.")
args = parser.parse_args()

RANDOM_SEED = 42
VERBOSE = True
DEBUG = True

def initialize_wandb(context_len, max_laps, max_steps, max_lap_path_length=None, exp_name=None):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=exp_name,
        config={
            "policy_type": "L2DisjointLinUCBRGBCamPolicy",
            "robot": "limo",
            "trajectory_type": "fixed_waypoint_following",
            "turn_per_lap": context_len,
            "context_period_steps": context_len,
            "lap_definition": "fixed_step_path_budget",
            "warmup_laps": args.warmup_laps,
            "path_speed": args.path_speed,
            "turn_path_speed": args.turn_path_speed,
            "max_path_speed": args.max_path_speed,
            "max_turn_path_speed": args.max_turn_path_speed,
            "min_motion_blur_speed": args.min_motion_blur_speed,
            "lap_speed_margin": args.lap_speed_margin,
            "forward_angle_threshold": args.forward_angle_threshold,
            "path_bounds": tuple(args.path_bounds),
            "path_bounds_margin": args.path_bounds_margin,
            "boundary_spin_velocity": args.boundary_spin_velocity,
            "max_lap_path_length": max_lap_path_length,
            "fixed_path_error": args.fixed_path_error,
            "fixed_recovery_speed": args.fixed_recovery_speed,
            "fixed_smoothing_passes": args.fixed_smoothing_passes,
            "mpc_horizon_steps": args.mpc_horizon_steps,
            "mpc_angular_samples": args.mpc_angular_samples,
            "mpc_path_error_weight": args.mpc_path_error_weight,
            "mpc_heading_error_weight": args.mpc_heading_error_weight,
            "mpc_progress_weight": args.mpc_progress_weight,
            "mpc_speed_weight": args.mpc_speed_weight,
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


def make_syn_data_cache() -> dict:
    return {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
        "imu": [],
    }


def build_fixed_path_waypoints(
    bounds: tuple[float, float, float, float],
    margin: float,
    waypoint_count:int=16
) -> list[Pose2D]:
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    margin = max(0.0, float(margin))
    x_margin = min(margin, max(0.0, 0.5 * (x_max - x_min) - 1e-6))
    y_margin = min(margin, max(0.0, 0.5 * (y_max - y_min) - 1e-6))
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    radius_x = max(1e-3, 0.5 * (x_max - x_min) - x_margin)
    radius_y = max(1e-3, 0.5 * (y_max - y_min) - y_margin)
    count = max(4, int(waypoint_count))
    waypoints = []
    for idx in range(count):
        angle = 2.0 * np.pi * idx / count
        x = center_x + radius_x * np.cos(angle)
        y = center_y + radius_y * np.sin(angle)
        theta = angle + np.pi / 2.0
        waypoints.append(Pose2D(x=float(x), y=float(y), theta=float(theta)))
    return waypoints


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
    max_lap_path_length = args.max_lap_path_length
    if max_lap_path_length is None:
        max_lap_path_length = args.max_path_speed * args.lap_period * render_config.rendering_dt / max(args.lap_speed_margin, 1e-6)

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
    fixed_waypoints = build_fixed_path_waypoints(
        bounds=tuple(args.path_bounds),
        margin=args.path_bounds_margin,
    )
    path_follower = FixedWayPointFollower(
        waypoints=fixed_waypoints,
        linear_speed=args.path_speed,
        lookahead_distance=args.lookahead_distance,
        angular_gain=args.angular_gain,
        cross_track_gain=args.boundary_turn_gain,
        max_angular_velocity=args.max_angular_velocity,
        max_path_error=args.fixed_path_error,
        recovery_linear_speed=args.fixed_recovery_speed,
        closed_path=True,
        smoothing_passes=args.fixed_smoothing_passes,
        use_mpc=True,
        mpc_horizon_steps=args.mpc_horizon_steps,
        mpc_angular_samples=args.mpc_angular_samples,
        mpc_path_error_weight=args.mpc_path_error_weight,
        mpc_heading_error_weight=args.mpc_heading_error_weight,
        mpc_progress_weight=args.mpc_progress_weight,
        mpc_speed_weight=args.mpc_speed_weight,
    )

    step = 0
    lap_idx = 0
    syn_data_cache = make_syn_data_cache()
    wandb_run = initialize_wandb(
        context_len=CHANGE_CONTEXT_EVERY,
        max_laps=MAX_LAPS,
        max_steps=MAX_STEPS,
        max_lap_path_length=max_lap_path_length,
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
            cmd = path_follower.step(
                pose,
                step_dt=render_config.rendering_dt,
            )
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
                motion_summary = average_motion_history(log_motion_history)
                completed_path_in_lap = path_follower.lap_count > route_lap_count_at_lap_start
                curr_motion_context = max(
                    motion_summary["linear_velocity"], 
                    motion_summary["abs_angular_velocity"]
                ) 
                is_warmup_lap = lap_idx < WARMUP_LAPS

                result = l2_policy.step(
                    context_information={
                        "light_intensity": curr_light,
                        "angular_velocity": curr_motion_context,
                        "iso_idx": curr_iso_idx,
                        "exposure_idx": curr_exposure_idx,
                        "tie_break_random": False,
                        "skip_update": is_warmup_lap,
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

                if not is_warmup_lap:
                    wandb_run.log(
                        {
                            "context/light_intensity": curr_light,
                            "context/agent_speed": motion_summary["linear_velocity"],
                            "context/angular_velocity": motion_summary["angular_velocity"],
                            "context/abs_angular_velocity": motion_summary["abs_angular_velocity"],
                            "context/path_id": motion_summary["path_id"],
                            "context/path_completed": float(completed_path_in_lap),
                            "context/iso_idx": curr_iso_idx,
                            "context/exposure_idx": curr_exposure_idx,
                            **get_eval_averages(log_reward_history, key_category="reward"),
                            **get_eval_averages(log_performance_history, key_category="performance"),
                        },
                        step=lap_idx - WARMUP_LAPS,
                        commit=True,
                    )
                    save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx - WARMUP_LAPS)

                context = trajectory.value_at(step + 1)
                curr_light = context["light_intensity"]
                my_scene.control_light_intensity(curr_light)

                syn_data_cache = make_syn_data_cache()
                log_reward_history = []
                log_performance_history = []
                log_motion_history = []
                lap_idx += 1
                route_lap_count_at_lap_start = path_follower.lap_count
                if VERBOSE:
                    lap_mode = "warmup" if is_warmup_lap else "linucb"
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
