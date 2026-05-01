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
parser.add_argument("--path_speed", type=float, default=0.45, help="LIMO nominal forward speed in m/s")
parser.add_argument("--lookahead_distance", type=float, default=0.8, help="Pure-pursuit lookahead distance in meters")
parser.add_argument("--angular_gain", type=float, default=1.6, help="Heading-error proportional gain")
parser.add_argument("--max_angular_velocity", type=float, default=1.4, help="Yaw-rate command limit in rad/s")
parser.add_argument("--path_bounds", type=float, nargs=4, default=(-4.0, 4.0, -4.0, 4.0), metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
parser.add_argument("--waypoint_count", type=int, default=8)
parser.add_argument("--spawn_random_objs", action="store_true", help="Spawn random scene objects. The built-in path follower does not avoid them.")
args = parser.parse_args()

RANDOM_SEED = 42
VERBOSE = True
DEBUG = True


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(slots=True)
class VelocityCommand:
    linear_velocity: float
    angular_velocity: float


def quat_wxyz_to_yaw(quat) -> float:
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def get_limo_pose_2d(scene: ATIDepthScene) -> Pose2D:
    position, orientation = scene.agent.get_world_pose()
    return Pose2D(
        x=float(position[0]),
        y=float(position[1]),
        theta=quat_wxyz_to_yaw(orientation),
    )


def vector_angle(w: np.ndarray, v: np.ndarray) -> float:
    return float(np.arctan2(w[1] * v[0] - w[0] * v[1], w[0] * v[0] + w[1] * v[1]))


def nearest_point_on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a2b = b - a
    a2c = c - a
    a2b_mag = float(np.sqrt(np.sum(a2b**2)))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = float(np.dot(a2c, a2b_norm))
    if dist < 0.0:
        return a, dist
    if dist > a2b_mag:
        return b, dist
    return a + a2b_norm * dist, dist


class PathHelper:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2:
            raise ValueError("points must be an Nx2 array with at least two points.")
        self._point_distances = self._init_point_distances()

    def _init_point_distances(self) -> np.ndarray:
        distances = np.zeros(len(self.points), dtype=float)
        length = 0.0
        for idx in range(len(self.points) - 1):
            distances[idx] = length
            length += float(np.linalg.norm(self.points[idx + 1] - self.points[idx]))
        distances[-1] = length
        return distances

    def get_path_length(self) -> float:
        return float(self._point_distances[-1])

    def find_nearest(self, point: np.ndarray):
        min_dist_to_seg = np.inf
        min_seg = (0, 1)
        min_point = self.points[0]
        min_dist_along_path = 0.0
        for a_idx in range(len(self.points) - 1):
            b_idx = a_idx + 1
            nearest_pt, dist_along_seg = nearest_point_on_segment(self.points[a_idx], self.points[b_idx], point)
            dist_to_seg = float(np.linalg.norm(point - nearest_pt))
            if dist_to_seg < min_dist_to_seg:
                min_seg = (a_idx, b_idx)
                min_dist_to_seg = dist_to_seg
                min_point = nearest_pt
                min_dist_along_path = self._point_distances[a_idx] + dist_along_seg
        return min_point, min_dist_along_path, min_seg, min_dist_to_seg

    def get_point_by_distance(self, distance: float, seg_id: int) -> np.ndarray:
        distance = float(np.clip(distance, 0.0, self.get_path_length()))
        seg_id = int(np.clip(seg_id, 0, len(self.points) - 2))
        if distance < self._point_distances[seg_id]:
            candidate_range = range(seg_id, 0, -1)
            for idx in candidate_range:
                if distance >= self._point_distances[idx - 1]:
                    seg_id = idx - 1
                    break
        else:
            for idx in range(seg_id, len(self.points) - 1):
                if distance <= self._point_distances[idx + 1]:
                    seg_id = idx
                    break
        a = self.points[seg_id]
        b = self.points[seg_id + 1]
        a_dist = self._point_distances[seg_id]
        b_dist = self._point_distances[seg_id + 1]
        ratio = np.clip((distance - a_dist) / ((b_dist - a_dist) + 1e-6), 0.0, 1.0)
        return a + ratio * (b - a)


class RandomPathFollower:
    """
    ROS-free random path follower.

    The behavior mirrors MobilityGen's path-following loop: choose a random
    path, pick a lookahead target on that path, then convert heading error into
    differential-drive linear/angular commands.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        waypoint_count: int,
        path_speed: float,
        lookahead_distance: float,
        angular_gain: float,
        max_angular_velocity: float,
        rng: np.random.Generator,
        stop_distance_threshold: float = 0.35,
        forward_angle_threshold: float = np.pi / 3.0,
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = (float(v) for v in bounds)
        self.waypoint_count = max(2, int(waypoint_count))
        self.path_speed = float(path_speed)
        self.lookahead_distance = float(lookahead_distance)
        self.angular_gain = float(angular_gain)
        self.max_angular_velocity = float(max_angular_velocity)
        self.stop_distance_threshold = float(stop_distance_threshold)
        self.forward_angle_threshold = float(forward_angle_threshold)
        self.rng = rng
        self.path = None
        self.path_helper = None
        self.path_id = 0

    def _sample_point(self) -> np.ndarray:
        return np.array([
            self.rng.uniform(self.x_min, self.x_max),
            self.rng.uniform(self.y_min, self.y_max),
        ], dtype=float)

    def _keep_inside_bounds(self, point: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(point[0], self.x_min, self.x_max),
            np.clip(point[1], self.y_min, self.y_max),
        ], dtype=float)

    def set_random_target_path(self, current_pose: Pose2D) -> None:
        start = self._keep_inside_bounds(np.array([current_pose.x, current_pose.y], dtype=float))
        random_points = [self._sample_point() for _ in range(self.waypoint_count - 1)]
        points = np.vstack([start, *random_points])
        self.path = self._smooth_polyline(points)
        self.path_helper = PathHelper(self.path)
        self.path_id += 1

    def _smooth_polyline(self, points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
        smoothed = [points[0]]
        for idx in range(len(points) - 1):
            start = points[idx]
            end = points[idx + 1]
            for sample_idx in range(1, samples_per_segment + 1):
                ratio = sample_idx / samples_per_segment
                eased = ratio * ratio * (3.0 - 2.0 * ratio)
                smoothed.append(start + eased * (end - start))
        return np.asarray(smoothed, dtype=float)

    def step(self, current_pose: Pose2D) -> VelocityCommand:
        if self.path is None or self.path_helper is None:
            self.set_random_target_path(current_pose)

        pt_robot = np.array([current_pose.x, current_pose.y], dtype=float)
        path_end = self.path[-1]
        dist_to_target = float(np.linalg.norm(pt_robot - path_end))
        if dist_to_target < self.stop_distance_threshold:
            self.set_random_target_path(current_pose)

        _, pt_path_length, pt_seg_idx, _ = self.path_helper.find_nearest(pt_robot)
        pt_target = self.path_helper.get_point_by_distance(
            distance=pt_path_length + self.lookahead_distance,
            seg_id=pt_seg_idx[0],
        )

        vec_robot_unit = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)], dtype=float)
        vec_target = pt_target - pt_robot
        target_norm = float(np.linalg.norm(vec_target))
        if target_norm < 1e-6:
            return VelocityCommand(linear_velocity=0.0, angular_velocity=0.0)

        vec_target_unit = vec_target / target_norm
        d_theta = vector_angle(vec_robot_unit, vec_target_unit)
        linear_velocity = 0.0 if abs(d_theta) > self.forward_angle_threshold else self.path_speed
        angular_velocity = -self.angular_gain * d_theta
        angular_velocity = float(np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity))
        return VelocityCommand(linear_velocity=float(linear_velocity), angular_velocity=angular_velocity)


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
            log_motion_history.append({
                "linear_velocity": cmd.linear_velocity,
                "angular_velocity": cmd.angular_velocity,
                "path_id": random_path_follower.path_id,
            })

            syn_data = my_scene.step(render=True)
            rgb_image: np.array = syn_data.get("rgb", None)
            gt_depth: np.array = syn_data.get(my_scene.get_anno("depth"), None)
            bbox_data: np.array = syn_data.get(my_scene.get_anno("2d_bounding_box"), None)
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
                curr_motion_context = motion_summary["abs_angular_velocity"]
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
