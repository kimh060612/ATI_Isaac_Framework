"""
Framework for Fine-grained Sensor Control Logic of ATI with Isaac Sim
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

# Scene Building
from ati_utils.log_utils import configure_isaac_sim_logging, save_synthetic_data, get_eval_averages
from robot_control import build_default_context_trajectory, build_step_context_trajectory
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle
from policy import SensorParamSpace
from policy.DiscreteUCBPolicy import L2DiscreteDisjointLinUCBRGBCamPolicy
import argparse
import fcntl
import json
from PIL import Image
import traceback
import wandb
import numpy as np
import random
import os 
import time

parser = argparse.ArgumentParser(description="ATI Sensor Control with L2-L3 Feedback Loop in Isaac Sim")
parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_depthany_oracle", help="Name of the experiment for logging purposes")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"], help="Type of reward function to use for the L2 policy")
parser.add_argument("--data_path", type=str, default="/home/kimh060612/ATI_research/dataset", help="Directory path to save synthetic data and logs")
parser.add_argument("--max_laps", type=int, default=600, help="Maximum number of laps (context changes) to run in the simulation")
parser.add_argument("--lap_period", type=int, default=30, help="Number of steps per lap (context change period)")
parser.add_argument("--exp_ratio", type=float, default=0.5, help="Ratio of exploration vs exploitation for the L2 policy's action selection")
parser.add_argument("--trajectory_mode", type=str, default="step", choices=["step", "smooth"], help="Context trajectory mode for discrete training")
parser.add_argument("--light_values", type=str, default="500,1000,3000,6000", help="Comma-separated light-intensity context values")
parser.add_argument("--angular_speed_values", type=str, default="2.5,5,7.5,10,12.5", help="Comma-separated angular-speed context values used by the policy")
parser.add_argument("--angular_speed_scale", type=float, default=float(np.pi / 12), help="Scale policy angular speed before sending it to robot_control")
parser.add_argument("--imu_speed_context_scale", type=float, default=None, help="Scale IMU gyro_magnitude before using it as the discrete policy speed context. Defaults to 1 / angular_speed_scale.")
parser.add_argument("--q_table_path", type=str, default=None, help="Shared JSON file that stores max context-action Q-values for inference")
parser.add_argument("--disable_q_table_save", action="store_true", help="Disable shared context-action Q-table saving")
args = parser.parse_args()

RANDOM_SEED = 42
VERBOSE=True
DEBUG = True

def initialize_wandb(context_len, max_laps, max_steps, exp_name=None, q_table_path=None):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=exp_name,
        config={
            "policy_type": "L2DiscreteDisjointLinUCBRGBCamPolicy",
            "turn_per_lap": context_len,
            "max_laps": max_laps,
            "max_steps": max_steps,
            "l3_mde_model": "Depth-Anything-V2-Small-hf",
            "trajectory_mode": args.trajectory_mode,
            "light_values": args.light_values,
            "angular_speed_values": args.angular_speed_values,
            "angular_speed_scale": args.angular_speed_scale,
            "imu_speed_context_scale": args.imu_speed_context_scale,
            "q_table_path": q_table_path,
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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_observed_policy_context(
    scene: ATIDepthScene,
    syn_data: dict | None = None,
    *,
    fallback_light: float,
    fallback_speed: float,
    speed_context_scale: float,
) -> dict:
    syn_data = syn_data or {}
    canonical_context = syn_data.get("canonical_context")
    if not isinstance(canonical_context, dict):
        canonical_context = scene.get_policy_context(
            imu_frame=syn_data.get("imu_sensor"),
            light_meter_info=syn_data.get("light_meter"),
        )

    imu_gyro_magnitude = float(canonical_context.get("gyro_magnitude", 0.0))
    policy_speed = imu_gyro_magnitude * float(speed_context_scale)
    if not np.isfinite(policy_speed):
        policy_speed = float(fallback_speed)

    light_intensity = float(canonical_context.get("light_intensity", fallback_light))
    if not np.isfinite(light_intensity):
        light_intensity = float(fallback_light)

    return {
        "light_intensity": light_intensity,
        "angular_velocity": float(policy_speed),
        "imu_gyro_magnitude": imu_gyro_magnitude,
        "imu_acceleration_magnitude": float(canonical_context.get("acceleration_magnitude", 0.0)),
    }


def action_to_key(action: tuple[int, int]) -> str:
    return f"{int(action[0])},{int(action[1])}"


def action_from_key(action_key: str) -> list[int]:
    de, di = action_key.split(",", maxsplit=1)
    return [int(de), int(di)]


def threshold_labels(thresholds: tuple[float, ...]) -> list[str]:
    labels = [f"<= {threshold:g}" for threshold in thresholds]
    labels.append(f"> {thresholds[-1]:g}" if thresholds else "all")
    return labels


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_q_table_payload(policy: L2DiscreteDisjointLinUCBRGBCamPolicy) -> dict:
    return {
        "version": 1,
        "policy_type": "L2DiscreteDisjointLinUCBRGBCamPolicy",
        "q_value_semantics": "LinUCB posterior mean x @ theta_a; each context-action entry keeps the maximum value observed across writers.",
        "angular_speed_thresholds": list(policy.ANGULAR_SPEED_THRESHOLDS),
        "light_intensity_thresholds": list(policy.LIGHT_INTENSITY_THRESHOLDS),
        "angular_speed_bin_labels": threshold_labels(policy.ANGULAR_SPEED_THRESHOLDS),
        "light_intensity_bin_labels": threshold_labels(policy.LIGHT_INTENSITY_THRESHOLDS),
        "actions": [
            {
                "key": action_to_key(action),
                "delta_exposure": int(action[0]),
                "delta_iso": int(action[1]),
            }
            for action in policy.action_space
        ],
        "contexts": {},
    }


def build_context_q_entry(
    policy: L2DiscreteDisjointLinUCBRGBCamPolicy,
    *,
    angular_velocity: float,
    light_intensity: float,
    lap_idx: int,
    exp_name: str,
    reward_type: str,
    selected_action: tuple[int, int] | None,
) -> tuple[str, dict]:
    context = policy.build_context(
        angular_velocity=angular_velocity,
        light_intensity=light_intensity,
    )
    angular_bin, light_bin = policy.context_bin_indices(
        angular_velocity=angular_velocity,
        light_intensity=light_intensity,
    )
    context_key = f"speed_bin_{angular_bin}__light_bin_{light_bin}"
    updated_at = utc_timestamp()

    q_values = {}
    for action in policy.action_space:
        score, mean, bonus = policy.ucb_score(context, action)
        if not np.isfinite(mean):
            continue
        q_values[action_to_key(action)] = {
            "action": [int(action[0]), int(action[1])],
            "q_value": float(mean),
            "ucb_score": float(score),
            "ucb_bonus": float(bonus),
            "lap_idx": int(lap_idx),
            "source_exp_name": exp_name,
            "source_reward_type": reward_type,
            "source_pid": int(os.getpid()),
            "updated_at": updated_at,
        }

    best_action_key = None
    best_q_value = None
    if q_values:
        best_action_key, best_payload = max(
            q_values.items(),
            key=lambda item: float(item[1]["q_value"]),
        )
        best_q_value = float(best_payload["q_value"])

    return context_key, {
        "angular_speed_bin": int(angular_bin),
        "light_intensity_bin": int(light_bin),
        "sample_angular_velocity": float(angular_velocity),
        "sample_light_intensity": float(light_intensity),
        "selected_action": list(selected_action) if selected_action is not None else None,
        "best_action_key": best_action_key,
        "best_action": action_from_key(best_action_key) if best_action_key is not None else None,
        "best_q_value": best_q_value,
        "q_values": q_values,
    }


def merge_context_q_table(
    q_table_path: str,
    policy: L2DiscreteDisjointLinUCBRGBCamPolicy,
    context_key: str,
    context_entry: dict,
) -> None:
    q_table_path = os.path.abspath(q_table_path)
    os.makedirs(os.path.dirname(q_table_path), exist_ok=True)
    lock_path = f"{q_table_path}.lock"
    updated_at = utc_timestamp()

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if os.path.exists(q_table_path):
                with open(q_table_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            else:
                payload = make_q_table_payload(policy)

            payload.setdefault("contexts", {})
            payload.setdefault("actions", make_q_table_payload(policy)["actions"])
            payload.setdefault("angular_speed_thresholds", list(policy.ANGULAR_SPEED_THRESHOLDS))
            payload.setdefault("light_intensity_thresholds", list(policy.LIGHT_INTENSITY_THRESHOLDS))

            stored_context = payload["contexts"].setdefault(
                context_key,
                {
                    "angular_speed_bin": int(context_entry["angular_speed_bin"]),
                    "light_intensity_bin": int(context_entry["light_intensity_bin"]),
                    "sample_angular_velocity": float(context_entry["sample_angular_velocity"]),
                    "sample_light_intensity": float(context_entry["sample_light_intensity"]),
                    "q_values": {},
                },
            )
            stored_context["angular_speed_bin"] = int(context_entry["angular_speed_bin"])
            stored_context["light_intensity_bin"] = int(context_entry["light_intensity_bin"])
            stored_context["sample_angular_velocity"] = float(context_entry["sample_angular_velocity"])
            stored_context["sample_light_intensity"] = float(context_entry["sample_light_intensity"])

            stored_q_values = stored_context.setdefault("q_values", {})
            for action_key, candidate in context_entry["q_values"].items():
                previous = stored_q_values.get(action_key)
                previous_q = -np.inf if previous is None else float(previous.get("q_value", -np.inf))
                candidate_q = float(candidate["q_value"])
                if candidate_q > previous_q:
                    stored_q_values[action_key] = candidate

            if stored_q_values:
                best_action_key, best_payload = max(
                    stored_q_values.items(),
                    key=lambda item: float(item[1]["q_value"]),
                )
                stored_context["best_action_key"] = best_action_key
                stored_context["best_action"] = action_from_key(best_action_key)
                stored_context["best_q_value"] = float(best_payload["q_value"])
                stored_context["best_source_exp_name"] = best_payload.get("source_exp_name")
                stored_context["best_source_pid"] = best_payload.get("source_pid")
                stored_context["best_lap_idx"] = best_payload.get("lap_idx")

            stored_context["updated_at"] = updated_at
            payload["updated_at"] = updated_at
            payload["num_contexts"] = len(payload["contexts"])

            tmp_path = f"{q_table_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, q_table_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

if __name__ == "__main__":
    # Which directory name will be cool and awesome?
    ## Plz recommend some fun, cool, sexy directory names...
    CHANGE_CONTEXT_EVERY = args.lap_period
    DATA_PATH = f"{args.data_path}/experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps"
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS
    RAD_COEFF = args.angular_speed_scale
    IMU_SPEED_CONTEXT_SCALE = args.imu_speed_context_scale
    if IMU_SPEED_CONTEXT_SCALE is None:
        IMU_SPEED_CONTEXT_SCALE = 1.0 / RAD_COEFF if abs(RAD_COEFF) > 1e-12 else 1.0
    Q_TABLE_PATH = args.q_table_path
    if Q_TABLE_PATH is None:
        Q_TABLE_PATH = os.path.join(
            args.data_path,
            f"discrete_context_q_table_{args.reward_type}.json",
        )
    if not args.disable_q_table_save:
        print(f"[Main] Shared discrete Q-table path: {Q_TABLE_PATH}")
    
    # Isaac Sim Scene Setup
    configure_isaac_sim_logging() # Set Isaac Sim logging level to Error to avoid cluttering
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_rendering_test",
        robot_config=kaya_config,
    )
    render_config.set_agent_sensor_controller("exposure_iso_controller")
    render_config.set_rendering_mode("realtime")
    render_config.set_pathtracing_param(spp=128, num_subsamples=32)
    my_scene = ATIDepthScene(
        simulation_app,
        config=render_config,
        physics_dt=render_config.physics_dt,
        rendering_dt=render_config.rendering_dt,
        stage_units_in_meters=render_config.stage_units_in_meters
    )
    
    ## L2 Policy and Reward Layer Setup
    set_deterministic(RANDOM_SEED)
    context_light = parse_float_list(args.light_values)
    context_agent_speed = parse_float_list(args.angular_speed_values)
    if args.trajectory_mode == "step":
        trajectory = build_step_context_trajectory(
            light_values=context_light,
            speed_values=context_agent_speed,
            light_hold_steps=args.lap_period,
            speed_hold_steps=args.lap_period,
            speed_phase_offset_steps=0,
        )
    else:
        trajectory = build_default_context_trajectory(
            light_values=context_light,
            speed_values=context_agent_speed,
            light_transition_steps=args.lap_period * 20,
            speed_transition_steps=args.lap_period * 5,
            light_hold_steps=args.lap_period,
            speed_hold_steps=args.lap_period,
            speed_phase_offset_steps=args.lap_period,
        )
    
    sensor_param_space = SensorParamSpace()
    policy_kwargs = {
        "sensor_names": "agent_camera",
        "sensor_config": sensor_param_space,
        "reward_function": select_reward_function(args.reward_type), # reward_flipped_img or reward_test_time_augment or reward_oracle
        "alpha": args.exp_ratio, # Exploration vs Exploitation ratio for LinUCB
        "random_seed": RANDOM_SEED,
    }
    l2_policy = L2DiscreteDisjointLinUCBRGBCamPolicy(**policy_kwargs)
    context = trajectory.value_at(0)
    curr_light = context["light_intensity"]
    curr_speed = context["angular_velocity"]
    
    curr_exposure_idx = len(sensor_param_space.exposure_values) // 2
    curr_iso_idx = len(sensor_param_space.iso_values) // 2
    my_scene.control_light_intensity(curr_light)
    my_scene.robot_control(
        time=my_scene.get_simulation_current_time,
        control_parameters={
            "angular_velocity": curr_speed * RAD_COEFF
        }
    )
    initial_policy_context = build_observed_policy_context(
        my_scene,
        fallback_light=curr_light,
        fallback_speed=curr_speed,
        speed_context_scale=IMU_SPEED_CONTEXT_SCALE,
    )
    initial_policy_result = l2_policy.step(
        context_information={
            "light_intensity": initial_policy_context["light_intensity"],
            "angular_velocity": initial_policy_context["angular_velocity"],
            "iso_idx": curr_iso_idx,
            "exposure_idx": curr_exposure_idx,
            "tie_break_random": True,
            "skip_update": True,
            "store_pending": True,
        },
        observations={
            "reward_info_override": get_avg_aggregation([]),
        },
    )
    initial_policy_result["imu_gyro_magnitude"] = initial_policy_context["imu_gyro_magnitude"]
    initial_policy_result["imu_acceleration_magnitude"] = initial_policy_context["imu_acceleration_magnitude"]
    initial_policy_result["env_light_intensity"] = curr_light
    initial_policy_result["env_agent_speed"] = curr_speed
    curr_exposure_idx = initial_policy_result["next_exposure_idx"]
    curr_iso_idx = initial_policy_result["next_iso_idx"]
    current_lap_policy_result = initial_policy_result

    ## Initial Sensor Control
    my_scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[curr_iso_idx],
            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx]
        }
    )
    
    ## L3 Perception Layer Setup
    l3_mde_config = L3MDEConfig(
        reward_type=args.reward_type, # "flipped" or "test_time_augment" or "oracle"
        model_name="depth-anything/Depth-Anything-V2-Small-hf",
        shift_ratios=[],
        zoom_factors=[],
        gaussian_noise_stds=(0.01, 0.02, 0.05),
        brightness_factors=(), # 0.8, 0.9, 1.1, 1.2
        color_jitter_strengths=(0.1, 0.15),
        disable_hflip=False,
        prediction_mode="identity",
    )
    mde_model = L3PLayerDepthAnythingv2(l3_config=l3_mde_config, device="cuda")
    rng = np.random.default_rng(RANDOM_SEED)
    
    ## Main L2-L3 FeedBack Loop for Sensor Control Logic
    step = 0
    lap_idx = 0
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": []
    }
    wandb_run = initialize_wandb(
        context_len=CHANGE_CONTEXT_EVERY,
        max_laps=MAX_LAPS,
        max_steps=MAX_STEPS,
        exp_name=f"ati_kaya_depthany_{l3_mde_config.reward_type}_{args.exp_name}",
        q_table_path=Q_TABLE_PATH,
    )
    log_context_history = []
    log_reward_history = []
    log_performance_history = []
    
    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            if VERBOSE: print(f"Step: {step+1}/{MAX_STEPS}, Simulation Time: {my_scene.get_simulation_current_time:.4f} seconds")
            ### Simulation Step and Synthetic Data Generation
            syn_data = my_scene.step(render=True)
            my_scene.robot_control(
                time=my_scene.get_simulation_current_time,
                control_parameters={
                    "angular_velocity": curr_speed * RAD_COEFF
                }
            )
            
            rgb_image: np.array = syn_data.get("rgb", None)
            gt_depth: np.array = syn_data.get(my_scene.get_anno("depth"), None)
            bbox_data: np.array = syn_data.get(my_scene.get_anno("2d_bounding_box"), None)
            if rgb_image is None or rgb_image.size == 0:
                if VERBOSE: print("Warning: Received empty RGB image. Skipping this step.")
                continue
            if gt_depth is None or gt_depth.size == 0:
                if VERBOSE: print("Warning: Received empty ground-truth depth image. Skipping this step.")
                continue
            # if bbox_data is None or bbox_data["data"].size == 0:
            #     print(bbox_data)
            #     if VERBOSE: print("Warning: Received empty bounding box data. Skipping this step.")
            #     continue
            syn_data_cache["rgb"].append(rgb_image)
            syn_data_cache["depth"].append(gt_depth)
            syn_data_cache["bbox"].append(bbox_data)
            observed_policy_context = build_observed_policy_context(
                my_scene,
                syn_data,
                fallback_light=curr_light,
                fallback_speed=curr_speed,
                speed_context_scale=IMU_SPEED_CONTEXT_SCALE,
            )
            log_context_history.append(observed_policy_context)
            
            pred_depths, metric_info = mde_model.predict_depth([Image.fromarray(rgb_image)], gt_depth)
            syn_data_cache["pred_depth"].append(pred_depths if isinstance(pred_depths, np.ndarray) else pred_depths[0])
            if DEBUG: print(f"Depth Prediction Metrics: {metric_info}")
            
            observation_info = build_observation_info(
                reward_type=l3_mde_config.reward_type,
                rgb_image=rgb_image,
                pred_depths=pred_depths,
                metric_info=metric_info
            )
            reward_info = l2_policy.reward_function(**observation_info)
            log_reward_history.append(reward_info)
            log_performance_history.append(metric_info)

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0 and step > 0:
                completed_env_light = curr_light
                completed_env_speed = curr_speed
                completed_policy_light = float(current_lap_policy_result["light_intensity"])
                completed_policy_speed = float(current_lap_policy_result["angular_velocity"])
                completed_iso_idx = curr_iso_idx
                completed_exposure_idx = curr_exposure_idx
                completed_policy_result = current_lap_policy_result
                lap_reward_info = get_avg_aggregation(log_reward_history)
                next_lap_idx = lap_idx + 1
                has_next_lap = next_lap_idx < MAX_LAPS
                next_step = step + 1
                next_env_context = trajectory.value_at(next_step)
                if has_next_lap:
                    my_scene.control_light_intensity(next_env_context["light_intensity"])
                    my_scene.robot_control(
                        time=my_scene.get_simulation_current_time,
                        control_parameters={
                            "angular_velocity": next_env_context["angular_velocity"] * RAD_COEFF
                        }
                    )
                    next_policy_context = build_observed_policy_context(
                        my_scene,
                        fallback_light=next_env_context["light_intensity"],
                        fallback_speed=next_env_context["angular_velocity"],
                        speed_context_scale=IMU_SPEED_CONTEXT_SCALE,
                    )
                else:
                    next_policy_context = observed_policy_context
                result = l2_policy.step(
                    context_information={
                        "light_intensity": next_policy_context["light_intensity"],
                        "angular_velocity": next_policy_context["angular_velocity"],
                        "iso_idx": curr_iso_idx,
                        "exposure_idx": curr_exposure_idx,
                        "tie_break_random": True,
                        "store_pending": has_next_lap,
                    },
                    observations={
                        "reward_info_override": lap_reward_info,
                    }
                )
                update_info = result.get("update_info")
                result["imu_gyro_magnitude"] = next_policy_context["imu_gyro_magnitude"]
                result["imu_acceleration_magnitude"] = next_policy_context["imu_acceleration_magnitude"]
                result["env_light_intensity"] = next_env_context["light_intensity"]
                result["env_agent_speed"] = next_env_context["angular_velocity"]
                q_table_context_key = None
                if not args.disable_q_table_save:
                    selected_action = None
                    if completed_policy_result is not None:
                        selected_action = completed_policy_result["action"]
                    q_table_context_key, q_table_entry = build_context_q_entry(
                        l2_policy,
                        angular_velocity=completed_policy_speed,
                        light_intensity=completed_policy_light,
                        lap_idx=lap_idx,
                        exp_name=args.exp_name,
                        reward_type=args.reward_type,
                        selected_action=selected_action,
                    )
                    merge_context_q_table(
                        Q_TABLE_PATH,
                        l2_policy,
                        q_table_context_key,
                        q_table_entry,
                    )
                # **get_eval_averages(log_context_history, key_category="context"),
                wandb_run.log(
                    {
                        "scenario/env_light_intensity": completed_env_light,
                        "scenario/env_agent_speed": completed_env_speed,
                        "scenario/env_robot_angular_velocity": completed_env_speed * RAD_COEFF,
                        "context/light_intensity": completed_policy_light,
                        "context/agent_speed": completed_policy_speed,
                        "context/imu_gyro_magnitude": float(completed_policy_result.get("imu_gyro_magnitude", 0.0)),
                        "context/imu_acceleration_magnitude": float(completed_policy_result.get("imu_acceleration_magnitude", 0.0)),
                        "context/imu_speed_context_scale": float(IMU_SPEED_CONTEXT_SCALE),
                        **({
                            "context/angular_speed_bin": float(completed_policy_result["angular_speed_bin"]),
                            "context/light_intensity_bin": float(completed_policy_result["light_intensity_bin"]),
                        } if completed_policy_result is not None else {}),
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
                            "policy/update_raw_reward": float(update_info.get("raw_reward", update_info["reward"])),
                            "policy/update_advantage": float(update_info.get("advantage", update_info["reward"])),
                            "policy/update_baseline": float(update_info.get("baseline", 0.0)),
                            "policy/update_baseline_count": float(update_info.get("baseline_count", 0)),
                            "policy/action_idx": float(update_info["action_idx"]),
                        } if update_info is not None else {}),
                        **({
                            "policy/next_env_light_intensity": float(next_env_context["light_intensity"]),
                            "policy/next_env_agent_speed": float(next_env_context["angular_velocity"]),
                            "policy/next_env_robot_angular_velocity": float(next_env_context["angular_velocity"] * RAD_COEFF),
                            "policy/next_light_intensity": float(next_policy_context["light_intensity"]),
                            "policy/next_agent_speed": float(next_policy_context["angular_velocity"]),
                            "policy/next_imu_gyro_magnitude": float(next_policy_context["imu_gyro_magnitude"]),
                            "policy/next_imu_acceleration_magnitude": float(next_policy_context["imu_acceleration_magnitude"]),
                            "policy/next_angular_speed_bin": float(result["angular_speed_bin"]),
                            "policy/next_light_intensity_bin": float(result["light_intensity_bin"]),
                            "policy/next_iso_idx": float(result["next_iso_idx"]),
                            "policy/next_exposure_idx": float(result["next_exposure_idx"]),
                        } if has_next_lap else {}),
                        **({
                            "policy/q_table_saved": 1.0,
                        } if q_table_context_key is not None else {
                            "policy/q_table_saved": 0.0,
                        }),
                        **get_eval_averages(log_reward_history, key_category="reward"),
                        **get_eval_averages(log_performance_history, key_category="performance"),
                        **get_eval_averages(log_context_history, key_category="context_avg"),
                    }, 
                    step=lap_idx,
                    commit=True
                )
                if DEBUG: print("[DEBUG]Policy Step Reward Result:", result["reward_info"])
                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)
                # "More smooth and Moderately changing the context for the agent to adapt to new conditions 
                # while avoiding drastic changes that could destabilize learning."
                if has_next_lap:
                    curr_light = next_env_context["light_intensity"]
                    curr_speed = next_env_context["angular_velocity"]
                    curr_exposure_idx = result["next_exposure_idx"]
                    curr_iso_idx = result["next_iso_idx"]
                    current_lap_policy_result = result

                    if DEBUG: print("[DEBUG] Sensor Control Action Taken - Exposure Index:", curr_exposure_idx, "ISO Index:", curr_iso_idx)
                    my_scene.sensor_control(
                        control_parameters={
                            "iso": sensor_param_space.iso_values[curr_iso_idx],
                            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx]
                        }
                    )
                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                    "pred_depth": []
                }
                log_reward_history = []
                log_performance_history = []
                log_context_history = []
                lap_idx += 1
                if VERBOSE: 
                    print(
                        f"Context changed at step {step+1}: "
                        f"Light Intensity set to {curr_light}, "
                        f"Command Speed set to {curr_speed}, "
                        f"Observed Policy Speed set to {result['angular_velocity']}, "
                        f"Robot Angular Velocity set to {curr_speed * RAD_COEFF}"
                    )
            
            if step >= MAX_STEPS - 1:
                print(f"All laps completed. Ending simulation")
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


# def wandb_log_step(
#     wandb_run: wandb.Run, 
#     lap_idx: int,
#     context_info: dict,
#     context_light: float,
#     context_agent_speed: float,
#     reward_info: dict, 
#     metric_info: dict
# ):
#     wandb_run.log({
#         "light_intensity": context_light,
#         "agent_speed": context_agent_speed,
#         **context_info,
#         **reward_info,
#         **metric_info,
#     }, step=lap_idx)

# wandb_log_step(
#     wandb_run=wandb_run, 
#     lap_idx=lap_idx, 
#     context_info=get_eval_averages(log_context_history, key_category="context"), 
#     reward_info=get_eval_averages(log_reward_history, key_category="reward"), 
#     metric_info=get_eval_averages(log_performance_history, key_category="performance"),  
#     context_light=curr_light, 
#     context_agent_speed=curr_speed 
# )
