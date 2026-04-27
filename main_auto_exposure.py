"""
Framework for evaluating ATI with Isaac Sim under renderer auto-exposure.
The loop mirrors main_egreed_control.py except camera exposure is handled by
the renderer and there is no explicit sensor control policy.
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

from ati_utils.log_utils import configure_isaac_sim_logging, save_synthetic_data, get_eval_averages
from robot_control import build_default_context_trajectory
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from policy import HighlightProtectedHistogramAE, HistogramAEExposureGain, SensorParams
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle

import argparse
from PIL import Image
import traceback
import wandb
import numpy as np


parser = argparse.ArgumentParser(description="ATI evaluation loop with renderer auto-exposure in Isaac Sim")
parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_depthany_autoexposure", help="Name of the experiment for logging purposes")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"], help="Type of reward function used for evaluation")
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype/kaya_autoexposure_eval", help="Directory path to save synthetic data and logs")
parser.add_argument("--max_laps", type=int, default=600, help="Maximum number of laps (context changes) to run in the simulation")
parser.add_argument("--lap_period", type=int, default=30, help="Number of steps per lap (context change period)")
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
            "policy_type": "autoexposure_baseline",
            "rendering_mode": "autoexposure",
            "turn_per_lap": context_len,
            "max_laps": max_laps,
            "max_steps": max_steps,
            "l3_mde_model": "Depth-Anything-V2-Small-hf",
            "reward_type": args.reward_type,
        },
    )

def make_center_weight_mask(h, w, center_ratio=0.6):
    mask = np.zeros((h, w), dtype=np.uint8)
    ch = int(h * center_ratio)
    cw = int(w * center_ratio)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    mask[y0:y0 + ch, x0:x0 + cw] = 1
    return mask


def select_reward_function(reward_type: str):
    if reward_type == "flipped":
        return reward_flipped_img
    if reward_type == "test_time_augment":
        return reward_test_time_augment
    if reward_type == "oracle":
        return reward_oracle
    raise ValueError(f"Invalid reward_type: {reward_type}. Must be one of ['flipped', 'test_time_augment', 'oracle']")


def build_observation_info(
    reward_type: str,
    rgb_image: np.ndarray,
    pred_depths,
    metric_info: dict,
) -> dict:
    if reward_type == "flipped":
        if not np.any(pred_depths[0]):
            raise ValueError("[Fatal Error] Predicted depth is empty or all zeros.")
        return {
            "original_rgb": np.array(rgb_image),
            "depth_original": pred_depths[0],
            "depth_flipped": pred_depths[1],
            "image_weight": 0.1,
            "depth_weight": 0.9,
        }
    if reward_type == "test_time_augment":
        return {
            "rgb": np.array(rgb_image),
            "inverse_depths": pred_depths,
            "uncertainty_reduction": "mean",
            "image_weight": 0.1,
            "depth_weight": 0.9,
        }
    if reward_type == "oracle":
        return {
            "original_rgb": np.array(rgb_image),
            "abs_rel_error": metric_info["abs_rel"],
            "delta_1": metric_info["a1"],
            "image_weight": 0.1,
            "depth_weight": 0.9,
        }
    raise ValueError(f"Unsupported reward_type: {reward_type}")


def average_history(metrics_list: list[dict], key_category: str) -> dict:
    if not metrics_list:
        return {}
    return get_eval_averages(metrics_list, key_category=key_category)


def build_lap_log_payload(
    curr_light: float,
    curr_speed: float,
    log_reward_history: list[dict],
    log_performance_history: list[dict],
) -> dict:
    payload = {
        "context/light_intensity": curr_light,
        "context/agent_speed": curr_speed,
        "baseline/auto_exposure_enabled": 1.0,
        "baseline/valid_steps": len(log_reward_history),
        **average_history(log_reward_history, key_category="reward"),
        **average_history(log_performance_history, key_category="performance"),
    }
    return payload


if __name__ == "__main__":
    CHANGE_CONTEXT_EVERY = args.lap_period
    DATA_PATH = f"{args.data_path}/experiment_ae_{args.exp_name}_{args.reward_type}"
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS
    RAD_COEFF = np.pi / 12
    
    configure_isaac_sim_logging()
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_rendering_autoexposure",
        robot_config=kaya_config,
    )
    render_config.set_rendering_mode("realtime") # Auto-exposure is only supported in realtime mode in this framework
    my_scene = ATIDepthScene(
        simulation_app,
        config=render_config,
        physics_dt=render_config.physics_dt,
        rendering_dt=render_config.rendering_dt,
        stage_units_in_meters=render_config.stage_units_in_meters,
    )

    set_deterministic(RANDOM_SEED)
    reward_function = select_reward_function(args.reward_type)
    context_light = [200, 1000, 3000, 6000, 9000]
    context_agent_speed = [1.5, 2.0, 1.5, 2.0, 1.5]
    # [0.2, 0.5, 1.0, 1.5, 2.0]
    trajectory = build_default_context_trajectory(
        light_values=[1000, 1000, 1000, 1000, 1000],
        speed_values=[s * RAD_COEFF for s in context_agent_speed],
        light_transition_steps=args.lap_period * 20,
        speed_transition_steps=args.lap_period * 10,
        light_hold_steps=args.lap_period,
        speed_hold_steps=args.lap_period,
        speed_phase_offset_steps=args.lap_period,
    )
    context = trajectory.value_at(0)
    curr_light = context["light_intensity"]
    curr_speed = context["angular_velocity"]
    context = {
        "light_intensity": curr_light,
        "angular_velocity": curr_speed,
    }
    my_scene.control_light_intensity(curr_light)
    
    l2_ae_policy = HighlightProtectedHistogramAE(
        target=0.45,
        low_percentile=5,
        high_percentile=95,
        min_exposure=0.001,
        max_exposure=0.03,
        smoothing=0.25,
        max_ev_step=0.5,
    )
    curr_cam_param = my_scene.get_sensor_control_params(sensor_name="agent_camera")
    curr_exposure = curr_cam_param.get("exposure", 0.008)
    curr_gain = curr_cam_param.get("iso", 400)
    
    l3_mde_config = L3MDEConfig(
        reward_type=args.reward_type,
        model_name="depth-anything/Depth-Anything-V2-Small-hf",
        shift_ratios=(0.02, 0.05, 0.1),
        zoom_factors=(0.9, 1.1),
        gaussian_noise_stds=(0.01, 0.02),
        brightness_factors=(0.8, 1.2),
        color_jitter_strengths=(0.05, 0.1),
        disable_hflip=False,
        prediction_mode="identity",
    )
    mde_model = L3PLayerDepthAnythingv2(l3_config=l3_mde_config, device="cuda")
    rng = np.random.default_rng(RANDOM_SEED)

    step = 0
    lap_idx = 0
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": [],
    }
    log_reward_history = []
    log_performance_history = []
    wandb_run = initialize_wandb(
        context_len=CHANGE_CONTEXT_EVERY,
        max_laps=MAX_LAPS,
        max_steps=MAX_STEPS,
        exp_name=f"ati_kaya_depthany_{l3_mde_config.reward_type}_{args.exp_name}",
    )

    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            if VERBOSE:
                print(f"Step: {step+1}/{MAX_STEPS}, Simulation Time: {my_scene.get_simulation_current_time:.4f} seconds")

            syn_data = my_scene.step(render=True)
            my_scene.robot_control(
                time=my_scene.get_simulation_current_time,
                control_parameters={
                    "angular_velocity": context["angular_velocity"],
                },
            )

            rgb_image: np.ndarray = syn_data.get("rgb", None)
            gt_depth: np.ndarray = syn_data.get(my_scene.get_anno("depth"), None)
            bbox_data: np.ndarray = syn_data.get(my_scene.get_anno("2d_bounding_box"), None)
            if rgb_image is None or rgb_image.size == 0:
                if VERBOSE:
                    print("Warning: Received empty RGB image. Skipping this step.")
                continue
            if gt_depth is None or gt_depth.size == 0:
                if VERBOSE:
                    print("Warning: Received empty ground-truth depth image. Skipping this step.")
                continue
            if bbox_data is None or bbox_data["data"].size == 0:
                if VERBOSE:
                    print("Warning: Received empty bounding box data. Skipping this step.")
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
            reward_info = reward_function(**observation_info)

            log_reward_history.append(reward_info)
            log_performance_history.append(metric_info)
            if DEBUG:
                print("[DEBUG] AutoExposure Reward Result:", reward_info)

            h, w = rgb_image.shape[:2]
            mask = make_center_weight_mask(h, w, center_ratio=0.6)   
            next_exposure, next_gain, info = l2_ae_policy.update(
                rgb_image=rgb_image,
                current_exposure=curr_exposure,
                current_gain=curr_gain,
                mask=mask,
            )
            my_scene.sensor_control(
                control_parameters={
                    "iso": next_gain,
                    "shutter_time": next_exposure,
                }
            )
            curr_exposure = next_exposure
            curr_gain = next_gain

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0 and step > 0:
                wandb_run.log(
                    build_lap_log_payload(
                        curr_light=context["light_intensity"],
                        curr_speed=context["angular_velocity"],
                        log_reward_history=log_reward_history,
                        log_performance_history=log_performance_history,
                    ),
                    step=lap_idx,
                    commit=True,
                )
                
                context = trajectory.value_at(lap_idx)
                curr_light = context["light_intensity"]
                curr_speed = context["angular_velocity"]
                my_scene.control_light_intensity(curr_light)
                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)                
                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                    "pred_depth": [],
                }
                log_reward_history = []
                log_performance_history = []
                lap_idx += 1

                if VERBOSE:
                    print(
                        f"Context changed at step {step+1}: "
                        f"Light Intensity set to {context['light_intensity']}, Agent Speed set to {context['angular_velocity']}"
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
        if wandb_run is not None and (log_reward_history or log_performance_history):
            wandb_run.log(
                build_lap_log_payload(
                    curr_light=context["light_intensity"],
                    curr_speed=context["angular_velocity"],
                    log_reward_history=log_reward_history,
                    log_performance_history=log_performance_history,
                ),
                step=lap_idx,
                commit=True,
            )
            if any(len(v) > 0 for v in syn_data_cache.values()):
                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)

        print("[Main] Shutting down...")
        try:
            if wandb_run is not None:
                wandb_run.finish()
        except Exception:
            print("")
        try:
            simulation_app.close()
        except Exception:
            print("")
        print("[Main] Done. If process hangs, run: pkill -f 'python.sh|kit'")

    simulation_app.close()
