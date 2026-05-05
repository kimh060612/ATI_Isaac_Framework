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

# Enable Livestream extension
from isaacsim.core.utils.extensions import enable_extension
simulation_app.set_setting("/app/window/drawMouse", True)
enable_extension("omni.services.livestream.nvcf")

# Scene Building
from ati_utils.log_utils import configure_isaac_sim_logging, save_synthetic_data, get_eval_averages
from robot_control import build_default_context_trajectory, build_step_context_trajectory
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle
from policy import L2SharedLinUCBRGBCamPolicy, L2DisjointLinUCBRGBCamPolicy, SensorParamSpace
import argparse
from PIL import Image
import traceback
import wandb
import numpy as np
import random
import os 

parser = argparse.ArgumentParser(description="ATI Sensor Control with L2-L3 Feedback Loop in Isaac Sim")
parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_depthany_oracle", help="Name of the experiment for logging purposes")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"], help="Type of reward function to use for the L2 policy")
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype/kaya_awesome_naming", help="Directory path to save synthetic data and logs")
parser.add_argument("--max_laps", type=int, default=600, help="Maximum number of laps (context changes) to run in the simulation")
parser.add_argument("--lap_period", type=int, default=30, help="Number of steps per lap (context change period)")
args = parser.parse_args()

RANDOM_SEED = 42
VERBOSE=True
DEBUG = True

def initialize_wandb(context_len, max_laps, max_steps, exp_name=None):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=exp_name,
        config={
            "policy_type": "L2DisjointLinUCBRGBCamPolicy",
            "turn_per_lap": context_len,
            "max_laps": max_laps,
            "max_steps": max_steps,
            "l3_mde_model": "Depth-Anything-V2-Small-hf",
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
            "image_weight": 0.0,
            "depth_weight": 1.0,
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

if __name__ == "__main__":
    # Which directory name will be cool and awesome?
    ## Plz recommend some fun, cool, sexy directory names...
    CHANGE_CONTEXT_EVERY = args.lap_period
    DATA_PATH = f"{args.data_path}/experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps"
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS
    RAD_COEFF = np.pi / 12
    
    # Isaac Sim Scene Setup
    configure_isaac_sim_logging() # Set Isaac Sim logging level to Error to avoid cluttering
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_rendering_test",
        robot_config=kaya_config,
    )
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
    context_light = [200, 1000, 3000, 6000, 9000]  # Example light intensity values for the agent's context
    context_agent_speed = [1.0, 1.0, 1.0, 1.0, 1.0]
    # [0.2, 0.5, 1.0, 1.5, 2.0]  # Example speed values for the agent's context
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
        reward_function=select_reward_function(args.reward_type), # reward_flipped_img or reward_test_time_augment or reward_oracle
        alpha=1.0,
        random_seed=RANDOM_SEED,
    )
    context = trajectory.value_at(0)
    curr_light = context["light_intensity"]
    curr_speed = context["angular_velocity"]
    context = {
        "light_intensity": curr_light,
        "angular_velocity": curr_speed,
    }
    
    curr_exposure_idx = len(sensor_param_space.exposure_values) // 2
    curr_iso_idx = len(sensor_param_space.iso_values) // 2
    # curr_light = context_light[len(context_light) // 2]
    # curr_speed = context_agent_speed[len(context_agent_speed) // 2] * np.pi / 12
    initial_policy_result = l2_policy.step(
        context_information={
            "light_intensity": curr_light,
            "angular_velocity": curr_speed,
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
    curr_exposure_idx = initial_policy_result["next_exposure_idx"]
    curr_iso_idx = initial_policy_result["next_iso_idx"]
    current_lap_policy_result = initial_policy_result

    ## Initial Sensor Control
    my_scene.control_light_intensity(curr_light) # Set initial light intensity
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
        brightness_factors=(0.8, 0.9),
        color_jitter_strengths=[],
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
        exp_name=f"ati_kaya_depthany_{l3_mde_config.reward_type}_{args.exp_name}"
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
                    "angular_velocity": curr_speed
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
                completed_light = curr_light
                completed_speed = curr_speed
                completed_iso_idx = curr_iso_idx
                completed_exposure_idx = curr_exposure_idx
                completed_policy_result = current_lap_policy_result
                lap_reward_info = get_avg_aggregation(log_reward_history)
                next_lap_idx = lap_idx + 1
                has_next_lap = next_lap_idx < MAX_LAPS
                next_step = step + 1
                next_context = trajectory.value_at(next_step)
                result = l2_policy.step(
                    context_information={
                        "light_intensity": next_context["light_intensity"],
                        "angular_velocity": next_context["angular_velocity"],
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
                # **get_eval_averages(log_context_history, key_category="context"),
                wandb_run.log(
                    {
                        "context/light_intensity": completed_light,
                        "context/agent_speed": completed_speed,
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
                            "policy/next_agent_speed": float(next_context["angular_velocity"]),
                            "policy/next_iso_idx": float(result["next_iso_idx"]),
                            "policy/next_exposure_idx": float(result["next_exposure_idx"]),
                        } if has_next_lap else {}),
                        **get_eval_averages(log_reward_history, key_category="reward"),
                        **get_eval_averages(log_performance_history, key_category="performance"),
                    }, 
                    step=lap_idx,
                    commit=True
                )
                if DEBUG: print("[DEBUG]Policy Step Reward Result:", result["reward_info"])
                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)
                # "More smooth and Moderately changing the context for the agent to adapt to new conditions 
                # while avoiding drastic changes that could destabilize learning."
                if has_next_lap:
                    curr_light = next_context["light_intensity"]
                    curr_speed = next_context["angular_velocity"]
                    curr_exposure_idx = result["next_exposure_idx"]
                    curr_iso_idx = result["next_iso_idx"]
                    current_lap_policy_result = result
                    log_context_history.append({
                        "iso_idx": curr_iso_idx,
                        "exposure_idx": curr_exposure_idx
                    })

                    if DEBUG: print("[DEBUG] Sensor Control Action Taken - Exposure Index:", curr_exposure_idx, "ISO Index:", curr_iso_idx)
                    my_scene.control_light_intensity(curr_light)
                    my_scene.sensor_control(
                        control_parameters={
                            "iso": sensor_param_space.iso_values[curr_iso_idx],
                            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
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
                lap_idx += 1
                if VERBOSE: 
                    print(f"Context changed at step {step+1}: Light Intensity set to {curr_light}, Agent Speed set to {curr_speed}")
            
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
