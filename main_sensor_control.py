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
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment
from policy import L2SharedLinUCBRGBCamPolicy, SensorParamSpace
from PIL import Image
from time import time
import traceback
import wandb
import numpy as np
import random
import os 

RANDOM_SEED = 42
CHANGE_CONTEXT_EVERY = 30
MAX_LAPS = 600
MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS
VERBOSE=True
DEBUG = True

# Which directory name will be cool and awesome?
## Plz recommend some fun, cool, sexy directory names...
DATA_PATH = "/issac-sim/dataset/experiment_mde_prototype/kaya_awesome_naming"

def initialize_wandb(exp_name=None):
    return wandb.init(
        entity="artificial_tripartite_intelligence_team",
        project="ati_sensor_control_prototype",
        name=exp_name,
        config={
            "policy_type": "L2SharedLinUCBRGBCamPolicy",
            "turn_per_lap": CHANGE_CONTEXT_EVERY,
            "max_laps": MAX_LAPS,
            "max_steps": MAX_STEPS,
            "l3_mde_model": "Depth-Anything-V2-Small-hf",
        },
    )

if __name__ == "__main__":
    
    # Isaac Sim Scene Setup
    configure_isaac_sim_logging() # Set Isaac Sim logging level to Error to avoid cluttering
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_rendering_test",
        robot_config=kaya_config,
    )
    render_config.set_rendering_mode("pathtracing")
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
    context_agent_speed = [0.2, 0.5, 1.0, 1.5, 2.0]  # Example speed values for the agent's context
    sensor_param_space = SensorParamSpace()
    l2_policy = L2SharedLinUCBRGBCamPolicy(
        sensor_names="agent_camera",
        sensor_config=sensor_param_space,
        reward_function=reward_test_time_augment, # reward_flipped_img or reward_test_time_augment
        alpha=1.0,
        random_seed=RANDOM_SEED,
    )
    curr_exposure_idx = len(sensor_param_space.exposure_values) // 2
    curr_iso_idx = len(sensor_param_space.iso_values) // 2
    curr_light = context_light[len(context_light) // 2]
    curr_speed = context_agent_speed[len(context_agent_speed) // 2] * np.pi / 12
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
        reward_type="test_time_augment", # "flipped" or "test_time_augment"
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
    
    ## Main L2-L3 FeedBack Loop for Sensor Control Logic
    step = 0
    lap_idx = 0
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        "pred_depth": []
    }
    wandb_run = initialize_wandb(exp_name=f"atil2l3_kaya_depthany_{l3_mde_config.reward_type}")
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
            if bbox_data is None or bbox_data["data"].size == 0:
                print(bbox_data)
                if VERBOSE: print("Warning: Received empty bounding box data. Skipping this step.")
                continue
            syn_data_cache["rgb"].append(rgb_image)
            syn_data_cache["depth"].append(gt_depth)
            syn_data_cache["bbox"].append(bbox_data)
            
            pred_depths, metric_info = mde_model.predict_depth([Image.fromarray(rgb_image)], gt_depth)
            syn_data_cache["pred_depth"].append(pred_depths if isinstance(pred_depths, np.ndarray) else pred_depths[0])
            if DEBUG: print(f"Depth Prediction Metrics: {metric_info}")
            
            if l3_mde_config.reward_type == "flipped":
                if not np.any(pred_depths[0]):
                    if DEBUG: print("[Fatal Error] Predicted depth is empty or all zeros.")
                    raise ValueError("[Fatal Error] Predicted depth is empty or all zeros.")    
                observation_info = {
                    "original_rgb": np.array(rgb_image),
                    "depth_original": pred_depths[0],
                    "depth_flipped": pred_depths[1],
                    "image_weight": 0.1,
                    "depth_weight": 0.9,
                }
            elif l3_mde_config.reward_type == "test_time_augment":
                observation_info = {
                    "rgb": np.array(rgb_image),
                    "inverse_depths": pred_depths,
                    "uncertainty_reduction": "mean",
                    "image_weight": 0.1,
                    "depth_weight": 0.9,
                }
            
            result = l2_policy.step(
                context_information={
                    "light_intensity": curr_light,
                    "angular_velocity": curr_speed,
                    "iso_idx": curr_iso_idx,
                    "exposure_idx": curr_exposure_idx
                },
                observations=observation_info
            )
            if DEBUG: print("[DEBUG]Policy Step Reward Result:", result["reward_info"])
            curr_exposure_idx = result["next_exposure_idx"]
            curr_iso_idx = result["next_iso_idx"]
            log_reward_history.append(result["reward_info"])
            log_performance_history.append(metric_info)
            log_context_history.append({
                "iso_idx": curr_iso_idx,
                "exposure_idx": curr_exposure_idx
            })
            
            if DEBUG: print("[DEBUG] Sensor Control Action Taken - Exposure Index:", curr_exposure_idx, "ISO Index:", curr_iso_idx)
            my_scene.sensor_control(
                control_parameters={
                    "iso": sensor_param_space.iso_values[curr_iso_idx],
                    "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx]
                }
            )

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0 and step > 0:
                wandb_run.log(
                    {
                        "context/light_intensity": curr_light,
                        "context/agent_speed": curr_speed,
                        **get_eval_averages(log_context_history, key_category="context"),
                        **get_eval_averages(log_reward_history, key_category="reward"),
                        **get_eval_averages(log_performance_history, key_category="performance"),
                    }, 
                    step=lap_idx,
                    commit=True
                )
                
                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)
                # "More smooth and Moderately changing the context for the agent to adapt to new conditions while avoiding drastic changes that could destabilize learning."
                curr_light = float(rng.choice(context_light))
                curr_speed = float(rng.choice(context_agent_speed)) * np.pi / 12
                my_scene.control_light_intensity(curr_light)
                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                }
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