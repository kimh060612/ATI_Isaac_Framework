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
    "headless": False,
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
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig
from transformers import pipeline
from PIL import Image
from time import time
import traceback
import numpy as np
import random
import os 

TIME_REPUTATION = 1200
ONE_LAP_PERIOD = 20
DATA_SAVE_PATH = "/issac-sim/dataset/experiment_isaac_rendering/exp_rt_kaya_uniform" # /home/ati/ATI_research/dataset/test_isaacsim_sdg/data/
MIN_DEPTH = 1e-3
MAX_DEPTH = 20.0

def compute_errors_numpy(
    gt,
    pred,
    min_depth=MIN_DEPTH,
    max_depth=MAX_DEPTH,
    align_mode="scale_shift",
    eps=1e-8,
):
    gt = np.asarray(gt).astype(np.float64)
    pred = np.asarray(pred).astype(np.float64)
    gt_inv = 1. / (gt + eps)
    
    valid = np.isfinite(gt) & np.isfinite(pred)
    valid &= (gt > min_depth) & (gt < max_depth)
    valid &= (pred > 0)
    if valid.sum() < 10:
        raise ValueError("No valid pixels found for evaluation.")

    gt_valid = gt[valid]
    gt_inv_valid = gt_inv[valid]
    pred_valid = pred[valid]

    # 2) optional alignment for relative-depth prediction
    if align_mode == "median":
        scale = np.median(gt_inv_valid) / (np.median(pred_valid) + eps)
        pred_valid = pred_valid * scale

    elif align_mode == "scale_shift":
        # solve: gt ≈ s * pred + t
        A = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)  # [N, 2]
        x, _, _, _ = np.linalg.lstsq(A, gt_inv_valid, rcond=None)
        s, t = x
        pred_valid = s * pred_valid + t
        pred_valid = np.maximum(pred_valid, 1e-6)

    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    # 3) clamp after alignment
    pred_valid = 1. / (pred_valid + 1e-8)
    pred_valid = np.clip(pred_valid, MIN_DEPTH, MAX_DEPTH)
    gt_valid = np.clip(gt_valid, MIN_DEPTH, MAX_DEPTH)

    thresh = np.maximum(gt_valid / (pred_valid + eps), pred_valid / (gt_valid + eps))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_valid + eps) - np.log(pred_valid + eps)) ** 2))
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / (gt_valid + eps))
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / (gt_valid + eps))

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "a1": float(a1),
        "a2": float(a2),
        "a3": float(a3),
    }

def evaluate_l3_depth(pred_depth, gt_depth, steps=0):
    # err_map = np.abs(gt_depth - pred_depth)
    
    pred_depth = pred_depth.flatten()
    gt_depth = gt_depth.flatten()

    metrics = compute_errors_numpy(gt_depth, pred_depth)
    print(
        "[MDE Result on step {:03d}] | abs_rel: {:.2f} | sq_rel {:.2f} | rmse {:.2f} | "
        "rmse_log {:.2f} | a1 {:.2f} | a2 {:.2f} | a3 {:.2f} |".format(
            steps,
            metrics["abs_rel"],
            metrics["sq_rel"],
            metrics["rmse"],
            metrics["rmse_log"],
            metrics["a1"],
            metrics["a2"],
            metrics["a3"],
        )
    )
    return metrics # , err_map

def dict_to_str(d: dict):
    return " | ".join([f"{k}: {v:.4f}" for k, v in d.items()])

def save_status(lap_idx, iso_idx, st_idx, speed_idx, light_idx, eval_result, d_time=None):
    with open(os.path.join(DATA_SAVE_PATH, "status_log.txt"), "a") as f:
        f.write(f"step: {lap_idx*ONE_LAP_PERIOD + 1} ~ {(lap_idx + 1) * ONE_LAP_PERIOD} | lap_idx: {lap_idx} | iso_idx: {iso_idx} | shutter_time_idx: {st_idx} | speed_idx: {speed_idx} | light_intensity_idx: {light_idx} | duration: {d_time:.2f} seconds | eval_result: {dict_to_str(eval_result)}\n")

def pred_mde_with_depthanything_v2(depth_pipe, rgb_image_list):
    pred_depth_list = depth_pipe([ Image.fromarray(x) for x in rgb_image_list ])
    return [
        np.array(pred_depth["depth"]) for pred_depth in pred_depth_list
    ]

if __name__ == "__main__":
    
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
        stage_units_in_meters=render_config.stage_units_in_meters,
        seed=20260420
    )
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(DATA_SAVE_PATH, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(DATA_SAVE_PATH, "depth"), exist_ok=True)
    os.makedirs(os.path.join(DATA_SAVE_PATH, "bbox"), exist_ok=True)
    
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cuda:0",
    )
    
    step = 0
    prev_step = 0
    lap_idx = 0
    agent_context = [0.2, 0.5, 1.0, 1.5, 2.0]
    agent_context_light = [1000, 2000, 3000, 4000, 5000]
    
    shutter_time_list = [0.002, 0.004, 0.008, 0.016, 0.024, 0.032]  # in seconds
    iso_list = [600, 600, 600, 600, 600] # [400, 600, 800, 1000, 1600]
    
    iso_idx = 0
    st_idx = 0
    speed_idx = 2
    light_idx = 2
    
    w_coeff = (np.pi / 12)
    omega = w_coeff * agent_context[speed_idx]
    
    # save_status(lap_idx, iso_idx, st_idx, speed_idx, light_idx)
    my_scene.control_light_intensity(agent_context_light[light_idx])
    my_scene.sensor_control(
        control_parameters={
            "iso": iso_list[iso_idx],
            "shutter_time": shutter_time_list[st_idx]
        }
    )
    
    synthetic_data_list = {
        "rgb": [],
        "depth": [],
        "bbox": []
    }
    
    s_time = time()
    e_time = -1
    VERBOSE = False
    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            syn_data = my_scene.step(render=True)
            print(f"Step: {step}, Simulation Time: {my_scene.get_simulation_current_time:.4f} seconds")
            my_scene.robot_control(
                time=my_scene.get_simulation_current_time,
                control_parameters={
                    "angular_velocity": omega
                }
            )
            rgb_data: np.array = syn_data.get("rgb", None)
            depth_data: np.array = syn_data.get(my_scene.get_anno("depth"), None)
            bbox_data: np.array = syn_data.get(my_scene.get_anno("2d_bounding_box"), None)
            if not (rgb_data is None or rgb_data.size == 0):
                synthetic_data_list["rgb"].append(rgb_data)
            if not (depth_data is None or depth_data.size == 0):
                synthetic_data_list["depth"].append(depth_data)
            if not (bbox_data is None or bbox_data["data"].size == 0):
                synthetic_data_list["bbox"].append(bbox_data)

            if (step + 1) % ONE_LAP_PERIOD == 0 and not (step == prev_step):
                e_time = time()
                pred_depth_list = pred_mde_with_depthanything_v2(depth_pipe, synthetic_data_list["rgb"])
                synthetic_data_list["pred_depth"] = pred_depth_list
                
                eval_result_list = [
                    evaluate_l3_depth(pred_depth, gt_depth, steps=step + 1) for pred_depth, gt_depth in zip(pred_depth_list, synthetic_data_list["depth"])
                ]
                save_status(lap_idx, iso_idx, st_idx, speed_idx, light_idx, get_eval_averages(eval_result_list), d_time=e_time - s_time)
                print(f"Lap {lap_idx} completed. Total steps: {step + 1}, Time for this lap: {e_time - s_time:.2f} seconds")
                s_time = e_time
                
                save_synthetic_data(DATA_SAVE_PATH, synthetic_data_list, lap_idx)
                synthetic_data_list = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                    "pred_depth": []
                }
                lap_idx += 1
                iso_idx = (lap_idx // len(shutter_time_list)) % len(iso_list) 
                st_idx = (lap_idx) % len(shutter_time_list) 
                speed_idx = 2 if step <= TIME_REPUTATION // 2 else 4 
                light_idx = 1 
                my_scene.sensor_control(
                    control_parameters={
                        "iso": iso_list[iso_idx],
                        "shutter_time": shutter_time_list[st_idx]
                    }
                )
                omega = w_coeff * agent_context[speed_idx]
                my_scene.control_light_intensity(agent_context_light[light_idx])
                prev_step = step

            if step >= TIME_REPUTATION:
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
    