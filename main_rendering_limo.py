"""
Framework for Fine-grained Sensor Control Logic of ATI with Isaac Sim
Reward testing for ATI-MDE pipeline. 
This script is used to test the reward function with different sensor control parameters, and collect data for reward function analysis and training. 
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
from ati_utils.log_utils import configure_isaac_sim_logging

# Enable Livestream extension
# from isaacsim.core.utils.extensions import enable_extension
# simulation_app.set_setting("/app/window/drawMouse", True)
# enable_extension("omni.services.livestream.nvcf")

# Scene Building
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, DEBUG
from time import time
import traceback
import numpy as np
import os

os.environ["PYOPENGL_PLATFORM"] = "egl" # For headless rendering with PyOpenGL.
os.environ["EGL_DEVICE_ID"] = "0" # Set to the appropriate GPU index if multiple GPUs are present.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATA_SAVE_PATH = "/issac-sim/dataset/experiment_isaac_rendering/ati_limo_rendering_test" # mb_iso_tradeoff_subsample16_camerafps

def save_status(lap_idx, iso_idx, st_idx, speed_idx, light_idx, d_time=None):
    with open(os.path.join(DATA_SAVE_PATH, "status_log.txt"), "a") as f:
        f.write(f"lap_idx: {lap_idx} | iso_idx: {iso_idx} | shutter_time_idx: {st_idx} | speed_idx: {speed_idx} | light_intensity_idx: {light_idx} | duration: {d_time:.2f} seconds\n")

def calculate_next_step(
    v_line, 
    radius, 
    camera_dt
):
    """
    Calculate the next step for the robot to complete a circular trajectory with given linear velocity and radius.
    """
    w_ang = v_line / radius
    period = 2 * np.pi * radius / v_line
    num_one_lap_steps = round(period / camera_dt)
    return w_ang, period, num_one_lap_steps

if __name__ == "__main__":
    
    configure_isaac_sim_logging() # Set Isaac Sim logging level to Error to avoid cluttering
    limo_config = ATIBaseRobotConfig(robot_name="limo")
    limo_config.set_limo_config()
    scene_config = ATIBaseConfig( 
        name="ati_rendering_limo_test",
        robot_config=limo_config,
    )
    scene_config.set_rendering_mode("pathtracing") # "pathtracing" or "realtime"
    scene_config.set_pathtracing_param(spp=128, num_subsamples=32) # Only effective when rendering_mode is "pathtracing"
    scene_config.set_random_obj_spawn(True)
    limo_scene = ATIDepthScene(
        simulation_app,
        config=scene_config,
        physics_dt=scene_config.physics_dt,
        rendering_dt=scene_config.rendering_dt,
        stage_units_in_meters=scene_config.stage_units_in_meters
    )
    
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(DATA_SAVE_PATH, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(DATA_SAVE_PATH, "depth"), exist_ok=True)
    os.makedirs(os.path.join(DATA_SAVE_PATH, "bbox"), exist_ok=True)
    
    step = 0
    prev_step = 0
    lap_idx = 0
    
    agent_linear_context = [2.0, 4.0]   # [m/s] LIMO max ≈ 1.5 m/s; 2.0 exceeds stable physics range for R=0.3m
    agent_context_light = [1000, 2000, 3000, 4000, 5000]
    
    shutter_time_list = [0.002, 0.004, 0.008, 0.016, 0.032] # in seconds
    iso_list = [200, 400, 600, 800, 1600]
    
    # The total number of laps is determined by controllable parameters: 
    # Shutter Time, ISO, Car Velocity -> with Fixed light intensity for now (can be added later)
    MAX_LAP = len(shutter_time_list) * len(iso_list) * len(agent_linear_context)
    
    iso_idx = 0
    st_idx = 0
    line_speed_idx = 0
    light_idx = 2
    
    R = 0.35 # (m) The radius of the circular trajectory. The linear velocity will determine the angular velocity.
    w_ang, T, NUM_ONE_LAP_STEPS = calculate_next_step(
        agent_linear_context[line_speed_idx], 
        R, 1. / scene_config.agent_camera_fps
    )
    
    limo_scene.control_light_intensity(agent_context_light[light_idx])
    limo_scene.sensor_control(
        control_parameters={
            "iso": iso_list[iso_idx],
            "shutter_time": shutter_time_list[st_idx]
        }
    )
    s_time = time()
    e_time = -1
    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            limo_scene.robot_control(
                time=limo_scene.get_simulation_current_time,
                control_parameters={
                    "linear_velocity": agent_linear_context[line_speed_idx], 
                    "angular_velocity": w_ang
                }
            )
            syn_data = limo_scene.step(render=True)
            if DEBUG: print(f"Step: {step}, Simulation Time: {limo_scene.get_simulation_current_time:.4f} seconds")
            rgb_data: np.array = syn_data.get("rgb", None)
            depth_data: np.array = syn_data.get(limo_scene.get_anno("depth"), None)
            bbox_data: np.array = syn_data.get(limo_scene.get_anno("2d_bounding_box"), None)
            if rgb_data is None or rgb_data.size == 0:
                if DEBUG:
                    print("There is no RGB rendering Data. Skipping...")
            else:
                np.save(os.path.join(DATA_SAVE_PATH, "rgb", f"rgb_lap{lap_idx:02d}_{step:03d}.npy"), rgb_data)
            if depth_data is None or depth_data.size == 0:
                if DEBUG:
                    print("There is no Depth rendering Data. Skipping...")
            else:
                np.save(os.path.join(DATA_SAVE_PATH, "depth", f"depth_lap{lap_idx:02d}_{step:03d}.npy"), depth_data)
            if bbox_data is None or bbox_data["data"].size == 0:
                if DEBUG:
                    print("There is no BBox rendering Data. Skipping...")
            else:
                np.save(os.path.join(DATA_SAVE_PATH, "bbox", f"bbox_lap{lap_idx:02d}_{step:03d}.npy"), bbox_data)

            if (step + 1) % NUM_ONE_LAP_STEPS == 0:
                e_time = time()
                save_status(lap_idx, iso_idx, st_idx, line_speed_idx, light_idx, d_time=e_time - s_time)
                print(f"Lap {lap_idx} completed. Total steps: {step + 1}, Time for this lap: {e_time - s_time:.2f} seconds")
                s_time = e_time
                
                lap_idx += 1
                st_idx = (lap_idx) % len(shutter_time_list) 
                iso_idx = (lap_idx // len(shutter_time_list)) % len(iso_list) 
                line_speed_idx = (lap_idx // (len(shutter_time_list) * len(iso_list))) % len(agent_linear_context) 
                w_ang, T, NUM_ONE_LAP_STEPS = calculate_next_step(
                    agent_linear_context[line_speed_idx], 
                    R, 1. / scene_config.agent_camera_fps
                )
                
                light_idx = 2 # random.randint(0, len(agent_context_light) - 1)
                limo_scene.sensor_control(
                    control_parameters={
                        "iso": iso_list[iso_idx],
                        "shutter_time": shutter_time_list[st_idx]
                    }
                )
                limo_scene.control_light_intensity(agent_context_light[light_idx])
                prev_step = limo_scene.num_steps  # cumulative count at lap boundary (not relative `step`)
                
            if lap_idx >= MAX_LAP:
                print(f"All laps completed. Ending simulation")
                break
            
            step = limo_scene.num_steps - prev_step
    
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
    
    
    