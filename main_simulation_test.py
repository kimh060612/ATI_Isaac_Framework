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
from ati_utils.log_utils import configure_isaac_sim_logging
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig
from time import time
import traceback
import numpy as np
import random
import os 

TIME_REPUTATION = 1000
ONE_LAP_PERIOD = 20
DATA_SAVE_PATH = "/issac-sim/dataset/experiment_isaac_rendering/exp_kaya_rt_custom_reward" # /home/ati/ATI_research/dataset/test_isaacsim_sdg/data/

def save_status(lap_idx, iso_idx, st_idx, speed_idx, light_idx, d_time=None):
    with open(os.path.join(DATA_SAVE_PATH, "status_log.txt"), "a") as f:
        f.write(f"step: {lap_idx*ONE_LAP_PERIOD + 1} ~ {(lap_idx + 1) * ONE_LAP_PERIOD} | lap_idx: {lap_idx} | iso_idx: {iso_idx} | shutter_time_idx: {st_idx} | speed_idx: {speed_idx} | light_intensity_idx: {light_idx} | duration: {d_time:.2f} seconds\n")

if __name__ == "__main__":
    
    configure_isaac_sim_logging() # Set Isaac Sim logging level to Error to avoid cluttering
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    scene_config = ATIBaseConfig( 
        name="ati_kaya_sim_test",
        robot_config=kaya_config,
    )
    scene_config.set_rendering_mode("realtime") # "pathtracing" or "realtime"
    scene_config.set_pathtracing_param(spp=128, num_subsamples=32) # Only effective when rendering_mode is "pathtracing"
    scene_config.set_random_obj_spawn(True)
    my_scene = ATIDepthScene(
        simulation_app,
        config=scene_config,
        physics_dt=scene_config.physics_dt,
        rendering_dt=scene_config.rendering_dt,
        stage_units_in_meters=scene_config.stage_units_in_meters
    )

    step = 0
    prev_step = 0
    lap_idx = 0
    agent_context = [0.2, 0.5, 1.0, 1.5, 2.0]
    agent_context_light = [1000, 2000, 3000, 4000, 5000]
    
    shutter_time_list = [0.002, 0.004, 0.008, 0.016, 0.032]  # in seconds
    iso_list = [400, 600, 800, 1000, 1600]
    
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
            if rgb_data is None or rgb_data.size == 0:
                if VERBOSE:
                    print("There is no RGB rendering Data. Skipping...")
            if depth_data is None or depth_data.size == 0:
                if VERBOSE:
                    print("There is no Depth rendering Data. Skipping...")
            if bbox_data is None or bbox_data["data"].size == 0:
                if VERBOSE:
                    print("There is no BBox rendering Data. Skipping...")
            print("IMU sensor output: ", syn_data.get("imu_sensor", None))
            imu_data = syn_data.get("imu_sensor", None)
            if not imu_data is None:
                np.save(os.path.join(DATA_SAVE_PATH, "imu", f"imu_{step:03d}.npy"), imu_data)
            
            if (step + 1) % ONE_LAP_PERIOD == 0 and not (step == prev_step):
                e_time = time()
                save_status(lap_idx, iso_idx, st_idx, speed_idx, light_idx, d_time=e_time - s_time)
                print(f"Lap {lap_idx} completed. Total steps: {step + 1}, Time for this lap: {e_time - s_time:.2f} seconds")
                s_time = e_time
                
                lap_idx += 1
                iso_idx = (lap_idx // len(shutter_time_list)) % len(iso_list) # random.randint(0, len(iso_list) - 1)
                st_idx = (lap_idx) % len(shutter_time_list) # random.randint(0, len(shutter_time_list) - 1)
                speed_idx = 2 if step <= TIME_REPUTATION // 2 else 4 # random.randint(0, len(agent_context) - 1)
                light_idx = 2 # random.randint(0, len(agent_context_light) - 1)
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
    