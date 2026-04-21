"""
Framework for Fine-grained Sensor Control Logic of ATI with Isaac Sim
using an epsilon-greedy CMAB control policy.
"""

from collections import Counter

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
from scene import ATIDepthScene
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3MDEConfig
from l3_perception_layer import L3PLayerDepthAnythingv2, set_deterministic
from policy.rewards.rewards import reward_flipped_img, reward_test_time_augment, reward_oracle
from policy import L2SharedEGreedyRGBCamPolicy, SensorParamSpace

import argparse
from PIL import Image
import traceback
import wandb
import numpy as np


parser = argparse.ArgumentParser(description="ATI Sensor Control with epsilon-greedy L2-L3 feedback loop in Isaac Sim")
parser.add_argument("--exp_name", type=str, default="atil2l3_kaya_depthany_egreedy", help="Name of the experiment for logging purposes")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["flipped", "test_time_augment", "oracle"], help="Type of reward function to use for the L2 policy")
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype/kaya_egreedy_control", help="Directory path to save synthetic data and logs")
parser.add_argument("--max_laps", type=int, default=600, help="Maximum number of laps (context changes) to run in the simulation")
parser.add_argument("--lap_period", type=int, default=30, help="Number of steps per lap (context change period)")
parser.add_argument("--epsilon_start", type=float, default=0.5, help="Initial exploration rate for epsilon-greedy control")
parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum exploration rate")
parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Per-update multiplicative epsilon decay")
parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for expected reward updates")
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
            "policy_type": "L2SharedEGreedyRGBCamPolicy",
            "turn_per_lap": context_len,
            "max_laps": max_laps,
            "max_steps": max_steps,
            "l3_mde_model": "Depth-Anything-V2-Small-hf",
            "reward_type": args.reward_type,
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "learning_rate": args.learning_rate,
        },
    )


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


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def build_policy_step_metrics(result: dict) -> dict:
    return {
        "epsilon": float(result["epsilon"]),
        "chosen_expected_reward": float(result["chosen_expected_reward"]),
        "chosen_count": float(result["chosen_count"]),
        "explore_flag": float(result["selection_mode"] == "explore"),
        "exploit_flag": float(result["selection_mode"] == "exploit"),
    }


def build_policy_update_metrics(update_info: dict | None) -> dict | None:
    if update_info is None:
        return None
    return {
        "assigned_reward": float(update_info["reward"]),
        "old_expected_reward": float(update_info["old_expected_reward"]),
        "new_expected_reward": float(update_info["new_expected_reward"]),
        "updated_action_count": float(update_info["count"]),
        "epsilon_after_update": float(update_info["epsilon"]),
    }


def build_lap_log_payload(
    curr_light: float,
    curr_speed: float,
    log_context_history: list[dict],
    log_reward_history: list[dict],
    log_performance_history: list[dict],
    log_policy_history: list[dict],
    log_policy_update_history: list[dict],
    log_state_history: list[str],
    policy: L2SharedEGreedyRGBCamPolicy,
) -> dict:
    payload = {
        "context/light_intensity": curr_light,
        "context/agent_speed": curr_speed,
        **average_history(log_context_history, key_category="context"),
        **average_history(log_reward_history, key_category="reward"),
        **average_history(log_performance_history, key_category="performance"),
        **average_history(log_policy_history, key_category="policy"),
        **average_history(log_policy_update_history, key_category="policy_update"),
        **prefix_metrics(policy.get_stats(), prefix="policy_stats"),
    }

    state_counter = Counter(log_state_history)
    if log_state_history:
        payload["policy/current_state"] = log_state_history[-1]
        payload["policy/most_visited_state"] = state_counter.most_common(1)[0][0]
        payload["policy/lap_unique_states"] = len(state_counter)

    for state in sorted(policy.expected_rewards.keys()):
        payload[f"policy_state_visits/{state}"] = int(state_counter.get(state, 0))

    return payload


if __name__ == "__main__":
    CHANGE_CONTEXT_EVERY = args.lap_period
    DATA_PATH = args.data_path
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS

    configure_isaac_sim_logging()
    kaya_config = ATIBaseRobotConfig(robot_name="kaya")
    kaya_config.set_kaya_config()
    render_config = ATIBaseConfig(
        name="ati_rendering_egreedy",
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
    )

    set_deterministic(RANDOM_SEED)
    context_light = [200, 1000, 3000, 6000, 9000]
    context_agent_speed = [0.2, 0.5, 1.0, 1.5, 2.0]
    sensor_param_space = SensorParamSpace()
    l2_policy = L2SharedEGreedyRGBCamPolicy(
        sensor_names="agent_camera",
        sensor_config=sensor_param_space,
        reward_function=select_reward_function(args.reward_type),
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        random_seed=RANDOM_SEED,
    )
    curr_exposure_idx = len(sensor_param_space.exposure_values) // 2
    curr_iso_idx = len(sensor_param_space.iso_values) // 2
    curr_light = context_light[len(context_light) // 2]
    curr_speed = context_agent_speed[len(context_agent_speed) // 2] * np.pi / 12

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
        # "bbox": [],
        "pred_depth": [],
    }
    log_context_history = []
    log_reward_history = []
    log_performance_history = []
    log_policy_history = []
    log_policy_update_history = []
    log_state_history = []
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
                    "angular_velocity": curr_speed,
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
            # if bbox_data is None or bbox_data["data"].size == 0:
            #     if VERBOSE:
            #         print("Warning: Received empty bounding box data. Skipping this step.")
            #     continue

            syn_data_cache["rgb"].append(rgb_image)
            syn_data_cache["depth"].append(gt_depth)
            # syn_data_cache["bbox"].append(bbox_data)

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

            result = l2_policy.step(
                context_information={
                    "light_intensity": curr_light,
                    "angular_velocity": curr_speed,
                    "iso_idx": curr_iso_idx,
                    "exposure_idx": curr_exposure_idx,
                },
                observations=observation_info,
            )
            if DEBUG:
                print("[DEBUG] Policy Reward Result:", result["reward_info"])
                print(
                    "[DEBUG] EGreedy Action:",
                    result["action_description"],
                    "State:",
                    result["state"],
                    "Mode:",
                    result["selection_mode"],
                    "Epsilon:",
                    result["epsilon"],
                )

            curr_exposure_idx = result["next_exposure_idx"]
            curr_iso_idx = result["next_iso_idx"]

            log_reward_history.append(result["reward_info"])
            log_performance_history.append(metric_info)
            log_context_history.append(
                {
                    "iso_idx": curr_iso_idx,
                    "exposure_idx": curr_exposure_idx,
                }
            )
            log_policy_history.append(build_policy_step_metrics(result))
            log_state_history.append(result["state"])
            update_metrics = build_policy_update_metrics(result["update_info"])
            if update_metrics is not None:
                log_policy_update_history.append(update_metrics)

            if DEBUG:
                print(
                    "[DEBUG] Sensor Control Action Taken - Exposure Index:",
                    curr_exposure_idx,
                    "ISO Index:",
                    curr_iso_idx,
                )

            # If selected action does not make any changes, we can skip sending redundant control commands to the simulator.
            ## Too frequent sensor control causes stale data issues in Isaac Sim, so we only send control commands when there is an actual change in parameters.
            current_control_params = my_scene.get_sensor_control_params(sensor_name="agent_camera")
            if current_control_params.get("iso", None) != sensor_param_space.iso_values[curr_iso_idx] or \
                current_control_params.get("shutter_time", None) != sensor_param_space.exposure_values[curr_exposure_idx]:
                    my_scene.sensor_control(
                        control_parameters={
                            "iso": sensor_param_space.iso_values[curr_iso_idx],
                            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
                        }
                    )

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0 and step > 0:
                wandb_run.log(
                    build_lap_log_payload(
                        curr_light=curr_light,
                        curr_speed=curr_speed,
                        log_context_history=log_context_history,
                        log_reward_history=log_reward_history,
                        log_performance_history=log_performance_history,
                        log_policy_history=log_policy_history,
                        log_policy_update_history=log_policy_update_history,
                        log_state_history=log_state_history,
                        policy=l2_policy,
                    ),
                    step=lap_idx,
                    commit=True,
                )

                save_synthetic_data(DATA_PATH, syn_data_cache, lap_idx)

                curr_light = float(rng.choice(context_light))
                curr_speed = float(rng.choice(context_agent_speed)) * np.pi / 12
                my_scene.control_light_intensity(curr_light)
                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    # "bbox": [],
                    "pred_depth": [],
                }
                log_context_history = []
                log_reward_history = []
                log_performance_history = []
                log_policy_history = []
                log_policy_update_history = []
                log_state_history = []
                lap_idx += 1

                if VERBOSE:
                    print(
                        f"Context changed at step {step+1}: "
                        f"Light Intensity set to {curr_light}, Agent Speed set to {curr_speed}"
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
        if wandb_run is not None and (
            log_context_history
            or log_reward_history
            or log_performance_history
            or log_policy_history
            or log_policy_update_history
            or log_state_history
        ):
            wandb_run.log(
                build_lap_log_payload(
                    curr_light=curr_light,
                    curr_speed=curr_speed,
                    log_context_history=log_context_history,
                    log_reward_history=log_reward_history,
                    log_performance_history=log_performance_history,
                    log_policy_history=log_policy_history,
                    log_policy_update_history=log_policy_update_history,
                    log_state_history=log_state_history,
                    policy=l2_policy,
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
