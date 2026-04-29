"""
Framework for Fine-grained Sensor Control Logic of ATI with Isaac Sim
using an epsilon-greedy CMAB control policy.
"""

from collections import Counter
import os

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
from ati_config import ATIBaseConfig, ATIBaseRobotConfig, L3ClassificationConfig
from l3_perception_layer import L3PLayerClassificiation, set_deterministic
from policy import L2SharedEGreedyRGBCamPolicy, SensorParamSpace
from policy import (
    build_reward_override, load_heuristic_memory, save_heuristic_memory,
    reward_classification_confidence, reward_classification_oracle
)

import argparse
from PIL import Image
import traceback
import wandb
import numpy as np


parser = argparse.ArgumentParser(description="ATI Sensor Control with epsilon-greedy L2-L3 feedback loop in Isaac Sim")
parser.add_argument("--exp_name", type=str, default="rt_egreedy", help="Name of the experiment for logging purposes")
parser.add_argument("--reward_type", type=str, default="oracle", choices=["confidence", "oracle"], help="Type of reward function to use for the L2 policy")
parser.add_argument("--data_path", type=str, default="/issac-sim/dataset/experiment_mde_prototype", help="Directory path to save synthetic data and logs")
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
WARMUP_LAPS = 5
HEURISTIC_UPDATE_THRESHOLD = 30
HEURISTIC_BLEND_ALPHA = 0.35
HEURISTIC_MEMORY_FILENAME = f"heuristic_offsets_{args.exp_name}_{args.reward_type}.json"


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
    if reward_type == "confidence":
        return reward_classification_confidence
    if reward_type == "oracle":
        return reward_classification_oracle
    raise ValueError(f"Invalid reward_type: {reward_type}. Must be one of ['confidence', 'oracle']")


def build_observation_info(
    reward_type: str,
    rgb_image: np.ndarray,
    confidence: float,
    correct: float,
) -> dict:
    if reward_type == "confidence":
        return {
            "rgb_image": np.array(rgb_image),
            "confidence": confidence,
            "image_weight": 0.1,
            "task_weight": 0.9,
        }
    if reward_type == "oracle":
        return {
            "rgb_image": np.array(rgb_image),
            "correct": correct,
            "image_weight": 0.1,
            "task_weight": 0.9,
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


def summarize_context_history(context_history: list[dict]) -> dict:
    if not context_history:
        return {
            "light_intensity": 0.0,
            "angular_velocity": 0.0,
            "light_intensity_start": 0.0,
            "light_intensity_end": 0.0,
            "angular_velocity_start": 0.0,
            "angular_velocity_end": 0.0,
            "light_intensity_std": 0.0,
            "angular_velocity_std": 0.0,
            "num_samples": 0.0,
        }

    light_values = np.asarray([sample["light_intensity"] for sample in context_history], dtype=np.float64)
    angular_values = np.asarray([sample["angular_velocity"] for sample in context_history], dtype=np.float64)
    return {
        "light_intensity": float(np.mean(light_values)),
        "angular_velocity": float(np.mean(angular_values)),
        "light_intensity_start": float(light_values[0]),
        "light_intensity_end": float(light_values[-1]),
        "angular_velocity_start": float(angular_values[0]),
        "angular_velocity_end": float(angular_values[-1]),
        "light_intensity_std": float(np.std(light_values)),
        "angular_velocity_std": float(np.std(angular_values)),
        "num_samples": float(len(context_history)),
    }


def sample_context_summary(trajectory, start_step: int, num_steps: int) -> dict:
    samples = [trajectory.value_at(start_step + offset) for offset in range(num_steps)]
    return summarize_context_history(samples)


def build_lap_log_payload(
    current_context_summary: dict,
    next_context_summary: dict,
    log_context_history: list[dict],
    log_reward_history: list[dict],
    log_performance_history: list[dict],
    log_policy_history: list[dict],
    log_policy_update_history: list[dict],
    log_state_history: list[str],
    policy: L2SharedEGreedyRGBCamPolicy,
) -> dict:
    payload = {
        "context/light_intensity": float(current_context_summary["light_intensity"]),
        "context/agent_speed": float(current_context_summary["angular_velocity"]),
        "context/light_intensity_start": float(current_context_summary["light_intensity_start"]),
        "context/light_intensity_end": float(current_context_summary["light_intensity_end"]),
        "context/agent_speed_start": float(current_context_summary["angular_velocity_start"]),
        "context/agent_speed_end": float(current_context_summary["angular_velocity_end"]),
        "context/light_intensity_std": float(current_context_summary["light_intensity_std"]),
        "context/agent_speed_std": float(current_context_summary["angular_velocity_std"]),
        "context/num_samples": float(current_context_summary["num_samples"]),
        "policy_context/next_light_intensity": float(next_context_summary["light_intensity"]),
        "policy_context/next_agent_speed": float(next_context_summary["angular_velocity"]),
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
    DATA_PATH = f"{args.data_path}/experiment_{args.exp_name}_{args.reward_type}_{args.lap_period}steps_decay{args.epsilon_decay}_lr{args.learning_rate}"
    MAX_LAPS = args.max_laps
    MAX_STEPS = CHANGE_CONTEXT_EVERY * MAX_LAPS
    HEURISTIC_MEMORY_PATH = os.path.join(DATA_PATH, HEURISTIC_MEMORY_FILENAME)
    RAD_COEFF = np.pi / 12

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
    context_agent_speed = [1.5, 2.0, 1.5, 2.0, 1.5]
    trajectory = build_default_context_trajectory(
        light_values=[1000, 1000, 1000, 1000, 1000],
        speed_values=[s * RAD_COEFF for s in context_agent_speed],
        light_transition_steps=args.lap_period * 20,
        speed_transition_steps=args.lap_period * 10,
        light_hold_steps=args.lap_period,
        speed_hold_steps=args.lap_period,
        speed_phase_offset_steps=args.lap_period,
    )
    
    sensor_param_space = SensorParamSpace()
    l2_policy = L2SharedEGreedyRGBCamPolicy(
        sensor_names="agent_camera",
        sensor_config=sensor_param_space,
        reward_function=select_reward_function(args.reward_type),
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        learning_rate=args.learning_rate,
        motion_thresholds=[ 
            ((s + e) / 2) * RAD_COEFF
            for (s, e) in zip(context_agent_speed[:-1], context_agent_speed[1:])
        ],
        light_thresholds=[
            ((s + e) / 2) for (s, e) in zip(context_light[:-1], context_light[1:])
        ],
        random_seed=RANDOM_SEED,
    )
    heuristic_offsets, state_update_counts = load_heuristic_memory(HEURISTIC_MEMORY_PATH)
    context = trajectory.value_at(0)
    curr_light = context["light_intensity"]
    curr_speed = context["angular_velocity"]
    context = {
        "light_intensity": curr_light,
        "angular_velocity": curr_speed,
    }
    curr_exposure_idx, curr_iso_idx, _ = l2_policy.calculate_base_indices(
        context_information=context,
        heuristic_offsets=heuristic_offsets,
    )

    my_scene.control_light_intensity(curr_light)
    my_scene.sensor_control(
        control_parameters={
            "iso": sensor_param_space.iso_values[curr_iso_idx],
            "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
        }
    )

    l3_classification_config = L3ClassificationConfig(
        model_name="mobilenetv2_100",
    )
    c_model = L3PLayerClassificiation(l3_config=l3_classification_config, device="cuda")

    step = 0
    lap_idx = 0
    syn_data_cache = {
        "rgb": [],
        "depth": [],
        "bbox": [],
        # "pred_depth": [],
    }
    lap_context_samples = []
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
        exp_name=f"ati_kaya_{l3_classification_config.model_name}_{args.exp_name}",
    )

    try:
        while simulation_app._app.is_running() and not simulation_app.is_exiting():
            if VERBOSE:
                print(f"Step: {step+1}/{MAX_STEPS}, Simulation Time: {my_scene.get_simulation_current_time:.4f} seconds")

            lap_context_samples.append(
                {
                    "light_intensity": float(context["light_intensity"]),
                    "angular_velocity": float(context["angular_velocity"]),
                }
            )
            my_scene.control_light_intensity(context["light_intensity"])
            my_scene.robot_control(
                time=my_scene.get_simulation_current_time,
                control_parameters={
                    "angular_velocity": context["angular_velocity"],
                },
            )
            syn_data = my_scene.step(render=True)

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

            syn_data_cache["rgb"].append(rgb_image)
            syn_data_cache["depth"].append(gt_depth)
            syn_data_cache["bbox"].append(bbox_data)
            
            correct, conf, pred_label = c_model.predict_image(rgb_image, gt_bbox=bbox_data)
            if DEBUG:
                print(f"[DEBUG] L3 Classification Prediction: {pred_label}, Confidence: {conf:.4f}, Correct in GT BBox: {correct:.4f}")

            if not correct == -1:
                observation_info = build_observation_info(
                    reward_type=args.reward_type,
                    rgb_image=rgb_image,
                    confidence=conf,
                    correct=correct,
                )
                reward_info = l2_policy.reward_function(**observation_info)
                log_reward_history.append(reward_info)
                log_performance_history.append({"accuracy": float(correct), "confidence": float(conf)})

            if (step + 1) % CHANGE_CONTEXT_EVERY == 0 and step > 0:
                current_context_summary = summarize_context_history(lap_context_samples)
                next_context_summary = sample_context_summary(
                    trajectory=trajectory,
                    start_step=step + 1,
                    num_steps=CHANGE_CONTEXT_EVERY,
                )
                lap_reward_info = build_reward_override(log_reward_history)
                base_exposure_idx, base_iso_idx, base_state = l2_policy.calculate_base_indices(
                    context_information=next_context_summary,
                    heuristic_offsets=heuristic_offsets,
                )
                is_warmup_lap = lap_idx < WARMUP_LAPS
                result = l2_policy.step(
                    context_information={
                        "light_intensity": next_context_summary["light_intensity"],
                        "angular_velocity": next_context_summary["angular_velocity"],
                        "iso_idx": base_iso_idx,
                        "exposure_idx": base_exposure_idx,
                        "tie_break_random": False,
                        "is_infer_mode": is_warmup_lap,
                        "skip_update": is_warmup_lap,
                    },
                    observations={
                        "reward_info_override": lap_reward_info,
                    },
                )

                if DEBUG:
                    print("[DEBUG] Lap Reward Result:", result["reward_info"])
                    print(
                        "[DEBUG] EGreedy Lap Action:",
                        result["action_description"],
                        "State:",
                        result["state"],
                        "Mode:",
                        result["selection_mode"],
                        "Epsilon:",
                        result["epsilon"],
                        "Warmup:",
                        is_warmup_lap,
                    )

                curr_exposure_idx = result["next_exposure_idx"]
                curr_iso_idx = result["next_iso_idx"]
                log_context_history.append(
                    {
                        "base_iso_idx": float(base_iso_idx),
                        "base_exposure_idx": float(base_exposure_idx),
                        "iso_idx": float(curr_iso_idx),
                        "exposure_idx": float(curr_exposure_idx),
                    }
                )
                log_policy_history.append(build_policy_step_metrics(result))
                log_state_history.append(result["state"])
                update_metrics = build_policy_update_metrics(result["update_info"])
                if update_metrics is not None:
                    log_policy_update_history.append(update_metrics)
                if l2_policy.update_long_term_memory(
                    heuristic_offsets=heuristic_offsets,
                    state_update_counts=state_update_counts,
                    update_info=result["update_info"],
                    update_threshold=HEURISTIC_UPDATE_THRESHOLD,
                    blend_alpha=HEURISTIC_BLEND_ALPHA,
                ):
                    save_heuristic_memory(
                        memory_path=HEURISTIC_MEMORY_PATH,
                        heuristic_offsets=heuristic_offsets,
                        state_update_counts=state_update_counts,
                    )

                if DEBUG:
                    print(
                        "[DEBUG] Base Sensor Indices - Exposure:",
                        base_exposure_idx,
                        "ISO:",
                        base_iso_idx,
                        "State:",
                        base_state,
                    )
                    print(
                        "[DEBUG] Sensor Control Action Taken - Exposure Index:",
                        curr_exposure_idx,
                        "ISO Index:",
                        curr_iso_idx,
                    )

                current_control_params = my_scene.get_sensor_control_params(sensor_name="agent_camera")
                if current_control_params.get("iso", None) != sensor_param_space.iso_values[curr_iso_idx] or \
                            current_control_params.get("shutter_time", None) != sensor_param_space.exposure_values[curr_exposure_idx]:
                        my_scene.sensor_control(
                            control_parameters={
                                "iso": sensor_param_space.iso_values[curr_iso_idx],
                                "shutter_time": sensor_param_space.exposure_values[curr_exposure_idx],
                            }
                        )

                wandb_run.log(
                    build_lap_log_payload(
                        current_context_summary=current_context_summary,
                        next_context_summary=next_context_summary,
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

                # curr_light = float(rng.choice(context_light))
                # curr_speed = float(rng.choice(context_agent_speed)) * np.pi / 12
                # my_scene.control_light_intensity(curr_light)
                syn_data_cache = {
                    "rgb": [],
                    "depth": [],
                    "bbox": [],
                    # "pred_depth": [],
                }
                lap_context_samples = []
                log_context_history = []
                log_reward_history = []
                log_performance_history = []
                log_policy_history = []
                log_policy_update_history = []
                log_state_history = []
                lap_idx += 1
                context = trajectory.value_at(step)
                
                if VERBOSE:
                    warmup_msg = "warmup" if is_warmup_lap else "cmab"
                    print(
                        f"Context changed at step {step+1}: "
                        f"Observed Lap Context (avg) = ({current_context_summary['light_intensity']:.2f}, {current_context_summary['angular_velocity']:.4f}), "
                        f"Next Policy Context (avg) = ({next_context_summary['light_intensity']:.2f}, {next_context_summary['angular_velocity']:.4f}), "
                        f"Mode={warmup_msg}. "
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
            current_context_summary = summarize_context_history(lap_context_samples)
            next_context_summary = sample_context_summary(
                trajectory=trajectory,
                start_step=step + 1,
                num_steps=CHANGE_CONTEXT_EVERY,
            )
            wandb_run.log(
                build_lap_log_payload(
                    current_context_summary=current_context_summary,
                    next_context_summary=next_context_summary,
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
        save_heuristic_memory(
            memory_path=HEURISTIC_MEMORY_PATH,
            heuristic_offsets=heuristic_offsets,
            state_update_counts=state_update_counts,
        )

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
