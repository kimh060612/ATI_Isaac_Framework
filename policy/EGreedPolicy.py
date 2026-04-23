from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from policy.BasePolicy import BaseCMABPolicy
from policy.SensorParams import SensorParamSpace


class L2SharedEGreedyRGBCamPolicy(BaseCMABPolicy):
    """
    State-based epsilon-greedy CMAB policy inspired by Grid2DRLAgent.kt.

    The policy keeps an expected-reward table per discrete context state and
    action. A small delayed-update buffer is used so the reward computed from
    the current frame can be assigned to the action that produced it in the
    previous control step.
    """

    MOTION_STATES = ("STATIC", "SLOW", "NORMAL", "FAST", "SHAKE")
    LIGHT_STATES = ("DARK", "DIM", "NORMAL", "BRIGHT", "OUTDOOR")

    ACTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    ACTION_DESCRIPTIONS: Dict[Tuple[int, int], str] = {
        (0, 0): "Stay",
        (1, 0): "FasterExp",
        (-1, 0): "SlowerExp",
        (0, 1): "HigherISO",
        (0, -1): "LowerISO",
        (1, 1): "FastExp_HighISO",
        (1, -1): "FastExp_LowISO",
        (-1, 1): "SlowExp_HighISO",
        (-1, -1): "SlowExp_LowISO",
    }

    def __init__(
        self,
        sensor_names,
        sensor_config: Optional[SensorParamSpace],
        reward_function,
        epsilon_start: float = 0.5,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.1,
        initial_expected_reward: float = 1.0,
        motion_thresholds: Tuple[float, float, float, float] = (0.08, 0.18, 0.32, 0.45),
        light_thresholds: Tuple[float, float, float, float] = (500.0, 2000.0, 4500.0, 7500.0),
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        self.cfg = sensor_config if sensor_config is not None else SensorParamSpace()
        self.n_exposure = len(self.cfg.exposure_values)
        self.n_iso = len(self.cfg.iso_values)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.learning_rate = float(learning_rate)
        self.initial_expected_reward = float(initial_expected_reward)
        self.motion_thresholds = motion_thresholds
        self.light_thresholds = light_thresholds

        super().__init__(
            sensor_names=sensor_names,
            action_space=list(self.ACTIONS),
            dim_context=1,
            reward_function=reward_function,
            alpha=alpha,
            lambda_reg=lambda_reg,
            random_seed=random_seed,
        )

        self.current_epsilon = self.epsilon_start
        self.update_count = 0
        self.expected_rewards: Dict[str, np.ndarray] = {}
        self.action_counts: Dict[str, np.ndarray] = {}
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_space)}
        self.pending_update: Optional[Dict[str, Union[str, int, Tuple[int, int]]]] = None

        self.initialize_expected_rewards()

    def initialize_expected_rewards(self) -> None:
        for motion in self.MOTION_STATES:
            for light in self.LIGHT_STATES:
                state = f"{motion}_{light}"
                self.expected_rewards[state] = np.full(
                    self.num_actions,
                    self.initial_expected_reward,
                    dtype=np.float64,
                )
                self.action_counts[state] = np.zeros(self.num_actions, dtype=np.int64)

    def _ensure_state(self, state: str) -> None:
        if state not in self.expected_rewards:
            self.expected_rewards[state] = np.full(
                self.num_actions,
                self.initial_expected_reward,
                dtype=np.float64,
            )
            self.action_counts[state] = np.zeros(self.num_actions, dtype=np.int64)

    def get_expected_rewards(self) -> Dict[str, np.ndarray]:
        return {state: values.copy() for state, values in self.expected_rewards.items()}

    def set_expected_rewards(self, new_rewards: Dict[str, np.ndarray]) -> None:
        self.expected_rewards = {}
        self.action_counts = {}
        for state, values in new_rewards.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape != (self.num_actions,):
                raise ValueError(
                    f"Expected reward vector for {state} must have shape {(self.num_actions,)}, got {arr.shape}"
                )
            self.expected_rewards[state] = arr.copy()
            self.action_counts[state] = np.zeros(self.num_actions, dtype=np.int64)

    def get_state_key(self, angular_velocity: float, light_intensity: float) -> str:
        motion_value = abs(float(angular_velocity))

        motion_level = 0
        for idx, threshold in enumerate(self.motion_thresholds):
            if motion_value < threshold:
                motion_level = idx
                break
        else:
            motion_level = len(self.motion_thresholds)

        light_value = float(light_intensity)
        light_level = 0
        for idx, threshold in enumerate(self.light_thresholds):
            if light_value < threshold:
                light_level = idx
                break
        else:
            light_level = len(self.light_thresholds)

        return f"{self.MOTION_STATES[motion_level]}_{self.LIGHT_STATES[light_level]}"

    @staticmethod
    def _linear_map_clipped(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        if abs(in_max - in_min) <= 1e-12:
            return float(out_min)
        ratio = float(np.clip((value - in_min) / (in_max - in_min), 0.0, 1.0))
        return float(out_min + ratio * (out_max - out_min))

    @classmethod
    def _log_map_clipped(cls, value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        safe_value = max(float(value), 1e-6)
        safe_min = max(float(in_min), 1e-6)
        safe_max = max(float(in_max), safe_min + 1e-6)
        return cls._linear_map_clipped(
            value=float(np.log(safe_value)),
            in_min=float(np.log(safe_min)),
            in_max=float(np.log(safe_max)),
            out_min=out_min,
            out_max=out_max,
        )

    @staticmethod
    def _closest_index(values: Tuple[float, ...] | Tuple[int, ...], target: float) -> int:
        arr = np.asarray(values, dtype=np.float64)
        return int(np.argmin(np.abs(arr - float(target))))

    def calculate_base_indices(
        self,
        context_information: dict,
        heuristic_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[int, int, str]:
        heuristic_offsets = heuristic_offsets or {}
        exposure_values = np.asarray(self.cfg.exposure_values, dtype=np.float64)
        iso_values = np.asarray(self.cfg.iso_values, dtype=np.float64)
        motion_value = abs(float(context_information["angular_velocity"]))
        light_value = float(context_information["light_intensity"])

        motion_min = 0.0
        motion_max = float(self.motion_thresholds[-1])
        light_min = max(1.0, float(self.light_thresholds[0]))
        light_max = float(self.light_thresholds[-1])

        target_exposure_motion = self._linear_map_clipped(
            value=motion_value,
            in_min=motion_min,
            in_max=motion_max,
            out_min=float(exposure_values[-1]),
            out_max=float(exposure_values[0]),
        )
        target_exposure_light = self._log_map_clipped(
            value=max(light_value, light_min),
            in_min=light_min,
            in_max=light_max,
            out_min=float(exposure_values[-1]),
            out_max=float(exposure_values[0]),
        )
        target_exposure = min(target_exposure_motion, target_exposure_light)
        base_exposure_idx = self._closest_index(self.cfg.exposure_values, target_exposure)

        base_iso_from_light = self._log_map_clipped(
            value=max(light_value, light_min),
            in_min=light_min,
            in_max=light_max,
            out_min=float(iso_values[-1]),
            out_max=float(iso_values[0]),
        )
        default_exposure = float(exposure_values[len(exposure_values) // 2])
        iso_scale = default_exposure / max(target_exposure, 1e-12)
        target_iso = base_iso_from_light * iso_scale
        base_iso_idx = self._closest_index(self.cfg.iso_values, target_iso)

        state = self.get_state_key(
            angular_velocity=context_information["angular_velocity"],
            light_intensity=context_information["light_intensity"],
        )
        if state in heuristic_offsets:
            base_exposure_idx += int(np.rint(heuristic_offsets[state][0]))
            base_iso_idx += int(np.rint(heuristic_offsets[state][1]))

        base_exposure_idx = int(np.clip(base_exposure_idx, 0, self.n_exposure - 1))
        base_iso_idx = int(np.clip(base_iso_idx, 0, self.n_iso - 1))
        return base_exposure_idx, base_iso_idx, state

    def update_long_term_memory(
        self,
        heuristic_offsets: Dict[str, Tuple[float, float]],
        state_update_counts: Dict[str, int],
        update_info: dict | None,
        update_threshold: int = 30,
        blend_alpha: float = 0.35,
    ) -> bool:
        if update_info is None:
            return False

        state = str(update_info["state"])
        state_update_counts[state] = int(state_update_counts.get(state, 0)) + 1
        if state_update_counts[state] < update_threshold:
            return False

        expected_rewards = self.expected_rewards.get(state)
        if expected_rewards is None:
            return False

        best_action_idx = int(np.argmax(expected_rewards))
        learned_action = tuple(self.action_space[best_action_idx])
        existing_offset = heuristic_offsets.get(state)
        if existing_offset is None:
            heuristic_offsets[state] = (float(learned_action[0]), float(learned_action[1]))
        else:
            heuristic_offsets[state] = (
                float(existing_offset[0] + (learned_action[0] * blend_alpha)),
                float(existing_offset[1] + (learned_action[1] * blend_alpha)),
            )

        self.reset_state_with_bias(state, learned_action)
        state_update_counts[state] = 0
        return True

    def valid_actions(self, exposure_idx: int, iso_idx: int) -> List[Tuple[int, int]]:
        valid = []
        for de, di in self.action_space:
            next_e = exposure_idx + de
            next_i = iso_idx + di
            if 0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso:
                valid.append((de, di))
        return valid

    def select_action(
        self,
        context_information: dict,
        tie_break_random: bool = True,
        is_infer_mode: bool = False,
    ) -> Dict[str, Union[str, float, int, Tuple[int, int], List[Dict[str, Union[float, int, Tuple[int, int], str]]]]]:
        state = self.get_state_key(
            angular_velocity=context_information["angular_velocity"],
            light_intensity=context_information["light_intensity"],
        )
        self._ensure_state(state)

        candidates = self.valid_actions(
            exposure_idx=context_information["exposure_idx"],
            iso_idx=context_information["iso_idx"],
        )
        if not candidates:
            raise RuntimeError("No valid actions available.")

        rows = []
        for action in candidates:
            action_idx = self.action_to_idx[action]
            rows.append(
                {
                    "action": action,
                    "action_idx": action_idx,
                    "description": self.ACTION_DESCRIPTIONS[action],
                    "expected_reward": float(self.expected_rewards[state][action_idx]),
                    "count": int(self.action_counts[state][action_idx]),
                }
            )

        should_explore = (not is_infer_mode) and (self.rng.random() < self.current_epsilon)
        if should_explore:
            chosen = rows[int(self.rng.integers(len(rows)))]
            mode = "explore"
        else:
            best_reward = max(row["expected_reward"] for row in rows)
            best_rows = [row for row in rows if abs(row["expected_reward"] - best_reward) <= 1e-12]
            chosen = best_rows[int(self.rng.integers(len(best_rows)))] if tie_break_random and len(best_rows) > 1 else best_rows[0]
            mode = "exploit"

        return {
            "state": state,
            "candidates": rows,
            "mode": mode,
            "chosen_action": chosen["action"],
            "chosen_action_idx": chosen["action_idx"],
            "chosen_description": chosen["description"],
            "chosen_expected_reward": chosen["expected_reward"],
            "chosen_count": chosen["count"],
            "epsilon": self.current_epsilon,
        }

    def update_parameters(
        self,
        state_action: Tuple[str, int],
        reward: float,
    ) -> Dict[str, Union[str, int, float]]:
        state, action_idx = state_action
        self._ensure_state(state)

        current_expected = float(self.expected_rewards[state][action_idx])
        new_expected = current_expected + self.learning_rate * (reward - current_expected)
        self.expected_rewards[state][action_idx] = new_expected
        self.action_counts[state][action_idx] += 1

        self.update_count += 1
        self.current_epsilon = max(self.epsilon_min, self.current_epsilon * self.epsilon_decay)

        return {
            "state": state,
            "action_idx": action_idx,
            "old_expected_reward": current_expected,
            "new_expected_reward": float(new_expected),
            "reward": float(reward),
            "epsilon": float(self.current_epsilon),
            "count": int(self.action_counts[state][action_idx]),
        }

    def transition(
        self,
        exposure_idx: int,
        iso_idx: int,
        action: Tuple[int, int],
    ) -> Tuple[int, int]:
        de, di = action
        next_e = exposure_idx + de
        next_i = iso_idx + di

        if not (0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso):
            raise ValueError(f"Invalid action {action} for current state {(exposure_idx, iso_idx)}")

        return next_e, next_i

    def reset_state_with_bias(self, state: str, learned_action: Tuple[int, int]) -> None:
        self._ensure_state(state)

        lx, ly = learned_action
        new_rewards = np.full(self.num_actions, self.initial_expected_reward, dtype=np.float64)

        for idx, (ax, ay) in enumerate(self.action_space):
            if ax == 0 and ay == 0:
                continue

            correlation = (lx * ax) + (ly * ay)
            if correlation > 0:
                new_rewards[idx] = 1.0
            elif correlation == 0:
                new_rewards[idx] = self.initial_expected_reward
            elif ax == -lx and ay == -ly:
                new_rewards[idx] = 0.1
            else:
                new_rewards[idx] = 0.5

        self.expected_rewards[state] = new_rewards
        self.action_counts[state] = np.zeros(self.num_actions, dtype=np.int64)

    def get_stats(self) -> Dict[str, Union[int, float]]:
        visited_states = sum(int(np.any(counts > 0)) for counts in self.action_counts.values())
        all_rewards = np.concatenate(list(self.expected_rewards.values())) if self.expected_rewards else np.array([0.0])
        return {
            "total_states": len(self.expected_rewards),
            "visited_states": visited_states,
            "avg_expected_reward": float(np.mean(all_rewards)),
            "total_actions": self.num_actions,
            "current_epsilon": float(self.current_epsilon),
            "update_count": self.update_count,
        }

    def reset_epsilon(self) -> None:
        self.current_epsilon = self.epsilon_start
        self.update_count = 0

    def step(
        self,
        context_information: dict,
        observations: Dict[str, Union[np.ndarray, List[np.ndarray], float, str]],
    ) -> Dict[str, Union[float, int, str, Tuple[int, int], Dict[str, Union[int, float, str]]]]:
        reward_info_override = observations.get("reward_info_override") if isinstance(observations, dict) else None
        if isinstance(reward_info_override, dict):
            reward_info = dict(reward_info_override)
        else:
            reward_info = self.reward_function(**observations)

        update_info = None
        skip_update = bool(context_information.get("skip_update", False))
        if (not skip_update) and self.pending_update is not None:
            update_info = self.update_parameters(
                (str(self.pending_update["state"]), int(self.pending_update["action_idx"])),
                float(reward_info["reward"]),
            )

        sel = self.select_action(
            context_information=context_information,
            tie_break_random=bool(context_information.get("tie_break_random", True)),
            is_infer_mode=bool(context_information.get("is_infer_mode", False)),
        )

        action = sel["chosen_action"]
        next_e_idx, next_i_idx = self.transition(
            exposure_idx=context_information["exposure_idx"],
            iso_idx=context_information["iso_idx"],
            action=action,
        )
        next_exposure_value = self.cfg.exposure_values[next_e_idx]
        next_iso_value = self.cfg.iso_values[next_i_idx]

        if not skip_update:
            self.pending_update = {
                "state": sel["state"],
                "action_idx": sel["chosen_action_idx"],
                "action": action,
            }

        record = {
            "state": sel["state"],
            "angular_velocity": context_information["angular_velocity"],
            "light_intensity": context_information["light_intensity"],
            "exposure_idx": context_information["exposure_idx"],
            "iso_idx": context_information["iso_idx"],
            "action": action,
            "action_description": sel["chosen_description"],
            "selection_mode": sel["mode"],
            "epsilon": sel["epsilon"],
            "chosen_expected_reward": sel["chosen_expected_reward"],
            "chosen_count": sel["chosen_count"],
            "next_exposure_idx": next_e_idx,
            "next_iso_idx": next_i_idx,
            "next_exposure_value": next_exposure_value,
            "next_iso_value": next_iso_value,
            "reward_info": reward_info,
            "update_info": update_info,
        }
        self.history.append(record)
        return record


L2EpsilonGreedyRGBCamPolicy = L2SharedEGreedyRGBCamPolicy
