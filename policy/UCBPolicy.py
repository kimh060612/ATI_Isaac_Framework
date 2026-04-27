from typing import Optional, Tuple, List, Dict, Union
from policy.SensorParams import SensorParamSpace
from policy.BasePolicy import BaseCMABPolicy
import numpy as np

class L2SharedLinUCBRGBCamPolicy(BaseCMABPolicy):
    """
    Shared Linear UCB policy for RGB camera control. Controlling only Exposure time and ISO with discrete actions.
    Action space: 9 discrete actions corresponding to changes in exposure and ISO.
    """
    def __init__(
        self, 
        sensor_names, 
        sensor_config: SensorParamSpace,   
        reward_function, 
        alpha = 1, 
        lambda_reg = 1,     
        random_seed = None
    ):
        self.cfg = sensor_config if sensor_config is not None else SensorParamSpace()
        self.n_exposure = len(self.cfg.exposure_values)
        self.n_iso = len(self.cfg.iso_values)

        all_actions = [(de, di) for de in (-1, 0, 1) for di in (-1, 0, 1)]
        # Feature dimension:
        # [1, w, log_light, e_norm, iso_norm, delta_e, delta_i]
        dim_context = 7
        
        super().__init__(
            sensor_names, 
            all_actions, 
            dim_context, 
            reward_function, 
            alpha, 
            lambda_reg, 
            random_seed
        )
    
    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _normalize_index(self, idx: int, max_idx: int) -> float:
        if max_idx <= 0:
            return 0.0
        return idx / max_idx

    def build_context(
        self,
        angular_velocity: float,
        light_intensity: float,
        exposure_idx: int,
        iso_idx: int,
    ) -> np.ndarray:
        """
        x = [1, w, log(light), e_idx_norm, iso_idx_norm]
        """
        if light_intensity <= 0:
            raise ValueError("light_intensity must be > 0 because log(light) is used.")

        e_norm = self._normalize_index(exposure_idx, self.n_exposure - 1)
        iso_norm = self._normalize_index(iso_idx, self.n_iso - 1)

        x = np.array(
            [
                1.0,
                float(angular_velocity),
                float(np.log(light_intensity)),
                e_norm,
                iso_norm,
            ],
            dtype=np.float64,
        )
        return x
    
    def valid_actions(
        self, 
        exposure_idx: int, 
        iso_idx: int
    ) -> List[Tuple[int, int]]:
        """
        Remove invalid actions instead of clipping.
        """
        valid = []
        for de, di in self.action_space:
            next_e = exposure_idx + de
            next_i = iso_idx + di
            if 0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso:
                valid.append((de, di))
        return valid
    
    def joint_feature(self, context: np.ndarray, action: Tuple[int, int]) -> np.ndarray:
        """
        z = [1, w, log_light, e_norm, iso_norm, delta_e, delta_i]
        """
        de, di = action
        z = np.concatenate(
            [context, np.array([float(de), float(di)], dtype=np.float64)]
        )
        return z
    
    def select_action(
        self, 
        context_information: dict,
        tie_break_random: bool = True,
    ) -> dict[str, float | int]:
        """
        Select one valid action using shared LinUCB.
        """
        context = self.build_context(
            angular_velocity=context_information["angular_velocity"],
            light_intensity=context_information["light_intensity"],
            exposure_idx=context_information["exposure_idx"],
            iso_idx=context_information["iso_idx"],
        )

        candidates = self.valid_actions(
            context_information["exposure_idx"], 
            context_information["iso_idx"]
        )
        if len(candidates) == 0:
            raise RuntimeError("No valid actions available.")

        rows = []
        best_score = -np.inf
        best_actions = []

        # print(candidates)
        for action in candidates:
            z = self.joint_feature(context, action)
            score, mean, bonus = self.ucb_score(z)

            row = {
                "action": action,
                "z": z,
                "score": score,
                "mean": mean,
                "bonus": bonus,
            }
            rows.append(row)

            if score > best_score + 1e-12:
                best_score = score
                best_actions = [row]
            elif abs(score - best_score) <= 1e-12:
                best_actions.append(row)

        if tie_break_random and len(best_actions) > 1:
            chosen = best_actions[self.rng.integers(len(best_actions))]
        else:
            chosen = best_actions[0]
            
        return {
            "context": context,
            "candidates": rows,
            "chosen_action": chosen["action"],
            "chosen_z": chosen["z"],
            "chosen_score": chosen["score"],
            "chosen_mean": chosen["mean"],
            "chosen_bonus": chosen["bonus"],
        }
    
    def update_parameters(
        self, 
        z: np.ndarray, 
        reward: float
    ):
        self.A += np.outer(z, z)
        self.b += reward * z
    
    def transition(
        self,
        exposure_idx: int,
        iso_idx: int,
        action: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Apply valid delta action.
        """
        de, di = action
        next_e = exposure_idx + de
        next_i = iso_idx + di

        if not (0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso):
            raise ValueError(f"Invalid action {action} for current state {(exposure_idx, iso_idx)}")

        return next_e, next_i
    
    def step(
        self, 
        context_information: dict,
        observations: Dict[str, Union[np.ndarray, List[np.ndarray], float, str]],
    ) -> Dict[str, Union[float, int, Tuple[int, int]]]:
        sel = self.select_action(
            context_information=context_information, 
            tie_break_random=True, 
        ) 
        
        action = sel["chosen_action"]
        z = sel["chosen_z"]
        next_e_idx, next_i_idx = self.transition( 
            context_information["exposure_idx"], 
            context_information["iso_idx"], 
            action 
        )
        next_exposure_value = self.cfg.exposure_values[next_e_idx]
        next_iso_value = self.cfg.iso_values[next_i_idx]

        r_t = self.reward_function(**observations)
        self.update_parameters(z, r_t["reward"])

        record = {
            "angular_velocity": context_information["angular_velocity"],
            "light_intensity": context_information["light_intensity"],
            "exposure_idx": context_information["exposure_idx"],
            "iso_idx": context_information["iso_idx"],
            "action": action,
            "next_exposure_idx": next_e_idx,
            "next_iso_idx": next_i_idx,
            "next_exposure_value": next_exposure_value,
            "next_iso_value": next_iso_value,
            "chosen_score": sel["chosen_score"],
            "chosen_mean": sel["chosen_mean"],
            "chosen_bonus": sel["chosen_bonus"],
            "reward_info": r_t
        }
        self.history.append(record)
        return record


class L2DisjointLinUCBRGBCamPolicy(BaseCMABPolicy):
    """
    Disjoint Linear UCB policy for RGB camera control. Controlling only Exposure time and ISO with discrete actions.
    Action space: 9 discrete actions corresponding to changes in exposure and ISO.

    Unlike L2SharedLinUCBRGBCamPolicy, this policy keeps an independent linear model for each action.
    """
    def __init__(
        self,
        sensor_names,
        sensor_config: SensorParamSpace,
        reward_function,
        alpha = 1,
        lambda_reg = 1,
        random_seed = None
    ):
        self.cfg = sensor_config if sensor_config is not None else SensorParamSpace()
        self.n_exposure = len(self.cfg.exposure_values)
        self.n_iso = len(self.cfg.iso_values)

        all_actions = [(de, di) for de in (-1, 0, 1) for di in (-1, 0, 1)]
        self.action_to_index = {action: idx for idx, action in enumerate(all_actions)}

        # Feature dimension:
        # [1, w, log_light]
        dim_context = 3

        super().__init__(
            sensor_names,
            all_actions,
            dim_context,
            reward_function,
            alpha,
            lambda_reg,
            random_seed
        )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _defining_parameters(self):
        """Initialize independent LinUCB parameters for each action."""
        self.A = np.stack(
            [
                self.lambda_reg * np.eye(self.dim_context, dtype=np.float64)
                for _ in range(self.num_actions)
            ],
            axis=0,
        )
        self.b = np.zeros((self.num_actions, self.dim_context), dtype=np.float64)

    def _normalize_index(self, idx: int, max_idx: int) -> float:
        if max_idx <= 0:
            return 0.0
        return idx / max_idx

    def _action_index(self, action: Tuple[int, int]) -> int:
        return self.action_to_index[action]

    def build_context(
        self,
        angular_velocity: float,
        light_intensity: float,
        # exposure_idx: int,
        # iso_idx: int,
    ) -> np.ndarray:
        """
        x = [1, w, log(light)]
        """
        if light_intensity <= 0:
            raise ValueError("light_intensity must be > 0 because log(light) is used.")

        # e_norm = self._normalize_index(exposure_idx, self.n_exposure - 1)
        # iso_norm = self._normalize_index(iso_idx, self.n_iso - 1)

        x = np.array(
            [
                1.0,
                float(angular_velocity),
                float(np.log(light_intensity))
            ],
            dtype=np.float64,
        )
        return x

    def valid_actions(
        self,
        exposure_idx: int,
        iso_idx: int
    ) -> List[Tuple[int, int]]:
        """
        Remove invalid actions instead of clipping.
        """
        valid = []
        for de, di in self.action_space:
            next_e = exposure_idx + de
            next_i = iso_idx + di
            if 0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso:
                valid.append((de, di))
        return valid

    def theta_hat(self, action: Tuple[int, int]) -> np.ndarray:
        """
        Solve A_a theta_a = b_a for the selected action.
        """
        action_idx = self._action_index(action)
        return np.linalg.solve(self.A[action_idx], self.b[action_idx])

    def ucb_score(self, x: np.ndarray, action: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Returns:
            score = mean + alpha * uncertainty
            mean
            bonus
        """
        action_idx = self._action_index(action)
        theta = self.theta_hat(action)
        mean = float(x @ theta)
        A_inv_x = np.linalg.solve(self.A[action_idx], x)
        bonus = float(self.alpha * np.sqrt(x @ A_inv_x))
        score = mean + bonus
        return score, mean, bonus

    def select_action(
        self,
        context_information: dict,
        tie_break_random: bool = True,
    ) -> dict[str, float | int]:
        """
        Select one valid action using disjoint LinUCB.
        """
        context = self.build_context(
            angular_velocity=context_information["angular_velocity"],
            light_intensity=context_information["light_intensity"],
            # exposure_idx=context_information["exposure_idx"],
            # iso_idx=context_information["iso_idx"],
        )

        candidates = self.valid_actions(
            context_information["exposure_idx"],
            context_information["iso_idx"]
        )
        if len(candidates) == 0:
            raise RuntimeError("No valid actions available.")

        rows = []
        best_score = -np.inf
        best_actions = []

        for action in candidates:
            score, mean, bonus = self.ucb_score(context, action)

            row = {
                "action": action,
                "z": context,
                "score": score,
                "mean": mean,
                "bonus": bonus,
            }
            rows.append(row)

            if score > best_score + 1e-12:
                best_score = score
                best_actions = [row]
            elif abs(score - best_score) <= 1e-12:
                best_actions.append(row)

        if tie_break_random and len(best_actions) > 1:
            chosen = best_actions[self.rng.integers(len(best_actions))]
        else:
            chosen = best_actions[0]

        return {
            "context": context,
            "candidates": rows,
            "chosen_action": chosen["action"],
            "chosen_z": chosen["z"],
            "chosen_score": chosen["score"],
            "chosen_mean": chosen["mean"],
            "chosen_bonus": chosen["bonus"],
        }

    def update_parameters(
        self,
        z: np.ndarray,
        reward: float,
        action: Tuple[int, int],
    ):
        action_idx = self._action_index(action)
        self.A[action_idx] += np.outer(z, z)
        self.b[action_idx] += reward * z

    def transition(
        self,
        exposure_idx: int,
        iso_idx: int,
        action: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Apply valid delta action.
        """
        de, di = action
        next_e = exposure_idx + de
        next_i = iso_idx + di

        if not (0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso):
            raise ValueError(f"Invalid action {action} for current state {(exposure_idx, iso_idx)}")

        return next_e, next_i

    def step(
        self,
        context_information: dict,
        observations: Dict[str, Union[np.ndarray, List[np.ndarray], float, str]],
    ) -> Dict[str, Union[float, int, Tuple[int, int]]]:
        sel = self.select_action(
            context_information=context_information,
            tie_break_random=True,
        )

        action = sel["chosen_action"]
        z = sel["chosen_z"]
        next_e_idx, next_i_idx = self.transition(
            context_information["exposure_idx"],
            context_information["iso_idx"],
            action
        )
        next_exposure_value = self.cfg.exposure_values[next_e_idx]
        next_iso_value = self.cfg.iso_values[next_i_idx]

        r_t = self.reward_function(**observations)
        self.update_parameters(z, r_t["reward"], action)

        record = {
            "angular_velocity": context_information["angular_velocity"],
            "light_intensity": context_information["light_intensity"],
            "exposure_idx": context_information["exposure_idx"],
            "iso_idx": context_information["iso_idx"],
            "action": action,
            "next_exposure_idx": next_e_idx,
            "next_iso_idx": next_i_idx,
            "next_exposure_value": next_exposure_value,
            "next_iso_value": next_iso_value,
            "chosen_score": sel["chosen_score"],
            "chosen_mean": sel["chosen_mean"],
            "chosen_bonus": sel["chosen_bonus"],
            "reward_info": r_t
        }
        self.history.append(record)
        return record
