from typing import Dict, Union, Tuple
from .BaseSensorController import BaseSensorController
import numpy as np
import math

class L1RGBCameraController(BaseSensorController):
    """
    Default camera controller for ATI Isaac framework.

    It manages the standard camera exposure parameters used by the current
    pipeline:
    - `iso`
    - `shutter_time`
    - `aperture`
    """
    def __init__(
        self,
        sensor_name,
        camera,
        camera_prim,
        camera_fps,
        control_parameters: Dict[str, Union[float, int]] | None = None,
    ):
        super().__init__(
            sensor_name=sensor_name,
            camera=camera,
            camera_prim=camera_prim,
            camera_fps=camera_fps,
            control_parameters=control_parameters,
        )
        self.exposure_values = [0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016]
        self.iso_values = [100, 400, 600, 800, 1200, 1600, 2400, 3200]
        
        self.EXP_T_MAX = max(self.exposure_values)
        self.EXP_T_MIN = min(self.exposure_values)
        self.ISO_MAX = max(self.iso_values)
        self.ISO_MIN = min(self.iso_values)
        
        # Exposure range in nanoseconds
        self.exposure_passive_ns: int = 16_666_666   # 1/60s
        self.exposure_active_max_ns: int = 1_000_000 # 1/1000s
        
        ## Safe exposure on velocity:
        ## LINEAR VELOCITY & ANGULAR VELOCITY
        ## HEURISTIC VALUE, Need to be fixed
        self.VEL_MAX = 2.0
        self.VEL_MIN = 0.0
        self.AVEL_MAX = 1.0
        self.AVEL_MIN = 0.0
        
        # Lux range used by mapLogLog()
        self.nit_min: float = 100.0
        self.nit_max: float = 6000.0
        
        ### Safe exposure on Light Intensity
        self.safe_exp_dark_ns: float = 16_666_666.0
        self.safe_exp_bright_ns: float = 1_000_000.0
        
        ### Safe ISO on Light Intensity
        self.base_iso_dark: float = 2000.0
        self.base_iso_bright: float = 100.0
        
        self.update_parameters(self.control_parameters)

    def __update_sensor_parameters(self, control_parameters: Dict[str, Union[float, int]] | None):
        control_parameters = control_parameters or {}
        if "iso" in control_parameters and control_parameters["iso"] is not None:
            self._set_iso(control_parameters["iso"])
        if "shutter_time" in control_parameters and control_parameters["shutter_time"] is not None:
            self._set_shutter_time(control_parameters["shutter_time"])
        if "aperture" in control_parameters and control_parameters["aperture"] is not None:
            self._set_aperture(control_parameters["aperture"])
       
    @staticmethod
    def _coerce(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
    
    def map_value(
        self,
        value: float,
        in_min: float,
        in_max: float,
        out_min: float,
        out_max: float,
    ) -> float:
        """
        Linear mapping with clipping.
        Equivalent to Kotlin mapValue().
        """
        if in_max == in_min:
            raise ValueError("in_max and in_min must be different.")
        result = (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return self._coerce(result, min(out_min, out_max), max(out_min, out_max))

    def map_log(
        self,
        value: float,
        in_init: float,
        in_end: float,
        out_init: float,
        out_end: float
    ):
        """
        Log-log interpolation.
        Equivalent to SensorDataManager.mapLogLog():
            value is clamped in input range,
            interpolation happens in log(input) -> log(output) space.
        """
        eps = 1e-6
        s_in_init = max(in_init, eps)
        s_in_end = max(in_end, eps)
        s_out_init = max(out_init, eps)
        s_out_end = max(out_end, eps)
        x = self._coerce(value, min(s_in_init, s_in_end), max(s_in_init, s_in_end))
        log_in_init = math.log(s_in_init)
        log_in_end = math.log(s_in_end)
        log_out_init = math.log(s_out_init)
        log_out_end = math.log(s_out_end)
        log_x = math.log(x)
        ratio = (log_x - log_in_init) / (log_in_end - log_in_init)
        log_y = log_out_init + ratio * (log_out_end - log_out_init)
        return math.exp(log_y)

    def get_safe_exposure_duration(self, lux: float) -> int:
        return int(
            self.map_log(
                lux,
                self.nit_min,
                self.nit_max,
                self.safe_exp_dark_ns,
                self.safe_exp_bright_ns,
            )
        )
        
    def get_base_iso_for_lux(self, lux: float) -> int:
        return int(
            round(
                self.map_log(
                    lux,
                    self.nit_min,
                    self.nit_max,
                    self.base_iso_dark,
                    self.base_iso_bright,
                )
            )
        )

    @staticmethod
    def closest_index(grid: Tuple[int, ...], target: float) -> int:
        arr = np.asarray(grid, dtype=np.float64)
        return int(np.argmin(np.abs(arr - target)))

    def calculate_base_indices(
        self,
        accel: float,
        gyro: float,
        lux: float,
    ) -> Tuple[int, int, dict]:
        """
        Equivalent to MainActivity.calculateBaseIndices()
        except this version does not apply L2/RL offsets.
        """
        # 1. Motion-based exposure candidates
        exp_from_accel = self.map_value(
            accel,
            self.VEL_MIN,
            self.VEL_MAX,
            self.exposure_passive_ns,
            self.exposure_active_max_ns,
        )
        exp_from_gyro = self.map_value(
            gyro,
            self.AVEL_MIN,
            self.AVEL_MAX,
            self.exposure_passive_ns,
            self.exposure_active_max_ns,
        )
        # 2. More aggressive motion signal wins.
        # Smaller exposure = less motion blur.
        raw_target_exp = min(exp_from_accel, exp_from_gyro)
        # 3. Lux-based safety floor.
        # Prevent too-short exposure in dark environments.
        safe_limit = self.get_safe_exposure_duration(lux)
        raw_target_exp = max(raw_target_exp, safe_limit)
        # 4. Base ISO from lux, then compensate for reduced exposure.
        base_iso_calc = self.get_base_iso_for_lux(lux)
        ratio = self.exposure_passive_ns / raw_target_exp
        raw_target_iso = int(base_iso_calc * ratio)
        # 5. Snap to camera grids.
        base_exp_idx = self.closest_index(self.exposure_values, raw_target_exp * 1e-9)
        base_iso_idx = self.closest_index(self.iso_values, raw_target_iso * 1e-9)
        # 6. Safety clipping.
        base_exp_idx = int(np.clip(base_exp_idx, 0, len(self.exposure_values) - 1))
        base_iso_idx = int(np.clip(base_iso_idx, 0, len(self.iso_values) - 1))
        meta = {
            "exp_from_accel_ns": exp_from_accel,
            "exp_from_gyro_ns": exp_from_gyro,
            "safe_limit_ns": safe_limit,
            "raw_target_exp_ns": raw_target_exp,
            "base_iso_from_lux": base_iso_calc,
            "exposure_compensation_ratio": ratio,
            "raw_target_iso": raw_target_iso,
            "base_exp_idx": base_exp_idx,
            "base_iso_idx": base_iso_idx,
        }
        return base_exp_idx, base_iso_idx, meta

    def update_parameters(
        self, 
        # current_context: Dict[str, Union[dict, float, int]] | None,
        # l2_action: Dict[str, Union[float, int]] | None,
        control_parameters: Dict[str, Union[float, int]] | None
    ):
        # control_parameters = control_parameters or {}
        l2_action = control_parameters.get("l2_action", {}) or {}
        current_context = control_parameters.get("current_context", {}) or {}
        
        lin_vel, ang_vel = current_context.get("lin_vel", 0.0), current_context.get("ang_vel", 0.0)
        lux = current_context.get("lux", 0.0)
        exp_idx, iso_idx, meta = self.calculate_base_indices(lin_vel, ang_vel, lux)
        
        d_exp, d_iso = l2_action.get("d_exp", 0), l2_action.get("d_iso", 0)
        exp_idx = int(np.clip(exp_idx + d_exp, 0, len(self.exposure_values) - 1))
        iso_idx = int(np.clip(iso_idx + d_iso, 0, len(self.iso_values) - 1))
        meta["l2_offset"] = l2_action
        
        self.__update_sensor_parameters(
            control_parameters={
                "iso": self.iso_values[iso_idx],
                "shutter_time": self.exposure_values[exp_idx],
            }
        )
        return self.get_control_parameters()

class L1ShortTermMemoryRGBController(BaseSensorController):
    def __init__(
        self,
        sensor_name,
        camera,
        camera_prim,
        camera_fps,
        control_parameters: Dict[str, Union[float, int]] | None = None,
    ):
        super().__init__(
            sensor_name=sensor_name,
            camera=camera,
            camera_prim=camera_prim,
            camera_fps=camera_fps,
            control_parameters=control_parameters,
        )
        self.exposure_values = [0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016]
        self.iso_values = [100, 400, 600, 800, 1200, 1600, 2400, 3200]

        ## Hyperparameters for differentiation logic
        self.short_diff_window = 2
        self.mid_diff_window = 6
        self.memory_length = 10  # Number of past steps to remember
        self.trend_history_window = 3
        self.min_trend_transitions = 2
        self.reward_abs_drop_threshold = 0.15
        self.reward_rel_drop_threshold = 0.25
        self.reward_slope_drop_threshold = -0.05
        self.exposure_product_trend_epsilon = 1e-9
        self.selection_history_window = 5

        self.step_memory: list[float] = []
        self.exposure_time_memory: list[float] = []
        self.iso_memory: list[float] = []
        self.exposure_product_memory: list[float] = []
        self.rel_exposure_value_memory: list[float] = []
        self.reward_memory: list[float] = []
        self.curr_idx = -1
        self.curr_steps = 0
        self.__update_sensor_parameters(self.control_parameters)
        
    def __update_sensor_parameters(self, control_parameters: Dict[str, Union[float, int]] | None):
        control_parameters = control_parameters or {}
        if "iso" in control_parameters and control_parameters["iso"] is not None:
            self._set_iso(control_parameters["iso"])
        if "shutter_time" in control_parameters and control_parameters["shutter_time"] is not None:
            self._set_shutter_time(control_parameters["shutter_time"])
        if "aperture" in control_parameters and control_parameters["aperture"] is not None:
            self._set_aperture(control_parameters["aperture"])
    
    @staticmethod
    def __linear_slope(memory: list[float]) -> float:
        y = np.asarray(memory, dtype=np.float64)
        x = np.arange(len(y), dtype=np.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = np.sum(x * x)
        if denom < 1e-12:
            return 0.0
        return float(np.sum(x * y) / denom)
    
    def __make_queue_into_linear_list(self, memory: list[float]) -> list[float]:
        return list(memory)

    @staticmethod
    def __exposure_product(exposure_time: float, iso: float) -> float:
        return float(exposure_time) * float(iso)

    @staticmethod
    def __rel_exposure_value(exposure_time: float, iso: float) -> float:
        ev_value = (float(exposure_time) * 1000.0) * (float(iso) / 100.0)
        return math.log2(max(ev_value, 1e-12))

    @staticmethod
    def __sign_with_deadband(value: float, epsilon: float) -> int:
        if value > epsilon:
            return 1
        if value < -epsilon:
            return -1
        return 0

    @staticmethod
    def __closest_index(grid: list[float], target: float) -> int:
        arr = np.asarray(grid, dtype=np.float64)
        return int(np.argmin(np.abs(arr - float(target))))

    def __append_memory(
        self,
        step: float,
        exposure_time: float,
        iso: float,
        reward: float,
    ):
        self.step_memory.append(float(step))
        self.exposure_time_memory.append(float(exposure_time))
        self.iso_memory.append(float(iso))
        self.exposure_product_memory.append(self.__exposure_product(exposure_time, iso))
        self.rel_exposure_value_memory.append(self.__rel_exposure_value(exposure_time, iso))
        self.reward_memory.append(float(reward))

        if len(self.reward_memory) > self.memory_length:
            self.step_memory.pop(0)
            self.exposure_time_memory.pop(0)
            self.iso_memory.pop(0)
            self.exposure_product_memory.pop(0)
            self.rel_exposure_value_memory.pop(0)
            self.reward_memory.pop(0)

        self.curr_steps += 1
        self.curr_idx = len(self.reward_memory) - 1

    def __reward_drop_stats(self, reward_linear: list[float]) -> dict:
        if len(reward_linear) < self.short_diff_window + 1:
            return {
                "reward_crashed": False,
                "abs_drop": 0.0,
                "rel_drop": 0.0,
                "reward_slope": 0.0,
                "prev_mean": np.nan,
                "recent_reward": reward_linear[-1] if reward_linear else np.nan,
            }

        prev_window = min(self.mid_diff_window, len(reward_linear) - 1)
        prev = np.asarray(reward_linear[-(prev_window + 1):-1], dtype=np.float64)
        recent_reward = float(reward_linear[-1])
        prev_mean = float(np.mean(prev))
        abs_drop = prev_mean - recent_reward
        rel_drop = abs_drop / (abs(prev_mean) + 1e-12)
        slope_window = reward_linear[-min(len(reward_linear), prev_window + 1):]
        reward_slope = self.__linear_slope(slope_window)
        reward_crashed = (
            abs_drop >= self.reward_abs_drop_threshold
            or (
                rel_drop >= self.reward_rel_drop_threshold
                and reward_slope <= self.reward_slope_drop_threshold
            )
        )

        return {
            "reward_crashed": bool(reward_crashed),
            "abs_drop": float(abs_drop),
            "rel_drop": float(rel_drop),
            "reward_slope": float(reward_slope),
            "prev_mean": float(prev_mean),
            "recent_reward": recent_reward,
        }

    def __continuing_exposure_product_trend(
        self,
        exposure_product_linear: list[float],
        proposed_product: float,
    ) -> dict:
        if len(exposure_product_linear) < 2:
            return {
                "trend_continues": False,
                "trend_direction": 0,
                "product_slope": 0.0,
                "trend_values": [],
            }

        history_values = exposure_product_linear[-self.trend_history_window:]
        trend_values = history_values + [float(proposed_product)]
        deltas = np.diff(np.asarray(trend_values, dtype=np.float64))
        signs = [
            self.__sign_with_deadband(delta, self.exposure_product_trend_epsilon)
            for delta in deltas
        ]
        nonzero_signs = [sign for sign in signs if sign != 0]
        trend_direction = nonzero_signs[-1] if nonzero_signs else 0
        trend_continues = (
            len(nonzero_signs) >= self.min_trend_transitions
            and all(sign == trend_direction for sign in nonzero_signs)
        )
        product_slope = self.__linear_slope(trend_values)

        return {
            "trend_continues": bool(trend_continues),
            "trend_direction": int(trend_direction if trend_continues else 0),
            "product_slope": float(product_slope),
            "trend_values": trend_values,
        }

    def __should_reject_command(
        self,
        proposed_exposure_time: float,
        proposed_iso: float,
    ) -> tuple[bool, dict]:
        proposed_product = self.__exposure_product(proposed_exposure_time, proposed_iso)
        reward_stats = self.__reward_drop_stats(self.reward_memory)
        trend_stats = self.__continuing_exposure_product_trend(
            self.exposure_product_memory,
            proposed_product,
        )
        should_reject = (
            reward_stats["reward_crashed"]
            and trend_stats["trend_continues"]
        )
        return bool(should_reject), {
            **reward_stats,
            **trend_stats,
            "proposed_exposure_product": float(proposed_product),
        }

    def __grid_candidates(self) -> list[dict]:
        candidates = []
        for exp_idx, exposure_time in enumerate(self.exposure_values):
            for iso_idx, iso in enumerate(self.iso_values):
                product = self.__exposure_product(exposure_time, iso)
                candidates.append({
                    "exposure_idx": exp_idx,
                    "iso_idx": iso_idx,
                    "shutter_time": float(exposure_time),
                    "iso": float(iso),
                    "exposure_product": float(product),
                    "rel_exposure": self.__rel_exposure_value(exposure_time, iso),
                })
        return candidates

    def __best_recent_reference(self) -> tuple[float, float, float]:
        if len(self.reward_memory) <= 1:
            return (
                float(self.exposure_time_memory[-1]),
                float(self.iso_memory[-1]),
                float(self.exposure_product_memory[-1]),
            )

        end_idx = len(self.reward_memory) - 1
        start_idx = max(0, end_idx - self.selection_history_window)
        history_indices = list(range(start_idx, end_idx))
        best_idx = max(history_indices, key=lambda idx: self.reward_memory[idx])
        return (
            float(self.exposure_time_memory[best_idx]),
            float(self.iso_memory[best_idx]),
            float(self.exposure_product_memory[best_idx]),
        )

    def __select_recovery_parameters(self, trend_direction: int) -> dict:
        current_exposure_time = float(self.exposure_time_memory[-1])
        current_iso = float(self.iso_memory[-1])
        current_product = float(self.exposure_product_memory[-1])
        current_exp_idx = self.__closest_index(self.exposure_values, current_exposure_time)
        current_iso_idx = self.__closest_index(self.iso_values, current_iso)
        target_exposure_time, target_iso, target_product = self.__best_recent_reference()
        target_exp_idx = self.__closest_index(self.exposure_values, target_exposure_time)
        target_iso_idx = self.__closest_index(self.iso_values, target_iso)

        candidates = self.__grid_candidates()
        if trend_direction > 0:
            recovery_candidates = [
                candidate for candidate in candidates
                if candidate["exposure_product"] < current_product - self.exposure_product_trend_epsilon
            ]
        elif trend_direction < 0:
            recovery_candidates = [
                candidate for candidate in candidates
                if candidate["exposure_product"] > current_product + self.exposure_product_trend_epsilon
            ]
        else:
            recovery_candidates = []

        if not recovery_candidates:
            recovery_candidates = [
                candidate for candidate in candidates
                if math.isclose(
                    candidate["shutter_time"],
                    current_exposure_time,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(candidate["iso"], current_iso, rel_tol=0.0, abs_tol=1e-9)
            ] or candidates

        log_target_product = math.log(max(target_product, 1e-12))

        def score(candidate: dict) -> tuple[float, int, int]:
            product_distance = abs(math.log(max(candidate["exposure_product"], 1e-12)) - log_target_product)
            target_index_distance = (
                abs(candidate["exposure_idx"] - target_exp_idx)
                + abs(candidate["iso_idx"] - target_iso_idx)
            )
            current_index_distance = (
                abs(candidate["exposure_idx"] - current_exp_idx)
                + abs(candidate["iso_idx"] - current_iso_idx)
            )
            return product_distance, target_index_distance, current_index_distance

        return min(recovery_candidates, key=score)
    
    def update_parameters(
        self, 
        control_parameters: Dict[str, Union[float, int]] | None
    ):
        if control_parameters is None:
            return {
                "is_sensor_updated": False,
                **self.get_control_parameters()
            }
        curr_step = control_parameters.get("step", 0)
        exposure_time = control_parameters.get("shutter_time", None)  # default 10ms
        gain = control_parameters.get("iso", None)  # default gain 1.0
        reward_value = control_parameters.get("reward", None)    
        if exposure_time is None or \
            gain is None or \
            curr_step is None or \
            reward_value is None:
            raise ValueError("exposure_time, gain, step, and reward must be provided in control_parameters.")
        
        current_params = self.get_control_parameters()
        current_exposure_time = current_params["shutter_time"]
        current_gain = current_params["iso"]
        self.__append_memory(
            step=curr_step,
            exposure_time=current_exposure_time,
            iso=current_gain,
            reward=reward_value,
        )

        should_reject, reject_stats = self.__should_reject_command(
            proposed_exposure_time=exposure_time,
            proposed_iso=gain,
        )

        action_accepted = not should_reject
        fallback_sensor_updated = False
        selected_parameters = {
            "shutter_time": float(exposure_time),
            "iso": float(gain),
        }

        if should_reject:
            selected_parameters = self.__select_recovery_parameters(
                trend_direction=reject_stats["trend_direction"],
            )
            selected_parameters = {
                "shutter_time": selected_parameters["shutter_time"],
                "iso": selected_parameters["iso"],
            }
            fallback_sensor_updated = (
                not math.isclose(selected_parameters["shutter_time"], current_exposure_time, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(selected_parameters["iso"], current_gain, rel_tol=0.0, abs_tol=1e-9)
            )
            if fallback_sensor_updated:
                self.__update_sensor_parameters(selected_parameters)
        else:
            self.__update_sensor_parameters(selected_parameters)

        return {
            "is_sensor_updated": action_accepted,
            "action_accepted": action_accepted,
            "fallback_sensor_updated": fallback_sensor_updated,
            "rejection_reason": "reward_drop_with_continuing_exposure_product_trend" if should_reject else None,
            "requested_shutter_time": float(exposure_time),
            "requested_iso": float(gain),
            "selected_shutter_time": float(selected_parameters["shutter_time"]),
            "selected_iso": float(selected_parameters["iso"]),
            "selected_exposure_idx": self.__closest_index(
                self.exposure_values,
                selected_parameters["shutter_time"],
            ),
            "selected_iso_idx": self.__closest_index(self.iso_values, selected_parameters["iso"]),
            "selected_exposure_product": self.__exposure_product(
                selected_parameters["shutter_time"],
                selected_parameters["iso"],
            ),
            "reward_drop_abs": float(reject_stats["abs_drop"]),
            "reward_drop_rel": float(reject_stats["rel_drop"]),
            "reward_slope": float(reject_stats["reward_slope"]),
            "exposure_product_slope": float(reject_stats["product_slope"]),
            "exposure_product_trend_direction": int(reject_stats["trend_direction"]),
            **self.get_control_parameters()
        }
