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
        ## Hyperparameters for differentiation logic
        self.short_diff_window = 2
        self.mid_diff_window = 6
        self.memory_length = 10  # Number of past steps to remember
        self.rel_exposure_value_memory: list[float] = [0.0] * self.memory_length
        self.reward_memory: list[float] = [0.0] * self.memory_length
        self.curr_idx = 0
        self.curr_steps = 0
        self.update_parameters({
            "step":0.0,
            "reward":0.0,
            **self.control_parameters
        })
        
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
        # Assuming self.curr_idx points to the most recent entry
        return memory[self.curr_idx + 1:] + memory[:self.curr_idx + 1]
    
    def __judge_significant_drop(
        self, 
        reward_linear: list[float], 
        rel_exp_linear: list[float]
    ) -> bool:
        # Threshold for significant reward drop, can be tuned based on empirical observations
        if self.curr_steps < self.mid_diff_window + self.short_diff_window:
            return False  # Not enough data to judge
        p_win = self.mid_diff_window
        r_win = self.short_diff_window
        reward_slope = self.__linear_slope(reward_linear[-(p_win + r_win):])
        rel_exp_slope = self.__linear_slope(rel_exp_linear[-(p_win + r_win):])
        prev = reward_linear[-(p_win + r_win):-r_win]
        recent = reward_linear[-r_win:]
        prev_mean = float(np.mean(prev))
        prev_std = float(np.std(prev)) + 1e-12
        recent_mean = float(np.mean(recent))
        abs_drop = prev_mean - recent_mean
        rel_drop = abs_drop / (abs(prev_mean) + 1e-12)
        z_drop = abs_drop / prev_std
        
        reward_crashed = (
            abs_drop >= 1e-6 or reward_slope < -0.2
            and rel_drop >= 0.25
            and z_drop >= 2.0
        )
        
        return reward_crashed and math.fabs(rel_exp_slope) > 0.2
    
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
        
        is_sensor_updated = False
        target_rel_exp = math.log2((exposure_time * 1000.) * (gain / 100.))
        self.curr_steps += 1
        self.curr_idx = (self.curr_idx + 1) % self.memory_length
        self.rel_exposure_value_memory[self.curr_idx] = target_rel_exp
        self.reward_memory[self.curr_idx] = reward_value
        
        ### If reward is significantly drops in short term, deny the l2 action by keeping the current sensor parameters.
        rel_exp_linear = self.__make_queue_into_linear_list(self.rel_exposure_value_memory)
        reward_linear = self.__make_queue_into_linear_list(self.reward_memory)
        # Threshold for significant reward drop, can be tuned based on empirical observations
        if not self.__judge_significant_drop(reward_linear, rel_exp_linear):
            is_sensor_updated = True
            self.__update_sensor_parameters(
                control_parameters={
                    "shutter_time": exposure_time,
                    "iso": gain,
                }
            )
        
        return {
            "is_sensor_updated": is_sensor_updated,
            **self.get_control_parameters()
        }