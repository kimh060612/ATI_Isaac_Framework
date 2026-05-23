from typing import Optional, Tuple
from .scenario_trajectory import *
import random
import numpy as np

class ScenarioManager:
    def __init__(
        self,
        light_range: Tuple[float, float],
        speed_range: Tuple[float, float],
        scenario_period: int,
        repeat_type: str,
        name: str = "",
        light_label: str = "",
        speed_label: str = "",
    ):
        self.scenario_period = scenario_period
        self.light_range = tuple(float(v) for v in light_range)
        self.speed_range = tuple(float(v) for v in speed_range)
        if repeat_type not in ["sin", "step"]:
            raise ValueError(f"Invalid repeat_type: {repeat_type}. Must be one of ['sin', 'step']")
        self.repeat_type = repeat_type
        self.name = name
        self.light_label = light_label
        self.speed_label = speed_label

    @staticmethod
    def _interpolate(value_range: Tuple[float, float], ratio: float) -> float:
        low, high = value_range
        return float(low + (high - low) * ratio)

    def _ratio_at(self, step_idx: int) -> float:
        if self.scenario_period <= 1:
            return 0.0
        step_idx = int(step_idx) % int(self.scenario_period)
        if self.repeat_type == "step":
            return 0.0 if step_idx < self.scenario_period // 2 else 1.0
        phase = step_idx / float(self.scenario_period - 1)
        return float(0.5 - 0.5 * np.cos(2.0 * np.pi * phase))
    
    
    def step(self, step_idx):
        ratio = self._ratio_at(step_idx)
        return {
            "light_intensity": self._interpolate(self.light_range, ratio),
            "angular_velocity": self._interpolate(self.speed_range, ratio),
            "scenario_name": self.name,
            "scenario_light_label": self.light_label,
            "scenario_speed_label": self.speed_label,
            "scenario_ratio": ratio,
        }
    

def episode_bank(
    context_ranges: list[dict[str, Tuple[float, float]]],
    scenario_period: int,
    repeat_type: str,
    random_seed: Optional[int] = None,
    shuffle: bool = True,
):
    scenario_list = [
        ScenarioManager(
            light_range=c["light_range"],
            speed_range=c["speed_range"],
            scenario_period=scenario_period,
            repeat_type=repeat_type,
            name=c.get("name", ""),
            light_label=c.get("light_label", ""),
            speed_label=c.get("speed_label", ""),
        ) for c in context_ranges
    ]
    if shuffle:
        rng = random.Random(random_seed)
        rng.shuffle(scenario_list)
    return scenario_list
    
    
