from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class SensorParamSpace:
    exposure_values: Tuple[float, ...] = (0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016)
    iso_values: Tuple[int, ...] = (100, 400, 600, 800, 1200, 1600, 2400, 3200)