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

# Enable Livestream extension
from isaacsim.core.utils.extensions import enable_extension
simulation_app.set_setting("/app/window/drawMouse", True)
enable_extension("omni.services.livestream.nvcf")

# Scene Building
from scene import ATIDepthScene
from ati_config import ATIBaseConfig
from time import time
import traceback
import numpy as np
import random
import os

if __name__ == "__main__":
    
    pass
        