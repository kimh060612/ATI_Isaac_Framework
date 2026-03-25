from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.controllers import BaseController
import numpy as np

class CircularController(BaseController):
    def __init__(
        self, 
        name,
        wheel_radius,  
        wheel_base,
    ):
        super().__init__(name)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        
    def forward(self, command, wheel_idx):
        linear_velocity = command["linear_velocity"]
        angular_velocity = command["angular_velocity"]
        
        left_w = ((2 * linear_velocity) - (angular_velocity * self.wheel_base)) / (2 * self.wheel_radius)
        right_w = ((2 * linear_velocity) + (angular_velocity * self.wheel_base)) / (2 * self.wheel_radius)
        
        joint_velocities = np.zeros(4, dtype=np.float32)  # Assuming 4 wheels: front_left, front_right, rear_left, rear_right
        joint_velocities[wheel_idx[0]] = left_w
        joint_velocities[wheel_idx[1]] = left_w
        joint_velocities[wheel_idx[2]] = right_w
        joint_velocities[wheel_idx[3]] = right_w
        return ArticulationAction(joint_velocities=joint_velocities)