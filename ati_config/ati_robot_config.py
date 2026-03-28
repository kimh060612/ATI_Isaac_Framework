from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np
import os

@dataclass
class ATIBaseRobotConfig:
    """Configuration for ATI robot."""
    robot_name: str
    
    ## Base Configurations
    agent_prim_path = "/World/Agent"
    agent_usd_path = "/Isaac/Robots/NVIDIA/Kaya/kaya.usd"
    agent_camera_usd_path = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    agent_camera_prim_path = "/World/Agent/base_link/realsense_d455"
    # For simulation, the camera prim path and asset prim path may be different. So we need to set them separately.
    agent_perspective_cam_prim_path = ""
    agent_camera_resolution = (640, 480)
    agent_camera_fps = 30
    require_external_camera: bool = True

    custom_robot: bool = False

    def __check_robot(self):
        if not self.require_external_camera and \
            (self.agent_camera_usd_path is None or \
            not self.agent_perspective_cam_prim_path):
            raise ValueError("External camera is not required, but agent_camera_usd_path is provided. \n"
                             "Please set require_external_camera to True or set agent_camera_usd_path to None.")

    def change_agent_camera_resolution(self, resolution: tuple):
        self.agent_camera_resolution = resolution   
    
    def set_limo_config(self):
        self.robot_name = "limo"
        self.change_agent_camera_resolution((640, 480))
        self.maxLinearSpeed = 1e6
        # wheel_base for DifferentialController = track width (left-right wheel center distance),
        # NOT the front-rear axle distance.
        # Real LIMO track width ≈ 0.172 m. Verify against custom USD before running.
        self.wheelDistance = 0.175   # [m] track width (좌우 바퀴 중심 간격)
        self.wheelRadius = 0.045
        self.front_jointNames = ["front_left_wheel", "front_right_wheel"]
        self.rear_jointNames = ["rear_left_wheel", "rear_right_wheel"]
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        _base_limo_usd_path = os.path.join(_base_dir, "../custom_usd/WegoLimo/Limo/limo_diff_thin.usd")
        self.agent_usd_path = _base_limo_usd_path
        self.agent_prim_path = "/World/Agent"
        self.agent_camera_prim_path = "/World/Agent/depth_link/limo_camera"
        self.agent_perspective_cam_prim_path = "/World/Agent/depth_link/limo_camera"
        self.require_external_camera = False
        self.custom_robot = True
        self.__check_robot()
    
    def set_kaya_config(self, perspective_cam_prim_path="RSD455/Camera_OmniVision_OV9782_Color"):
        self.robot_name = "kaya"
        self.change_agent_camera_resolution((640, 480))
        self.kaya_wheels = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
        self.agent_perspective_cam_prim_path = perspective_cam_prim_path
        self.__check_robot()