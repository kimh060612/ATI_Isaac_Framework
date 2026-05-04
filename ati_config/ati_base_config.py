from typing import List, Tuple
from .ati_robot_config import ATIBaseRobotConfig
import numpy as np
import os

class ATIBaseConfig:
    """Configuration for rendering test."""
    exp_name: str
    # Base physics dt requested by the experiment. The scene may refine this into
    # a smaller internal step so camera_fps-aligned shutter integration has
    # enough temporal samples.
    physics_dt: float = 1. / 30 
    # Maximum rendering dt requested by the experiment. The scene will clamp this
    # against camera_fps and motion-blur sampling needs.
    rendering_dt: float = 1. / 30 
    stage_units_in_meters: float = 1.0
    rendering_mode: str = "pathtracing"  # "realtime" or "pathtracing" or "autoexposure"
    capture_motion_blur: bool = True
    pt_spp: int = 128  
    enable_mb_adaptive_sampling: bool = True  # Whether to enable adaptive sampling for pathtracing
    num_subsamples: int = 32  # Maximum motion-blur samples rendered inside one shutter interval
    min_motion_blur_subsamples: int = 2  # Minimum samples used for any non-zero shutter interval
    
    
    scene_usd: str = f"{os.getcwd()}/custom_usd/ATI_MDE_Scene_002/World0.usd"
    # "/Isaac/Environments/Simple_Room/simple_room.usd"
    # "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    # "/Isaac/Environments/Grid/gridroom_curved.usd"
    # "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    # "/Isaac/Environments/Grid/default_environment.usd"
    # "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd" -> Warehouse full
    agent_prim_path = "/World/Agent"
    agent_usd_path = "/Isaac/Robots/NVIDIA/Kaya/kaya.usd"
    agent_camera_usd_path = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    agent_camera_prim_path = "/World/Agent/base_link/realsense_d455"
    agent_perspective_cam_prim_path = ""
    agent_camera_resolution = (640, 480)
    agent_origin_position = (0.0, 0.0, -0.7) # x, y, z
    require_external_camera: bool = True
    spawn_random_objs: bool = True
    min_distance_from_agent: float = 3.0 # Minimum distance from the agent for randomly spawned objects
    robot_config: ATIBaseRobotConfig
    camera_controller: str = "exposure_iso_controller"
    
    single_object_usd_paths: List[Tuple[str, int]] = [] # field(default_factory=list)
    props_object_usd_paths: List[Tuple[str, int]] = [] # field(default_factory=list)
    external_cameras: List[Tuple[str, str, np.ndarray, np.ndarray]] = [] # field(default_factory=list) # camera name, prim path, position, target position
    
    def __init__(
        self, name,
        robot_config: ATIBaseRobotConfig,
    ):
        self.exp_name = name
        # BaseScene emits one synthetic frame every 1 / agent_camera_fps seconds,
        # independent of the finer internal simulation step it may use.
        self.rendering_targets = [
            "depth",
            "2d_bounding_box",
            "motion_vectors"
        ]
        self.single_object_usd_paths = [
            ("/Isaac/Props/Dolly/dolly.usd", 0),
        ]
        self.props_object_usd_paths = [
            ("/Isaac/Props/YCB/Axis_Aligned_Physics", 15),
        ]
        self.external_cameras: List[Tuple[str, str, np.ndarray, np.ndarray]] = [ # camera name, prim path, position, target position
            (
                "overhead", 
                "/World/OverheadCam",
                np.array([0.0, -2.0, 3.0]),
                np.array([0.0, 0.0, 0.0])
            ),
        ]
        
        self.robot_config = robot_config
        self.agent_camera_fps = robot_config.agent_camera_fps
        self.agent_camera_resolution = robot_config.agent_camera_resolution
        self.agent_usd_path = robot_config.agent_usd_path
        self.agent_prim_path = robot_config.agent_prim_path
        self.agent_camera_usd_path = robot_config.agent_camera_usd_path
        self.agent_camera_prim_path = robot_config.agent_camera_prim_path
        self.agent_perspective_cam_prim_path = robot_config.agent_perspective_cam_prim_path
        self.require_external_camera = robot_config.require_external_camera
        
        # enforce physics_dt and rendering_dt to be no larger than 1 / agent_camera_fps for correct motion blur sampling and camera-aligned stepping.
        self.physics_dt = self.rendering_dt = 1. / self.agent_camera_fps
        
    
    def set_agent_camera_usd_path(self, camera_usd_path):
        self.agent_camera_usd_path = camera_usd_path
    
    def set_rendering_targets(self, targets):
        self.rendering_targets = targets    
        
    def set_camera_resolution(self, resolution: tuple):
        self.agent_camera_resolution = resolution
    
    def set_camera_fps(self, fps):
        self.agent_camera_fps = fps
    
    def set_rendering_mode(self, mode):
        if mode not in ["realtime", "pathtracing"]:
            raise ValueError(
                f"Unsupported rendering mode: {mode}. "
                "Supported modes are 'realtime' and 'pathtracing'."
            )
        self.rendering_mode = mode
    
    def set_pathtracing_param(self, spp, num_subsamples, min_motion_blur_subsamples=None):
        self.pt_spp = spp
        self.num_subsamples = num_subsamples
        if min_motion_blur_subsamples is not None:
            self.min_motion_blur_subsamples = min_motion_blur_subsamples
    
    def set_random_obj_spawn(self, spawn_random_objs: bool):
        self.spawn_random_objs = spawn_random_objs
