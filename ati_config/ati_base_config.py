from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ATIBaseConfig:
    """Configuration for rendering test."""
    exp_name: str
    # physical FPS must be larger than rendering step to ensure motion blur effect.
    physics_dt: float = 1. / 60 
    # For accuracy, the physical FPS should be "num sub samepls" X "rendering FPS". 
    # However, we set just 2 times for shortcut. 
    rendering_dt: float = 1. / 60
    stage_units_in_meters: float = 1.0
    rendering_mode: str = "pathtracing"  # "realtime" or "pathtracing"
    capture_motion_blur: bool = True     # Turn off this parameter is not recommended due to Sim2Real Gap.
    pt_spp: int = 128  
    num_subsamples: int = 16  # Samples per pixel for path tracing
    
    scene_usd: str = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    agent_prim_path = "/World/Agent"
    agent_usd_path = "/Isaac/Robots/NVIDIA/Kaya/kaya.usd"
    agent_camera_usd_path = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    agent_camera_prim_path = "/World/Agent/base_link/realsense_d455"
    agent_perspective_cam_prim_path = ""
    agent_camera_resolution = (640, 480)
    spawn_random_objs: bool = True
    
    single_object_usd_paths: List[Tuple[str, int]] = field(default_factory=list)
    props_object_usd_paths: List[Tuple[str, int]] = field(default_factory=list)
    external_cameras: List[Tuple[str, str, np.ndarray, np.ndarray]] = field(default_factory=list) # camera name, prim path, position, target position
    
    def __init__(
        self, 
        name,
        agent_perspective_cam_prim_path
    ):
        self.exp_name = name
        # No effect for Camera objects. We control camera FPS with rendering frequency (rendering_dt) in the scene.
        self.agent_camera_fps = 60
        self.rendering_targets = [
            "depth",
            "2d_bounding_box",
            "motion_vectors"
        ]
        self.single_object_usd_paths = [
            ("/Isaac/Props/Dolly/dolly.usd", 5),
        ]
        self.props_object_usd_paths = [
            ("/Isaac/Props/YCB/Axis_Aligned_Physics", 30),
        ]
        self.external_cameras: List[Tuple[str, str, np.ndarray, np.ndarray]] = [ # camera name, prim path, position, target position
            (
                "overhead", 
                "/World/OverheadCam",
                np.array([0.0, -2.0, 9.0]),
                np.array([0.0, 0.0, 0.0])
            ),
        ]
        
        # For simulation, the camera prim path and asset prim path may be different. So we need to set them separately.
        self.agent_perspective_cam_prim_path = agent_perspective_cam_prim_path
    
    def set_agent_camera_usd_path(self, camera_usd_path):
        self.agent_camera_usd_path = camera_usd_path
    
    def set_agent_prim_path(self, prim_path):
        self.agent_prim_path = prim_path
        self.agent_camera_prim_path = f"{prim_path}/base_link/camera"
    
    def set_rendering_targets(self, targets):
        self.rendering_targets = targets    
        
    def set_camera_resolution(self, resolution: tuple):
        self.agent_camera_resolution = resolution
    
    def set_camera_fps(self, fps):
        self.agent_camera_fps = fps