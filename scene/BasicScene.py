from abc import *
from typing import Union, Dict
from isaacsim.core.api import World
from isaacsim.simulation_app import SimulationApp

# --- All imports AFTER SimulationApp init ---
import carb
import carb.settings
import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdLux, PhysxSchema
import omni.replicator.core as rep

## RGB-D Sensor
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
import omni.timeline
import cv2
import carb
import numpy as np
import random
import os

from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view

from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.sensors.camera import Camera

from ati_config import ATIBaseConfig


class BaseScene(metaclass=ABCMeta):
    
    RENDERING_ANNOTATOR_TYPES = {
        "depth": "distance_to_image_plane",
        "instance_segmentation": "semantic_segmentation",
        "2d_bounding_box": "bounding_box_2d_tight",
        "3d_bounding_box": "bounding_box_3d",
        "motion_vectors": "motion_vectors"
    }
    
    def __init__(
        self,
        simulation_app: SimulationApp,
        config: ATIBaseConfig,
        physics_dt: float = 1 / 60,
        rendering_dt: float = 1 / 60, 
        stage_units_in_meters=1.0,
        seed: int = 20260318
    ):
        self.world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters
        )
        
        self.sim_app = simulation_app
        self.config = config
        self.robot_config = config.robot_config
        self.rendering_mode = config.rendering_mode
        self.capture_motion_blur = config.capture_motion_blur
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.cameras: Dict[str, Camera] = {}
        self.robot_agent = None
        self.agent = None
        self.n_rendered_frames = 0
        self.assets_root_path = None
        self._timeline = None
        self._num_frame_steps = 0
        self._num_steps = 0
        self.__reset_needed = False
        self.__warmup_steps = 60
        
        self.__capture_on_play = True
        
        # Control simulation timeline for rendering control based on shutter time parameter.
        # You can configure these parameters by configuration file (YAML).
        self.scene_usd = config.scene_usd
        self.agent_prim_path = config.agent_prim_path
        self.agent_usd_path = config.agent_usd_path
        self.agent_camera_usd_path = config.agent_camera_usd_path
        self.agent_camera_prim_path = config.agent_camera_prim_path
        self.agent_perspective_cam_prim_path = config.agent_perspective_cam_prim_path
        self.agent_camera_fps = config.agent_camera_fps
        self.agent_camera_resolution = config.agent_camera_resolution
        self.rendering_targets = config.rendering_targets
        self.spawn_random_objs = config.spawn_random_objs
        self.external_cameras = config.external_cameras
        self.single_object_usd_paths = config.single_object_usd_paths
        self.props_object_usd_paths = config.props_object_usd_paths
        self.wheel_indices = None
        
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Scene will be built simultaneously when the class is initialized, so no need to call build_scene() separately in main_rendering_test.py
        self.build_scene()
        
    
    def __rendering_settings(self):
        # ── Disable auto-exposure so camera exposure attributes take effect ──
        carb.settings.get_settings().set_bool("/rtx/post/histogram/enabled", False)        # disable auto-exposure
        carb.settings.get_settings().set_int("/rtx/post/tonemap/op", 1)                    # Linear: applies exposure, no tone curve
        carb.settings.get_settings().set_bool("/rtx/post/tonemap/enableSrgbToGamma", False)
        
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", self.__capture_on_play) # True
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", self.capture_motion_blur)
        
        # Make sure fixed time stepping is set (the timeline will be advanced with the same delta time)
        carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
        
        if self.rendering_mode == "pathtracing":
            print(f"[RenderingSettings] Setting PathTracing render mode settings")
            carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
            # (int): Total number of samples for each rendered pixel, per frame.
            carb.settings.get_settings().set("/rtx/pathtracing/spp", self.config.pt_spp)
            # (int): Maximum number of samples to accumulate per pixel. 
            # When this count is reached the rendering stops until a scene or setting change is detected, restarting the rendering process. 
            # Set to 0 to remove this limit.
            carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", self.config.pt_spp)
            carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)
            # Number of sub samples to render if in PathTracing render mode and motion blur is enabled.
            carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", self.config.num_subsamples)
        else:
            print(f"[RenderingSettings] Setting RayTracedLighting render mode motion blur settings")
            carb.settings.get_settings().set("/rtx/rendermode", "RayTracedLighting")
            # 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
            carb.settings.get_settings().set("/rtx/post/aa/op", 2)
            # (float): The fraction of the largest screen dimension to use as the maximum motion blur diameter.
            carb.settings.get_settings().set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.02)
            # (float): Exposure time fraction in frames (1.0 = one frame duration) to sample.
            carb.settings.get_settings().set("/rtx/post/motionblur/exposureFraction", 1.0)
            # (int): Number of samples to use in the filter. A higher number improves quality at the cost of performance.
            carb.settings.get_settings().set("/rtx/post/motionblur/numSamples", 8)
        
        physx_scene = None
        for prim in self.world.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                break
        if physx_scene is None:
            print(f"[MotionBlur] Creating a new PhysicsScene")
            physics_scene = UsdPhysics.Scene.Define(self.world.stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(self.world.stage.GetPrimAtPath("/PhysicsScene"))
            # Check the target physics depending on the custom delta time and the render mode
        
        target_physics_fps = 1 / self.physics_dt
        if self.config.rendering_mode == "pathtracing":
            target_physics_fps *= self.config.num_subsamples
            self.physics_dt /= self.config.num_subsamples
            # Check if the physics FPS needs to be increased to match the custom delta time
        orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
        if target_physics_fps > orig_physics_fps:
            print(f"[MotionBlur] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
            physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)
        
    def reset(self):
        self.world.reset()
        self._timeline = None
        self.build_scene()
    
    @property
    def get_simulation_current_time(self):
        if self._timeline is not None:
            return self._timeline.get_current_time()
        else:
            return -1
    
    """
    Build the scene by configuration file. 
    Given YAML file must contain the following fields:
        - Robot Agent: Robot Type(Wheeled, Holonomic, etc...), Robot USD, Initial Pose, etc..
        - Initial Sensor State: Shutter Time, ISO, Aperture, etc.
        - Initial Scene Configuration: Lighting, Scene USD, etc..
        - objects: list of objects to be added to the scene, number of object, minimum/maximum distance between robot agent.
        - Rendering Targets: GT Depth, GT Segmentation, GT Object Pose, GT 2D Bounding Box, etc..
    """
    def build_scene(self):
        self._load_scene_essentials(self.scene_usd)
        if self.spawn_random_objs:
            self.spawn_random_objects(min_dist_from_agent=4)
        
        self.__rendering_settings()
        # ── Physics must be initialized (world.reset) BEFORE any tensor API use ──
        # Standard Isaac Sim pattern: add prims → world.reset() → warmup steps → play
        self.world.reset()
        print("[Main] Waiting for assets to load...")
        for _ in range(self.__warmup_steps):
            self.sim_app.update()
        print("[Main] Assets settled & Synthetic Data Generation Ready.")
        
        self._attach_annotators_to_camera("agent_camera")
        print("[Timeline] Timeline setup for rendering control")
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.set_current_time(0.0)
        self._timeline.play()
        self._timeline.commit()
        self._previous_time = 0.0
        self._elapsed_time = 0.0
        return     
    
    @property
    def num_steps(self):
        return self._num_frame_steps
    
    """
    Most important function. -> Reducing Sim2Real gap by step time and rendering time control. 
    We should render image by shutter time parameter. 
    For example, if the shutter time is 0.001 second, we should render image every 0.001 second (in the simulation time). 
    But we can step physics with larger time step to have more accurate simulation.  
    For example, we can step physics with 0.01 second and render image every 0.001 second.  
        -> That means we should render image 10 times in one physics step. 
    """
    def step(self, render=True):
        """
        Output Parameters:
            - RGB Images: NxHxWx3 numpy array
            - (Optional) Ground Truth Depth Images: NxHxW numpy array
            - (Optional) Ground Truth Segmentation Images: NxHxW numpy array
            - (Optional) Ground Truth 2D Object Bounding Box: Nx4 numpy array (x_top, y_top, width, height)
            - (Optional) Ground Truth 3D Object Bounding Box: Nx8 numpy array (x, y, z, width, height, depth, roll, pitch, yaw)
        OR 
            - Empty Dictionary: (No rendered output. The sensor did not capture during the timestep, or the rendering is disabled.)
        """
        if self.rendering_mode == "pathtracing":
            rendered_data = self.render_time_control(render=render)
        elif self.rendering_mode == "realtime":
            self.world.step(render=render)
            rendered_data = {
                "rgb": self.agent_camera.get_rgb(),
                **self.agent_camera.get_current_frame()
            }
        
        if self.world.is_stopped() and not self.__reset_needed:
            self.__reset_needed = True
        elif self.world.is_playing():
            if self.__reset_needed:
                self.reset()
                self.__reset_needed = False
        
        # # Rendering Agent Camera Only for now. Need to be generalized for multiple cameras.
        # rendered_data = {
        #     "rgb": self.agent_camera.get_rgb(),
        #     **self.agent_camera.get_current_frame()
        # }
        if rendered_data and \
            rendered_data.get("rgb", None) is not None and \
            rendered_data["rgb"].size != 0:
            self._num_frame_steps += 1
        return rendered_data 
    
    @abstractmethod
    def robot_control(self, control_parameters: dict):
        """
        Control the robot agent based on the control_parameters dictionary. 
        For example, if the control_parameters contains linear velocity and angular velocity, we should update the robot agent's velocity.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sensor_control(self, sensor_name: str, control_parameters: dict):
        """
        Control the sensor parameters based on the control_parameters dictionary. 
        For example, if the control_parameters contains shutter time, we should update the shutter time of the sensor.
        """
        raise NotImplementedError
    
    @abstractmethod
    def spawn_random_objects(self, num_objects: int, object_usd_paths: list, min_dist_from_agent=4):
        """Randomly spawn objects in the scene based on the number of objects and object USD paths."""
        raise NotImplementedError 
    
    
    def control_light_intensity(self, intensity):
        sun_prim = self.world.stage.GetPrimAtPath("/World/ExtraSun")
        sun = UsdLux.DistantLight(sun_prim)
        sun.GetIntensityAttr().Set(intensity)
    
    
    def render_time_control(self, render=True):
        """
        Enforcing rendering frequency based on camera FPS.
        In path tracing mode with motion blur, rep.orchestrator.step() internally
        handles all sub-frame stepping (num_subsamples physics sub-steps per call).
        Each call advances the simulation by exactly 1/camera_fps seconds and
        produces one composited camera frame with motion blur.
        """
        self.world.step(render=render)
        current_time = self._timeline.get_current_time()
        print(f"[RenderControl] Current Time: {current_time:.4f} seconds, Elapsed Time: {self._elapsed_time:.4f} seconds")
        delta_time = current_time - self._previous_time
        self._elapsed_time += delta_time
        
        agent_dt = 1.0 / self.agent_camera_fps
        if self._elapsed_time >= agent_dt - 1e-9:
            self._elapsed_time -= agent_dt
            rendered_data = {
                "rgb": self.agent_camera.get_rgb(),
                **self.agent_camera.get_current_frame()
            }
        else: 
            rendered_data = {}
        
        self._num_steps += 1
        self._previous_time = current_time
        return rendered_data
    
    def _define_robot_agent(
        self,
        robot_type: str,
        robot_name: str,
        robot_usd_path: str,
        initial_position: np.array,
        initial_orientation: np.array,
        **kwargs
    ):
        if robot_type == "kaya":
            robot_agent = self.world.scene.add(
                WheeledRobot(
                    prim_path=self.agent_prim_path,
                    name=robot_name,
                    wheel_dof_names=kwargs.get( # Default setting is Kaya.
                        "wheel_dof_names", 
                        ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
                    ),
                    create_robot=True,
                    usd_path=robot_usd_path,
                    position=initial_position,
                    orientation=initial_orientation,
                )
            )
        elif robot_type == "limo":
            add_reference_to_stage(usd_path=robot_usd_path, prim_path=self.agent_prim_path)
            robot_agent = SingleArticulation(
                prim_path=self.agent_prim_path,
                name=robot_name,
                position=initial_position,
                orientation=initial_orientation,
            )
            self.world.scene.add(robot_agent)
            wheel_joint_names = kwargs.get("wheel_joint_names", [])
            self.wheel_indices = [robot_agent.get_dof_index(name) for name in wheel_joint_names]
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")
        return robot_agent
    
    ## Always render agent in (0, 0)
    def _load_scene_essentials(self, scene_usd_path: str):
        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
        print(f"[Assets] Root: {self.assets_root_path}")
        
        try:
            add_reference_to_stage(usd_path=self.assets_root_path + scene_usd_path, prim_path="/World/Environment")
        except Exception as e:
            carb.log_error(f"Failed to load scene USD: {scene_usd_path}. Error: {e}")
            raise e
        
        # Robot Definition Part: This need to be generalized into various robot class.
        robot_usd_path = self.agent_usd_path if self.robot_config.custom_robot else self.assets_root_path + self.agent_usd_path
        kwargs = {}
        if self.robot_config.robot_name == "kaya":
            kwargs["wheel_dof_names"] = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
        elif self.robot_config.robot_name == "limo":
            kwargs["wheel_joint_names"] = [*self.robot_config.front_jointNames, *self.robot_config.rear_jointNames]
        self.agent = self._define_robot_agent(
            robot_type=self.robot_config.robot_name,
            robot_name="my_agent",
            robot_usd_path=robot_usd_path,
            initial_position=np.array([0.0, 0.0, 0.02]), # -3.0, -3.0
            initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            **kwargs
        )
        
        self.agent_camera = self._define_agent_camera(
            self.agent_camera_prim_path, 
            self.assets_root_path + self.agent_camera_usd_path,
            self.config.require_external_camera
        )
        
        # --- Create overhead camera ---
        for e_cam_props in self.external_cameras:
            cam_name, cam_prim_path, cam_position, cam_t_position = e_cam_props
            self._define_external_camera(
                cam_name, 
                cam_prim_path,
                cam_position,
                camera_usd_path=None
            )
            self._modify_external_camera(cam_prim_path, cam_position, cam_t_position)
        
        # --- Extra sun light ---: Things to modify for the lighting variations
        sun = UsdLux.DistantLight.Define(self.world.stage, "/World/ExtraSun")
        sun.CreateIntensityAttr(3000) # Need to check the unit of this parameter and the range of it.
        sun.CreateAngleAttr(1.0)
        sun_xf = UsdGeom.Xformable(sun.GetPrim())
        sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 0))
        
        # --- Verify ---
        for path in ["/World/Environment", "/World/Agent", "/World/OverheadCam", self.agent_camera_prim_path]:
            prim = self.world.stage.GetPrimAtPath(path)
            print(f"[Verify] {path}: {'OK' if prim.IsValid() else 'MISSING'}")
            if not prim.IsValid():
                print("[Error] There is something going wrong....")
        print("[Scene] Build complete")
        
    
    def _define_external_camera(
        self, cam_name, camera_prim_path: str, camera_position: np.array,
        camera_usd_path: Union[str, None] = None
    ):
        if camera_usd_path is None:
            # Create a default pinhole camera if no USD path is provided
            camera = UsdGeom.Camera.Define(self.world.stage, camera_prim_path)
            camera.CreateFocalLengthAttr(18.0) 
            camera.CreateFStopAttr(0.0)
            camera.CreateFocusDistanceAttr(3.0)
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateVerticalApertureAttr(15.2908)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 10000))
            camera.GetPrim().CreateAttribute("shutter:open", Sdf.ValueTypeNames.Double).Set(0.0)
            camera.GetPrim().CreateAttribute("shutter:close", Sdf.ValueTypeNames.Double).Set(0.5)
        else:
            prim_utils.create_prim(
                camera_prim_path, "Xform",
                translation=(0.1, 0.0, 0.1),
            )  # reference를 걸 컨테이너
            camera = add_reference_to_stage(usd_path=camera_usd_path, prim_path=camera_prim_path)
        self.cameras[cam_name] = Camera(
            prim_path=camera_prim_path,
            frequency=self.agent_camera_fps,
            resolution=self.agent_camera_resolution,
            position=camera_position, # external camera의 position을 그대로 사용
        )
        self.cameras[cam_name].initialize(attach_rgb_annotator=True)
        return camera
    
    def _modify_external_camera(
        self, 
        camera_prim_path, 
        position: np.array, 
        t_position: np.array
    ):
        self.__force_viewport_camera(camera_prim_path, retries=10)
        set_camera_view(
            eye=position,
            target=t_position,
            camera_prim_path=camera_prim_path,
        )
        print(f"[Camera] Overhead at {position} looking at {t_position}")
        # Force camera again after positioning (belt and suspenders)
        self.__force_viewport_camera(camera_prim_path, retries=5)
    
    
    def _define_agent_camera(
        self,
        camera_prim_path: str, 
        camera_usd_path: Union[str, None] = None,
        require_external_camera: bool = True
    ):
        if require_external_camera:
            # Define camera prim for agent perspective view. This camera will be attached to the robot agent and will move together with the agent.
            if camera_usd_path is None:
                # Create a default pinhole camera if no USD path is provided
                agent_camera = UsdGeom.Camera.Define(self.world.stage, camera_prim_path)
                agent_camera.CreateFocalLengthAttr(35.0)
                agent_camera.CreateHorizontalApertureAttr(32.0)
                agent_camera.CreateVerticalApertureAttr(18.0)
            else:
                prim_utils.create_prim(
                    camera_prim_path, "Xform",
                    translation=(0.1, 0.0, 0.1),
                )  # reference를 걸 컨테이너
                agent_camera = add_reference_to_stage(usd_path=camera_usd_path, prim_path=camera_prim_path)
            
            # For ease of controlling the agent camera parameters, we used native camera class in Isaac Sim.
            cam_real_prim_path = f"{camera_prim_path}/{self.agent_perspective_cam_prim_path}"
            cam_rel_position = np.array([0.1, 0.0, 0.1])
        else:
            # This case, the agent USD natively supports RGB(-D) camera on the platform.
            cam_real_prim_path = camera_prim_path
            cam_rel_position = None

        self.cameras["agent_camera"] = Camera(
            prim_path=cam_real_prim_path,
            frequency=self.agent_camera_fps,
            resolution=self.agent_camera_resolution,
            position=cam_rel_position,
            annotator_device="cpu"
        )
        self.cameras["agent_camera"].initialize(attach_rgb_annotator=True) # 
        
        ## Set Exposure API
        cam_prim = self.world.stage.GetPrimAtPath(cam_real_prim_path)
        cam_prim.ApplyAPI("OmniRtxCameraExposureAPI_1")
        
        return self.cameras["agent_camera"]
    
    def _attach_annotators_to_camera(self, camera_name:str):
        camera = self.cameras[camera_name]
        camera.add_rgb_to_frame()
        for target in self.rendering_targets:
            annotator_type = self.RENDERING_ANNOTATOR_TYPES.get(target)
            if annotator_type is None:
                print(f"[Warning] Unsupported rendering target: {target}. Skipping annotator attachment.")
                continue
            camera.attach_annotator(annotator_type)
    
    def _object_spawn_randomization(self, obj_prim, agent_pos, min_dist_from_agent=4):
        """
        Randomize the number of objects, type of objects, and initial pose of objects based on the configuration file.
        """
        if not obj_prim.GetAttribute("xformOp:translate"):
            UsdGeom.Xformable(obj_prim).AddTranslateOp()
        if not obj_prim.GetAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(obj_prim).AddRotateXYZOp()
        for _ in range(100):
            x, y = random.uniform(-6, 6), random.uniform(-6, 6)
            dist = (Gf.Vec2f(x, y) - Gf.Vec2f(agent_pos[0], agent_pos[1])).GetLength()
            if dist > min_dist_from_agent:
                obj_prim.GetAttribute("xformOp:translate").Set((x, y, 0))
                break
        obj_prim.GetAttribute("xformOp:rotateXYZ").Set((0, 0, random.uniform(-180, 180)))
    
    def __force_viewport_camera(self, camera_path, retries=5):
        """Set viewport camera with retries — Leatherback's Camera_Chase can override."""
        try:
            import omni.kit.viewport.utility as vu
            viewport = vu.get_active_viewport()
            if not viewport:
                print("[Viewport] No active viewport")
                return

            for i in range(retries):
                viewport.set_active_camera(camera_path)
                self.sim_app.update()

            current = viewport.get_active_camera()
            print(f"[Viewport] Set camera → {camera_path} (active: {current})")
        except Exception as e:
            print(f"[Viewport] Error: {e}")
    



# # ── Agent prim에 붙어 있는 Camera prim path 자동 탐색 ─────────────────────
#         agent_prim = self.world.stage.GetPrimAtPath(self.agent_prim_path)
#         discovered_cam_path = None
#         if agent_prim.IsValid():
#             for prim in Usd.PrimRange(agent_prim):
#                 if prim.IsA(UsdGeom.Camera):
#                     discovered_cam_path = str(prim.GetPath())
#                     print(f"[Camera] Agent prim 하위에서 Camera prim 발견: {discovered_cam_path}")
#                     break
#         if discovered_cam_path is None:
#             print(f"[Camera] Agent prim({self.agent_prim_path}) 하위에 Camera prim 없음. 설정값 사용: {cam_real_prim_path}")
#         # ─────────────────────────────────────────────────────────────────────────