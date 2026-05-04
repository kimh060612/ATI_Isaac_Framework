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

from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import IMUSensor

from ati_config import ATIBaseConfig, DEBUG
from sensor_control import BaseSensorController, ShutterExposureSensorController
from ati_utils.iso_noise_processing import add_d455_noise
from tqdm import tqdm

class BaseScene(metaclass=ABCMeta):
    SENSOR_CONTROLLER_CLS = {
        "exposure_iso_controller": ShutterExposureSensorController,
        "name_of_controller": None, # Replace with actual example controller class
    }
    
    RENDERING_ANNOTATOR_TYPES = {
        # refere to: https://docs.omniverse.nvidia.com/py/replicator/latest/source/extensions/omni.replicator.core/docs/API.html#default-annotators
        "depth": "distance_to_image_plane", # "distance_to_image_plane", "distance_to_camera"
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
        self.sim_app = simulation_app
        self.config = config
        self.robot_config = config.robot_config
        self.rendering_mode = config.rendering_mode
        self.capture_motion_blur = config.capture_motion_blur
        self.agent_camera_fps = config.agent_camera_fps
        self.sensor_control_algorithm = config.camera_controller
        
        if self.agent_camera_fps <= 0:
            raise ValueError(f"agent_camera_fps must be positive, got {self.agent_camera_fps}.")
        self._camera_frame_dt = 1.0 / float(self.agent_camera_fps)
        self._base_simulation_dt, self._sim_steps_per_camera_frame = self._compute_internal_simulation_dt(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
        )
        self._simulation_dt = self._base_simulation_dt
        self._internal_render_fps = 1.0 / self._simulation_dt
        
        self.world = World(
            physics_dt=self._simulation_dt,
            rendering_dt=self._simulation_dt,
            stage_units_in_meters=stage_units_in_meters
        )

        self.requested_physics_dt = physics_dt
        self.requested_rendering_dt = rendering_dt
        self.physics_dt = self._simulation_dt
        self.rendering_dt = self._simulation_dt
        self.motion_blur_physics_dt = self._simulation_dt
        
        self.cameras: Dict[str, Camera] = {}
        self.sensor_controllers: Dict[str, BaseSensorController] = {}
        self.robot_agent = None
        self.agent = None
        self.n_rendered_frames = 0
        self.assets_root_path = None
        self._timeline = None
        self._num_frame_steps = 0
        self._num_steps = 0
        self.__reset_needed = False
        self.__warmup_steps = 60
        self._camera_capture_start_time = 0.0
        self._pt_external_frame_counter = 0
        
        self.__capture_on_play = True
        
        # Control simulation timeline for rendering control based on shutter time parameter.
        # You can configure these parameters by configuration file (YAML).
        self.scene_usd = config.scene_usd
        self.agent_prim_path = config.agent_prim_path
        self.agent_usd_path = config.agent_usd_path
        self.agent_camera_usd_path = config.agent_camera_usd_path
        self.agent_camera_prim_path = config.agent_camera_prim_path
        self.agent_perspective_cam_prim_path = config.agent_perspective_cam_prim_path
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
        
    def _compute_internal_simulation_dt(self, physics_dt: float, rendering_dt: float):
        candidate_dt = min(float(physics_dt), float(rendering_dt), self._camera_frame_dt)
        if self.capture_motion_blur:
            max_motion_blur_samples = max(int(self.config.num_subsamples), 1)
            candidate_dt = min(candidate_dt, self._camera_frame_dt / float(max_motion_blur_samples))
        if candidate_dt <= 0.0:
            raise ValueError(
                f"Computed internal simulation dt must be positive, got {candidate_dt}."
            )
        steps_per_camera_frame = max(1, int(np.ceil(self._camera_frame_dt / candidate_dt - 1e-9)))
        simulation_dt = self._camera_frame_dt / float(steps_per_camera_frame)
        print(
            "[Timing] "
            f"camera_fps={self.agent_camera_fps}, "
            f"camera_dt={self._camera_frame_dt:.6f}, "
            f"internal_sim_dt={simulation_dt:.6f}, "
            f"steps_per_camera_frame={steps_per_camera_frame}"
        )
        return simulation_dt, steps_per_camera_frame

    def _get_physx_scene_api(self):
        for prim in self.world.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                return PhysxSchema.PhysxSceneAPI.Apply(prim)
        return None    
    
    def __rendering_settings(self):
        if self.rendering_mode == "autoexposure":
            print(f"[RenderingSettings] This configuration function is not for auto-exposure")
            raise ValueError(f"rendering_mode must be 'realtime' or 'pathtracing' to use this rendering settings function, got '{self.rendering_mode}'.")
        
        # ── Disable auto-exposure so camera exposure attributes take effect ──
        carb.settings.get_settings().set_bool("/rtx/post/histogram/enabled", False)        # disable auto-exposure
        carb.settings.get_settings().set_int("/rtx/post/tonemap/op", 1)                    # Linear: applies exposure, no tone curve
        carb.settings.get_settings().set_bool("/rtx/post/tonemap/enableSrgbToGamma", False)
        
        carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", True)
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", self._internal_render_fps)
        carb.settings.get_settings().set("/app/stage/timeCodesPerSecond", float(self._internal_render_fps))
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", self.__capture_on_play) # True 
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", False)
        carb.settings.get_settings().set_bool("/rtx/post/motionblur/enabled", False)
        
        # Make sure fixed time stepping is set (the timeline will be advanced with the same delta time)
        carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
        
        if self.rendering_mode == "pathtracing":
            print(f"[RenderingSettings] Setting PathTracing render mode settings")
            carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
            # (int): Total number of samples for each rendered pixel, per frame.
            carb.settings.get_settings().set("/rtx/pathtracing/spp", self.config.pt_spp)
            # Keep accumulation capped to a single frame's spp so each internal
            # simulation step advances time once, matching the successful
            # manual-shutter experiment in test_mb_exp_time.py.
            carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", self.config.pt_spp)
            carb.settings.get_settings().set("/rtx/pathtracing/clampSpp", self.config.pt_spp)
            carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 1)
            carb.settings.get_settings().destroy_item("/omni/replicator/pathTracedMotionBlurSubSamples")
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
        
        physx_scene = self._get_physx_scene_api()
        if physx_scene is None:
            print(f"[MotionBlur] Creating a new PhysicsScene")
            UsdPhysics.Scene.Define(self.world.stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(self.world.stage.GetPrimAtPath("/PhysicsScene"))
            # Check the target physics depending on the custom delta time and the render mode
        
        target_physics_fps = 1 / self.physics_dt
        self.motion_blur_physics_dt = self.physics_dt
        orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
        if orig_physics_fps is None or abs(float(target_physics_fps) - float(orig_physics_fps)) > 1e-6:
            print(f"[MotionBlur] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
            physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)
        
    def reset(self):
        self.world.reset()
        self._timeline = None
        self.build_scene()

    @property
    def get_simulation_current_time(self):
        return float(self.world.current_time)
    
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
        self.cameras = {}
        self.sensor_controllers = {}
        self._load_scene_essentials(self.scene_usd)
        if self.spawn_random_objs:
            self.spawn_random_objects(min_dist_from_agent=self.config.min_distance_from_agent)
        
        # Deprecated Auto-Exposure Rendering Mode -> Implemented by separated AE sensor control policy.
        # if not self.rendering_mode == "autoexposure":
        self.__rendering_settings()
        # else:
        #     self.__ae_rendering_settings()
            
        # ── Physics must be initialized (world.reset) BEFORE any tensor API use ──
        # Standard Isaac Sim pattern: add prims → world.reset() → warmup steps → play
        self.world.reset()
        self._initialize_cameras()
        # self._initialize_sensor_controllers()
        print("[Main] Waiting for assets to load...")
        for _ in tqdm(range(self.__warmup_steps)):
            # if DEBUG: print(f"[Warmup] Step {i+1}/{self.__warmup_steps}"),
            self.sim_app.update()
        print("[Main] Assets settled & Synthetic Data Generation Ready.")
        
        self._attach_annotators_to_camera("agent_camera")
        for _ in tqdm(range(self.__warmup_steps)):
            self.sim_app.update()
        print("[Timeline] Timeline setup for rendering control")
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.set_current_time(0.0)
        self._timeline.play()
        self._timeline.commit()
        self._camera_capture_start_time = self.get_simulation_current_time
        return     
    
    @property
    def num_steps(self):
        return self._num_frame_steps
    
    """
    Most important function. -> Reducing Sim2Real gap by step time and rendering time control. 
    In this function, you should control the time of simulation world. 
    Currently, we control the time in criteria of camera FPS. For every 1 / camera_fps seconds, we will capture the synthetic data from the camera sensor.
    And, That is the one 'step' of the simulation.  
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
        rendered_data = self.__render_time_control(render=render)
        rendered_data["imu_sensor"] = self.agent_imu.get_current_frame()
        
        if self.world.is_stopped() and not self.__reset_needed:
            self.__reset_needed = True
        elif self.world.is_playing():
            if self.__reset_needed:
                self.reset()
                self.__reset_needed = False
        
        # # Rendering Agent Camera Only for now. Need to be generalized for multiple cameras.
        if not render:
            return rendered_data
        if self.__check_valid_synthetic_data(rendered_data):
            self._num_frame_steps += 1
        else:
            print("[Warning] No valid synthetic data captured at step {}.\nCheck the camera settings and rendering mode.".format(self.num_steps))
            # raise ValueError("No RGB data captured. Check the camera settings and rendering mode.")
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
    
    def __check_valid_synthetic_data(self, data: dict):
        if not data:
            return False
        if data.get("rgb", None) is None or data["rgb"].size == 0:
            return False
        if data.get("imu_sensor", None) is None:
            return False
        return True
    
    def __render_time_control(self, render=True):
        """
        Advance physics with the internal simulation dt, but only emit one camera
        frame per 1 / camera_fps seconds. RGB is integrated manually across the
        shutter interval so motion blur is controlled by shutter_time rather than
        Replicator's built-in subframe combiner.
        """
        frame_start_time = self._camera_capture_start_time
        frame_end_time = frame_start_time + self._camera_frame_dt

        if not render:
            while self.get_simulation_current_time + 1e-9 < frame_end_time:
                self.world.step(render=False)
                self._num_steps += 1
            self._camera_capture_start_time = frame_end_time
            return {}

        rendered_data = self._capture_camera_frame(
            sensor_name="agent_camera",
            frame_start_time=frame_start_time,
            frame_end_time=frame_end_time,
        )
        self._camera_capture_start_time = frame_end_time
        return rendered_data

    def _initialize_cameras(self):
        """
        Initialize all cameras in the scene and set up sensor controllers for them.
        """
        for camera in self.cameras.values():
            camera.set_dt(self._simulation_dt)
            camera.initialize(attach_rgb_annotator=True)
        
        for sensor_name, camera in self.cameras.items():
            self.sensor_controllers[sensor_name] = self.SENSOR_CONTROLLER_CLS[self.sensor_control_algorithm](
                sensor_name=sensor_name,
                camera=camera,
                camera_prim=self._get_camera_prim(sensor_name),
                camera_fps=self.agent_camera_fps,
            )

    def _get_camera_prim(self, sensor_name: str):
        if sensor_name not in self.cameras:
            raise ValueError(f"Camera '{sensor_name}' is not registered.")
        camera_prim_path = self.cameras[sensor_name].prim_path
        cam_prim = self.world.stage.GetPrimAtPath(camera_prim_path)
        if not cam_prim.IsValid():
            raise ValueError(f"Camera prim path {camera_prim_path} is not valid.")
        return cam_prim

    def _get_camera_shutter_window_seconds(self, sensor_name: str):
        return self.sensor_controllers[sensor_name].get_shutter_window_seconds(
            max_frame_duration=self._camera_frame_dt
        )

    def _advance_simulation_without_render(self, target_time: float, leave_step_for_render: bool = False):
        """
        Advance simulation time without rendering until the target time.
        If leave_step_for_render is True, stop one internal step before the target
        so the next render=True step lands on the requested sample time.
        """
        while True:
            current_time = self.get_simulation_current_time
            next_time = current_time + self._simulation_dt
            if leave_step_for_render:
                if next_time >= float(target_time) - 1e-9:
                    break
            elif current_time >= float(target_time) - 1e-9:
                break
            self.world.step(render=False)
            self._num_steps += 1

    def _average_rgb_samples(self, rgb_samples: list, fallback_rgb: np.ndarray):
        if rgb_samples:
            averaged_rgb = np.mean(np.stack(rgb_samples, axis=0), axis=0)
        elif fallback_rgb is not None and fallback_rgb.size != 0:
            averaged_rgb = np.asarray(fallback_rgb, dtype=np.float32)
        else:
            return None

        if fallback_rgb is not None and np.issubdtype(np.asarray(fallback_rgb).dtype, np.integer):
            info = np.iinfo(np.asarray(fallback_rgb).dtype)
            return np.clip(averaged_rgb, info.min, info.max).astype(np.asarray(fallback_rgb).dtype)
        return averaged_rgb

    def _extract_rgb_from_frame(self, frame: dict):
        if not frame:
            return None
        frame_rgb = frame.get("rgb", None)
        if frame_rgb is None:
            return None
        frame_rgb = np.asarray(frame_rgb)
        if frame_rgb.size == 0:
            return None
        if frame_rgb.ndim >= 3 and frame_rgb.shape[-1] >= 3:
            return frame_rgb[..., :3]
        return frame_rgb

    def _get_motion_blur_samples(self, shutter_duration: float, available_samples: int) -> int:
        if not self.config.enable_mb_adaptive_sampling:
            # Hueristic threshold for motion blur sampling: 
            # if shutter duration is very short, just do 1 sample to save computation. 
            # Otherwise, use the configured number of subsamples.
            if shutter_duration <= 0.003:
                return 1
            return max(1, min(int(self.config.num_subsamples), int(available_samples)))
        if shutter_duration <= 1e-9 or available_samples <= 0:
            return 1
        max_samples = max(int(self.config.num_subsamples), 1)
        if max_samples <= 1:
            return 1

        min_samples = max(1, min(int(self.config.min_motion_blur_subsamples), max_samples))
        available_samples = max(0, int(available_samples))
        scaled_samples = int(np.floor(max_samples * float(shutter_duration) / self._camera_frame_dt + 1e-9))
        if scaled_samples <= 0:
            return 1

        effective_samples = min(max_samples, available_samples, scaled_samples)
        if available_samples >= min_samples:
            effective_samples = max(min_samples, effective_samples)
            effective_samples = min(effective_samples, available_samples, max_samples)
        return max(1, effective_samples)

    def _build_render_sample_schedule(
        self,
        frame_start_time: float,
        frame_end_time: float,
        shutter_start_time: float,
        shutter_end_time: float,
        gt_reference_time: float,
    ):
        step_offsets = np.arange(1, self._sim_steps_per_camera_frame + 1, dtype=np.float64)
        candidate_times = frame_start_time + step_offsets * self._simulation_dt
        candidate_times[-1] = frame_end_time

        if not self.capture_motion_blur or shutter_end_time <= shutter_start_time + 1e-9:
            return [float(frame_end_time)]

        shutter_mask = np.logical_and(
            candidate_times >= shutter_start_time - 1e-9,
            candidate_times <= shutter_end_time + 1e-9,
        )
        shutter_candidate_times = candidate_times[shutter_mask]
        available_samples = int(shutter_candidate_times.size)
        requested_samples = self._get_motion_blur_samples(
            shutter_duration=shutter_end_time - shutter_start_time,
            available_samples=available_samples,
        )

        if available_samples <= 0:
            nearest_index = int(np.argmin(np.abs(candidate_times - gt_reference_time)))
            return [float(candidate_times[nearest_index])]

        if requested_samples >= available_samples:
            return [float(sample_time) for sample_time in shutter_candidate_times]

        sampled_indices = np.linspace(
            0,
            available_samples - 1,
            num=requested_samples,
            dtype=int,
        )
        return [float(shutter_candidate_times[index]) for index in sampled_indices]

    def _capture_camera_frame(self, sensor_name: str, frame_start_time: float, frame_end_time: float):
        """
        TODO: Apply stereo camera settings
        """
        
        camera = self.cameras[sensor_name]
        shutter_start_offset, shutter_end_offset = self._get_camera_shutter_window_seconds(sensor_name)
        shutter_duration = max(0.0, shutter_end_offset - shutter_start_offset)
        shutter_end_time = frame_end_time
        shutter_start_time = max(frame_start_time, frame_end_time - shutter_duration)
        
        rgb_samples = []
        latest_rgb = None
        latest_frame = camera.get_current_frame(clone=True)
        representative_frame = None
        last_consumed_rendering_time = (
            float(latest_frame.get("rendering_time"))
            if latest_frame and latest_frame.get("rendering_time", None) is not None
            else float("-inf")
        )
        gt_reference_time = frame_end_time
        render_schedule = self._build_render_sample_schedule(
            frame_start_time=frame_start_time,
            frame_end_time=frame_end_time,
            shutter_start_time=shutter_start_time,
            shutter_end_time=shutter_end_time,
            gt_reference_time=gt_reference_time,
        )
        representative_frame_distance = float("inf")

        if DEBUG: 
            print("[RenderControl] Capturing frame: frame_time={:.3f}, shutter_window=({:.3f}, {:.3f}), current_time={:.3f}, frame_end_time={:.3f}, scheduled_renders={}".format(
                frame_start_time,
                shutter_start_time,
                shutter_end_time,
                self.get_simulation_current_time,
                frame_end_time,
                len(render_schedule),
            ))

        for scheduled_render_time in render_schedule:
            self._advance_simulation_without_render(
                target_time=scheduled_render_time,
                leave_step_for_render=True,
            )
            self.world.step(render=True)
            self._num_steps += 1
            current_time = self.get_simulation_current_time
            if DEBUG:
                print(
                    f"[RenderControl] Render sample: "
                    f"current_time={current_time:.6f}, "
                    f"scheduled_time={scheduled_render_time:.6f}, "
                    f"simulation_dt={self._simulation_dt:.6f}"
                )
            latest_frame = camera.get_current_frame(clone=True)
            render_time = (
                float(latest_frame.get("rendering_time"))
                if latest_frame and latest_frame.get("rendering_time", None) is not None
                else float("-inf")
            )
            if render_time <= last_consumed_rendering_time + 1e-9:
                if DEBUG:
                    print(
                        "[RenderControl] Skipping stale camera frame at "
                        f"render_time={render_time:.6f}, last_consumed={last_consumed_rendering_time:.6f}"
                    )
                continue
            last_consumed_rendering_time = render_time

            current_rgb = self._extract_rgb_from_frame(latest_frame)
            if current_rgb is not None and current_rgb.size != 0:
                latest_rgb = np.asarray(current_rgb)
            sample_time = render_time if np.isfinite(render_time) else current_time
            frame_distance = abs(sample_time - gt_reference_time)
            if frame_distance <= representative_frame_distance + 1e-9:
                representative_frame = latest_frame
                representative_frame_distance = frame_distance
            if shutter_start_time - 1e-9 <= sample_time <= shutter_end_time + 1e-9:
                if latest_rgb is None or latest_rgb.size == 0:
                    return
                rgb_samples.append(np.asarray(latest_rgb, dtype=np.float32))
            elif not self.capture_motion_blur and latest_rgb is not None and latest_rgb.size != 0:
                rgb_samples = [np.asarray(latest_rgb, dtype=np.float32)]

        self._advance_simulation_without_render(target_time=frame_end_time)

        final_rgb = self._average_rgb_samples(rgb_samples, latest_rgb)
        rendered_data = dict(representative_frame) if representative_frame else dict(latest_frame)
        rendered_data["rgb"] = add_d455_noise(
            final_rgb, 
            iso=self.sensor_controllers[sensor_name].get_control_parameters().get("iso", 100)
        ) 
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
            first_directory = scene_usd_path.split("/")[1]
            if first_directory == "Isaac":
                add_reference_to_stage(usd_path=self.assets_root_path + scene_usd_path, prim_path="/World/Environment")
            else:
                add_reference_to_stage(usd_path=scene_usd_path, prim_path="/World/Environment")
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
        initial_pos = self.config.agent_origin_position
        self.agent = self._define_robot_agent(
            robot_type=self.robot_config.robot_name,
            robot_name="my_agent",
            robot_usd_path=robot_usd_path,
            initial_position=np.array(initial_pos), # -3.0, -3.0
            initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            **kwargs
        )
        
        self.agent_camera = self._define_agent_camera(
            self.agent_camera_prim_path, 
            self.assets_root_path + self.agent_camera_usd_path,
            self.config.require_external_camera
        )
        
        self.agent_imu = IMUSensor(
            prim_path=self.robot_config.agent_imu_prim_path,
            name="agent_imu",
            frequency=int(1. / self._simulation_dt),
            translation=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            linear_acceleration_filter_size = 10,
            angular_velocity_filter_size = 10,
            orientation_filter_size = 10,
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
            resolution=self.agent_camera_resolution,
            position=camera_position, # external camera의 position을 그대로 사용
        )
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
            resolution=self.agent_camera_resolution,
            position=cam_rel_position,
            annotator_device="cpu"
        )
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
            # Object will be randomly spawned in the range of 3 times minimum distance from the agent to ensure enough distribution
            spawn_rage = 3 * min_dist_from_agent
            min_x, min_y = agent_pos[0] - spawn_rage, agent_pos[1] - spawn_rage
            max_x, max_y = agent_pos[0] + spawn_rage, agent_pos[1] + spawn_rage
            x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
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
