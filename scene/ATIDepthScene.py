from .BasicScene import BaseScene
from isaacsim import SimulationApp
from ati_config import ATIBaseConfig
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdLux

from isaacsim.core.utils.stage import add_reference_to_stage
import omni.replicator.core as rep
import omni
import carb
import carb.settings
import numpy as np
from itertools import cycle
import os

from robot_control.define_controller import define_agent_controller
from ati_config import DEBUG

class ATIDepthScene(BaseScene):
    def __init__(
        self,
        simulation_app: SimulationApp,
        config: ATIBaseConfig,
        physics_dt: float = 1 / 60,
        rendering_dt: float = 1 / 60, 
        stage_units_in_meters=1.0,
        seed: int = 20260319
    ):
        super().__init__(
            simulation_app=simulation_app,
            config=config,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=stage_units_in_meters,
            seed=seed
        )

        ## Robot Control Code
        if DEBUG: print("[DEBUG] Checking Robot Prim Path:", self.agent.prim_path)
        self.agent_controller = define_agent_controller(
            agent_name=self.robot_config.robot_name,
            agent_prim_path=self.agent.prim_path,
            robot_config=self.robot_config
        )
        # world.reset() is now called inside build_scene() (before warmup).
        # Calling it again here would invalidate the PhysX SimView handles that
        # were just created, causing "Simulation view object is invalidated" errors.
        self.agent_controller.reset()

    def reset(self):
        super().reset()
        self.agent_controller.reset()

    def get_anno(self, anno_name):
        return self.RENDERING_ANNOTATOR_TYPES[anno_name]

    def get_sensor_control_params(self, sensor_name="agent_camera") -> dict:
        if sensor_name not in self.sensor_controllers:
            raise ValueError(f"Sensor '{sensor_name}' does not have a controller.")
        controller = self.sensor_controllers[sensor_name]
        return controller.get_control_parameters()
    
    def sensor_control(
        self, 
        sensor_name="agent_camera", 
        control_parameters: dict=None
    ):
        """
        You should control your camera(sensor) parameters here only.
        For now, I'll implement the control logic for...
        ISO, Shutter Time, Aperture
        """
        if self.rendering_mode == "autoexposure":
            raise RuntimeError("Manual sensor control is not allowed in 'autoexposure' rendering mode.")
        if sensor_name not in self.sensor_controllers:
            raise ValueError(f"Sensor '{sensor_name}' does not have a controller.")
        # If selected action does not make any changes, we can skip sending redundant control commands to the simulator.
        ## Too frequent sensor control causes stale data issues in Isaac Sim, so we only send control commands when there is an actual change in parameters.
        curr_params = self.get_sensor_control_params(sensor_name="agent_camera")
        if curr_params.get("iso", None) != control_parameters["iso"] or \
            curr_params.get("shutter_time", None) != control_parameters["exposure"]:
            controller = self.sensor_controllers[sensor_name]
            controller.update_parameters(control_parameters)
        return 
    
    def robot_control(
        self, 
        time, 
        control_parameters: dict = None
    ):
        """
        Agent(Robot) Control or Navigation Logic must be here.
        """
        if self.robot_config.robot_name == "limo":
            linear_vel = control_parameters["linear_velocity"]
            angular_vel = control_parameters["angular_velocity"]
            
            # print(f"[Robot Control] time: {time:.2f}, linear_vel: {linear_vel:.2f}, angular_vel: {angular_vel:.2f}, left_w: {left_w:.2f}, right_w: {right_w:.2f}")
            left_front_idx  = self.agent.get_dof_index("front_left_wheel")
            right_front_idx = self.agent.get_dof_index("front_right_wheel")
            left_rear_idx   = self.agent.get_dof_index("rear_left_wheel")
            right_rear_idx  = self.agent.get_dof_index("rear_right_wheel")
            self.agent.apply_action(self.agent_controller.forward(
                command={
                    "linear_velocity": linear_vel,
                    "angular_velocity": angular_vel
                },
                wheel_idx=[left_front_idx, left_rear_idx, right_front_idx, right_rear_idx]
            ))

        elif self.robot_config.robot_name == "kaya":
            # HolonomicController expects body-frame [forward, lateral, yaw] velocity.
            # For in-place rotation we should keep the planar components at zero.
            forward_velocity = float(
                control_parameters.get(
                    "linear_velocity_x",
                    control_parameters.get("forward_velocity", 0.0),
                )
            )
            lateral_velocity = float(
                control_parameters.get(
                    "linear_velocity_y",
                    control_parameters.get("lateral_velocity", 0.0),
                )
            )
            omega = float(control_parameters["angular_velocity"])

            self.agent.apply_wheel_actions(
                self.agent_controller.forward(command=[forward_velocity, lateral_velocity, omega])
            )
        
        return 
    
    def spawn_random_objects(self, min_dist_from_agent=6.0):
        car_prim = self.world.stage.GetPrimAtPath(self.agent_prim_path)
        car_loc = car_prim.GetAttribute("xformOp:translate").Get()
        for s_obj, num in self.single_object_usd_paths:
            for i in range(num):
                add_reference_to_stage(usd_path=self.assets_root_path + s_obj, prim_path=f"/World/Dolly_{i}")
                
                # Pose Randomization for Dolly
                dolly_prim = self.world.stage.GetPrimAtPath(f"/World/Dolly_{i}")
                if not dolly_prim.GetAttribute("xformOp:translate"):
                    UsdGeom.Xformable(dolly_prim).AddTranslateOp()
                if not dolly_prim.GetAttribute("xformOp:rotateXYZ"):
                    UsdGeom.Xformable(dolly_prim).AddRotateXYZOp()        
                self._object_spawn_randomization(
                    dolly_prim, 
                    agent_pos=car_loc, 
                    min_dist_from_agent=min_dist_from_agent
                )

        for p_obj, num in self.props_object_usd_paths:
            props_urls = []
            props_folder_path = self.assets_root_path + p_obj
            result, entries = omni.client.list(props_folder_path)
            if result != omni.client.Result.OK:
                carb.log_error(f"Could not list assets in path: {props_folder_path}")
            for entry in entries:
                _, ext = os.path.splitext(entry.relative_path)
                if ext == ".usd":
                    props_urls.append(f"{props_folder_path}/{entry.relative_path}")
            
            min_dist_from_car_props = 1
            cycled_props_url = cycle(props_urls)
            for i in range(num):
                prop_url = next(cycled_props_url)
                prop_name = os.path.splitext(os.path.basename(prop_url))[0]
                path = f"/World/Props/Prop_{prop_name}_{i}"
                prim = self.world.stage.DefinePrim(path, "Xform")
                prim.GetReferences().AddReference(prop_url)
                self._object_spawn_randomization(
                    obj_prim=prim, 
                    agent_pos=car_loc, 
                    min_dist_from_agent=min_dist_from_car_props
                )
        
        return 
