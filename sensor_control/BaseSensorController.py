from abc import *
from typing import Union, Dict
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom

"""
BaseSensorController defines the interface for controlling camera parameters in Isaac Sim.
It includes an abstract method `update_parameters` that must be implemented by any subclass to update the camera parameters based on the provided control parameters. 
The `get_control_parameters` method allows retrieval of the current control parameters for the sensor.
"""
class BaseSensorController(metaclass=ABCMeta):
    def __init__(
        self, 
        camera: Camera,
        camera_prim: UsdGeom.Camera,
        control_parameters: Dict[str, Union[float, int]]
    ):
        self.camera = camera
        self.camera_prim = camera_prim
        self.control_parameters = control_parameters

    @abstractmethod
    def update_parameters(
        self, 
        control_parameters: Dict[str, Union[float, int]]
    ):
        raise NotImplementedError()

    def get_control_parameters(self) -> Dict[str, Union[float, int]]:
        return self.control_parameters
    