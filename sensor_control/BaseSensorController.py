from abc import *
from typing import Union, Dict
from isaacsim.sensors.camera import Camera
from pxr import Sdf, Usd

"""
BaseSensorController defines the interface for controlling camera parameters in Isaac Sim.
It includes an abstract method `update_parameters` that must be implemented by any subclass to update the camera parameters based on the provided control parameters. 
The `get_control_parameters` method allows retrieval of the current control parameters for the sensor.
"""
class BaseSensorController(metaclass=ABCMeta):
    def __init__(
        self, 
        sensor_name: str,
        camera: Camera,
        camera_prim: Usd.Prim,
        camera_fps: float,
        control_parameters: Dict[str, Union[float, int]] | None = None
    ):
        if camera_fps <= 0:
            raise ValueError(f"camera_fps must be positive, got {camera_fps}.")
        if not camera_prim.IsValid():
            raise ValueError(f"Camera prim for sensor '{sensor_name}' is not valid.")
        self.sensor_name = sensor_name
        self.camera = camera
        self.camera_prim = camera_prim
        self.camera_fps = float(camera_fps)
        self.control_parameters: Dict[str, Union[float, int]] = {
            "iso": 400.0,
            "aperture": 0.0,
            "shutter_time": 0.008,
        }
        self.__ensure_exposure_api()
        if control_parameters:
            self.control_parameters.update(control_parameters)

    @abstractmethod
    def update_parameters(
        self, 
        control_parameters: Dict[str, Union[float, int]]
    ):
        raise NotImplementedError()

    def get_control_parameters(self) -> Dict[str, Union[float, int]]:
        return dict(self.control_parameters)
    
    def get_shutter_window_frames(self) -> tuple[float, float]:
        shutter_time = max(0.0, float(self.control_parameters.get("shutter_time", 0.0)))
        return 0.0, shutter_time * self.camera_fps
    
    def get_shutter_window_seconds(self, max_frame_duration: float | None = None) -> tuple[float, float]:
        shutter_open_frames, shutter_close_frames = self.get_shutter_window_frames()
        shutter_open_seconds = shutter_open_frames / self.camera_fps
        shutter_close_seconds = shutter_close_frames / self.camera_fps
        if max_frame_duration is not None:
            shutter_open_seconds = max(0.0, min(shutter_open_seconds, max_frame_duration))
            shutter_close_seconds = max(
                shutter_open_seconds,
                min(shutter_close_seconds, max_frame_duration)
            )
        return shutter_open_seconds, shutter_close_seconds
    
    def __ensure_exposure_api(self):
        self.camera_prim.ApplyAPI("OmniRtxCameraExposureAPI_1")
        if not self.camera_prim.GetAttribute("shutter:open"):
            self.camera_prim.CreateAttribute("shutter:open", Sdf.ValueTypeNames.Double).Set(0.0)
        if not self.camera_prim.GetAttribute("shutter:close"):
            self.camera_prim.CreateAttribute("shutter:close", Sdf.ValueTypeNames.Double).Set(0.0)
        if not self.camera_prim.GetAttribute("exposure:time"):
            self.camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(0.0)
        if not self.camera_prim.GetAttribute("exposure:iso"):
            self.camera_prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(100.0)
        if not self.camera_prim.GetAttribute("exposure:fStop"):
            self.camera_prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(0.0)
        if not self.camera_prim.GetAttribute("exposure"):
            self.camera_prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(0.0)
        if not self.camera_prim.GetAttribute("exposure:responsivity"):
            self.camera_prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(1.0)
    
    def _set_iso(self, iso: Union[float, int]):
        target_iso = float(iso)
        self.camera_prim.GetAttribute("exposure:iso").Set(target_iso)
        self.control_parameters["iso"] = target_iso
    
    def _set_shutter_time(self, shutter_time: Union[float, int]):
        target_shutter_time = max(0.0, float(shutter_time))
        shutter_open_frames, shutter_close_frames = self.get_shutter_window_frames_from_seconds(target_shutter_time)
        self.camera_prim.GetAttribute("shutter:open").Set(shutter_open_frames)
        self.camera_prim.GetAttribute("shutter:close").Set(shutter_close_frames)
        self.camera_prim.GetAttribute("exposure:time").Set(target_shutter_time)
        self.control_parameters["shutter_time"] = target_shutter_time
    
    def _set_aperture(self, aperture: Union[float, int]):
        target_aperture = float(aperture)
        self.camera_prim.GetAttribute("exposure:fStop").Set(target_aperture)
        self.camera.set_lens_aperture(target_aperture)
        self.control_parameters["aperture"] = target_aperture
    
    def get_shutter_window_frames_from_seconds(self, shutter_time: float) -> tuple[float, float]:
        shutter_time = max(0.0, float(shutter_time))
        return 0.0, shutter_time * self.camera_fps
    
