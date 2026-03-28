from typing import Dict, Union

from .BaseSensorController import BaseSensorController


class ShutterExposureSensorController(BaseSensorController):
    """
    Default camera controller for ATI Isaac framework.

    It manages the standard camera exposure parameters used by the current
    pipeline:
    - `iso`
    - `shutter_time`
    - `aperture`
    """
    def __init__(
        self,
        sensor_name,
        camera,
        camera_prim,
        camera_fps,
        control_parameters: Dict[str, Union[float, int]] | None = None,
    ):
        super().__init__(
            sensor_name=sensor_name,
            camera=camera,
            camera_prim=camera_prim,
            camera_fps=camera_fps,
            control_parameters=control_parameters,
        )
        self.update_parameters(self.control_parameters)

    def update_parameters(self, control_parameters: Dict[str, Union[float, int]] | None):
        control_parameters = control_parameters or {}

        if "iso" in control_parameters and control_parameters["iso"] is not None:
            self._set_iso(control_parameters["iso"])

        if "shutter_time" in control_parameters and control_parameters["shutter_time"] is not None:
            self._set_shutter_time(control_parameters["shutter_time"])

        if "aperture" in control_parameters and control_parameters["aperture"] is not None:
            self._set_aperture(control_parameters["aperture"])

        return self.get_control_parameters()
