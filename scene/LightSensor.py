import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

import omni.usd
import omni.physx
from pxr import Usd, UsdGeom, UsdLux, Gf


@dataclass
class DomeLightInfo:
    path: str
    intensity: float
    exposure: float
    color: Tuple[float, float, float]
    texture_file: Optional[str]


class DomeLightLuxProxy:
    """
    DomeLight 기반 simulated illumination proxy.

    - 입력: camera prim path
    - 출력: camera 위치에서의 dome light illumination proxy
    - 특징:
        1. exposure time / ISO / gain에 독립적
        2. 여러 개의 DomeLight를 자동으로 합산
        3. PhysX raycast로 sky visibility를 근사
        4. 절대 lux가 아니라 relative proxy임
    """

    def __init__(
        self,
        camera_prim_path: str,
        num_rays: int = 128,
        max_distance: float = 1000.0,
        upward_only: bool = True,
        seed: int = 0,
    ):
        self.camera_prim_path = camera_prim_path
        self.num_rays = int(num_rays)
        self.max_distance = float(max_distance)
        self.upward_only = bool(upward_only)

        random.seed(seed)
        np.random.seed(seed)

        self.stage = omni.usd.get_context().get_stage()
        self.scene_query = omni.physx.get_physx_scene_query_interface()

        self._directions = self._make_sample_directions(
            num_rays=self.num_rays,
            upward_only=self.upward_only,
        )

    def compute(self) -> dict:
        """
        Returns:
            {
                "illumination_proxy": float,
                "sky_visibility": float,
                "num_dome_lights": int,
                "dome_lights": List[DomeLightInfo],
            }
        """
        cam_pos = self._get_world_position(self.camera_prim_path)

        dome_lights = self._find_dome_lights()
        sky_visibility = self._compute_sky_visibility(cam_pos)

        total = 0.0
        for light in dome_lights:
            # UsdLux convention: exposure scales emission by 2^exposure.
            # color_luma는 RGB light color가 있을 때 밝기 계수로 반영.
            color_luma = (
                0.2126 * light.color[0]
                + 0.7152 * light.color[1]
                + 0.0722 * light.color[2]
            )

            light_power = light.intensity * (2.0 ** light.exposure) * color_luma
            total += light_power * sky_visibility

        return {
            "illumination_proxy": float(total),
            "sky_visibility": float(sky_visibility),
            "num_dome_lights": len(dome_lights),
            "dome_lights": dome_lights,
        }

    def _find_dome_lights(self) -> List[DomeLightInfo]:
        dome_lights = []

        for prim in self.stage.Traverse():
            if not prim.IsValid():
                continue

            if prim.GetTypeName() != "DomeLight":
                continue

            dome = UsdLux.DomeLight(prim)

            intensity = self._get_attr_or_default(
                dome.GetIntensityAttr(),
                default=1.0,
            )
            exposure = self._get_attr_or_default(
                dome.GetExposureAttr(),
                default=0.0,
            )
            color = self._get_attr_or_default(
                dome.GetColorAttr(),
                default=Gf.Vec3f(1.0, 1.0, 1.0),
            )

            # texture file은 있을 수도, 없을 수도 있음.
            texture_file = None
            try:
                texture_attr = dome.GetTextureFileAttr()
                if texture_attr and texture_attr.HasAuthoredValueOpinion():
                    texture_asset = texture_attr.Get()
                    if texture_asset is not None:
                        texture_file = str(texture_asset)
            except Exception:
                texture_file = None

            dome_lights.append(
                DomeLightInfo(
                    path=str(prim.GetPath()),
                    intensity=float(intensity),
                    exposure=float(exposure),
                    color=(float(color[0]), float(color[1]), float(color[2])),
                    texture_file=texture_file,
                )
            )

        return dome_lights

    def _compute_sky_visibility(self, origin: np.ndarray) -> float:
        """
        camera 위치에서 여러 방향으로 raycast.
        ray가 아무 collider에도 맞지 않으면 해당 방향은 sky/environment visible.
        """
        visible_count = 0

        # self-collision 방지를 위해 ray origin을 아주 조금 띄움
        origin = np.asarray(origin, dtype=np.float32)
        origin_tuple = tuple(float(x) for x in origin)

        for direction in self._directions:
            direction = np.asarray(direction, dtype=np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # 카메라 자신 또는 mount와 바로 충돌하는 경우를 줄이기 위한 offset
            ray_origin = origin + direction * 0.03

            hit = self._raycast_closest(
                origin=ray_origin,
                direction=direction,
                max_distance=self.max_distance,
            )

            if not hit:
                visible_count += 1

        return visible_count / max(len(self._directions), 1)

    def _raycast_closest(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> bool:
        """
        Returns:
            True  -> 뭔가에 막힘
            False -> 안 막힘, environment visible

        Isaac Sim 버전에 따라 raycast_closest 반환 타입이 조금 다를 수 있어서
        dict / tuple / object 형태를 모두 방어적으로 처리.
        """
        origin_tuple = tuple(float(x) for x in origin)
        direction_tuple = tuple(float(x) for x in direction)

        try:
            result = self.scene_query.raycast_closest(
                origin_tuple,
                direction_tuple,
                float(max_distance),
            )
        except TypeError:
            # 일부 버전에서는 max distance 인자명이 다르거나 signature가 다를 수 있음.
            result = self.scene_query.raycast_closest(
                origin=origin_tuple,
                dir=direction_tuple,
                distance=float(max_distance),
            )

        return self._parse_raycast_hit(result)

    @staticmethod
    def _parse_raycast_hit(result) -> bool:
        if result is None:
            return False

        if isinstance(result, bool):
            return result

        if isinstance(result, dict):
            # Isaac Sim에서 흔한 형태: {"hit": bool, ...}
            if "hit" in result:
                return bool(result["hit"])
            if "rigidBody" in result or "collision" in result:
                return True
            return False

        # object style result
        if hasattr(result, "hit"):
            return bool(result.hit)

        # tuple/list style result
        if isinstance(result, (tuple, list)) and len(result) > 0:
            if isinstance(result[0], bool):
                return bool(result[0])
            return bool(result)

        return bool(result)

    def _get_world_position(self, prim_path: str) -> np.ndarray:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")

        xformable = UsdGeom.Xformable(prim)
        world_mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = world_mat.ExtractTranslation()

        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

    @staticmethod
    def _get_attr_or_default(attr, default):
        if attr is None:
            return default
        value = attr.Get()
        if value is None:
            return default
        return value

    @staticmethod
    def _make_sample_directions(
        num_rays: int,
        upward_only: bool = True,
    ) -> List[np.ndarray]:
        """
        Fibonacci sphere/hemisphere sampling.

        upward_only=True이면 z > 0 방향만 샘플링.
        로봇 위쪽으로 들어오는 DomeLight/sky visibility를 조도 센서처럼 근사.
        """
        dirs = []

        if upward_only:
            # upper hemisphere: z in [0, 1]
            for i in range(num_rays):
                z = (i + 0.5) / num_rays
                theta = math.pi * (3.0 - math.sqrt(5.0)) * i
                r = math.sqrt(max(0.0, 1.0 - z * z))

                x = r * math.cos(theta)
                y = r * math.sin(theta)

                dirs.append(np.array([x, y, z], dtype=np.float32))
        else:
            # full sphere: z in [-1, 1]
            for i in range(num_rays):
                z = 1.0 - 2.0 * (i + 0.5) / num_rays
                theta = math.pi * (3.0 - math.sqrt(5.0)) * i
                r = math.sqrt(max(0.0, 1.0 - z * z))

                x = r * math.cos(theta)
                y = r * math.sin(theta)

                dirs.append(np.array([x, y, z], dtype=np.float32))

        return dirs