from scene import ATIDepthScene
import numpy as np
from dataclasses import dataclass  


@dataclass(slots=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(slots=True)
class VelocityCommand:
    linear_velocity: float
    angular_velocity: float


def quat_wxyz_to_yaw(quat) -> float:
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def get_limo_pose_2d(scene: ATIDepthScene) -> Pose2D:
    position, orientation = scene.agent.get_world_pose()
    return Pose2D(
        x=float(position[0]),
        y=float(position[1]),
        theta=quat_wxyz_to_yaw(orientation),
    )

def vector_angle(w: np.ndarray, v: np.ndarray) -> float:
    return float(np.arctan2(w[1] * v[0] - w[0] * v[1], w[0] * v[0] + w[1] * v[1]))

def nearest_point_on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a2b = b - a
    a2c = c - a
    a2b_mag = float(np.sqrt(np.sum(a2b**2)))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = float(np.dot(a2c, a2b_norm))
    if dist < 0.0:
        return a, dist
    if dist > a2b_mag:
        return b, dist
    return a + a2b_norm * dist, dist

class StraightLineLapFollower:
    """
    Follower for a fixed straight out-and-back lap.

    One lap is:
      (0, 0) -> (1, 0), rotate 180 deg,
      (1, 0) -> (-1, 0), rotate 180 deg,
      (-1, 0) -> (0, 0).
    """

    def __init__(
        self,
        straight_speed: float = 0.6,
        linear_acceleration: float = 2.0,
        linear_deceleration: float = 2.0,
        endpoint_turn_speed: float = 0.8,
        angular_acceleration: float = 2.0,
        angular_deceleration: float = 2.0,
        endpoint_distance: float = 1.0,
        position_threshold: float = 0.04,
        heading_threshold: float = 0.04,
        heading_gain: float = 2.5,
        max_heading_correction: float | None = None,
        forward_angle_threshold: float = np.pi / 3.0,
        recovery_speed: float = 0.15,
    ):
        self.max_straight_speed = max(0.0, float(straight_speed))
        self.linear_acceleration = max(1e-6, float(linear_acceleration))
        self.linear_deceleration = max(1e-6, float(linear_deceleration))
        self.endpoint_turn_speed = max(0.0, float(endpoint_turn_speed))
        self.angular_acceleration = max(1e-6, float(angular_acceleration))
        self.angular_deceleration = max(1e-6, float(angular_deceleration))
        self.endpoint_distance = max(0.0, float(endpoint_distance))
        self.position_threshold = max(0.0, float(position_threshold))
        self.heading_threshold = max(0.0, float(heading_threshold))
        self.heading_gain = float(heading_gain)
        self.max_heading_correction = (
            self.endpoint_turn_speed
            if max_heading_correction is None
            else max(0.0, float(max_heading_correction))
        )
        self.forward_angle_threshold = float(np.clip(forward_angle_threshold, 0.0, np.pi))
        self.recovery_speed = max(0.0, float(recovery_speed))
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.phase_index = 0
        self.lap_count = 0
        self.phase_names = (
            "drive_to_positive_x",
            "turn_to_negative_x",
            "drive_to_negative_x",
            "turn_to_positive_x",
            "drive_to_origin",
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def reset(self) -> None:
        self.phase_index = 0
        self.lap_count = 0
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

    @property
    def phase_name(self) -> str:
        return self.phase_names[self.phase_index]

    def _advance_phase(self) -> None:
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.phase_index += 1
        if self.phase_index >= len(self.phase_names):
            self.phase_index = 0
            self.lap_count += 1

    def _approach_velocity(
        self,
        current_velocity: float,
        target_velocity: float,
        acceleration: float,
        deceleration: float,
        step_dt: float | None,
    ) -> float:
        if step_dt is None or step_dt <= 0.0:
            return float(target_velocity)
        delta = float(target_velocity) - float(current_velocity)
        rate = acceleration if abs(target_velocity) > abs(current_velocity) else deceleration
        max_delta = rate * float(step_dt)
        if abs(delta) <= max_delta:
            return float(target_velocity)
        return float(current_velocity + np.sign(delta) * max_delta)

    def _turn_command(self, current_pose: Pose2D, target_heading: float, step_dt: float | None) -> VelocityCommand:
        heading_error = self._wrap_angle(target_heading - current_pose.theta)
        heading_epsilon = 1e-2
        if (
            abs(heading_error) <= heading_epsilon
            or (
                abs(heading_error) <= self.heading_threshold
                and abs(self.current_angular_velocity) <= heading_epsilon
            )
        ):
            self._advance_phase()
            return VelocityCommand(linear_velocity=0.0, angular_velocity=0.0)

        stop_angle = max(0.0, abs(heading_error))
        braking_speed = float(np.sqrt(2.0 * self.angular_deceleration * stop_angle))
        target_angular_speed = min(self.endpoint_turn_speed, braking_speed)
        target_angular_velocity = np.sign(heading_error) * target_angular_speed
        angular_velocity = self._approach_velocity(
            self.current_angular_velocity,
            target_angular_velocity,
            self.angular_acceleration,
            self.angular_deceleration,
            step_dt,
        )
        if step_dt is not None and step_dt > 0.0:
            max_safe_velocity = stop_angle / float(step_dt)
            angular_velocity = np.sign(heading_error) * min(abs(angular_velocity), max_safe_velocity)
        self.current_angular_velocity = float(angular_velocity)
        return VelocityCommand(linear_velocity=0.0, angular_velocity=float(angular_velocity))

    def _drive_command(
        self,
        current_pose: Pose2D,
        line_start: np.ndarray,
        target: np.ndarray,
        travel_heading: float,
        step_dt: float | None,
    ) -> VelocityCommand:
        robot_point = np.array([current_pose.x, current_pose.y], dtype=float)
        tangent_unit = np.array([np.cos(travel_heading), np.sin(travel_heading)], dtype=float)
        target_vec = target - robot_point
        signed_remaining = float(np.dot(target_vec, tangent_unit))
        distance_to_target = float(np.linalg.norm(target_vec))
        position_epsilon = 1e-4
        if (
            signed_remaining <= position_epsilon
            or (
                distance_to_target <= self.position_threshold
                and self.current_linear_velocity <= position_epsilon
            )
        ):
            self._advance_phase()
            return VelocityCommand(linear_velocity=0.0, angular_velocity=0.0)

        segment = target - line_start
        segment_length = float(np.linalg.norm(segment))
        segment_progress = float(np.clip(np.dot(robot_point - line_start, tangent_unit), 0.0, segment_length))
        nearest_point = line_start + tangent_unit * segment_progress
        path_to_robot = robot_point - nearest_point
        cross_track_error = float(
            tangent_unit[0] * path_to_robot[1] - tangent_unit[1] * path_to_robot[0]
        )

        heading_error = self._wrap_angle(travel_heading - current_pose.theta)
        angular_velocity = float(np.clip(
            self.heading_gain * heading_error - self.heading_gain * cross_track_error,
            -self.max_heading_correction,
            self.max_heading_correction,
        ))

        stop_distance = max(0.0, signed_remaining)
        braking_speed = float(np.sqrt(2.0 * self.linear_deceleration * stop_distance))
        target_linear_velocity = min(self.max_straight_speed, braking_speed)
        if abs(heading_error) > self.forward_angle_threshold:
            target_linear_velocity = min(target_linear_velocity, self.recovery_speed)
        linear_velocity = self._approach_velocity(
            self.current_linear_velocity,
            target_linear_velocity,
            self.linear_acceleration,
            self.linear_deceleration,
            step_dt,
        )
        if step_dt is not None and step_dt > 0.0:
            max_safe_velocity = stop_distance / float(step_dt)
            linear_velocity = min(linear_velocity, max_safe_velocity)
        self.current_linear_velocity = float(linear_velocity)
        return VelocityCommand(linear_velocity=float(linear_velocity), angular_velocity=angular_velocity)

    def step(self, current_pose: Pose2D, step_dt: float | None = None) -> VelocityCommand:
        origin = np.array([0.0, 0.0], dtype=float)
        positive_endpoint = np.array([self.endpoint_distance, 0.0], dtype=float)
        negative_endpoint = np.array([-self.endpoint_distance, 0.0], dtype=float)

        if self.phase_name == "drive_to_positive_x":
            return self._drive_command(current_pose, origin, positive_endpoint, 0.0, step_dt)
        if self.phase_name == "turn_to_negative_x":
            return self._turn_command(current_pose, np.pi, step_dt)
        if self.phase_name == "drive_to_negative_x":
            return self._drive_command(current_pose, positive_endpoint, negative_endpoint, np.pi, step_dt)
        if self.phase_name == "turn_to_positive_x":
            return self._turn_command(current_pose, 0.0, step_dt)
        return self._drive_command(current_pose, negative_endpoint, origin, 0.0, step_dt)
