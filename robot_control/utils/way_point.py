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


class PathHelper:
    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2:
            raise ValueError("points must be an Nx2 array with at least two points.")
        self._point_distances = self._init_point_distances()

    def _init_point_distances(self) -> np.ndarray:
        distances = np.zeros(len(self.points), dtype=float)
        length = 0.0
        for idx in range(len(self.points) - 1):
            distances[idx] = length
            length += float(np.linalg.norm(self.points[idx + 1] - self.points[idx]))
        distances[-1] = length
        return distances

    def get_path_length(self) -> float:
        return float(self._point_distances[-1])

    def find_nearest(self, point: np.ndarray):
        min_dist_to_seg = np.inf
        min_seg = (0, 1)
        min_point = self.points[0]
        min_dist_along_path = 0.0
        for a_idx in range(len(self.points) - 1):
            b_idx = a_idx + 1
            nearest_pt, dist_along_seg = nearest_point_on_segment(self.points[a_idx], self.points[b_idx], point)
            dist_to_seg = float(np.linalg.norm(point - nearest_pt))
            if dist_to_seg < min_dist_to_seg:
                min_seg = (a_idx, b_idx)
                min_dist_to_seg = dist_to_seg
                min_point = nearest_pt
                min_dist_along_path = self._point_distances[a_idx] + dist_along_seg
        return min_point, min_dist_along_path, min_seg, min_dist_to_seg

    def get_point_by_distance(self, distance: float, seg_id: int) -> np.ndarray:
        distance = float(np.clip(distance, 0.0, self.get_path_length()))
        seg_id = int(np.clip(seg_id, 0, len(self.points) - 2))
        if distance < self._point_distances[seg_id]:
            candidate_range = range(seg_id, 0, -1)
            for idx in candidate_range:
                if distance >= self._point_distances[idx - 1]:
                    seg_id = idx - 1
                    break
        else:
            for idx in range(seg_id, len(self.points) - 1):
                if distance <= self._point_distances[idx + 1]:
                    seg_id = idx
                    break
        a = self.points[seg_id]
        b = self.points[seg_id + 1]
        a_dist = self._point_distances[seg_id]
        b_dist = self._point_distances[seg_id + 1]
        ratio = np.clip((distance - a_dist) / ((b_dist - a_dist) + 1e-6), 0.0, 1.0)
        return a + ratio * (b - a)


class RandomPathFollower:
    """
    ROS-free random path follower.

    The behavior mirrors MobilityGen's path-following loop: choose a random
    path, pick a lookahead target on that path, then convert heading error into
    differential-drive linear/angular commands.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        waypoint_count: int,
        path_speed: float,
        lookahead_distance: float,
        angular_gain: float,
        max_angular_velocity: float,
        rng: np.random.Generator,
        stop_distance_threshold: float = 0.35,
        forward_angle_threshold: float = np.pi / 3.0,
        turn_path_speed: float = 0.15,
        max_path_speed: float = 2.0,
        max_turn_path_speed: float = 1.0,
        min_motion_blur_speed: float = 1.2,
        lap_speed_margin: float = 1.25,
        bounds_margin: float = 0.15,
        boundary_turn_gain: float = 2.5,
        boundary_recovery_speed: float = 0.2,
        boundary_spin_velocity: float = 0.8,
        max_path_length: float | None = None,
        auto_resample_on_completion: bool = True,
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = (float(v) for v in bounds)
        self.waypoint_count = max(2, int(waypoint_count))
        self.lookahead_distance = float(lookahead_distance)
        self.rng = rng
        self.path = None
        self.path_helper = None
        self.path_id = 0
        self.path_completed = False
        self.auto_resample_on_completion = bool(auto_resample_on_completion)
        self.set_motion_limits(
            path_speed=path_speed,
            turn_path_speed=turn_path_speed,
            max_path_speed=max_path_speed,
            max_turn_path_speed=max_turn_path_speed,
            min_motion_blur_speed=min_motion_blur_speed,
            lap_speed_margin=lap_speed_margin,
            bounds_margin=bounds_margin,
            boundary_turn_gain=boundary_turn_gain,
            boundary_recovery_speed=boundary_recovery_speed,
            boundary_spin_velocity=boundary_spin_velocity,
            max_path_length=max_path_length,
            angular_gain=angular_gain,
            max_angular_velocity=max_angular_velocity,
            stop_distance_threshold=stop_distance_threshold,
            forward_angle_threshold=forward_angle_threshold,
        )

    def set_motion_limits(
        self,
        path_speed: float | None = None,
        turn_path_speed: float | None = None,
        max_path_speed: float | None = None,
        max_turn_path_speed: float | None = None,
        min_motion_blur_speed: float | None = None,
        lap_speed_margin: float | None = None,
        bounds_margin: float | None = None,
        boundary_turn_gain: float | None = None,
        boundary_recovery_speed: float | None = None,
        boundary_spin_velocity: float | None = None,
        max_path_length: float | None = None,
        angular_gain: float | None = None,
        max_angular_velocity: float | None = None,
        stop_distance_threshold: float | None = None,
        forward_angle_threshold: float | None = None,
    ) -> None:
        if path_speed is not None:
            self.path_speed = max(0.0, float(path_speed))
        if turn_path_speed is not None:
            self.turn_path_speed = max(0.0, float(turn_path_speed))
        if max_path_speed is not None:
            self.max_path_speed = max(0.0, float(max_path_speed))
        if max_turn_path_speed is not None:
            self.max_turn_path_speed = max(0.0, float(max_turn_path_speed))
        if min_motion_blur_speed is not None:
            self.min_motion_blur_speed = max(0.0, float(min_motion_blur_speed))
        if lap_speed_margin is not None:
            self.lap_speed_margin = max(1.0, float(lap_speed_margin))
        if bounds_margin is not None:
            self.bounds_margin = max(0.0, float(bounds_margin))
        if boundary_turn_gain is not None:
            self.boundary_turn_gain = max(0.0, float(boundary_turn_gain))
        if boundary_recovery_speed is not None:
            self.boundary_recovery_speed = max(0.0, float(boundary_recovery_speed))
        if boundary_spin_velocity is not None:
            self.boundary_spin_velocity = max(0.0, float(boundary_spin_velocity))
        if max_path_length is not None:
            self.max_path_length = max(0.0, float(max_path_length))
        elif not hasattr(self, "max_path_length"):
            self.max_path_length = None
        if angular_gain is not None:
            self.angular_gain = float(angular_gain)
        if max_angular_velocity is not None:
            self.max_angular_velocity = max(0.0, float(max_angular_velocity))
        if stop_distance_threshold is not None:
            self.stop_distance_threshold = max(0.0, float(stop_distance_threshold))
        if forward_angle_threshold is not None:
            self.forward_angle_threshold = float(np.clip(forward_angle_threshold, 0.0, np.pi))

    def _safe_bounds(self) -> tuple[float, float, float, float]:
        x_margin = min(self.bounds_margin, max(0.0, 0.5 * (self.x_max - self.x_min) - 1e-6))
        y_margin = min(self.bounds_margin, max(0.0, 0.5 * (self.y_max - self.y_min) - 1e-6))
        return (
            self.x_min + x_margin,
            self.x_max - x_margin,
            self.y_min + y_margin,
            self.y_max - y_margin,
        )

    def _sample_point(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self._safe_bounds()
        return np.array([
            self.rng.uniform(x_min, x_max),
            self.rng.uniform(y_min, y_max),
        ], dtype=float)

    def _keep_inside_bounds(self, point: np.ndarray) -> np.ndarray:
        x_min, x_max, y_min, y_max = self._safe_bounds()
        return np.array([
            np.clip(point[0], x_min, x_max),
            np.clip(point[1], y_min, y_max),
        ], dtype=float)

    def _is_inside_safe_bounds(self, point: np.ndarray) -> bool:
        x_min, x_max, y_min, y_max = self._safe_bounds()
        return bool(x_min <= point[0] <= x_max and y_min <= point[1] <= y_max)

    def _distance_to_safe_boundary_along_heading(self, point: np.ndarray, heading: np.ndarray) -> float:
        x_min, x_max, y_min, y_max = self._safe_bounds()
        distances = []
        if heading[0] > 1e-6:
            distances.append((x_max - point[0]) / heading[0])
        elif heading[0] < -1e-6:
            distances.append((x_min - point[0]) / heading[0])
        if heading[1] > 1e-6:
            distances.append((y_max - point[1]) / heading[1])
        elif heading[1] < -1e-6:
            distances.append((y_min - point[1]) / heading[1])
        positive_distances = [dist for dist in distances if dist >= 0.0]
        if not positive_distances:
            return 0.0
        return float(min(positive_distances))

    def _sample_bounded_path_points(self, start: np.ndarray) -> np.ndarray:
        if self.max_path_length is None or self.waypoint_count <= 2:
            random_points = [self._sample_point() for _ in range(self.waypoint_count - 1)]
            return np.vstack([start, *random_points])

        points = [start]
        segment_budget = self.max_path_length / max(1, self.waypoint_count - 1)
        for _ in range(self.waypoint_count - 1):
            prev = points[-1]
            candidate = prev
            for _ in range(16):
                angle = self.rng.uniform(-np.pi, np.pi)
                radius = self.rng.uniform(0.35 * segment_budget, segment_budget)
                candidate = self._keep_inside_bounds(
                    prev + radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
                )
                if np.linalg.norm(candidate - prev) > 1e-3:
                    break
            points.append(candidate)
        return np.vstack(points)

    def set_random_target_path(self, current_pose: Pose2D) -> None:
        start = self._keep_inside_bounds(np.array([current_pose.x, current_pose.y], dtype=float))
        points = self._sample_bounded_path_points(start)
        self.path = self._smooth_polyline(points)
        self.path_helper = PathHelper(self.path)
        self.path_id += 1
        self.path_completed = False

    def _smooth_polyline(self, points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
        smoothed = [points[0]]
        for idx in range(len(points) - 1):
            start = points[idx]
            end = points[idx + 1]
            for sample_idx in range(1, samples_per_segment + 1):
                ratio = sample_idx / samples_per_segment
                eased = ratio * ratio * (3.0 - 2.0 * ratio)
                smoothed.append(start + eased * (end - start))
        return np.asarray(smoothed, dtype=float)

    def _remaining_path_distance(self, pt_path_length: float) -> float:
        return max(0.0, self.path_helper.get_path_length() - float(pt_path_length))

    def distance_to_path_end(self, current_pose: Pose2D) -> float:
        if self.path is None:
            return np.inf
        pt_robot = np.array([current_pose.x, current_pose.y], dtype=float)
        return float(np.linalg.norm(pt_robot - self.path[-1]))

    def update_completion_status(self, current_pose: Pose2D) -> bool:
        self.path_completed = self.distance_to_path_end(current_pose) < self.stop_distance_threshold
        return self.path_completed

    def _target_linear_velocity(
        self,
        d_theta: float,
        remaining_distance: float,
        remaining_steps: int | None,
        step_dt: float | None,
    ) -> float:
        full_speed = max(self.path_speed, self.min_motion_blur_speed)
        turn_speed = self.turn_path_speed
        if remaining_steps is not None and step_dt is not None:
            remaining_time = max(float(remaining_steps) * float(step_dt), float(step_dt))
            required_speed = self.lap_speed_margin * remaining_distance / remaining_time
            full_speed = max(full_speed, required_speed)
            turn_speed = max(turn_speed, required_speed)

        if abs(d_theta) > self.forward_angle_threshold:
            return float(np.clip(turn_speed, 0.0, self.max_turn_path_speed))
        return float(np.clip(full_speed, 0.0, self.max_path_speed))

    def _apply_bounds_guard(
        self,
        current_pose: Pose2D,
        linear_velocity: float,
        angular_velocity: float,
        step_dt: float | None,
    ) -> VelocityCommand:
        if step_dt is None or step_dt <= 0.0:
            return VelocityCommand(linear_velocity=float(linear_velocity), angular_velocity=float(angular_velocity))

        pt_robot = np.array([current_pose.x, current_pose.y], dtype=float)
        heading = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)], dtype=float)
        requested_linear_velocity = float(linear_velocity)
        safe_distance = self._distance_to_safe_boundary_along_heading(pt_robot, heading)
        max_safe_velocity = max(0.0, safe_distance / float(step_dt))
        linear_velocity = min(requested_linear_velocity, max_safe_velocity)

        projected = pt_robot + heading * linear_velocity * float(step_dt)
        is_guard_active = (
            (not self._is_inside_safe_bounds(pt_robot))
            or (not self._is_inside_safe_bounds(projected))
            or max_safe_velocity < requested_linear_velocity
        )
        if not is_guard_active:
            return VelocityCommand(
                linear_velocity=float(linear_velocity),
                angular_velocity=float(angular_velocity)
            )

        safe_target = self._keep_inside_bounds(pt_robot)
        if np.linalg.norm(safe_target - pt_robot) < 1e-6:
            x_min, x_max, y_min, y_max = self._safe_bounds()
            safe_target = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=float)
        target_vec = safe_target - pt_robot
        target_norm = float(np.linalg.norm(target_vec))
        if target_norm > 1e-6:
            target_unit = target_vec / target_norm
            d_theta = vector_angle(heading, target_unit)
            angular_velocity = -self.boundary_turn_gain * d_theta

        angular_velocity = float(np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity))
        if abs(angular_velocity) < self.boundary_spin_velocity:
            spin_direction = 1.0 if angular_velocity >= 0.0 else -1.0
            angular_velocity = spin_direction * min(self.boundary_spin_velocity, self.max_angular_velocity)
        return VelocityCommand(
            linear_velocity=0.0,
            angular_velocity=angular_velocity,
        )

    def _spin_in_place(self) -> VelocityCommand:
        return VelocityCommand(
            linear_velocity=0.0,
            angular_velocity=float(min(self.boundary_spin_velocity, self.max_angular_velocity)),
        )

    def step(
        self,
        current_pose: Pose2D,
        remaining_steps: int | None = None,
        step_dt: float | None = None,
    ) -> VelocityCommand:
        if self.path is None or self.path_helper is None:
            self.set_random_target_path(current_pose)

        pt_robot = np.array([current_pose.x, current_pose.y], dtype=float)
        if self.update_completion_status(current_pose):
            if self.auto_resample_on_completion:
                self.set_random_target_path(current_pose)
            else:
                return self._spin_in_place()

        _, pt_path_length, pt_seg_idx, _ = self.path_helper.find_nearest(pt_robot)
        remaining_distance = self._remaining_path_distance(pt_path_length)
        pt_target = self.path_helper.get_point_by_distance(
            distance=pt_path_length + self.lookahead_distance,
            seg_id=pt_seg_idx[0],
        )

        vec_robot_unit = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)], dtype=float)
        vec_target = pt_target - pt_robot
        target_norm = float(np.linalg.norm(vec_target))
        if target_norm < 1e-6:
            return VelocityCommand(linear_velocity=0.0, angular_velocity=0.0)

        vec_target_unit = vec_target / target_norm
        d_theta = vector_angle(vec_robot_unit, vec_target_unit)
        linear_velocity = self._target_linear_velocity(
            d_theta=d_theta,
            remaining_distance=remaining_distance,
            remaining_steps=remaining_steps,
            step_dt=step_dt,
        )
        angular_velocity = -self.angular_gain * d_theta
        angular_velocity = float(np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity))
        return self._apply_bounds_guard(
            current_pose=current_pose,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            step_dt=step_dt,
        )


class FixedWayPointFollower:
    """
    High-speed follower for a fixed closed waypoint path.

    This controller tracks the continuous path distance on the waypoint
    polyline instead of switching between discrete waypoint targets. That keeps
    the lookahead target on the path and avoids the corner-cutting behavior that
    shows up when the robot drives directly toward sparse waypoints.
    """

    def __init__(
        self,
        waypoints: list[Pose2D],
        waypoint_threshold: float = 0.05,
        linear_speed: float = 2.0,
        lookahead_distance: float = 0.6,
        angular_gain: float = 2.5,
        cross_track_gain: float = 3.0,
        max_angular_velocity: float = 1.0,
        max_path_error: float = 0.08,
        recovery_linear_speed: float = 0.8,
        closed_path: bool = True,
        smoothing_passes: int = 2,
        use_mpc: bool = True,
        mpc_horizon_steps: int = 12,
        mpc_angular_samples: int = 9,
        mpc_path_error_weight: float = 80.0,
        mpc_heading_error_weight: float = 8.0,
        mpc_progress_weight: float = 2.0,
        mpc_speed_weight: float = 1.0,
        mpc_angular_weight: float = 0.05,
    ):
        if len(waypoints) < 2:
            raise ValueError("waypoints must contain at least two poses.")
        self.waypoints = waypoints
        self.waypoint_threshold = max(0.0, float(waypoint_threshold))
        self.linear_speed = max(0.0, float(linear_speed))
        self.lookahead_distance = max(0.0, float(lookahead_distance))
        self.angular_gain = float(angular_gain)
        self.cross_track_gain = float(cross_track_gain)
        self.max_angular_velocity = max(0.0, float(max_angular_velocity))
        self.max_path_error = max(0.0, float(max_path_error))
        self.recovery_linear_speed = max(0.0, float(recovery_linear_speed))
        self.closed_path = bool(closed_path)
        self.smoothing_passes = max(0, int(smoothing_passes))
        self.use_mpc = bool(use_mpc)
        self.mpc_horizon_steps = max(1, int(mpc_horizon_steps))
        self.mpc_angular_samples = max(3, int(mpc_angular_samples))
        self.mpc_path_error_weight = max(0.0, float(mpc_path_error_weight))
        self.mpc_heading_error_weight = max(0.0, float(mpc_heading_error_weight))
        self.mpc_progress_weight = max(0.0, float(mpc_progress_weight))
        self.mpc_speed_weight = max(0.0, float(mpc_speed_weight))
        self.mpc_angular_weight = max(0.0, float(mpc_angular_weight))
        self.path_distance = 0.0
        self.lap_count = 0
        self.path = self._build_path()
        self.path_helper = PathHelper(self.path)

    def _build_path(self) -> np.ndarray:
        points = np.array([[pose.x, pose.y] for pose in self.waypoints], dtype=float)
        if self.closed_path and np.linalg.norm(points[0] - points[-1]) > 1e-6:
            points = np.vstack([points, points[0]])
        for _ in range(self.smoothing_passes):
            points = self._chaikin_smooth(points)
        return points

    def _chaikin_smooth(self, points: np.ndarray) -> np.ndarray:
        smoothed = [points[0]]
        end_idx = len(points) - 1
        for idx in range(end_idx):
            p0 = points[idx]
            p1 = points[idx + 1]
            smoothed.append(0.75 * p0 + 0.25 * p1)
            smoothed.append(0.25 * p0 + 0.75 * p1)
        smoothed.append(points[-1])
        return np.asarray(smoothed, dtype=float)

    def reset(self) -> None:
        self.path_distance = 0.0
        self.lap_count = 0

    def _advance_distance(self, distance: float) -> float:
        path_length = self.path_helper.get_path_length()
        if path_length <= 1e-6:
            return 0.0
        if self.closed_path:
            laps, wrapped = divmod(float(distance), path_length)
            self.lap_count += int(laps)
            return wrapped
        return float(np.clip(distance, 0.0, path_length))

    def _progress_delta(self, start_distance: float, end_distance: float) -> float:
        path_length = self.path_helper.get_path_length()
        delta = float(end_distance) - float(start_distance)
        if self.closed_path and path_length > 1e-6 and delta < -0.5 * path_length:
            delta += path_length
        return max(0.0, delta)

    def _point_at_distance(self, distance: float, seg_id: int = 0) -> np.ndarray:
        path_length = self.path_helper.get_path_length()
        if self.closed_path and path_length > 1e-6:
            distance = float(distance) % path_length
        return self.path_helper.get_point_by_distance(distance, seg_id)

    def _tangent_at_distance(self, distance: float, seg_id: int = 0) -> np.ndarray:
        eps = max(0.05, self.lookahead_distance * 0.25)
        p0 = self._point_at_distance(distance, seg_id)
        p1 = self._point_at_distance(distance + eps, seg_id)
        tangent = p1 - p0
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return tangent / tangent_norm

    def _sync_progress_from_pose(self, nearest_distance: float, path_error: float) -> None:
        path_length = self.path_helper.get_path_length()
        if path_length <= 1e-6:
            self.path_distance = 0.0
            return
        if self.closed_path:
            forward_delta = (float(nearest_distance) - self.path_distance) % path_length
            if forward_delta < max(self.lookahead_distance, self.linear_speed * 0.25) or path_error > self.max_path_error:
                self.path_distance = float(nearest_distance)
        elif nearest_distance >= self.path_distance or path_error > self.max_path_error:
            self.path_distance = float(nearest_distance)

    def _rollout_pose(self, pose: Pose2D, linear_velocity: float, angular_velocity: float, dt: float) -> Pose2D:
        if abs(angular_velocity) < 1e-6:
            next_x = pose.x + linear_velocity * np.cos(pose.theta) * dt
            next_y = pose.y + linear_velocity * np.sin(pose.theta) * dt
        else:
            next_theta = pose.theta + angular_velocity * dt
            radius = linear_velocity / angular_velocity
            next_x = pose.x + radius * (np.sin(next_theta) - np.sin(pose.theta))
            next_y = pose.y - radius * (np.cos(next_theta) - np.cos(pose.theta))
        next_theta = pose.theta + angular_velocity * dt
        next_theta = float(np.arctan2(np.sin(next_theta), np.cos(next_theta)))
        return Pose2D(x=float(next_x), y=float(next_y), theta=next_theta)

    def _mpc_velocity_candidates(self, path_error: float) -> np.ndarray:
        if path_error > self.max_path_error:
            return np.array([
                self.recovery_linear_speed,
                0.5 * self.linear_speed,
                0.75 * self.linear_speed,
                self.linear_speed,
            ], dtype=float)
        return np.array([
            0.6 * self.linear_speed,
            0.8 * self.linear_speed,
            self.linear_speed,
        ], dtype=float)

    def _mpc_command(
        self,
        current_pose: Pose2D,
        step_dt: float,
        path_error: float,
    ) -> VelocityCommand:
        linear_candidates = np.clip(
            self._mpc_velocity_candidates(path_error),
            0.0,
            self.linear_speed,
        )
        angular_candidates = np.linspace(
            -self.max_angular_velocity,
            self.max_angular_velocity,
            self.mpc_angular_samples,
            dtype=float,
        )
        best_cost = np.inf
        best_command = VelocityCommand(linear_velocity=self.linear_speed, angular_velocity=0.0)
        start_distance = self.path_distance

        for linear_velocity in linear_candidates:
            for angular_velocity in angular_candidates:
                rollout_pose = current_pose
                rollout_distance = start_distance
                cost = 0.0
                for horizon_idx in range(1, self.mpc_horizon_steps + 1):
                    rollout_pose = self._rollout_pose(
                        rollout_pose,
                        float(linear_velocity),
                        float(angular_velocity),
                        step_dt,
                    )
                    rollout_point = np.array([rollout_pose.x, rollout_pose.y], dtype=float)
                    _, nearest_distance, nearest_seg, rollout_error = self.path_helper.find_nearest(rollout_point)
                    if self.closed_path:
                        forward_delta = self._progress_delta(rollout_distance, nearest_distance)
                        if forward_delta < self.linear_speed * step_dt * 2.0:
                            rollout_distance = nearest_distance
                    else:
                        rollout_distance = max(rollout_distance, nearest_distance)

                    tangent_unit = self._tangent_at_distance(rollout_distance, nearest_seg[0])
                    heading_unit = np.array([np.cos(rollout_pose.theta), np.sin(rollout_pose.theta)], dtype=float)
                    heading_error = vector_angle(heading_unit, tangent_unit)
                    progress = self._progress_delta(start_distance, rollout_distance)
                    cost += self.mpc_path_error_weight * rollout_error * rollout_error
                    cost += self.mpc_heading_error_weight * heading_error * heading_error
                    cost -= self.mpc_progress_weight * progress
                    cost += self.mpc_speed_weight * (self.linear_speed - linear_velocity) ** 2
                    cost += self.mpc_angular_weight * angular_velocity * angular_velocity
                    if horizon_idx == self.mpc_horizon_steps:
                        cost += 2.0 * self.mpc_path_error_weight * rollout_error * rollout_error

                if cost < best_cost:
                    best_cost = cost
                    best_command = VelocityCommand(
                        linear_velocity=float(linear_velocity),
                        angular_velocity=float(angular_velocity),
                    )
        return best_command

    def step(self, current_pose: Pose2D, step_dt: float | None = None) -> VelocityCommand:
        robot_point = np.array([current_pose.x, current_pose.y], dtype=float)
        nearest_point, nearest_distance, nearest_seg, path_error = self.path_helper.find_nearest(robot_point)
        self._sync_progress_from_pose(nearest_distance, path_error)

        if step_dt is not None and step_dt > 0.0 and self.use_mpc:
            command = self._mpc_command(
                current_pose=current_pose,
                step_dt=float(step_dt),
                path_error=path_error,
            )
            self.path_distance = self._advance_distance(self.path_distance + command.linear_velocity * float(step_dt))
            return command

        if step_dt is not None and step_dt > 0.0:
            self.path_distance = self._advance_distance(self.path_distance + self.linear_speed * float(step_dt))

        lookahead_distance = self.path_distance + max(self.lookahead_distance, self.linear_speed * 0.3)
        target_point = self._point_at_distance(lookahead_distance, nearest_seg[0])
        tangent_unit = self._tangent_at_distance(self.path_distance, nearest_seg[0])
        heading_unit = np.array([np.cos(current_pose.theta), np.sin(current_pose.theta)], dtype=float)

        target_vec = target_point - robot_point
        target_norm = float(np.linalg.norm(target_vec))
        if target_norm < 1e-6:
            target_unit = tangent_unit
        else:
            target_unit = target_vec / target_norm

        heading_error = vector_angle(heading_unit, target_unit)
        path_to_robot = robot_point - nearest_point
        signed_cross_track_error = float(
            tangent_unit[0] * path_to_robot[1] - tangent_unit[1] * path_to_robot[0]
        )
        angular_velocity = (
            -self.angular_gain * heading_error
            -self.cross_track_gain * signed_cross_track_error
        )
        angular_velocity = float(np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity))

        linear_velocity = self.linear_speed
        if path_error > self.max_path_error:
            linear_velocity = min(linear_velocity, self.recovery_linear_speed)
        if abs(heading_error) > np.pi / 2.0:
            linear_velocity = min(linear_velocity, self.recovery_linear_speed)

        return VelocityCommand(
            linear_velocity=float(linear_velocity),
            angular_velocity=angular_velocity,
        )


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
        endpoint_turn_speed: float = 0.8,
        endpoint_distance: float = 1.0,
        position_threshold: float = 0.04,
        heading_threshold: float = 0.04,
        heading_gain: float = 2.5,
        max_heading_correction: float | None = None,
        forward_angle_threshold: float = np.pi / 3.0,
        recovery_speed: float = 0.15,
    ):
        self.straight_speed = max(0.0, float(straight_speed))
        self.endpoint_turn_speed = max(0.0, float(endpoint_turn_speed))
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

    @property
    def phase_name(self) -> str:
        return self.phase_names[self.phase_index]

    def _advance_phase(self) -> None:
        self.phase_index += 1
        if self.phase_index >= len(self.phase_names):
            self.phase_index = 0
            self.lap_count += 1

    def _turn_command(self, current_pose: Pose2D, target_heading: float, step_dt: float | None) -> VelocityCommand:
        heading_error = self._wrap_angle(target_heading - current_pose.theta)
        if abs(heading_error) <= self.heading_threshold:
            self._advance_phase()
            return VelocityCommand(linear_velocity=0.0, angular_velocity=0.0)

        angular_velocity = np.sign(heading_error) * self.endpoint_turn_speed
        if step_dt is not None and step_dt > 0.0:
            angular_velocity = np.sign(heading_error) * min(abs(angular_velocity), abs(heading_error) / step_dt)
        return VelocityCommand(linear_velocity=0.0, angular_velocity=float(angular_velocity))

    def _drive_command(self, current_pose: Pose2D, target: np.ndarray) -> VelocityCommand:
        robot_point = np.array([current_pose.x, current_pose.y], dtype=float)
        target_vec = target - robot_point
        distance = float(np.linalg.norm(target_vec))
        if distance <= self.position_threshold:
            self._advance_phase()
            return VelocityCommand(linear_velocity=0.0, angular_velocity=0.0)

        target_heading = float(np.arctan2(target_vec[1], target_vec[0]))
        heading_error = self._wrap_angle(target_heading - current_pose.theta)
        angular_velocity = float(np.clip(
            self.heading_gain * heading_error,
            -self.max_heading_correction,
            self.max_heading_correction,
        ))

        linear_velocity = self.straight_speed
        if abs(heading_error) > self.forward_angle_threshold:
            linear_velocity = min(linear_velocity, self.recovery_speed)
        return VelocityCommand(linear_velocity=float(linear_velocity), angular_velocity=angular_velocity)

    def step(self, current_pose: Pose2D, step_dt: float | None = None) -> VelocityCommand:
        positive_endpoint = np.array([self.endpoint_distance, 0.0], dtype=float)
        negative_endpoint = np.array([-self.endpoint_distance, 0.0], dtype=float)
        origin = np.array([0.0, 0.0], dtype=float)

        if self.phase_name == "drive_to_positive_x":
            return self._drive_command(current_pose, positive_endpoint)
        if self.phase_name == "turn_to_negative_x":
            return self._turn_command(current_pose, np.pi, step_dt)
        if self.phase_name == "drive_to_negative_x":
            return self._drive_command(current_pose, negative_endpoint)
        if self.phase_name == "turn_to_positive_x":
            return self._turn_command(current_pose, 0.0, step_dt)
        return self._drive_command(current_pose, origin)
