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
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = (float(v) for v in bounds)
        self.waypoint_count = max(2, int(waypoint_count))
        self.lookahead_distance = float(lookahead_distance)
        self.rng = rng
        self.path = None
        self.path_helper = None
        self.path_id = 0
        self.set_motion_limits(
            path_speed=path_speed,
            turn_path_speed=turn_path_speed,
            angular_gain=angular_gain,
            max_angular_velocity=max_angular_velocity,
            stop_distance_threshold=stop_distance_threshold,
            forward_angle_threshold=forward_angle_threshold,
        )

    def set_motion_limits(
        self,
        path_speed: float | None = None,
        turn_path_speed: float | None = None,
        angular_gain: float | None = None,
        max_angular_velocity: float | None = None,
        stop_distance_threshold: float | None = None,
        forward_angle_threshold: float | None = None,
    ) -> None:
        if path_speed is not None:
            self.path_speed = max(0.0, float(path_speed))
        if turn_path_speed is not None:
            self.turn_path_speed = max(0.0, float(turn_path_speed))
        if angular_gain is not None:
            self.angular_gain = float(angular_gain)
        if max_angular_velocity is not None:
            self.max_angular_velocity = max(0.0, float(max_angular_velocity))
        if stop_distance_threshold is not None:
            self.stop_distance_threshold = max(0.0, float(stop_distance_threshold))
        if forward_angle_threshold is not None:
            self.forward_angle_threshold = float(np.clip(forward_angle_threshold, 0.0, np.pi))

    def _sample_point(self) -> np.ndarray:
        return np.array([
            self.rng.uniform(self.x_min, self.x_max),
            self.rng.uniform(self.y_min, self.y_max),
        ], dtype=float)

    def _keep_inside_bounds(self, point: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(point[0], self.x_min, self.x_max),
            np.clip(point[1], self.y_min, self.y_max),
        ], dtype=float)

    def set_random_target_path(self, current_pose: Pose2D) -> None:
        start = self._keep_inside_bounds(np.array([current_pose.x, current_pose.y], dtype=float))
        random_points = [self._sample_point() for _ in range(self.waypoint_count - 1)]
        points = np.vstack([start, *random_points])
        self.path = self._smooth_polyline(points)
        self.path_helper = PathHelper(self.path)
        self.path_id += 1

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

    def step(self, current_pose: Pose2D) -> VelocityCommand:
        if self.path is None or self.path_helper is None:
            self.set_random_target_path(current_pose)

        pt_robot = np.array([current_pose.x, current_pose.y], dtype=float)
        path_end = self.path[-1]
        dist_to_target = float(np.linalg.norm(pt_robot - path_end))
        if dist_to_target < self.stop_distance_threshold:
            self.set_random_target_path(current_pose)

        _, pt_path_length, pt_seg_idx, _ = self.path_helper.find_nearest(pt_robot)
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
        linear_velocity = self.path_speed
        if abs(d_theta) > self.forward_angle_threshold:
            linear_velocity = self.turn_path_speed
        angular_velocity = -self.angular_gain * d_theta
        angular_velocity = float(np.clip(angular_velocity, -self.max_angular_velocity, self.max_angular_velocity))
        return VelocityCommand(linear_velocity=float(linear_velocity), angular_velocity=angular_velocity)
