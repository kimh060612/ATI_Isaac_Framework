from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi
from typing import Literal, Sequence


EaseMode = Literal["cosine", "smoothstep", "linear"]


def _validate_anchor_values(anchor_values: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in anchor_values)
    if len(values) < 2:
        raise ValueError("anchor_values must contain at least two values.")
    return values


def _ping_pong_values(anchor_values: Sequence[float]) -> tuple[float, ...]:
    values = _validate_anchor_values(anchor_values)
    if len(values) == 2:
        return values
    return values + values[-2:0:-1]


def _ease_01(ratio: float, easing: EaseMode) -> float:
    x = min(max(float(ratio), 0.0), 1.0)
    if easing == "linear":
        return x
    if easing == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if easing == "cosine":
        return 0.5 - 0.5 * cos(pi * x)
    raise ValueError(f"Unsupported easing mode: {easing}")

@dataclass(slots=True)
class StepTrajectory:
    """
    Traverses anchor values in a stepwise pattern.

    Example:
        [200, 1000, 3000, 6000, 9000]
        -> 200 -> 1000 -> 3000 -> 6000 -> 9000 -> repeat
    Each value is held for hold_steps before jumping to the next value.
    """
    anchor_values: Sequence[float]
    hold_steps: int
    phase_offset_steps: int = 0
    _cycle_values: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cycle_values = _validate_anchor_values(self.anchor_values)
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be zero or a positive integer.")

    @property
    def num_segments(self) -> int:
        return len(self._cycle_values)

    @property
    def cycle_steps(self) -> int:
        return self.num_segments * self.hold_steps

    def value_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be zero or a positive integer.")

        shifted_step = (int(step) + int(self.phase_offset_steps)) % self.cycle_steps
        segment_idx = shifted_step // self.hold_steps
        return float(self._cycle_values[segment_idx])


@dataclass(slots=True)
class PingPongTrajectory:
    """
    Smoothly traverses anchor values in a ping-pong pattern.

    Example:
        [200, 1000, 3000, 6000, 9000]
        -> 200 -> 1000 -> 3000 -> 6000 -> 9000 -> 6000 -> 3000 -> 1000 -> repeat

    Each transition is interpolated with an ease-in-out curve so the value
    slows down near turning points instead of jumping abruptly.
    """

    anchor_values: Sequence[float]
    transition_steps: int
    hold_steps: int = 0
    phase_offset_steps: int = 0
    easing: EaseMode = "cosine"
    _cycle_values: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cycle_values = _ping_pong_values(self.anchor_values)
        if self.transition_steps <= 0:
            raise ValueError("transition_steps must be a positive integer.")
        if self.hold_steps < 0:
            raise ValueError("hold_steps must be zero or a positive integer.")

    @property
    def num_segments(self) -> int:
        return len(self._cycle_values)

    @property
    def segment_steps(self) -> int:
        return self.transition_steps + self.hold_steps

    @property
    def cycle_steps(self) -> int:
        return self.num_segments * self.segment_steps

    def value_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be zero or a positive integer.")

        shifted_step = (int(step) + int(self.phase_offset_steps)) % self.cycle_steps
        segment_idx = shifted_step // self.segment_steps
        step_in_segment = shifted_step % self.segment_steps

        start_value = self._cycle_values[segment_idx]
        end_value = self._cycle_values[(segment_idx + 1) % self.num_segments]

        if step_in_segment < self.hold_steps:
            return float(start_value)

        transition_step = step_in_segment - self.hold_steps
        if self.transition_steps == 1:
            return float(end_value)

        ratio = transition_step / (self.transition_steps - 1)
        eased_ratio = _ease_01(ratio, self.easing)
        value = start_value + (end_value - start_value) * eased_ratio
        return float(value)

    def sample(self, start_step: int, num_steps: int) -> list[float]:
        if num_steps < 0:
            raise ValueError("num_steps must be zero or a positive integer.")
        return [self.value_at(start_step + offset) for offset in range(num_steps)]


@dataclass(slots=True)
class ContextTrajectory:
    """
    Bundles smooth light and angular-velocity trajectories for control loops.

    This wrapper is intentionally thin so it can be dropped into existing
    simulation code with minimal changes.
    """

    light_trajectory: PingPongTrajectory
    speed_trajectory: PingPongTrajectory
    speed_scale: float = 1.0

    def value_at(self, step: int) -> dict[str, float]:
        return {
            "light_intensity": self.light_trajectory.value_at(step),
            "angular_velocity": self.speed_trajectory.value_at(step) * float(self.speed_scale),
        }


def build_step_context_trajectory(
    light_values: Sequence[float],
    speed_values: Sequence[float],
    *,
    light_hold_steps: int = 30,
    speed_hold_steps: int = 15,
    speed_phase_offset_steps: int = 0,
) -> ContextTrajectory:
    light_trajectory = StepTrajectory(
        anchor_values=light_values,
        hold_steps=light_hold_steps,
        phase_offset_steps=0,
    )
    speed_trajectory = StepTrajectory(
        anchor_values=speed_values,
        hold_steps=speed_hold_steps,
        phase_offset_steps=speed_phase_offset_steps,
    )
    return ContextTrajectory(
        light_trajectory=light_trajectory,
        speed_trajectory=speed_trajectory,
        speed_scale=1.0,
    )
    

def build_default_context_trajectory(
    light_values: Sequence[float],
    speed_values: Sequence[float],
    *,
    light_transition_steps: int = 120,
    speed_transition_steps: int = 60,
    light_hold_steps: int = 30,
    speed_hold_steps: int = 15,
    speed_scale: float = 1.0,
    speed_phase_offset_steps: int = 0,
    easing: EaseMode = "cosine",
) -> ContextTrajectory:
    """
    Convenience builder tuned for ATI experiments.

    By default, light changes more slowly than angular velocity so the two
    contexts do not peak at exactly the same cadence.
    """

    light_trajectory = PingPongTrajectory(
        anchor_values=light_values,
        transition_steps=light_transition_steps,
        hold_steps=light_hold_steps,
        easing=easing,
    )
    speed_trajectory = PingPongTrajectory(
        anchor_values=speed_values,
        transition_steps=speed_transition_steps,
        hold_steps=speed_hold_steps,
        phase_offset_steps=speed_phase_offset_steps,
        easing=easing,
    )
    return ContextTrajectory(
        light_trajectory=light_trajectory,
        speed_trajectory=speed_trajectory,
        speed_scale=speed_scale,
    )


def sample_context_trajectory(
    trajectory: ContextTrajectory,
    num_steps: int,
    sampling_period: int = 30,
    start_step: int = 0,
) -> dict[str, list[float]]:
    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")

    light_values = []
    speed_values = []
    for step in range(start_step, start_step + num_steps):
        context = trajectory.value_at(step)
        light_values.append(context["light_intensity"])
        speed_values.append(context["angular_velocity"])

    return {
        "steps": list(range(num_steps // sampling_period)),
        "light_intensity": light_values[::sampling_period],
        "angular_velocity": speed_values[::sampling_period],
    }


def plot_context_trajectory(
    trajectory: ContextTrajectory,
    num_steps: int,
    sampling_period: int = 30,
    start_step: int = 0,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt

    sampled = sample_context_trajectory(
        trajectory=trajectory,
        num_steps=num_steps,
        sampling_period=sampling_period,
        start_step=start_step,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axes[0].plot(sampled["steps"], sampled["light_intensity"], color="darkorange", linewidth=2.0)
    axes[0].set_title("Light Intensity")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sampled["steps"], sampled["angular_velocity"], color="royalblue", linewidth=2.0)
    axes[1].set_title("Speed")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Angular Velocity")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Context Trajectories", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    LAP_PERIOD = 30
    import numpy as np
    speed_vals = [1.0, 1.0, 1.0, 1.0, 1.0]# [1.0, 2.0, 1.0, 2.0, 1.0]
    light_vals = [500, 6000]
    step_trajectory = build_step_context_trajectory(
        light_values=light_vals,
        speed_values=[s * np.pi / 12 for s in speed_vals],
        light_hold_steps=LAP_PERIOD,
        speed_hold_steps=LAP_PERIOD,
        speed_phase_offset_steps=0,
    )
    # trajectory = build_default_context_trajectory(
    #     light_values=light_vals,
    #     speed_values=[s * np.pi / 12 for s in speed_vals],
    #     light_transition_steps=LAP_PERIOD * 200,
    #     speed_transition_steps=LAP_PERIOD * 10,
    #     light_hold_steps=LAP_PERIOD,
    #     speed_hold_steps=LAP_PERIOD,
    #     speed_phase_offset_steps=LAP_PERIOD,
    # )
    plot_context_trajectory(
        trajectory=step_trajectory,
        num_steps=LAP_PERIOD * 20,
        sampling_period=1,
        save_path="./context_trajectories.png",
    )
