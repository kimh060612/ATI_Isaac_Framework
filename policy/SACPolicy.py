from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


@dataclass(frozen=True)
class SACBoxSpec:
    low: tuple[float, ...]
    high: tuple[float, ...]
    log_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if len(self.low) != len(self.high):
            raise ValueError("SACBoxSpec low/high must have the same length.")
        if len(self.low) == 0:
            raise ValueError("SACBoxSpec must contain at least one dimension.")
        for idx in self.log_indices:
            if idx < 0 or idx >= len(self.low):
                raise ValueError(f"log index {idx} is out of range for box dimension {len(self.low)}.")

    @property
    def dim(self) -> int:
        return len(self.low)

    def normalize(self, values: np.ndarray | Iterable[float]) -> np.ndarray:
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1).copy()
        low_arr = np.asarray(self.low, dtype=np.float32).copy()
        high_arr = np.asarray(self.high, dtype=np.float32).copy()
        if values_arr.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} values, got shape {values_arr.shape}.")

        for idx in self.log_indices:
            values_arr[idx] = np.log(max(float(values_arr[idx]), EPS))
            low_arr[idx] = np.log(max(float(low_arr[idx]), EPS))
            high_arr[idx] = np.log(max(float(high_arr[idx]), EPS))

        denom = np.maximum(high_arr - low_arr, EPS)
        normalized = 2.0 * (values_arr - low_arr) / denom - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def clamp(self, values: np.ndarray | Iterable[float]) -> np.ndarray:
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        low_arr = np.asarray(self.low, dtype=np.float32)
        high_arr = np.asarray(self.high, dtype=np.float32)
        if values_arr.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} values, got shape {values_arr.shape}.")
        return np.clip(values_arr, low_arr, high_arr).astype(np.float32)

    def to_unit(self, values: np.ndarray | Iterable[float]) -> np.ndarray:
        values_arr = self.clamp(values)
        low_arr = np.asarray(self.low, dtype=np.float32)
        high_arr = np.asarray(self.high, dtype=np.float32)
        denom = np.maximum(high_arr - low_arr, EPS)
        return np.clip(2.0 * (values_arr - low_arr) / denom - 1.0, -1.0, 1.0).astype(np.float32)

    def from_unit(self, unit_values: np.ndarray | Iterable[float]) -> np.ndarray:
        unit_arr = np.asarray(unit_values, dtype=np.float32).reshape(-1)
        if unit_arr.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} unit values, got shape {unit_arr.shape}.")
        low_arr = np.asarray(self.low, dtype=np.float32)
        high_arr = np.asarray(self.high, dtype=np.float32)
        clipped = np.clip(unit_arr, -1.0, 1.0)
        return (low_arr + 0.5 * (clipped + 1.0) * (high_arr - low_arr)).astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, context_dim: int, action_dim: int):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.context = np.zeros((capacity, context_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_context = np.zeros((capacity, context_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        context: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_context: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.ptr] = state
        self.context[self.ptr] = context
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = float(reward)
        self.next_state[self.ptr] = next_state
        self.next_context[self.ptr] = next_context
        self.done[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "state": torch.as_tensor(self.state[indices], device=device),
            "context": torch.as_tensor(self.context[indices], device=device),
            "action": torch.as_tensor(self.action[indices], device=device),
            "reward": torch.as_tensor(self.reward[indices], device=device),
            "next_state": torch.as_tensor(self.next_state[indices], device=device),
            "next_context": torch.as_tensor(self.next_context[indices], device=device),
            "done": torch.as_tensor(self.done[indices], device=device),
        }

    def __len__(self) -> int:
        return self.size


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dims, hidden_dims[-1])
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mean)
        return action, log_prob, deterministic_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class L2ContextualSACRGBCamPolicy:
    """
    Lightweight online off-policy SAC for RGB camera control.

    State is the controllable camera state: [exposure_time, iso].
    Context is action-independent deployment information: for the current scene
    this is expected to be [acceleration_magnitude, gyro_magnitude, lux].
    Action is a continuous camera delta: [delta_exposure_time, delta_iso].
    """

    def __init__(
        self,
        state_spec: SACBoxSpec,
        context_spec: SACBoxSpec,
        action_spec: SACBoxSpec,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        batch_size: int = 64,
        replay_size: int = 100_000,
        random_steps: int = 10,
        update_after: int = 64,
        update_every: int = 1,
        gradient_steps: int = 1,
        reward_scale: float = 1.0,
        target_entropy: float | None = None,
        device: str | torch.device | None = None,
        random_seed: int | None = None,
    ):
        self.state_spec = state_spec
        self.context_spec = context_spec
        self.action_spec = action_spec
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.random_steps = int(random_steps)
        self.update_after = int(update_after)
        self.update_every = int(update_every)
        self.gradient_steps = int(gradient_steps)
        self.reward_scale = float(reward_scale)
        self.rng = np.random.default_rng(random_seed)
        if random_seed is not None:
            torch.manual_seed(int(random_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(random_seed))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.obs_dim = self.state_spec.dim + self.context_spec.dim
        self.action_dim = self.action_spec.dim

        self.actor = GaussianActor(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.q1 = QNetwork(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.q1_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=float(critic_lr))
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=float(critic_lr))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(alpha_lr))
        self.target_entropy = float(target_entropy if target_entropy is not None else -self.action_dim)

        self.replay = ReplayBuffer(
            capacity=int(replay_size),
            state_dim=self.state_spec.dim,
            context_dim=self.context_spec.dim,
            action_dim=self.action_dim,
        )
        self.total_steps = 0
        self.total_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def build_observation(self, state: np.ndarray | Iterable[float], context: np.ndarray | Iterable[float]) -> np.ndarray:
        state_norm = self.state_spec.normalize(state)
        context_norm = self.context_spec.normalize(context)
        return np.concatenate([state_norm, context_norm], axis=0).astype(np.float32)

    def _obs_tensor(self, state: np.ndarray | Iterable[float], context: np.ndarray | Iterable[float]) -> torch.Tensor:
        obs = self.build_observation(state, context)
        return torch.as_tensor(obs, device=self.device).unsqueeze(0)

    def _batch_obs(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state_np = state.detach().cpu().numpy()
        context_np = context.detach().cpu().numpy()
        obs = [
            self.build_observation(s, c)
            for s, c in zip(state_np, context_np)
        ]
        return torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)

    def select_action(
        self,
        state: np.ndarray | Iterable[float],
        context: np.ndarray | Iterable[float],
        *,
        deterministic: bool = False,
        explore: bool = True,
    ) -> tuple[np.ndarray, dict[str, float]]:
        use_random = explore and not deterministic and self.total_steps < self.random_steps
        if use_random:
            unit_action = self.rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)
            raw_action = self.action_spec.from_unit(unit_action)
            return raw_action, {
                "action_unit_0": float(unit_action[0]),
                "action_unit_1": float(unit_action[1]) if self.action_dim > 1 else 0.0,
                "log_prob": 0.0,
                "mode": "random_warmup",
            }

        with torch.no_grad():
            obs = self._obs_tensor(state, context)
            sampled_action, log_prob, deterministic_action = self.actor.sample(obs)
            unit_action_tensor = deterministic_action if deterministic else sampled_action
            unit_action = unit_action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            raw_action = self.action_spec.from_unit(unit_action)
            return raw_action, {
                "action_unit_0": float(unit_action[0]),
                "action_unit_1": float(unit_action[1]) if self.action_dim > 1 else 0.0,
                "log_prob": float(log_prob.item()),
                "mode": "deterministic" if deterministic else "policy",
            }

    def observe(
        self,
        *,
        state: np.ndarray | Iterable[float],
        context: np.ndarray | Iterable[float],
        action: np.ndarray | Iterable[float],
        reward: float,
        next_state: np.ndarray | Iterable[float],
        next_context: np.ndarray | Iterable[float],
        done: bool,
    ) -> dict[str, float]:
        self.replay.add(
            state=self.state_spec.clamp(state),
            context=self.context_spec.clamp(context),
            action=self.action_spec.clamp(action),
            reward=float(reward),
            next_state=self.state_spec.clamp(next_state),
            next_context=self.context_spec.clamp(next_context),
            done=bool(done),
        )
        self.total_steps += 1
        if len(self.replay) < self.update_after or self.total_steps % self.update_every != 0:
            return {
                "replay_size": float(len(self.replay)),
                "updates": 0.0,
            }
        return self.update(self.gradient_steps)

    def update(self, gradient_steps: int | None = None) -> dict[str, float]:
        gradient_steps = int(gradient_steps if gradient_steps is not None else self.gradient_steps)
        metrics: dict[str, float] = {}
        if len(self.replay) < max(1, self.batch_size):
            return {
                "replay_size": float(len(self.replay)),
                "updates": 0.0,
            }

        for _ in range(gradient_steps):
            batch = self.replay.sample(self.batch_size, self.device)
            obs = self._batch_obs(batch["state"], batch["context"])
            next_obs = self._batch_obs(batch["next_state"], batch["next_context"])
            action_unit = torch.as_tensor(
                np.asarray([self.action_spec.to_unit(a) for a in batch["action"].detach().cpu().numpy()], dtype=np.float32),
                device=self.device,
            )
            reward = batch["reward"] * self.reward_scale
            done = batch["done"]

            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_obs)
                target_q1 = self.q1_target(next_obs, next_action)
                target_q2 = self.q2_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
                backup = reward + (1.0 - done) * self.gamma * target_q

            q1_loss = F.mse_loss(self.q1(obs, action_unit), backup)
            q2_loss = F.mse_loss(self.q2(obs, action_unit), backup)
            self.q1_optimizer.zero_grad(set_to_none=True)
            q1_loss.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.zero_grad(set_to_none=True)
            q2_loss.backward()
            self.q2_optimizer.step()

            sampled_action, log_prob, _ = self.actor.sample(obs)
            q_pi = torch.min(self.q1(obs, sampled_action), self.q2(obs, sampled_action))
            actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            self.total_updates += 1

            metrics = {
                "replay_size": float(len(self.replay)),
                "updates": float(self.total_updates),
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha_loss": float(alpha_loss.item()),
                "alpha": float(self.alpha.item()),
                "mean_log_prob": float(log_prob.mean().item()),
                "mean_q_pi": float(q_pi.mean().item()),
            }
        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str | torch.device | None = None) -> None:
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"])
        self.q2.load_state_dict(payload["q2"])
        self.q1_target.load_state_dict(payload["q1_target"])
        self.q2_target.load_state_dict(payload["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.q1_optimizer.load_state_dict(payload["q1_optimizer"])
        self.q2_optimizer.load_state_dict(payload["q2_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.total_steps = int(payload.get("total_steps", 0))
        self.total_updates = int(payload.get("total_updates", 0))
