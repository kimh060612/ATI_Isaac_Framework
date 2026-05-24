from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad_and_value, vmap

from policy.BasePolicy import BaseCMABPolicy
from policy.SensorParams import SensorParamSpace


class NeuralBanditReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")
        self.capacity = int(capacity)
        self.rng = rng
        self._storage = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        action_feature: np.ndarray,
        base_context: np.ndarray,
        action: Tuple[int, int],
        reward: float,
    ) -> None:
        self._storage.append(
            {
                "action_feature": np.asarray(action_feature, dtype=np.float32),
                "base_context": np.asarray(base_context, dtype=np.float32),
                "action": (int(action[0]), int(action[1])),
                "reward": float(reward),
            }
        )

    def sample(self, batch_size: int) -> list[dict]:
        if not self._storage:
            return []
        size = min(int(batch_size), len(self._storage))
        indices = self.rng.choice(len(self._storage), size=size, replace=False)
        return [self._storage[int(idx)] for idx in indices]

    def all(self) -> list[dict]:
        return list(self._storage)


class RewardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        dims = (int(input_dim),) + tuple(int(dim) for dim in hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralLinearRewardModel(nn.Module):
    def __init__(
        self,
        base_context_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...],
        feature_dim: int,
    ):
        super().__init__()
        dims = (int(base_context_dim),) + tuple(int(dim) for dim in hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], int(feature_dim)))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.reward_head = nn.Sequential(
            nn.Linear(int(feature_dim) + int(action_dim), int(feature_dim)),
            nn.ReLU(),
            nn.Linear(int(feature_dim), 1),
        )

    def encode(self, base_context: torch.Tensor) -> torch.Tensor:
        return self.encoder(base_context)

    def forward(self, base_context: torch.Tensor, action_values: torch.Tensor) -> torch.Tensor:
        phi = self.encode(base_context)
        x = torch.cat([phi, action_values], dim=-1)
        return self.reward_head(x).squeeze(-1)


class _BaseNeuralBanditPolicy(BaseCMABPolicy, metaclass=ABCMeta):
    ACTIONS: Tuple[Tuple[int, int], ...] = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def _init_common(
        self,
        *,
        sensor_config: Optional[SensorParamSpace],
        forced_exploration_prob: float,
        max_acceleration_context: float,
        max_gyro_context: float,
        max_light_context: float,
        hidden_dims: Tuple[int, ...],
        replay_capacity: int,
        batch_size: int,
        train_every: int,
        gradient_steps: int,
        learning_rate: float,
        weight_decay: float,
        device: Optional[str],
    ) -> None:
        self.cfg = sensor_config if sensor_config is not None else SensorParamSpace()
        self.n_exposure = len(self.cfg.exposure_values)
        self.n_iso = len(self.cfg.iso_values)
        self.action_to_index = {
            action: idx
            for idx, action in enumerate(self.ACTIONS)
        }
        self.forced_exploration_prob = float(forced_exploration_prob)
        self.max_acceleration_context = float(max_acceleration_context)
        self.max_gyro_context = float(max_gyro_context)
        self.max_light_context = float(max_light_context)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.train_every = max(int(train_every), 1)
        self.gradient_steps = max(int(gradient_steps), 0)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.base_context_dim = 3
        self.action_dim = 2
        self.action_feature_dim = self.base_context_dim + self.action_dim
        self.update_count = 0
        self.last_train_loss = None

    @staticmethod
    def _scale_clipped(value: float, max_value: float) -> float:
        if max_value <= 0:
            return 0.0
        return float(np.clip(float(value) / float(max_value), 0.0, 1.0))

    def _normalize_index(self, idx: int, max_idx: int) -> float:
        if max_idx <= 0:
            return 0.0
        return float(np.clip(float(idx) / float(max_idx), 0.0, 1.0))

    def build_base_context(self, context_information: dict) -> np.ndarray:
        acceleration = float(context_information.get("acceleration_magnitude", 0.0))
        gyro = float(
            context_information.get(
                "gyro_magnitude",
                context_information.get("angular_velocity", 0.0),
            )
        )
        light = max(float(context_information.get("light_intensity", 1.0)), 1.0)
        # exposure_idx = int(context_information.get("exposure_idx", 0))
        # iso_idx = int(context_information.get("iso_idx", 0))

        return np.asarray(
            [
                self._scale_clipped(acceleration, self.max_acceleration_context),
                self._scale_clipped(abs(gyro), self.max_gyro_context),
                self._scale_clipped(
                    np.log1p(light),
                    np.log1p(max(self.max_light_context, 1.0)),
                ),
                # self._normalize_index(exposure_idx, self.n_exposure - 1),
                # self._normalize_index(iso_idx, self.n_iso - 1),
            ],
            dtype=np.float32,
        )

    def build_action_feature(self, context_information: dict, action: Tuple[int, int]) -> np.ndarray:
        base_context = self.build_base_context(context_information)
        action_values = np.asarray([float(action[0]), float(action[1])], dtype=np.float32)
        return np.concatenate([base_context, action_values]).astype(np.float32)

    def _action_index(self, action: Tuple[int, int]) -> int:
        return self.action_to_index[action]

    def valid_actions(self, exposure_idx: int, iso_idx: int) -> List[Tuple[int, int]]:
        valid = []
        for de, di in self.action_space:
            next_e = int(exposure_idx) + int(de)
            next_i = int(iso_idx) + int(di)
            if 0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso:
                valid.append((int(de), int(di)))
        return valid

    @abstractmethod
    def observe(self, selection, action):
        raise NotImplementedError("This method is not used in this policy. Rewards are observed externally and passed to observe().")

    def transition(
        self,
        exposure_idx: int,
        iso_idx: int,
        action: Tuple[int, int],
    ) -> Tuple[int, int]:
        de, di = action
        next_e = int(exposure_idx) + int(de)
        next_i = int(iso_idx) + int(di)
        if not (0 <= next_e < self.n_exposure and 0 <= next_i < self.n_iso):
            raise ValueError(f"Invalid action {action} for current state {(exposure_idx, iso_idx)}")
        return next_e, next_i

    def step(self, context_information: dict, observations: dict) -> dict:
        reward = float(observations.get("reward", 0.0))
        selection = self.select_action(context_information)
        update_info = self.observe(selection, reward)
        next_exposure_idx, next_iso_idx = self.transition(
            context_information["exposure_idx"],
            context_information["iso_idx"],
            selection["chosen_action"],
        )
        record = {
            **context_information,
            "action": selection["chosen_action"],
            "next_exposure_idx": next_exposure_idx,
            "next_iso_idx": next_iso_idx,
            "chosen_score": selection["chosen_score"],
            "chosen_mean": selection["chosen_mean"],
            "chosen_bonus": selection["chosen_bonus"],
            "selection_mode": selection["selection_mode"],
            "forced_explore": selection["forced_explore"],
            "update_info": update_info,
        }
        self.history.append(record)
        return record

    def _train_batch_tensors(self, batch: list[dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_features = np.stack([item["action_feature"] for item in batch], axis=0)
        base_contexts = np.stack([item["base_context"] for item in batch], axis=0)
        rewards = np.asarray([item["reward"] for item in batch], dtype=np.float32)
        return (
            torch.as_tensor(action_features, dtype=torch.float32, device=self.device),
            torch.as_tensor(base_contexts, dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
        )

    def _common_metadata(self, metadata: Optional[Dict]) -> Dict:
        payload = {
            "policy_type": self.__class__.__name__,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "forced_exploration_prob": self.forced_exploration_prob,
            "max_acceleration_context": self.max_acceleration_context,
            "max_gyro_context": self.max_gyro_context,
            "max_light_context": self.max_light_context,
            "hidden_dims": list(self.hidden_dims),
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "train_every": self.train_every,
            "gradient_steps": self.gradient_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "action_space": [list(action) for action in self.action_space],
            "exposure_values": [float(value) for value in self.cfg.exposure_values],
            "iso_values": [float(value) for value in self.cfg.iso_values],
        }
        if metadata:
            payload.update(metadata)
        return payload


class L2NeuralUCBPolicy(_BaseNeuralBanditPolicy):
    """
    NeuralUCB with an MLP reward predictor and gradient-based UCB features.

    For each candidate action, the score is:
        f(x_a; theta) + alpha * sqrt(g_a^T Z^{-1} g_a)
    where g_a is the flattened gradient of f with respect to the MLP
    parameters.  The MLP is trained from replay after observed rewards.
    """

    def __init__(
        self,
        sensor_names,
        sensor_config: Optional[SensorParamSpace],
        reward_function,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        random_seed: Optional[int] = None,
        forced_exploration_prob: float = 0.05,
        max_acceleration_context: float = 5.0,
        max_gyro_context: float = 3.0,
        max_light_context: float = 10000.0,
        hidden_dims: Tuple[int, ...] = (32,),
        replay_capacity: int = 5000,
        batch_size: int = 64,
        train_every: int = 1,
        gradient_steps: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self._init_common(
            sensor_config=sensor_config,
            forced_exploration_prob=forced_exploration_prob,
            max_acceleration_context=max_acceleration_context,
            max_gyro_context=max_gyro_context,
            max_light_context=max_light_context,
            hidden_dims=hidden_dims,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            train_every=train_every,
            gradient_steps=gradient_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        super().__init__(
            sensor_names=sensor_names,
            action_space=list(self.ACTIONS),
            dim_context=self.action_feature_dim,
            reward_function=reward_function,
            alpha=alpha,
            lambda_reg=lambda_reg,
            random_seed=random_seed,
        )

    def _defining_parameters(self) -> None:
        self.model = RewardMLP(self.action_feature_dim, self.hidden_dims).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.replay = NeuralBanditReplayBuffer(self.replay_capacity, self.rng)
        self.num_model_params = sum(param.numel() for param in self.model.parameters())
        self.Z = self.lambda_reg * np.eye(self.num_model_params, dtype=np.float64)

    def _batched_ucb_scores(
        self,
        action_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        x = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        param_names = tuple(params.keys())

        def model_output(params, buffers, action_feature):
            return functional_call(
                self.model,
                (params, buffers),
                (action_feature.unsqueeze(0),),
            ).squeeze(0)

        grads, means = vmap(
            grad_and_value(model_output),
            in_dims=(None, None, 0),
        )(params, buffers, x)
        gradient_tensor = torch.cat(
            [grads[name].reshape(x.shape[0], -1) for name in param_names],
            dim=1,
        )
        width = float(self.hidden_dims[0]) if self.hidden_dims else 1.0
        gradient_features = (
            gradient_tensor.detach().cpu().numpy().astype(np.float64)
            / np.sqrt(max(width, 1.0))
        )
        means_np = means.detach().cpu().numpy().astype(np.float64)
        z_inv_g = np.linalg.solve(self.Z, gradient_features.T).T
        bonuses = self.alpha * np.sqrt(
            np.maximum(np.einsum("ij,ij->i", gradient_features, z_inv_g), 0.0)
        )
        scores = means_np + bonuses
        return scores, means_np, bonuses, gradient_features

    def ucb_score(self, action_feature: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        scores, means, bonuses, gradient_features = self._batched_ucb_scores(action_feature[None, :])
        return (
            float(scores[0]),
            float(means[0]),
            float(bonuses[0]),
            gradient_features[0],
        )

    def select_action(
        self,
        context_information: dict,
        tie_break_random: bool = True,
        force_explore: Optional[bool] = None,
    ) -> dict:
        base_context = self.build_base_context(context_information)
        candidates = self.valid_actions(
            context_information["exposure_idx"],
            context_information["iso_idx"],
        )
        if not candidates:
            raise RuntimeError("No valid actions available.")

        action_features = np.concatenate(
            [
                np.repeat(base_context[None, :], len(candidates), axis=0),
                np.asarray(candidates, dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        scores, means, bonuses, gradient_features = self._batched_ucb_scores(action_features)

        rows = []
        best_score = -np.inf
        best_rows = []
        for idx, action in enumerate(candidates):
            score = float(scores[idx])
            row = {
                "action": action,
                "z": gradient_features[idx],
                "train_x": action_features[idx],
                "base_context": base_context,
                "score": score,
                "mean": float(means[idx]),
                "bonus": float(bonuses[idx]),
            }
            rows.append(row)
            if score > best_score + 1e-12:
                best_score = score
                best_rows = [row]
            elif abs(score - best_score) <= 1e-12:
                best_rows.append(row)

        if force_explore is None:
            force_explore = bool(self.rng.random() < self.forced_exploration_prob)

        if force_explore:
            chosen = rows[int(self.rng.integers(len(rows)))]
            selection_mode = "forced_explore"
        elif tie_break_random and len(best_rows) > 1:
            chosen = best_rows[int(self.rng.integers(len(best_rows)))]
            selection_mode = "ucb_tie_random"
        else:
            chosen = best_rows[0]
            selection_mode = "neural_ucb"

        return {
            "context": base_context,
            "candidates": rows,
            "chosen_action": chosen["action"],
            "chosen_z": chosen["z"],
            "chosen_train_x": chosen["train_x"],
            "chosen_base_context": chosen["base_context"],
            "chosen_score": chosen["score"],
            "chosen_mean": chosen["mean"],
            "chosen_bonus": chosen["bonus"],
            "selection_mode": selection_mode,
            "forced_explore": bool(force_explore),
        }

    def train_reward_model(self) -> Optional[float]:
        if len(self.replay) == 0 or self.gradient_steps <= 0:
            return None
        losses = []
        self.model.train()
        for _ in range(self.gradient_steps):
            batch = self.replay.sample(self.batch_size)
            action_features, _, rewards = self._train_batch_tensors(batch)
            pred = self.model(action_features)
            loss = F.mse_loss(pred, rewards)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        self.last_train_loss = float(np.mean(losses)) if losses else None
        return self.last_train_loss

    def observe(self, selection: dict, reward: float) -> dict:
        action = selection["chosen_action"]
        action_idx = self._action_index(action)
        g = np.asarray(selection["chosen_z"], dtype=np.float64)
        self.Z += np.outer(g, g)
        self.replay.add(
            selection["chosen_train_x"],
            selection["chosen_base_context"],
            action,
            reward,
        )
        self.update_count += 1
        train_loss = None
        if self.update_count % self.train_every == 0:
            train_loss = self.train_reward_model()
        return {
            "action": action,
            "action_idx": action_idx,
            "reward": float(reward),
            "train_loss": train_loss,
            "replay_size": len(self.replay),
            "num_model_params": self.num_model_params,
        }

    def update_parameters(self, z: np.ndarray, reward: float, action: Tuple[int, int]) -> dict:
        selection = {
            "chosen_action": action,
            "chosen_z": z,
            "chosen_train_x": z,
            "chosen_base_context": np.asarray(z[: self.base_context_dim], dtype=np.float32),
        }
        return self.observe(selection, reward)

    def save(self, checkpoint_path: str, metadata: Optional[Dict] = None) -> str:
        checkpoint_path = os.path.abspath(checkpoint_path)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        payload = {
            "metadata": self._common_metadata(metadata),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "Z": torch.as_tensor(self.Z, dtype=torch.float64),
            "update_count": self.update_count,
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load(self, checkpoint_path: str) -> dict:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.Z = payload["Z"].detach().cpu().numpy().astype(np.float64)
        self.update_count = int(payload.get("update_count", 0))
        return dict(payload.get("metadata", {}))


class L2NeuralLinearUCBPolicy(_BaseNeuralBanditPolicy):
    """
    Neural-linear UCB variant.

    A replay-trained MLP encoder maps the observed context into phi(context).
    The exploration and deployed policy remain action-disjoint LinUCB:
        theta_a^T phi + alpha * sqrt(phi^T A_a^{-1} phi).
    """

    def __init__(
        self,
        sensor_names,
        sensor_config: Optional[SensorParamSpace],
        reward_function,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        random_seed: Optional[int] = None,
        forced_exploration_prob: float = 0.05,
        max_acceleration_context: float = 5.0,
        max_gyro_context: float = 3.0,
        max_light_context: float = 10000.0,
        hidden_dims: Tuple[int, ...] = (64,),
        neural_feature_dim: int = 32,
        replay_capacity: int = 5000,
        batch_size: int = 64,
        train_every: int = 1,
        gradient_steps: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.neural_feature_dim = int(neural_feature_dim)
        self._init_common(
            sensor_config=sensor_config,
            forced_exploration_prob=forced_exploration_prob,
            max_acceleration_context=max_acceleration_context,
            max_gyro_context=max_gyro_context,
            max_light_context=max_light_context,
            hidden_dims=hidden_dims,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            train_every=train_every,
            gradient_steps=gradient_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        super().__init__(
            sensor_names=sensor_names,
            action_space=list(self.ACTIONS),
            dim_context=self.neural_feature_dim,
            reward_function=reward_function,
            alpha=alpha,
            lambda_reg=lambda_reg,
            random_seed=random_seed,
        )

    def _defining_parameters(self) -> None:
        self.model = NeuralLinearRewardModel(
            base_context_dim=self.base_context_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            feature_dim=self.neural_feature_dim,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.replay = NeuralBanditReplayBuffer(self.replay_capacity, self.rng)
        self.A = np.stack(
            [
                self.lambda_reg * np.eye(self.neural_feature_dim, dtype=np.float64)
                for _ in range(self.num_actions)
            ],
            axis=0,
        )
        self.b = np.zeros((self.num_actions, self.neural_feature_dim), dtype=np.float64)

    def encode_context(self, base_context: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(base_context[None, :], dtype=torch.float32, device=self.device)
            phi = self.model.encode(x).squeeze(0)
        return phi.detach().cpu().numpy().astype(np.float64)

    def _reward_prediction(self, base_context: np.ndarray, action: Tuple[int, int]) -> float:
        self.model.eval()
        with torch.no_grad():
            base = torch.as_tensor(base_context[None, :], dtype=torch.float32, device=self.device)
            action_values = torch.as_tensor([[float(action[0]), float(action[1])]], dtype=torch.float32, device=self.device)
            return float(self.model(base, action_values).item())

    def theta_hat(self, action: Tuple[int, int]) -> np.ndarray:
        action_idx = self._action_index(action)
        return np.linalg.solve(self.A[action_idx], self.b[action_idx])

    def ucb_score(self, phi: np.ndarray, action: Tuple[int, int]) -> Tuple[float, float, float]:
        action_idx = self._action_index(action)
        theta = self.theta_hat(action)
        mean = float(phi @ theta)
        A_inv_phi = np.linalg.solve(self.A[action_idx], phi)
        bonus = float(self.alpha * np.sqrt(max(float(phi @ A_inv_phi), 0.0)))
        return mean + bonus, mean, bonus

    def select_action(
        self,
        context_information: dict,
        tie_break_random: bool = True,
        force_explore: Optional[bool] = None,
    ) -> dict:
        base_context = self.build_base_context(context_information)
        phi = self.encode_context(base_context)
        candidates = self.valid_actions(
            context_information["exposure_idx"],
            context_information["iso_idx"],
        )
        if not candidates:
            raise RuntimeError("No valid actions available.")

        rows = []
        best_score = -np.inf
        best_rows = []
        for action in candidates:
            action_feature = self.build_action_feature(context_information, action)
            score, mean, bonus = self.ucb_score(phi, action)
            reward_prediction = self._reward_prediction(base_context, action)
            row = {
                "action": action,
                "z": phi,
                "train_x": action_feature,
                "base_context": base_context,
                "score": score,
                "mean": mean,
                "bonus": bonus,
                "reward_prediction": reward_prediction,
            }
            rows.append(row)
            if score > best_score + 1e-12:
                best_score = score
                best_rows = [row]
            elif abs(score - best_score) <= 1e-12:
                best_rows.append(row)

        if force_explore is None:
            force_explore = bool(self.rng.random() < self.forced_exploration_prob)

        if force_explore:
            chosen = rows[int(self.rng.integers(len(rows)))]
            selection_mode = "forced_explore"
        elif tie_break_random and len(best_rows) > 1:
            chosen = best_rows[int(self.rng.integers(len(best_rows)))]
            selection_mode = "ucb_tie_random"
        else:
            chosen = best_rows[0]
            selection_mode = "neural_linear_ucb"

        return {
            "context": base_context,
            "candidates": rows,
            "chosen_action": chosen["action"],
            "chosen_z": chosen["z"],
            "chosen_train_x": chosen["train_x"],
            "chosen_base_context": chosen["base_context"],
            "chosen_score": chosen["score"],
            "chosen_mean": chosen["mean"],
            "chosen_bonus": chosen["bonus"],
            "chosen_reward_prediction": chosen["reward_prediction"],
            "selection_mode": selection_mode,
            "forced_explore": bool(force_explore),
        }

    def train_reward_model(self) -> Optional[float]:
        if len(self.replay) == 0 or self.gradient_steps <= 0:
            return None
        losses = []
        self.model.train()
        for _ in range(self.gradient_steps):
            batch = self.replay.sample(self.batch_size)
            _, base_contexts, rewards = self._train_batch_tensors(batch)
            action_values = np.asarray(
                [[float(item["action"][0]), float(item["action"][1])] for item in batch],
                dtype=np.float32,
            )
            action_values = torch.as_tensor(action_values, dtype=torch.float32, device=self.device)
            pred = self.model(base_contexts, action_values)
            loss = F.mse_loss(pred, rewards)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        self.last_train_loss = float(np.mean(losses)) if losses else None
        self._rebuild_linear_models_from_replay()
        return self.last_train_loss

    def _rebuild_linear_models_from_replay(self) -> None:
        self.A = np.stack(
            [
                self.lambda_reg * np.eye(self.neural_feature_dim, dtype=np.float64)
                for _ in range(self.num_actions)
            ],
            axis=0,
        )
        self.b = np.zeros((self.num_actions, self.neural_feature_dim), dtype=np.float64)
        for item in self.replay.all():
            action = item["action"]
            action_idx = self._action_index(action)
            phi = self.encode_context(np.asarray(item["base_context"], dtype=np.float32))
            self.A[action_idx] += np.outer(phi, phi)
            self.b[action_idx] += float(item["reward"]) * phi

    def observe(self, selection: dict, reward: float) -> dict:
        action = selection["chosen_action"]
        action_idx = self._action_index(action)
        phi = np.asarray(selection["chosen_z"], dtype=np.float64)
        self.A[action_idx] += np.outer(phi, phi)
        self.b[action_idx] += float(reward) * phi
        self.replay.add(
            selection["chosen_train_x"],
            selection["chosen_base_context"],
            action,
            reward,
        )
        self.update_count += 1
        train_loss = None
        if self.update_count % self.train_every == 0:
            train_loss = self.train_reward_model()
        return {
            "action": action,
            "action_idx": action_idx,
            "reward": float(reward),
            "train_loss": train_loss,
            "replay_size": len(self.replay),
            "neural_feature_dim": self.neural_feature_dim,
        }

    def update_parameters(self, z: np.ndarray, reward: float, action: Tuple[int, int]) -> dict:
        selection = {
            "chosen_action": action,
            "chosen_z": z,
            "chosen_train_x": np.zeros(self.action_feature_dim, dtype=np.float32),
            "chosen_base_context": np.zeros(self.base_context_dim, dtype=np.float32),
        }
        return self.observe(selection, reward)

    def save(self, checkpoint_path: str, metadata: Optional[Dict] = None) -> str:
        checkpoint_path = os.path.abspath(checkpoint_path)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        payload_metadata = self._common_metadata(metadata)
        payload_metadata["neural_feature_dim"] = self.neural_feature_dim
        payload = {
            "metadata": payload_metadata,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "A": torch.as_tensor(self.A, dtype=torch.float64),
            "b": torch.as_tensor(self.b, dtype=torch.float64),
            "update_count": self.update_count,
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load(self, checkpoint_path: str) -> dict:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.A = payload["A"].detach().cpu().numpy().astype(np.float64)
        self.b = payload["b"].detach().cpu().numpy().astype(np.float64)
        self.update_count = int(payload.get("update_count", 0))
        return dict(payload.get("metadata", {}))
