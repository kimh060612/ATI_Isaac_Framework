"""
ATI Sensor Control CMAB Policy Abstract Base Class
- Roles
    - Define the interface for sensor control policies.
    - Provide sensor parameter definition interfaces for different types of sensors (e.g. cameras).
    - Allow for different policy implementations (e.g. UCB, random, etc.) that can be swapped in and out.
"""
from typing import List, Dict, Callable, Tuple
from abc import ABCMeta, abstractmethod
import numpy as np

class BaseCMABPolicy(metaclass=ABCMeta):
    def __init__(
        self,
        sensor_names: list[str],
        action_space: list[tuple],
        dim_context: int,
        reward_function: Callable[[dict], Dict[str, float]],
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        random_seed: int | None = None,
    ):
        self.sensor_names = sensor_names
        self.action_space = action_space
        self.dim_context = dim_context
        self.num_actions = len(action_space)
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Defining parameters
        self.history: List[Dict] = []
        self.reward_function = reward_function
        self.rng = np.random.default_rng(random_seed)
        self._defining_parameters()
                
    def _defining_parameters(self):
        """Initialize parameters for the policy. This can be overridden by subclasses if needed."""
        self.A = self.lambda_reg * np.eye(self.dim_context, dtype=np.float64)
        self.b = np.zeros(self.dim_context, dtype=np.float64)

    def theta_hat(self) -> np.ndarray:
        """
        Solve A theta = b instead of explicitly computing inverse for theta.
        """
        return np.linalg.solve(self.A, self.b)

    def ucb_score(self, z: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns:
            score = mean + alpha * bonus
            mean
            bonus
        """
        theta = self.theta_hat()
        mean = float(z @ theta)
        # solve(A, z) is numerically preferable to inv(A) @ z
        A_inv_z = np.linalg.solve(self.A, z)
        bonus = float(self.alpha * np.sqrt(z @ A_inv_z))
        score = mean + bonus
        return score, mean, bonus

    @abstractmethod
    def select_action(
        self, 
        context_information: dict,
        tie_break_random: bool = True,
    ) -> dict[str, float | int]:
        raise NotImplementedError()
    
    @abstractmethod
    def update_parameters(
        self, 
        z: np.ndarray, 
        reward: float
    ):
        # Default implementation does nothing, can be overridden by subclasses if needed
        raise NotImplementedError()
    
    @abstractmethod
    def step(
        self,
        context_information: dict,
        observations: np.ndarray,
    ):
        # Function for one step of interaction: select action, apply it, observe reward, and update parameters
        raise NotImplementedError()
    