"""Bandit environment implementation."""

from typing import Optional, Tuple, Callable
import numpy as np


class BanditEnvironment:
    """
    Represents a multi-armed bandit environment.

    Attributes:
        n_arms: Number of available arms
        arm_means: True means for each arm (if known/simulated)
        costs: Cost per pull for each arm
        arm_groups: Group membership for representation constraints
        reward_fn: Custom reward function (arm_idx, rng) -> reward
    """

    def __init__(
        self,
        n_arms: int,
        arm_means: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        arm_groups: Optional[np.ndarray] = None,
        reward_fn: Optional[Callable[[int, np.random.Generator], float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize bandit environment.

        Args:
            n_arms: Number of arms
            arm_means: Mean reward for each arm (for Bernoulli bandits)
            costs: Cost to pull each arm (default: all 1.0)
            arm_groups: Group membership array for representation constraints
            reward_fn: Custom reward function, overrides arm_means if provided
            seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.arm_means = arm_means
        self.costs = costs if costs is not None else np.ones(n_arms)
        self.arm_groups = arm_groups
        self.reward_fn = reward_fn
        self.rng = np.random.default_rng(seed)

        # Validate inputs
        if arm_means is not None:
            assert len(arm_means) == n_arms, "arm_means length must match n_arms"
            assert np.all((arm_means >= 0) & (arm_means <= 1)), "arm_means must be in [0,1]"

        if costs is not None:
            assert len(costs) == n_arms, "costs length must match n_arms"
            assert np.all(costs > 0), "costs must be positive"

        if arm_groups is not None:
            assert len(arm_groups) == n_arms, "arm_groups length must match n_arms"

    def pull(self, arm: int) -> Tuple[float, float]:
        """
        Pull an arm and observe reward.

        Args:
            arm: Index of arm to pull (0 to n_arms-1)

        Returns:
            Tuple of (reward, cost)
        """
        assert 0 <= arm < self.n_arms, f"Invalid arm index: {arm}"

        if self.reward_fn is not None:
            reward = self.reward_fn(arm, self.rng)
        elif self.arm_means is not None:
            # Bernoulli reward
            reward = float(self.rng.binomial(1, self.arm_means[arm]))
        else:
            raise ValueError("Either arm_means or reward_fn must be provided")

        cost = float(self.costs[arm])
        return reward, cost

    def get_optimal_arm(self, objective: str = "reward") -> int:
        """
        Get the optimal arm for a given objective.

        Args:
            objective: Either 'reward', 'cost', or 'reward_per_cost'

        Returns:
            Index of optimal arm
        """
        if self.arm_means is None:
            raise ValueError("Cannot determine optimal arm without arm_means")

        if objective == "reward":
            return int(np.argmax(self.arm_means))
        elif objective == "cost":
            return int(np.argmin(self.costs))
        elif objective == "reward_per_cost":
            return int(np.argmax(self.arm_means / self.costs))
        else:
            raise ValueError(f"Unknown objective: {objective}")

    def get_optimal_reward(self, objective: str = "reward") -> float:
        """Get the expected reward from optimal arm."""
        if self.arm_means is None:
            raise ValueError("Cannot determine optimal reward without arm_means")

        optimal_arm = self.get_optimal_arm(objective)
        return float(self.arm_means[optimal_arm])

    def reset_rng(self, seed: Optional[int] = None):
        """Reset the random number generator."""
        self.rng = np.random.default_rng(seed)
