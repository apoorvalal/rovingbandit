"""Epsilon-greedy policy."""

from typing import Optional
import numpy as np
from rovingbandit.core.policy import Policy


class EpsilonGreedy(Policy):
    """
    Epsilon-greedy policy.

    With probability epsilon: explore (select random arm)
    With probability 1-epsilon: exploit (select best arm)

    Reference: Sutton & Barto (2018), Reinforcement Learning
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        decay: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize epsilon-greedy policy.

        Args:
            n_arms: Number of arms
            epsilon: Exploration probability
            decay: If True, epsilon decays as epsilon_0 / sqrt(t)
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.epsilon_0 = epsilon
        self.epsilon = epsilon
        self.decay = decay

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using epsilon-greedy strategy.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # Update epsilon if decaying
        if self.decay and self.total_pulls > 0:
            self.epsilon = self.epsilon_0 / np.sqrt(self.total_pulls)

        # Explore with probability epsilon
        if self.rng.random() < self.epsilon:
            return self._select_random_arm()
        else:
            # Exploit - pick best arm (or random if all zeros)
            if self.total_pulls == 0:
                return self._select_random_arm()
            return self._select_greedy_arm()

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update counts and value estimates.

        Args:
            arm: Index of pulled arm
            reward: Observed reward
            cost: Cost incurred (ignored)
        """
        self.counts[arm] += 1
        self.total_pulls += 1
        self._update_value_incremental(arm, reward)
