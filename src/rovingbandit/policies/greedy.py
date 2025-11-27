"""Greedy policy - always exploit best arm."""

from typing import Optional
import numpy as np
from rovingbandit.core.policy import Policy


class GreedyPolicy(Policy):
    """
    Greedy policy - always selects arm with highest estimated value.

    Pure exploitation with no exploration.
    Ties broken randomly.
    """

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm with highest estimated value.

        Args:
            context: Ignored

        Returns:
            Greedy arm index
        """
        # Handle initial case where all values are zero
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
