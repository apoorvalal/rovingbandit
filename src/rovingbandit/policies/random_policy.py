"""Random policy - uniform random arm selection."""

from typing import Optional
import numpy as np
from rovingbandit.core.policy import Policy


class RandomPolicy(Policy):
    """
    Random policy - selects arms uniformly at random.

    Serves as baseline for comparison.
    """

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm uniformly at random.

        Args:
            context: Ignored

        Returns:
            Random arm index
        """
        return self._select_random_arm()

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
