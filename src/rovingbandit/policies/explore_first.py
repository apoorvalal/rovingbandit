"""Explore-first (explore-then-commit) policy."""

from typing import Optional
import numpy as np
from rovingbandit.core.policy import Policy


class ExploreFirst(Policy):
    """
    Explore-first (explore-then-commit) policy.

    Pure exploration for first exploration_fraction of steps,
    then pure exploitation afterwards.

    Reference: Perchet et al. (2016)
    """

    def __init__(
        self,
        n_arms: int,
        exploration_fraction: float = 0.1,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize explore-first policy.

        Args:
            n_arms: Number of arms
            exploration_fraction: Fraction of horizon to explore
            horizon: Total number of pulls (if known)
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.exploration_fraction = exploration_fraction
        self.horizon = horizon

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using explore-first strategy.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # If horizon known, check if in exploration phase
        if self.horizon is not None:
            exploration_steps = int(self.horizon * self.exploration_fraction)
            if self.total_pulls < exploration_steps:
                return self._select_random_arm()
            else:
                return self._select_greedy_arm()
        else:
            # Without known horizon, always exploit after some initial pulls
            # Use n_arms * some multiple as proxy
            exploration_steps = int(self.n_arms * (1 / self.exploration_fraction))
            if self.total_pulls < exploration_steps:
                return self._select_random_arm()
            else:
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

    def set_horizon(self, horizon: int):
        """
        Set the horizon (useful when it's determined at runtime).

        Args:
            horizon: Total number of pulls
        """
        self.horizon = horizon
