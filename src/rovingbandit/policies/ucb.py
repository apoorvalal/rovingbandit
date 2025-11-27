"""Upper Confidence Bound (UCB) policies."""

from typing import Optional
import numpy as np
from rovingbandit.core.policy import Policy


class UCB1(Policy):
    """
    UCB1 policy - Upper Confidence Bound algorithm.

    Selects arm maximizing: value_i + sqrt(exploration_factor * log(t) / n_i)

    Achieves logarithmic regret: O(log T)

    Reference: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
    Finite-time analysis of the multiarmed bandit problem.
    Machine Learning, 47(2-3), 235-256.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_factor: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize UCB1 policy.

        Args:
            n_arms: Number of arms
            exploration_factor: Exploration constant (default 2.0 for UCB1)
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.exploration_factor = exploration_factor

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using UCB1 strategy.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # Pull each arm once initially
        if self.total_pulls < self.n_arms:
            for arm in range(self.n_arms):
                if self.counts[arm] == 0:
                    return arm

        # Compute UCB values
        ucb_values = self._compute_ucb_values()

        # Select arm with highest UCB value
        max_ucb = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_ucb)[0]
        return int(self.rng.choice(best_arms))

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

    def _compute_ucb_values(self) -> np.ndarray:
        """
        Compute UCB values for all arms.

        UCB_i = value_i + sqrt(exploration_factor * log(t) / n_i)

        Returns:
            Array of UCB values
        """
        # Exploration bonus
        exploration_bonus = np.sqrt(
            self.exploration_factor * np.log(self.total_pulls + 1) / (self.counts + 1e-10)
        )

        ucb_values = self.values + exploration_bonus
        return ucb_values
