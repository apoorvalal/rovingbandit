"""Lower-Upper Confidence Bound (LUCB) policy for best-arm identification."""

from typing import Optional
import numpy as np
from rovingbandit.core.policy import Policy


class LUCB(Policy):
    """
    LUCB policy for best-arm identification.

    At each step, identifies the empirical best arm and the most competitive
    challenger based on upper confidence bounds, then samples the arm with the
    larger uncertainty among the pair. This targets rapid elimination of
    suboptimal arms while tightening confidence around the leader.
    """

    def __init__(self, n_arms: int, exploration_factor: float = 2.0, seed: Optional[int] = None):
        """
        Initialize LUCB policy.

        Args:
            n_arms: Number of arms
            exploration_factor: Multiplier inside the confidence radius
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.exploration_factor = exploration_factor

    def _confidence_radius(self) -> np.ndarray:
        """Compute confidence radii for each arm."""
        pulls = np.maximum(self.counts, 1e-10)
        t = max(self.total_pulls, 1)
        return np.sqrt(self.exploration_factor * np.log(t) / pulls)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm according to LUCB rule.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # Ensure each arm is pulled at least once
        if self.total_pulls < self.n_arms:
            for arm in range(self.n_arms):
                if self.counts[arm] == 0:
                    return arm

        radii = self._confidence_radius()
        means = self.values

        # Identify leader and challenger
        leader = int(np.argmax(means))
        ucb = means + radii

        competitors = np.array([i for i in range(self.n_arms) if i != leader])
        if len(competitors) == 0:
            return leader
        challenger = int(competitors[np.argmax(ucb[competitors])])

        # Sample the arm with greater uncertainty among the pair
        if radii[leader] >= radii[challenger]:
            return leader
        return challenger

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
