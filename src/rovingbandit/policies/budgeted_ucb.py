"""Budgeted UCB policy."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.policies.ucb import UCB1


class BudgetedUCB(UCB1):
    """
    Budgeted UCB policy - Cost-aware Upper Confidence Bound.

    Selects arm maximizing: (value_i + exploration_bonus_i) / cost_i

    Reference: Tran-Thanh, L., Chapman, A., Rogers, A., & Jennings, N. R. (2012).
    Knapsack based optimal policies for budget-limited multi-armed bandits. AAAI.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_factor: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize BudgetedUCB policy.

        Args:
            n_arms: Number of arms
            exploration_factor: Exploration constant
            seed: Random seed
        """
        super().__init__(n_arms, exploration_factor, seed)
        self.avg_costs = np.zeros(n_arms)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using Budgeted UCB strategy.

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

        # Compute UCB values (regular UCB numerator)
        ucb_values = self._compute_ucb_values()

        # Adjust by cost
        # Avoid division by zero by using a small epsilon if cost is 0
        safe_costs = np.maximum(self.avg_costs, 1e-10)
        budgeted_scores = ucb_values / safe_costs

        # Select arm with highest budgeted score
        max_score = np.max(budgeted_scores)
        best_arms = np.where(budgeted_scores == max_score)[0]
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update counts, value estimates, and cost estimates.

        Args:
            arm: Index of pulled arm
            reward: Observed reward
            cost: Cost incurred
        """
        self.counts[arm] += 1
        self.total_pulls += 1

        # Update value estimate
        self._update_value_incremental(arm, reward)

        # Update cost estimate incrementally
        # counts[arm] is already updated
        self.avg_costs[arm] += (cost - self.avg_costs[arm]) / self.counts[arm]

    def get_state(self) -> Dict[str, Any]:
        """Get state including cost estimates."""
        state = super().get_state()
        state.update({
            "avg_costs": self.avg_costs.copy(),
        })
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state including cost estimates."""
        super().set_state(state)
        self.avg_costs = state["avg_costs"].copy()

    def reset(self):
        """Reset policy to initial state."""
        super().reset()
        self.avg_costs = np.zeros(self.n_arms)
