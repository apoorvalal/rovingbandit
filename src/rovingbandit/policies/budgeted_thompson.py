"""Budgeted Thompson Sampling policy."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.policies.thompson_sampling import ThompsonSampling


class BudgetedThompsonSampling(ThompsonSampling):
    """
    Budgeted Thompson Sampling.

    Adapts Thompson Sampling for budget-constrained settings where arms have different costs.
    Instead of selecting the arm with the highest posterior sample, it selects the arm
    with the highest "bang-for-buck" ratio: (posterior_sample / cost).

    Algorithm:
    1. Sample theta_a ~ Beta(alpha_a, beta_a) for each arm.
    2. Select arm a = argmax (theta_a / cost_a).

    Reference:
    - Lal, A. (2022). Multi-armed Bandits for Budget-Constrained Data Collection.
      (Algorithm 2: Budgeted Thompson Sampling)
    - Xia, Y., et al. (2015). Thompson sampling for budgeted multi-armed bandits.
    """

    def __init__(
        self,
        n_arms: int,
        costs: Optional[np.ndarray] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Budgeted Thompson Sampling.

        Args:
            n_arms: Number of arms
            costs: Array of costs for each arm. If None, assumes all costs are 1.0.
                   (Can be updated dynamically if costs are unknown/stochastic,
                    but this implementation primarily supports fixed known costs).
            prior_alpha: Prior alpha parameter
            prior_beta: Prior beta parameter
            seed: Random seed
        """
        super().__init__(n_arms, prior_alpha, prior_beta, seed)
        if costs is not None:
            if len(costs) != n_arms:
                raise ValueError(f"Length of costs ({len(costs)}) must match n_arms ({n_arms})")
            self.costs = np.array(costs, dtype=float)
        else:
            self.costs = np.ones(n_arms, dtype=float)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm maximizing sample/cost ratio.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # Sample from posterior for each arm
        samples = self._sample_posteriors()

        # Compute ratio
        # Avoid division by zero
        safe_costs = np.maximum(self.costs, 1e-10)
        ratios = samples / safe_costs

        # Select arm with highest ratio
        return int(np.argmax(ratios))

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update posterior distributions.

        Args:
            arm: Index of pulled arm
            reward: Observed reward
            cost: Cost incurred (can be used to update cost estimates if they were not fixed)
        """
        super().update(arm, reward, cost)
        
        # If we wanted to learn costs (e.g. if self.costs was not provided initally),
        # we could update self.costs here. 
        # For now, we assume costs are provided or updated manually if needed,
        # to match the paper's Algorithm 2 which takes vector C as input.

    def get_state(self) -> Dict[str, Any]:
        """Get state including costs."""
        state = super().get_state()
        state.update({
            "costs": self.costs.copy(),
        })
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state including costs."""
        super().set_state(state)
        self.costs = state["costs"].copy()
