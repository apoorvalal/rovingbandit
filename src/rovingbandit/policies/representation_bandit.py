"""Representation Bandit policy."""

from typing import Optional, Dict, Any, List
import numpy as np
from rovingbandit.policies.budgeted_thompson import BudgetedThompsonSampling


class RepresentationBandit(BudgetedThompsonSampling):
    """
    Representation Bandit.

    Extends Budgeted Thompson Sampling to target specific representation shares
    for different groups of arms. It dynamically adjusts the effective cost
    of arms based on how over- or under-represented their group is.

    Cost Adjustment:
    effective_cost = base_cost * (1 + fraction_spent * (current_share - target_share)) ** gamma

    - If a group is over-represented (current > target), effective cost increases.
    - If a group is under-represented, effective cost decreases.
    - The adjustment magnitude scales with how much of the budget has been spent.

    Reference:
    - Lal, A. (2022). Multi-armed Bandits for Budget-Constrained Data Collection.
      (Section 2.2.3: Targeting Representativeness)
    """

    def __init__(
        self,
        n_arms: int,
        arm_groups: np.ndarray,
        target_shares: np.ndarray,
        total_budget: float,
        costs: Optional[np.ndarray] = None,
        gamma: float = 2.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Representation Bandit.

        Args:
            n_arms: Number of arms
            arm_groups: Array of length n_arms, containing group index for each arm.
                        Groups should be 0-indexed integers: 0, 1, ..., n_groups-1.
            target_shares: Array of length n_groups, summing to 1.0.
            total_budget: Total budget available (used to calculate progress).
            costs: Base costs for each arm.
            gamma: Representativeness prioritization parameter (default 2.0).
            prior_alpha: Prior alpha parameter
            prior_beta: Prior beta parameter
            seed: Random seed
        """
        super().__init__(n_arms, costs, prior_alpha, prior_beta, seed)
        
        self.arm_groups = np.array(arm_groups, dtype=int)
        self.target_shares = np.array(target_shares, dtype=float)
        self.total_budget = float(total_budget)
        self.gamma = float(gamma)
        
        self.n_groups = len(target_shares)
        self.total_cost_incurred = 0.0
        
        # Validation
        if len(self.arm_groups) != n_arms:
            raise ValueError(f"arm_groups length ({len(self.arm_groups)}) != n_arms ({n_arms})")
        if not np.isclose(np.sum(self.target_shares), 1.0):
            raise ValueError(f"target_shares must sum to 1.0, got {np.sum(self.target_shares)}")
        if np.max(self.arm_groups) >= self.n_groups:
             raise ValueError("arm_groups indices exceed number of target_shares")

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using adjusted costs.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # Calculate current shares
        current_counts = np.zeros(self.n_groups)
        if self.total_pulls > 0:
            for arm in range(self.n_arms):
                group = self.arm_groups[arm]
                current_counts[group] += self.counts[arm]
            current_shares = current_counts / self.total_pulls
        else:
            current_shares = np.zeros(self.n_groups)

        # Calculate budget fraction spent
        # Clamp to [0, 1] to avoid instability if budget is slightly exceeded
        fraction_spent = min(1.0, max(0.0, self.total_cost_incurred / self.total_budget))

        # Calculate adjusted costs
        # c_adjusted = c_base * (1 + fraction * (share - target)) ** gamma
        adjusted_costs = self.costs.copy()
        
        for group in range(self.n_groups):
            psi = current_shares[group] - self.target_shares[group]
            multiplier = (1.0 + fraction_spent * psi) ** self.gamma
            
            # Apply to all arms in this group
            # Identify arms belonging to this group
            group_arms = np.where(self.arm_groups == group)[0]
            adjusted_costs[group_arms] *= multiplier

        # Use Thompson Sampling logic with adjusted costs
        # 1. Sample posteriors
        samples = self._sample_posteriors()
        
        # 2. Compute ratio with adjusted costs
        safe_costs = np.maximum(adjusted_costs, 1e-10)
        ratios = samples / safe_costs
        
        return int(np.argmax(ratios))

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update state and total cost.

        Args:
            arm: Index of pulled arm
            reward: Observed reward
            cost: Cost incurred
        """
        super().update(arm, reward, cost)
        self.total_cost_incurred += cost

    def get_state(self) -> Dict[str, Any]:
        """Get state including accumulated cost."""
        state = super().get_state()
        state.update({
            "total_cost_incurred": self.total_cost_incurred,
            "arm_groups": self.arm_groups.copy(),
            "target_shares": self.target_shares.copy(),
            "total_budget": self.total_budget,
            "gamma": self.gamma
        })
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state."""
        super().set_state(state)
        self.total_cost_incurred = state["total_cost_incurred"]
        self.arm_groups = state["arm_groups"].copy()
        self.target_shares = state["target_shares"].copy()
        self.total_budget = state["total_budget"]
        self.gamma = state["gamma"]
