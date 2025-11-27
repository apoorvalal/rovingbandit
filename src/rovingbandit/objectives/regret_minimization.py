"""Regret minimization objective."""

from typing import Dict, Any
import numpy as np
from rovingbandit.core.objective import Objective
from rovingbandit.core.result import History
from rovingbandit.core.policy import Policy


class RegretMinimization(Objective):
    """
    Regret minimization objective.

    Cumulative regret = sum_t (optimal_reward - actual_reward_t)

    Reference: Lai & Robbins (1985), Auer et al. (2002)
    """

    def __init__(self, optimal_reward: float = None):
        """
        Initialize regret minimization objective.

        Args:
            optimal_reward: Known optimal reward (if available)
        """
        self.optimal_reward = optimal_reward

    def compute_metric(self, history: History, optimal_reward: float = None, **kwargs) -> float:
        """
        Compute cumulative regret.

        Args:
            history: Complete history
            optimal_reward: Optimal expected reward (overrides init value)
            **kwargs: Ignored

        Returns:
            Cumulative regret
        """
        opt_reward = optimal_reward or self.optimal_reward
        if opt_reward is None:
            raise ValueError("optimal_reward must be provided")

        T = len(history)
        optimal_rewards = np.full(T, opt_reward)
        regret = np.sum(optimal_rewards - history.rewards_array)
        return float(regret)

    def stopping_criterion(self, policy: Policy, history: History, **kwargs) -> bool:
        """
        Regret minimization typically runs for fixed horizon.

        Returns:
            Always False (no early stopping)
        """
        return False

    def get_metadata(self, policy: Policy, history: History, **kwargs) -> Dict[str, Any]:
        """Return optimal reward for regret calculation."""
        if self.optimal_reward is not None:
            return {"optimal_reward": self.optimal_reward}
        return {}
