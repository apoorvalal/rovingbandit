"""Base class for bandit policies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class Policy(ABC):
    """
    Abstract base class for bandit policies.

    All policies maintain:
    - counts: Number of times each arm has been pulled
    - values: Estimated value for each arm
    - total_pulls: Total number of arm pulls

    Subclasses implement select_arm() and update() methods.
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None):
        """
        Initialize policy.

        Args:
            n_arms: Number of arms in the bandit
            seed: Random seed for reproducibility
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select the next arm to pull.

        Args:
            context: Optional contextual information

        Returns:
            Index of selected arm
        """
        pass

    @abstractmethod
    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update policy state after observing reward.

        Args:
            arm: Index of pulled arm
            reward: Observed reward
            cost: Cost incurred (optional)
        """
        pass

    def reset(self):
        """Reset policy to initial state."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_pulls = 0

    def get_state(self) -> Dict[str, Any]:
        """
        Get current policy state for serialization.

        Returns:
            Dictionary containing policy state
        """
        return {
            "n_arms": self.n_arms,
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_pulls": self.total_pulls,
        }

    def set_state(self, state: Dict[str, Any]):
        """
        Restore policy state from serialization.

        Args:
            state: Dictionary containing policy state
        """
        self.n_arms = state["n_arms"]
        self.counts = state["counts"].copy()
        self.values = state["values"].copy()
        self.total_pulls = state["total_pulls"]

    def _update_value_incremental(self, arm: int, reward: float):
        """
        Update value estimate using incremental mean formula.

        value_new = value_old + (reward - value_old) / count

        Args:
            arm: Index of arm
            reward: Observed reward
        """
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def _select_random_arm(self) -> int:
        """Select a random arm uniformly."""
        return int(self.rng.integers(0, self.n_arms))

    def _select_greedy_arm(self) -> int:
        """Select arm(s) with highest estimated value, breaking ties randomly."""
        max_value = np.max(self.values)
        best_arms = np.where(self.values == max_value)[0]
        return int(self.rng.choice(best_arms))
