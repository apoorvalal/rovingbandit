"""Kasy-Sautmann variance-oriented allocation with welfare constraint."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.core.policy import Policy


class KasySautmann(Policy):
    """
    Variance-minimizing allocation with a welfare threshold.

    Allocates probability proportional to empirical standard deviations while
    suppressing arms whose estimated mean falls below a welfare threshold
    relative to the current best. Designed for adaptive experiments where
    variance reduction must respect minimum welfare.
    """

    def __init__(
        self,
        n_arms: int,
        welfare_threshold: float = 0.8,
        smoothing: float = 1e-3,
        seed: Optional[int] = None,
    ):
        """
        Initialize Kasy-Sautmann policy.

        Args:
            n_arms: Number of arms
            welfare_threshold: Fraction of the current best mean below which arms are down-weighted
            smoothing: Small constant to avoid zero probabilities
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.welfare_threshold = welfare_threshold
        self.smoothing = smoothing
        self._last_probabilities: Optional[np.ndarray] = None

    def _allocation_probabilities(self) -> np.ndarray:
        """Compute welfare-respecting allocation probabilities."""
        sigmas = np.sqrt(np.clip(self.values * (1 - self.values), 0.0, None))

        # Early exploration safeguard
        unpulled = self.counts == 0
        if np.any(unpulled):
            probs = np.zeros(self.n_arms)
            probs[unpulled] = 1.0 / np.sum(unpulled)
            self._last_probabilities = probs
            return probs

        best_mean = float(np.max(self.values))
        welfare_floor = self.welfare_threshold * best_mean if best_mean > 0 else 0.0

        weights = sigmas.copy()
        weights[self.values < welfare_floor] = 0.0

        if np.sum(weights) <= 0:
            weights = sigmas

        weights = weights + self.smoothing
        probabilities = weights / np.sum(weights)
        self._last_probabilities = probabilities
        return probabilities

    def get_allocation_probabilities(self) -> Optional[np.ndarray]:
        """Return last computed allocation probabilities (if any)."""
        return self._last_probabilities

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm according to welfare-constrained variance allocation.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        probabilities = self._allocation_probabilities()
        return int(self.rng.choice(self.n_arms, p=probabilities))

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
        self._last_probabilities = None
        self._update_value_incremental(arm, reward)

    def get_state(self) -> Dict[str, Any]:
        """Get state including welfare settings."""
        state = super().get_state()
        state.update(
            {
                "welfare_threshold": self.welfare_threshold,
                "smoothing": self.smoothing,
                "allocation_probabilities": None
                if self._last_probabilities is None
                else self._last_probabilities.copy(),
            }
        )
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state including welfare settings."""
        super().set_state(state)
        self.welfare_threshold = state["welfare_threshold"]
        self.smoothing = state["smoothing"]
        self._last_probabilities = (
            None
            if state.get("allocation_probabilities") is None
            else state["allocation_probabilities"].copy()
        )

    def reset(self):
        """Reset policy to initial state."""
        super().reset()
        self._last_probabilities = None
