"""Epsilon-Neyman allocation policy for variance minimization."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.core.policy import Policy


class EpsilonNeymanAllocation(Policy):
    """
    Epsilon-Neyman allocation policy.

    Explore uniformly for the first `exploration_fraction` of the horizon,
    then allocate probability mass proportional to the estimated standard
    deviation of each arm: p_i ∝ sigma_i where sigma_i = sqrt(p_i (1 - p_i)).

    Suitable for variance minimization in Bernoulli settings. When no
    information is available for an arm (zero pulls), it defaults to the
    maximum Bernoulli standard deviation (0.5) to avoid starving the arm.

    Intended for adaptive experimental designs with K >= 3; for two-arm
    experiments, uniform randomization is usually adequate.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_fraction: float = 0.2,
        horizon: Optional[int] = None,
        min_variance: float = 1e-6,
        seed: Optional[int] = None,
    ):
        """
        Initialize epsilon-Neyman policy.

        Args:
            n_arms: Number of arms
            exploration_fraction: Fraction of the horizon to explore uniformly
            horizon: Total number of planned pulls (required for clean splits)
            min_variance: Floor to avoid zero standard deviations
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        if exploration_fraction < 0 or exploration_fraction > 1:
            raise ValueError("exploration_fraction must be in [0, 1]")
        self.exploration_fraction = exploration_fraction
        self.horizon = horizon
        self.min_variance = min_variance
        self._last_probabilities: Optional[np.ndarray] = None

    def set_horizon(self, horizon: int):
        """Set the planning horizon (used by OnlineRunner)."""
        self.horizon = horizon

    def _allocation_probabilities(self) -> np.ndarray:
        """Compute Neyman allocation probabilities based on estimated sigmas."""
        sigmas = np.sqrt(np.clip(self.values * (1 - self.values), self.min_variance, None))
        # Encourage sampling unobserved arms during allocation phase
        sigmas[self.counts == 0] = 0.5

        total_sigma = float(np.sum(sigmas))
        if total_sigma <= 0:
            probabilities = np.full(self.n_arms, 1.0 / self.n_arms)
        else:
            probabilities = sigmas / total_sigma

        self._last_probabilities = probabilities
        return probabilities

    def get_allocation_probabilities(self) -> Optional[np.ndarray]:
        """Return last computed allocation probabilities (if any)."""
        return self._last_probabilities

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using epsilon-Neyman strategy.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        if self.horizon is None:
            raise ValueError("EpsilonNeymanAllocation requires horizon to be set before running.")

        exploration_steps = int(self.horizon * self.exploration_fraction)
        if self.total_pulls < exploration_steps:
            return self._select_random_arm()

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
        self._update_value_incremental(arm, reward)

    def get_state(self) -> Dict[str, Any]:
        """Get state including exploration settings."""
        state = super().get_state()
        state.update(
            {
                "exploration_fraction": self.exploration_fraction,
                "horizon": self.horizon,
                "min_variance": self.min_variance,
                "allocation_probabilities": None
                if self._last_probabilities is None
                else self._last_probabilities.copy(),
            }
        )
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state including exploration settings."""
        super().set_state(state)
        self.exploration_fraction = state["exploration_fraction"]
        self.horizon = state["horizon"]
        self.min_variance = state["min_variance"]
        self._last_probabilities = (
            None
            if state.get("allocation_probabilities") is None
            else state["allocation_probabilities"].copy()
        )

    def reset(self):
        """Reset policy state."""
        super().reset()
        self._last_probabilities = None
