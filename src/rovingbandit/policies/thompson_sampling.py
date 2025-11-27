"""Thompson Sampling policy."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.core.policy import Policy


class ThompsonSampling(Policy):
    """
    Thompson Sampling for Bernoulli bandits.

    Maintains Beta(alpha, beta) posterior for each arm.
    Samples from each posterior and selects argmax.

    Achieves optimal regret and excellent empirical performance.

    Reference:
    - Thompson, W. R. (1933). On the likelihood that one unknown
      probability exceeds another. Biometrika.
    - Chapelle, O., & Li, L. (2011). An empirical evaluation of
      thompson sampling. NIPS.
    """

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Thompson Sampling policy.

        Args:
            n_arms: Number of arms
            prior_alpha: Prior alpha parameter (pseudo-successes)
            prior_beta: Prior beta parameter (pseudo-failures)
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Track successes and failures for Beta posterior
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using Thompson Sampling.

        Samples from Beta(alpha_i, beta_i) for each arm and picks argmax.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # Sample from posterior for each arm
        samples = self._sample_posteriors()

        # Select arm with highest sample
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update posterior distributions.

        Args:
            arm: Index of pulled arm
            reward: Observed reward (0 or 1 for Bernoulli)
            cost: Cost incurred (ignored)
        """
        self.counts[arm] += 1
        self.total_pulls += 1

        # Update successes/failures
        self.successes[arm] += reward
        self.failures[arm] += (1 - reward)

        # Update value estimate
        self._update_value_incremental(arm, reward)

    def _sample_posteriors(self) -> np.ndarray:
        """
        Sample from Beta posteriors for all arms.

        Beta(alpha, beta) where:
        - alpha = prior_alpha + successes
        - beta = prior_beta + failures

        Returns:
            Array of samples, one per arm
        """
        alpha = self.prior_alpha + self.successes
        beta = self.prior_beta + self.failures

        samples = np.array([
            self.rng.beta(alpha[i], beta[i])
            for i in range(self.n_arms)
        ])

        return samples

    def get_state(self) -> Dict[str, Any]:
        """Get state including posterior parameters."""
        state = super().get_state()
        state.update({
            "successes": self.successes.copy(),
            "failures": self.failures.copy(),
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
        })
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state including posterior parameters."""
        super().set_state(state)
        self.successes = state["successes"].copy()
        self.failures = state["failures"].copy()
        self.prior_alpha = state["prior_alpha"]
        self.prior_beta = state["prior_beta"]

    def reset(self):
        """Reset policy to initial state."""
        super().reset()
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)
