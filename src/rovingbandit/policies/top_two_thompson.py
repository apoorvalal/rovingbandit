"""Top-Two Thompson Sampling policy."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.policies.thompson_sampling import ThompsonSampling


class TopTwoThompson(ThompsonSampling):
    """
    Top-Two Thompson Sampling (TTTS) for Best Arm Identification.

    Algorithm:
    1. Sample from posterior to find candidate best arm I.
    2. With probability psi, play I.
    3. Else, resample until a different arm J != I is found. Play J.

    This encourages exploration of the "challenger" arm (second likely best).

    Reference:
    - Russo, D. (2016). Simple Bayesian algorithms for best arm identification.
      Operations Research, 68(6), 1625-1647.
    """

    def __init__(
        self,
        n_arms: int,
        psi: float = 0.5,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        max_resamples: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize Top-Two Thompson Sampling.

        Args:
            n_arms: Number of arms
            psi: Probability of playing the leader (0 < psi < 1)
            prior_alpha: Prior alpha parameter
            prior_beta: Prior beta parameter
            max_resamples: Maximum attempts to find challenger before fallback
            seed: Random seed
        """
        super().__init__(n_arms, prior_alpha, prior_beta, seed)
        if not 0.0 <= psi <= 1.0:
            raise ValueError(f"psi must be between 0 and 1, got {psi}")
        self.psi = psi
        self.max_resamples = max_resamples

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using Top-Two Thompson Sampling.

        Args:
            context: Ignored

        Returns:
            Selected arm index
        """
        # 1. Sample leader I
        samples_leader = self._sample_posteriors()
        leader = int(np.argmax(samples_leader))

        # 2. Decide whether to play leader
        if self.rng.random() < self.psi:
            return leader

        # 3. Find challenger J
        for _ in range(self.max_resamples):
            samples_challenger = self._sample_posteriors()
            challenger = int(np.argmax(samples_challenger))
            if challenger != leader:
                return challenger

        # Fallback if rejection sampling fails (posterior too concentrated)
        # Return second best from the last sample
        # Note: samples_challenger is from the last iteration
        sorted_indices = np.argsort(samples_challenger)
        # Best is at [-1], second best is at [-2]
        # But we need to make sure [-1] is indeed the leader or we just pick the best distinct from leader
        # If posterior is concentrated, samples_challenger argmax is likely leader.
        # So we pick the best arm that is NOT the leader from the last sample.

        # We can just mask the leader and pick argmax
        samples_challenger[leader] = -np.inf
        fallback_challenger = int(np.argmax(samples_challenger))
        return fallback_challenger

    def get_state(self) -> Dict[str, Any]:
        """Get state including TTTS parameters."""
        state = super().get_state()
        state.update(
            {
                "psi": self.psi,
                "max_resamples": self.max_resamples,
            }
        )
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore state including TTTS parameters."""
        super().set_state(state)
        self.psi = state["psi"]
        self.max_resamples = state["max_resamples"]
