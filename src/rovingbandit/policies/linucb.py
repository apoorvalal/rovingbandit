"""LinUCB policy for linear contextual bandits."""

from typing import Optional, Dict, Any
import numpy as np
from rovingbandit.core.policy import Policy


class LinUCB(Policy):
    """
    Linear Upper Confidence Bound (LinUCB) for contextual bandits.

    Assumes rewards follow r = x^T theta + noise. Maintains a ridge-regression
    estimate per arm and selects according to an upper confidence bound.
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float = 1.0,
        dim: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize LinUCB policy.

        Args:
            n_arms: Number of arms
            alpha: Exploration parameter
            dim: Feature dimension (optional; inferred from first contexts if None)
            seed: Random seed
        """
        super().__init__(n_arms, seed)
        self.alpha = alpha
        self.dim = dim
        self._A: Optional[np.ndarray] = None  # shape (n_arms, d, d)
        self._A_inv: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None  # shape (n_arms, d)
        self._last_context: Optional[np.ndarray] = None

    def _ensure_matrices(self, dim: int):
        """Initialize parameter matrices if not yet created."""
        if self._A is None or self._A.shape[1] != dim:
            self._A = np.array([np.eye(dim) for _ in range(self.n_arms)])
            self._A_inv = np.array([np.eye(dim) for _ in range(self.n_arms)])
            self._b = np.zeros((self.n_arms, dim))

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Select arm using LinUCB.

        Args:
            context: Matrix of shape (n_arms, d) with per-arm features.

        Returns:
            Selected arm index
        """
        if context is None:
            raise ValueError("LinUCB requires per-arm context features.")

        if context.shape[0] != self.n_arms:
            raise ValueError("Context first dimension must match number of arms.")

        dim = context.shape[1]
        self._ensure_matrices(dim)

        # Pull any unobserved arm at least once
        if np.any(self.counts == 0):
            candidates = np.where(self.counts == 0)[0]
            chosen_arm = int(self.rng.choice(candidates))
            self._last_context = context[chosen_arm]
            return chosen_arm

        p_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            x = context[arm]
            A_inv = self._A_inv[arm]
            theta_hat = A_inv @ self._b[arm]
            uncertainty = np.sqrt(x.T @ A_inv @ x)
            p_values[arm] = theta_hat.T @ x + self.alpha * uncertainty

        best_value = np.max(p_values)
        best_arms = np.where(p_values == best_value)[0]
        chosen_arm = int(self.rng.choice(best_arms))

        # Store context of chosen arm for update
        self._last_context = context[chosen_arm]
        return chosen_arm

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """
        Update ridge estimates with observed reward.

        Args:
            arm: Index of pulled arm
            reward: Observed reward
            cost: Cost incurred (ignored)
        """
        if self._last_context is None:
            raise ValueError("Context required for LinUCB update but not found.")

        x = self._last_context
        x = x.reshape(-1, 1)

        self.counts[arm] += 1
        self.total_pulls += 1

        # Update A and b
        self._A[arm] += x @ x.T
        self._A_inv[arm] = np.linalg.inv(self._A[arm])
        self._b[arm] += float(reward) * x.flatten()

        # Update mean reward estimate for compatibility with Result/Objective
        self._update_value_incremental(arm, reward)

        # Clear stored context
        self._last_context = None

    def get_state(self) -> Dict[str, Any]:
        """Serialize policy state."""
        state = super().get_state()
        state.update(
            {
                "alpha": self.alpha,
                "dim": self.dim,
                "A": None if self._A is None else self._A.copy(),
                "A_inv": None if self._A_inv is None else self._A_inv.copy(),
                "b": None if self._b is None else self._b.copy(),
            }
        )
        return state

    def set_state(self, state: Dict[str, Any]):
        """Restore policy state."""
        super().set_state(state)
        self.alpha = state["alpha"]
        self.dim = state["dim"]
        self._A = None if state.get("A") is None else state["A"].copy()
        self._A_inv = None if state.get("A_inv") is None else state["A_inv"].copy()
        self._b = None if state.get("b") is None else state["b"].copy()
        self._last_context = None

    def reset(self):
        """Reset policy to initial state."""
        super().reset()
        self._A = None
        self._A_inv = None
        self._b = None
        self._last_context = None
