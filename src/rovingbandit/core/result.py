"""Result and history tracking for bandit simulations."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class History:
    """
    Tracks the complete history of a bandit run.

    Attributes:
        arms: Sequence of arms pulled
        rewards: Sequence of rewards received
        costs: Sequence of costs incurred
        contexts: Sequence of contexts (if contextual)
    """

    arms: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    contexts: List[Optional[np.ndarray]] = field(default_factory=list)

    def add(
        self,
        arm: int,
        reward: float,
        cost: float = 0.0,
        context: Optional[np.ndarray] = None,
    ):
        """Add a single observation to history."""
        self.arms.append(arm)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.contexts.append(context)

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self.arms)

    @property
    def arms_array(self) -> np.ndarray:
        """Return arms as numpy array."""
        return np.array(self.arms)

    @property
    def rewards_array(self) -> np.ndarray:
        """Return rewards as numpy array."""
        return np.array(self.rewards)

    @property
    def costs_array(self) -> np.ndarray:
        """Return costs as numpy array."""
        return np.array(self.costs)

    @property
    def cumulative_rewards(self) -> np.ndarray:
        """Return cumulative rewards."""
        return np.cumsum(self.rewards_array)

    @property
    def cumulative_costs(self) -> np.ndarray:
        """Return cumulative costs."""
        return np.cumsum(self.costs_array)

    @property
    def average_reward(self) -> np.ndarray:
        """Return average reward over time."""
        cum_rewards = self.cumulative_rewards
        steps = np.arange(1, len(self) + 1)
        return cum_rewards / steps


@dataclass
class Result:
    """
    Results from a bandit simulation run.

    Attributes:
        history: Complete history of pulls
        policy_state: Final state of the policy
        metadata: Additional information about the run
    """

    history: History
    policy_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Number of steps taken."""
        return len(self.history)

    @property
    def total_reward(self) -> float:
        """Total reward accumulated."""
        return float(np.sum(self.history.rewards_array))

    @property
    def total_cost(self) -> float:
        """Total cost incurred."""
        return float(np.sum(self.history.costs_array))

    @property
    def average_reward(self) -> float:
        """Average reward per step."""
        return self.total_reward / self.n_steps if self.n_steps > 0 else 0.0

    @property
    def cumulative_regret(self) -> Optional[np.ndarray]:
        """Cumulative regret over time (if optimal reward known)."""
        if "optimal_reward" not in self.metadata:
            return None
        optimal = self.metadata["optimal_reward"]
        optimal_rewards = np.full(self.n_steps, optimal)
        return np.cumsum(optimal_rewards - self.history.rewards_array)

    @property
    def final_regret(self) -> Optional[float]:
        """Final cumulative regret."""
        regret = self.cumulative_regret
        return float(regret[-1]) if regret is not None else None

    @property
    def best_arm(self) -> int:
        """Best arm according to final policy estimates."""
        return int(np.argmax(self.policy_state["values"]))

    @property
    def confidence(self) -> Optional[float]:
        """Confidence in best arm (if available)."""
        return self.metadata.get("confidence")

    @property
    def group_shares(self) -> Optional[np.ndarray]:
        """Group representation shares (if applicable)."""
        return self.metadata.get("group_shares")

    @property
    def estimation_variance(self) -> Optional[float]:
        """Estimation variance (if applicable)."""
        return self.metadata.get("estimation_variance")

    def plot(
        self,
        metric: str = "cumulative_reward",
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot results.

        Args:
            metric: What to plot - 'cumulative_reward', 'average_reward',
                   'cumulative_regret', 'arm_pulls'
            ax: Matplotlib axes to plot on (creates new if None)
            **kwargs: Additional arguments passed to plot()

        Returns:
            Matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        steps = np.arange(1, self.n_steps + 1)

        if metric == "cumulative_reward":
            ax.plot(steps, self.history.cumulative_rewards, **kwargs)
            ax.set_ylabel("Cumulative Reward")
            ax.set_title("Cumulative Reward Over Time")
        elif metric == "average_reward":
            ax.plot(steps, self.history.average_reward, **kwargs)
            ax.set_ylabel("Average Reward")
            ax.set_title("Average Reward Over Time")
        elif metric == "cumulative_regret":
            if self.cumulative_regret is not None:
                ax.plot(steps, self.cumulative_regret, **kwargs)
                ax.set_ylabel("Cumulative Regret")
                ax.set_title("Cumulative Regret Over Time")
            else:
                raise ValueError("Cumulative regret not available")
        elif metric == "arm_pulls":
            # Plot cumulative pulls per arm
            n_arms = self.policy_state["n_arms"]
            pull_counts = np.zeros((self.n_steps, n_arms))
            for t, arm in enumerate(self.history.arms):
                if t > 0:
                    pull_counts[t] = pull_counts[t - 1]
                pull_counts[t, arm] += 1

            for arm in range(n_arms):
                ax.plot(steps, pull_counts[:, arm], label=f"Arm {arm}", **kwargs)
            ax.set_ylabel("Cumulative Pulls")
            ax.set_title("Arm Pulls Over Time")
            ax.legend()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        ax.set_xlabel("Steps")
        ax.grid(alpha=0.3)

        return ax

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        summary = {
            "n_steps": self.n_steps,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "total_cost": self.total_cost,
            "best_arm": self.best_arm,
        }

        if self.final_regret is not None:
            summary["final_regret"] = self.final_regret

        if self.confidence is not None:
            summary["confidence"] = self.confidence

        if self.group_shares is not None:
            summary["group_shares"] = self.group_shares

        if self.estimation_variance is not None:
            summary["estimation_variance"] = self.estimation_variance

        return summary
