"""Variance minimization objective."""

from typing import Dict, Any, Optional
import numpy as np
from rovingbandit.core.objective import Objective
from rovingbandit.core.result import History
from rovingbandit.core.policy import Policy


class VarianceMinimization(Objective):
    """
    Variance minimization objective.

    Goal: Minimize variance of treatment effect estimates while
    potentially maintaining a welfare threshold.

    Reference: Kasy & Sautmann (2021), Offer-Westort et al. (2021)
    """

    def __init__(
        self,
        target_shares: Optional[np.ndarray] = None,
        welfare_threshold: float = 0.0,
    ):
        """
        Initialize variance minimization objective.

        Args:
            target_shares: Target allocation shares for groups (optional)
            welfare_threshold: Minimum fraction of optimal welfare to maintain
        """
        self.target_shares = target_shares
        self.welfare_threshold = welfare_threshold

    def compute_metric(
        self,
        history: History,
        policy: Policy = None,
        arm_groups: np.ndarray = None,
        **kwargs,
    ) -> float:
        """
        Compute estimation variance.

        For binary treatment, variance of ATE estimate.

        Args:
            history: Complete history
            policy: Policy state
            arm_groups: Group membership for arms
            **kwargs: Ignored

        Returns:
            Estimation variance
        """
        if policy is None:
            raise ValueError("policy required for variance minimization")

        # Compute variance of arm estimates
        variances = policy.values * (1 - policy.values)  # Bernoulli variance
        sample_variances = variances / np.maximum(policy.counts, 1)

        # Overall estimation variance (sum of variances)
        total_variance = float(np.sum(sample_variances))

        return total_variance

    def stopping_criterion(
        self, policy: Policy, history: History, **kwargs
    ) -> bool:
        """
        Variance minimization typically runs for fixed horizon.

        Returns:
            Always False (no early stopping)
        """
        return False

    def get_metadata(
        self,
        policy: Policy,
        history: History,
        arm_groups: np.ndarray = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Return variance and group shares.

        Args:
            policy: Final policy state
            history: Complete history
            arm_groups: Group membership array
            **kwargs: Ignored

        Returns:
            Metadata dictionary
        """
        metadata = {
            "estimation_variance": self.compute_metric(history, policy=policy),
        }

        # Compute group shares if groups defined
        if arm_groups is not None:
            arms_array = history.arms_array
            n_groups = len(np.unique(arm_groups))
            group_counts = np.zeros(n_groups)

            for arm in arms_array:
                group = arm_groups[arm]
                group_counts[group] += 1

            total = np.sum(group_counts)
            if total > 0:
                group_shares = group_counts / total
                metadata["group_shares"] = group_shares

        return metadata
