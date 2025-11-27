"""Batched runner for parallel decision-making."""

from typing import Optional
import numpy as np
from rovingbandit.core.environment import BanditEnvironment
from rovingbandit.core.policy import Policy
from rovingbandit.core.objective import Objective
from rovingbandit.core.result import Result, History


class BatchedRunner:
    """
    Batched runner - parallel decision making.

    Selects multiple arms per batch, updates policy after all rewards observed.
    Useful for parallelizable experiments or when decisions must be made in batches.
    """

    def run(
        self,
        policy: Policy,
        environment: BanditEnvironment,
        batch_size: int,
        n_batches: int,
        objective: Optional[Objective] = None,
    ) -> Result:
        """
        Run policy in batched mode.

        Args:
            policy: Policy to run
            environment: Bandit environment
            batch_size: Number of arms to pull per batch
            n_batches: Number of batches to run
            objective: Objective function (optional)

        Returns:
            Result object
        """
        # Reset policy
        policy.reset()

        # Initialize history
        history = History()

        # Set horizon for policies that need it
        if hasattr(policy, "set_horizon"):
            policy.set_horizon(batch_size * n_batches)

        # Main loop over batches
        for batch in range(n_batches):
            # Select arms for this batch
            arms_batch = []
            for _ in range(batch_size):
                arm = policy.select_arm()
                arms_batch.append(arm)

            # Pull all arms and collect rewards
            rewards_batch = []
            costs_batch = []
            for arm in arms_batch:
                reward, cost = environment.pull(arm)
                rewards_batch.append(reward)
                costs_batch.append(cost)

                # Record in history immediately
                history.add(arm, reward, cost)

            # Update policy with all observations from batch
            for arm, reward, cost in zip(arms_batch, rewards_batch, costs_batch):
                policy.update(arm, reward, cost)

        # Collect metadata
        metadata = {"batch_size": batch_size, "n_batches": n_batches}

        if objective is not None:
            metadata.update(
                objective.get_metadata(
                    policy,
                    history,
                    arm_groups=environment.arm_groups,
                )
            )
        else:
            if environment.arm_means is not None:
                metadata["optimal_reward"] = environment.get_optimal_reward()

        # Create result
        result = Result(
            history=history,
            policy_state=policy.get_state(),
            metadata=metadata,
        )

        return result
