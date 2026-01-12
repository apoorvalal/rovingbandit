"""Online runner for sequential decision-making."""

from typing import Optional
import numpy as np
from rovingbandit.core.environment import BanditEnvironment
from rovingbandit.core.policy import Policy
from rovingbandit.core.objective import Objective
from rovingbandit.core.result import Result, History


class OnlineRunner:
    """
    Online runner - sequential decision making, one arm at a time.

    Pulls arms sequentially based on policy, updating after each observation.
    """

    def run(
        self,
        policy: Policy,
        environment: BanditEnvironment,
        n_steps: int,
        objective: Optional[Objective] = None,
        early_stopping: bool = False,
    ) -> Result:
        """
        Run policy on environment for n_steps.

        Args:
            policy: Policy to run
            environment: Bandit environment
            n_steps: Maximum number of steps to run
            objective: Objective function (optional)
            early_stopping: If True, stop when objective criterion met

        Returns:
            Result object with history and diagnostics
        """
        # Reset policy
        policy.reset()

        # Initialize history
        history = History()

        # Set horizon for policies that need it
        if hasattr(policy, "set_horizon"):
            policy.set_horizon(n_steps)

        # Main loop
        for step in range(n_steps):
            # Retrieve contexts if available
            contexts = environment.get_contexts() if hasattr(environment, "get_contexts") else None

            # Select arm (pass contexts if policy can use them)
            arm = policy.select_arm(contexts)

            # Pull arm and observe reward
            reward, cost = environment.pull(arm)

            # Update policy
            policy.update(arm, reward, cost)

            # Record history
            chosen_context = None
            if contexts is not None:
                try:
                    chosen_context = contexts[arm]
                except Exception:
                    chosen_context = None
            history.add(arm, reward, cost, context=chosen_context)

            # Check stopping criterion
            if early_stopping and objective is not None:
                if objective.stopping_criterion(policy, history):
                    break

        # Collect metadata from objective
        metadata = {}
        if objective is not None:
            metadata = objective.get_metadata(
                policy,
                history,
                arm_groups=environment.arm_groups,
            )
        else:
            # Add optimal reward if available
            if environment.arm_means is not None:
                metadata["optimal_reward"] = environment.get_optimal_reward()

        # Create result
        result = Result(
            history=history,
            policy_state=policy.get_state(),
            metadata=metadata,
        )

        return result

    def run_with_budget(
        self,
        policy: Policy,
        environment: BanditEnvironment,
        budget: float,
        objective: Optional[Objective] = None,
        pay_on_success: bool = False,
    ) -> Result:
        """
        Run policy with budget constraint.

        Stops when budget exhausted.

        Args:
            policy: Policy to run
            environment: Bandit environment with costs
            budget: Total budget available
            objective: Objective function (optional)
            pay_on_success: If True, only pay cost when reward=1

        Returns:
            Result object
        """
        # Reset policy
        policy.reset()

        # Initialize history
        history = History()

        remaining_budget = budget

        # Main loop
        while remaining_budget > 0:
            # Select arm
            arm = policy.select_arm()

            # Pull arm and observe reward
            reward, cost = environment.pull(arm)

            # Check if we can afford this pull
            actual_cost = cost if not pay_on_success else (cost if reward > 0 else 0.0)

            # Only record if we can afford it
            if remaining_budget >= actual_cost:
                # Update policy
                policy.update(arm, reward, cost)

                # Record history
                history.add(arm, reward, cost)

                # Deduct cost from budget
                remaining_budget -= actual_cost
            else:
                # Can't afford this pull, stop
                break

            # Safety check - don't run forever
            if len(history) > budget * 100:
                break

        # Collect metadata
        metadata = {"budget": budget, "pay_on_success": pay_on_success}

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
