"""
RovingBandit: A flexible library for multi-armed bandit algorithms.

Supports multiple objectives (regret minimization, best-arm identification,
variance minimization) in both online and batched modes.
"""

from rovingbandit.core import (
    BanditEnvironment,
    Policy,
    Objective,
    Result,
    History,
)

from rovingbandit.policies import (
    RandomPolicy,
    GreedyPolicy,
    EpsilonGreedy,
    ExploreFirst,
    UCB1,
    BudgetedUCB,
    ThompsonSampling,
    BudgetedThompsonSampling,
    EpsilonNeymanAllocation,
    LUCB,
    KasySautmann,
    LinUCB,
    TopTwoThompson,
    RepresentationBandit,
)

from rovingbandit.objectives import (
    RegretMinimization,
    BestArmIdentification,
    VarianceMinimization,
)

from rovingbandit.runners import (
    OnlineRunner,
    BatchedRunner,
)

from rovingbandit.banditry import (
    pick_arm,
    sim_runner,
    arm_sequence,
    pull_sequence,
    best_arm,
    rep_bandit_cost,
    rep_bandit_rake,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "BanditEnvironment",
    "Policy",
    "Objective",
    "Result",
    "History",
    # Policies
    "RandomPolicy",
    "GreedyPolicy",
    "EpsilonGreedy",
    "ExploreFirst",
    "UCB1",
    "BudgetedUCB",
    "ThompsonSampling",
    "BudgetedThompsonSampling",
    "RepresentationBandit",
    "EpsilonNeymanAllocation",
    "LUCB",
    "KasySautmann",
    "LinUCB",
    "TopTwoThompson",
    # Objectives
    "RegretMinimization",
    "BestArmIdentification",
    "VarianceMinimization",
    # Runners
    "OnlineRunner",
    "BatchedRunner",
    # Legacy functions (backward compatibility)
    "pick_arm",
    "sim_runner",
    "arm_sequence",
    "pull_sequence",
    "best_arm",
    "rep_bandit_cost",
    "rep_bandit_rake",
]
