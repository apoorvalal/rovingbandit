"""
RovingBandit: A flexible library for multi-armed bandit algorithms.

Supports multiple objectives (regret minimization, best-arm identification,
variance minimization) in both online and batched modes.
"""

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
    "pick_arm",
    "sim_runner",
    "arm_sequence",
    "pull_sequence",
    "best_arm",
    "rep_bandit_cost",
    "rep_bandit_rake",
]
