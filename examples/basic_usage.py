"""
Basic usage examples for the RovingBandit library.

Demonstrates the new OOP API for regret minimization, best-arm identification,
and variance minimization objectives.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

from rovingbandit import (
    BanditEnvironment,
    UCB1,
    ThompsonSampling,
    EpsilonGreedy,
    RandomPolicy,
    RegretMinimization,
    BestArmIdentification,
    VarianceMinimization,
    OnlineRunner,
    BatchedRunner,
)

# %%


def example_1_regret_minimization():
    """Example 1: Regret minimization with multiple policies."""
    print("\n" + "=" * 70)
    print("Example 1: Regret Minimization")
    print("=" * 70)

    env = BanditEnvironment(
        n_arms=5,
        arm_means=np.array([0.1, 0.3, 0.5, 0.4, 0.2]),
        seed=42,
    )

    policies = {
        "Random": RandomPolicy(n_arms=5, seed=42),
        "Greedy": EpsilonGreedy(n_arms=5, epsilon=0.0, seed=42),
        "Epsilon-Greedy": EpsilonGreedy(n_arms=5, epsilon=0.1, seed=42),
        "UCB1": UCB1(n_arms=5, seed=42),
        "Thompson": ThompsonSampling(n_arms=5, seed=42),
    }

    objective = RegretMinimization(optimal_reward=0.5)
    runner = OnlineRunner()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, policy in policies.items():
        env.reset_rng(42)
        result = runner.run(policy, env, n_steps=1000, objective=objective)

        print(
            f"{name:15} | Final Regret: {result.final_regret:.2f} | Avg Reward: {result.average_reward:.3f}"
        )

        result.plot(metric="cumulative_regret", ax=axes[0], label=name)
        result.plot(metric="average_reward", ax=axes[1], label=name)

    axes[0].legend()
    axes[0].set_title("Cumulative Regret Over Time")
    axes[1].legend()
    axes[1].set_title("Average Reward Over Time")
    plt.tight_layout()
    plt.savefig("example_1_regret_minimization.png", dpi=150)
    print("Plot saved to: example_1_regret_minimization.png")


# %%


def example_2_best_arm_identification():
    """Example 2: Best-arm identification with early stopping."""
    print("\n" + "=" * 70)
    print("Example 2: Best-Arm Identification")
    print("=" * 70)

    env = BanditEnvironment(
        n_arms=3,
        arm_means=np.array([0.3, 0.5, 0.9]),
        seed=42,
    )

    policy = ThompsonSampling(n_arms=3, seed=42)
    objective = BestArmIdentification(confidence_threshold=0.95, n_mc_samples=500, seed=42)
    runner = OnlineRunner()

    result = runner.run(policy, env, n_steps=2000, objective=objective, early_stopping=True)

    print(f"Samples used: {result.n_steps}")
    print(f"Best arm identified: {result.metadata['best_arm']}")
    print(f"Confidence: {result.metadata['confidence']:.3f}")
    print(f"True best arm: {env.get_optimal_arm()}")
    print(f"Correct: {result.metadata['best_arm'] == env.get_optimal_arm()}")

    fig, ax = plt.subplots(figsize=(10, 6))
    result.plot(metric="arm_pulls", ax=ax)
    ax.set_title("Arm Pulls Over Time (Best-Arm Identification)")
    plt.tight_layout()
    plt.savefig("example_2_best_arm_identification.png", dpi=150)
    print("Plot saved to: example_2_best_arm_identification.png")

# %%

def example_3_budget_constraint():
    """Example 3: Budget-constrained bandits."""
    print("\n" + "=" * 70)
    print("Example 3: Budget-Constrained Bandits")
    print("=" * 70)

    env = BanditEnvironment(
        n_arms=4,
        arm_means=np.array([0.3, 0.5, 0.7, 0.6]),
        costs=np.array([1.0, 2.0, 3.0, 1.5]),
        seed=42,
    )

    policy = ThompsonSampling(n_arms=4, seed=42)
    runner = OnlineRunner()

    result = runner.run_with_budget(policy, env, budget=100.0)

    print(f"Budget: 100.0")
    print(f"Total cost: {result.total_cost:.2f}")
    print(f"Pulls: {result.n_steps}")
    print(f"Total reward: {result.total_reward:.1f}")
    print(f"Avg reward per unit cost: {result.total_reward / result.total_cost:.3f}")

    print("\nPulls per arm:")
    for i, count in enumerate(result.policy_state["counts"]):
        mean = env.arm_means[i]
        cost = env.costs[i]
        value = result.policy_state["values"][i]
        print(
            f"  Arm {i}: {int(count):3d} pulls | True mean: {mean:.2f} | Cost: {cost:.2f} | Estimated: {value:.3f}"
        )

# %%

def example_4_batched_mode():
    """Example 4: Batched (parallel) mode."""
    print("\n" + "=" * 70)
    print("Example 4: Batched Mode")
    print("=" * 70)

    env = BanditEnvironment(
        n_arms=5,
        arm_means=np.array([0.1, 0.3, 0.5, 0.4, 0.2]),
        seed=42,
    )

    policy = UCB1(n_arms=5, seed=42)
    runner = BatchedRunner()

    result = runner.run(policy, env, batch_size=10, n_batches=50)

    print(f"Batch size: 10")
    print(f"Number of batches: 50")
    print(f"Total pulls: {result.n_steps}")
    print(f"Total reward: {result.total_reward:.1f}")
    print(f"Average reward: {result.average_reward:.3f}")


def example_5_variance_minimization():
    """Example 5: Variance minimization with group constraints."""
    print("\n" + "=" * 70)
    print("Example 5: Variance Minimization")
    print("=" * 70)

    env = BanditEnvironment(
        n_arms=6,
        arm_means=np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        arm_groups=np.array([0, 0, 0, 1, 1, 1]),
        seed=42,
    )

    policy = ThompsonSampling(n_arms=6, seed=42)
    objective = VarianceMinimization(target_shares=np.array([0.5, 0.5]))
    runner = OnlineRunner()

    result = runner.run(policy, env, n_steps=500, objective=objective)

    print(f"Total pulls: {result.n_steps}")
    print(f"Estimation variance: {result.metadata['estimation_variance']:.4f}")

    if "group_shares" in result.metadata:
        shares = result.metadata["group_shares"]
        print(f"Group 0 share: {shares[0]:.3f} (target: 0.5)")
        print(f"Group 1 share: {shares[1]:.3f} (target: 0.5)")

# %%
if __name__ == "__main__":
    print("\nRovingBandit Library - Usage Examples")
    print("=" * 70)

    example_1_regret_minimization()
    example_2_best_arm_identification()
    example_3_budget_constraint()
    example_4_batched_mode()
    example_5_variance_minimization()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)

# %%
