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
    EpsilonNeymanAllocation,
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


def example_6_epsilon_neyman_variance():
    """Example 6: Neyman allocation for lower standard errors."""
    print("\n" + "=" * 70)
    print("Example 6: Epsilon-Neyman Allocation")
    print("=" * 70)

    # Two-arm Bernoulli experiment with different variances
    # Arm 0: p = 0.05 (sigma ~ 0.218), Arm 1: p = 0.35 (sigma ~ 0.477)
    arm_means = np.array([0.05, 0.35])
    n_steps = 1000
    exploration_fraction = 0.2

    env = BanditEnvironment(n_arms=2, arm_means=arm_means, seed=123)
    runner = OnlineRunner()

    def ate_standard_error(values, counts) -> float:
        """Wald SE for difference in Bernoulli means."""
        p0, p1 = values
        n0, n1 = np.maximum(counts, 1)  # avoid division by zero
        var = p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1
        return float(np.sqrt(var))

    # Uniform random assignment baseline
    env.reset_rng(123)
    random_result = runner.run(
        RandomPolicy(n_arms=2, seed=123),
        env,
        n_steps=n_steps,
        objective=VarianceMinimization(),
    )

    # Epsilon-Neyman allocation
    neyman_policy = EpsilonNeymanAllocation(
        n_arms=2,
        exploration_fraction=exploration_fraction,
        horizon=n_steps,
        seed=123,
    )
    env.reset_rng(123)
    neyman_result = runner.run(
        neyman_policy,
        env,
        n_steps=n_steps,
        objective=VarianceMinimization(),
    )

    random_counts = random_result.policy_state["counts"]
    neyman_counts = neyman_result.policy_state["counts"]

    random_se = ate_standard_error(random_result.policy_state["values"], random_counts)
    neyman_se = ate_standard_error(neyman_result.policy_state["values"], neyman_counts)

    print(f"Arm means (true): {arm_means}")
    print(f"Random counts: {random_counts.astype(int)} | share: {random_counts / n_steps}")
    print(f"Neyman counts: {neyman_counts.astype(int)} | share: {neyman_counts / n_steps}")
    print(f"SE (random): {random_se:.4f}")
    print(f"SE (epsilon-Neyman): {neyman_se:.4f}")
    print(
        f"SE reduction vs random: {(1 - neyman_se / random_se) * 100:.1f}% "
        f"(epsilon={exploration_fraction})"
    )


def example_7_epsilon_neyman_multiarm():
    """Example 7: Multi-arm Neyman allocation (variance-oriented design)."""
    print("\n" + "=" * 70)
    print("Example 7: Multi-Arm Epsilon-Neyman Allocation")
    print("=" * 70)

    # Five arms with heterogeneous variances
    arm_means = np.array([0.05, 0.15, 0.35, 0.6, 0.8])
    n_steps = 3000
    exploration_fraction = 0.2

    env = BanditEnvironment(n_arms=5, arm_means=arm_means, seed=202)
    runner = OnlineRunner()

    def variance_score(values, counts) -> float:
        """Sum of per-arm Bernoulli variances over allocated samples."""
        sigmas_sq = np.clip(values * (1 - values), 1e-8, None)
        counts = np.maximum(counts, 1)
        return float(np.sum(sigmas_sq / counts))

    # Baseline: uniform randomization
    env.reset_rng(202)
    random_result = runner.run(
        RandomPolicy(n_arms=5, seed=202),
        env,
        n_steps=n_steps,
        objective=VarianceMinimization(),
    )

    # Epsilon-Neyman allocation
    neyman_policy = EpsilonNeymanAllocation(
        n_arms=5,
        exploration_fraction=exploration_fraction,
        horizon=n_steps,
        seed=202,
    )
    env.reset_rng(202)
    neyman_result = runner.run(
        neyman_policy,
        env,
        n_steps=n_steps,
        objective=VarianceMinimization(),
    )

    random_counts = random_result.policy_state["counts"]
    neyman_counts = neyman_result.policy_state["counts"]

    random_score = variance_score(random_result.policy_state["values"], random_counts)
    neyman_score = variance_score(neyman_result.policy_state["values"], neyman_counts)

    print(f"Arm means (true): {arm_means}")
    print(f"Random counts:  {random_counts.astype(int)} | shares: {random_counts / n_steps}")
    print(f"Neyman counts:  {neyman_counts.astype(int)} | shares: {neyman_counts / n_steps}")
    print(f"Variance score (lower is better) - random: {random_score:.6f}")
    print(f"Variance score (lower is better) - epsilon-Neyman: {neyman_score:.6f}")
    print(
        f"Variance score reduction vs random: {(1 - neyman_score / random_score) * 100:.1f}% "
        f"(epsilon={exploration_fraction})"
    )

# %%
if __name__ == "__main__":
    print("\nRovingBandit Library - Usage Examples")
    print("=" * 70)

    example_1_regret_minimization()
    example_2_best_arm_identification()
    example_3_budget_constraint()
    example_4_batched_mode()
    example_5_variance_minimization()
    example_6_epsilon_neyman_variance()
    example_7_epsilon_neyman_multiarm()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)

# %%
