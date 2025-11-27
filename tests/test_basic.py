"""Basic smoke tests for core functionality."""

import numpy as np
import pytest

from rovingbandit import (
    BanditEnvironment,
    RandomPolicy,
    GreedyPolicy,
    EpsilonGreedy,
    ExploreFirst,
    UCB1,
    ThompsonSampling,
    RegretMinimization,
    BestArmIdentification,
    VarianceMinimization,
    OnlineRunner,
    BatchedRunner,
)


class TestBanditEnvironment:
    """Test BanditEnvironment class."""

    def test_init(self):
        """Test initialization."""
        env = BanditEnvironment(
            n_arms=5,
            arm_means=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        assert env.n_arms == 5
        assert len(env.arm_means) == 5

    def test_pull(self):
        """Test pulling an arm."""
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.1, 0.5, 0.9]),
            seed=42,
        )
        reward, cost = env.pull(1)
        assert reward in [0.0, 1.0]
        assert cost == 1.0

    def test_optimal_arm(self):
        """Test finding optimal arm."""
        env = BanditEnvironment(
            n_arms=5,
            arm_means=np.array([0.1, 0.2, 0.5, 0.4, 0.3]),
        )
        assert env.get_optimal_arm() == 2
        assert env.get_optimal_reward() == 0.5

    def test_costs(self):
        """Test arm costs."""
        costs = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
        env = BanditEnvironment(
            n_arms=5,
            arm_means=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            costs=costs,
        )
        _, cost = env.pull(1)
        assert cost == 2.0


class TestPolicies:
    """Test policy implementations."""

    def test_random_policy(self):
        """Test random policy."""
        policy = RandomPolicy(n_arms=5, seed=42)
        arm = policy.select_arm()
        assert 0 <= arm < 5

        policy.update(arm, 1.0)
        assert policy.counts[arm] == 1
        assert policy.values[arm] == 1.0

    def test_greedy_policy(self):
        """Test greedy policy."""
        policy = GreedyPolicy(n_arms=5, seed=42)

        for i in range(5):
            policy.update(i, i * 0.1)

        arm = policy.select_arm()
        assert arm == 4

    def test_epsilon_greedy(self):
        """Test epsilon-greedy policy."""
        policy = EpsilonGreedy(n_arms=5, epsilon=0.1, seed=42)

        for i in range(5):
            policy.update(i, i * 0.1)

        arm = policy.select_arm()
        assert 0 <= arm < 5

    def test_explore_first(self):
        """Test explore-first policy."""
        policy = ExploreFirst(n_arms=5, exploration_fraction=0.2, horizon=100, seed=42)

        arms = [policy.select_arm() for _ in range(30)]
        for arm in arms[:20]:
            policy.update(arm, np.random.random())

        assert policy.total_pulls == 20

    def test_ucb1(self):
        """Test UCB1 policy."""
        policy = UCB1(n_arms=5, seed=42)

        for i in range(10):
            arm = policy.select_arm()
            policy.update(arm, np.random.random())

        assert policy.total_pulls == 10

    def test_thompson_sampling(self):
        """Test Thompson Sampling."""
        policy = ThompsonSampling(n_arms=5, seed=42)

        for i in range(10):
            arm = policy.select_arm()
            reward = 1.0 if i % 2 == 0 else 0.0
            policy.update(arm, reward)

        assert policy.total_pulls == 10
        assert np.sum(policy.successes) > 0


class TestObjectives:
    """Test objective implementations."""

    def test_regret_minimization(self):
        """Test regret minimization objective."""
        obj = RegretMinimization(optimal_reward=0.9)

        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]), seed=42)
        policy = RandomPolicy(n_arms=3, seed=42)
        runner = OnlineRunner()

        result = runner.run(policy, env, n_steps=100, objective=obj)

        assert result.final_regret is not None
        assert result.final_regret >= 0

    def test_best_arm_identification(self):
        """Test best-arm identification objective."""
        obj = BestArmIdentification(confidence_threshold=0.8, n_mc_samples=100, seed=42)

        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]), seed=42)
        policy = ThompsonSampling(n_arms=3, seed=42)
        runner = OnlineRunner()

        result = runner.run(policy, env, n_steps=200, objective=obj)

        assert "confidence" in result.metadata
        assert "best_arm" in result.metadata

    def test_variance_minimization(self):
        """Test variance minimization objective."""
        obj = VarianceMinimization()

        env = BanditEnvironment(n_arms=3, arm_means=np.array([0.1, 0.5, 0.9]), seed=42)
        policy = UCB1(n_arms=3, seed=42)
        runner = OnlineRunner()

        result = runner.run(policy, env, n_steps=100, objective=obj)

        assert "estimation_variance" in result.metadata


class TestRunners:
    """Test runner implementations."""

    def test_online_runner(self):
        """Test online runner."""
        env = BanditEnvironment(
            n_arms=5,
            arm_means=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            seed=42,
        )
        policy = UCB1(n_arms=5, seed=42)
        runner = OnlineRunner()

        result = runner.run(policy, env, n_steps=100)

        assert result.n_steps == 100
        assert len(result.history.arms) == 100
        assert result.total_reward >= 0

    def test_online_runner_with_budget(self):
        """Test online runner with budget constraint."""
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.1, 0.5, 0.9]),
            costs=np.array([1.0, 2.0, 1.5]),
            seed=42,
        )
        policy = ThompsonSampling(n_arms=3, seed=42)
        runner = OnlineRunner()

        result = runner.run_with_budget(policy, env, budget=50.0)

        assert result.total_cost <= 50.0
        assert "budget" in result.metadata

    def test_batched_runner(self):
        """Test batched runner."""
        env = BanditEnvironment(
            n_arms=5,
            arm_means=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            seed=42,
        )
        policy = EpsilonGreedy(n_arms=5, epsilon=0.1, seed=42)
        runner = BatchedRunner()

        result = runner.run(policy, env, batch_size=10, n_batches=5)

        assert result.n_steps == 50
        assert "batch_size" in result.metadata
        assert result.metadata["batch_size"] == 10


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_regret_minimization_workflow(self):
        """Test complete regret minimization workflow."""
        env = BanditEnvironment(
            n_arms=5,
            arm_means=np.array([0.1, 0.3, 0.5, 0.4, 0.2]),
            seed=42,
        )

        policies = {
            "Random": RandomPolicy(n_arms=5, seed=42),
            "Greedy": GreedyPolicy(n_arms=5, seed=42),
            "UCB1": UCB1(n_arms=5, seed=42),
            "Thompson": ThompsonSampling(n_arms=5, seed=42),
        }

        objective = RegretMinimization(optimal_reward=0.5)
        runner = OnlineRunner()

        results = {}
        for name, policy in policies.items():
            env.reset_rng(42)
            result = runner.run(policy, env, n_steps=500, objective=objective)
            results[name] = result

        for name, result in results.items():
            assert result.final_regret is not None
            assert result.final_regret >= 0

        assert results["Thompson"].final_regret < results["Random"].final_regret

    def test_best_arm_identification_workflow(self):
        """Test best-arm identification workflow."""
        env = BanditEnvironment(
            n_arms=3,
            arm_means=np.array([0.3, 0.5, 0.9]),
            seed=42,
        )

        policy = ThompsonSampling(n_arms=3, seed=42)
        objective = BestArmIdentification(
            confidence_threshold=0.9, n_mc_samples=500, seed=42
        )
        runner = OnlineRunner()

        result = runner.run(
            policy, env, n_steps=1000, objective=objective, early_stopping=True
        )

        assert result.n_steps <= 1000
        assert result.metadata["confidence"] >= 0.9
        assert result.metadata["best_arm"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
