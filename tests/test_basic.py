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
    BudgetedUCB,
    ThompsonSampling,
    BudgetedThompsonSampling,
    EpsilonNeymanAllocation,
    LUCB,
    KasySautmann,
    LinUCB,
    RepresentationBandit,
    TopTwoThompson,
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

    def test_budgeted_ucb(self):
        """Test Budgeted UCB policy."""
        policy = BudgetedUCB(n_arms=3, seed=42)
        
        # Test initialization
        assert np.all(policy.avg_costs == 0.0)
        
        # Pull each arm once
        for i in range(3):
            arm = policy.select_arm()
            assert arm == i
            # Arm 0: low cost, high reward
            # Arm 1: high cost, high reward
            # Arm 2: low cost, low reward
            costs = [1.0, 10.0, 1.0]
            rewards = [1.0, 1.0, 0.0]
            policy.update(arm, rewards[i], costs[i])
            
        # Verify updates
        assert np.allclose(policy.avg_costs, [1.0, 10.0, 1.0])
        assert np.allclose(policy.values, [1.0, 1.0, 0.0])
        
        # Next pull should favor arm 0 (high value / low cost) over arm 1 (high value / high cost)
        # UCB numerator will be similar for arm 0 and 1, but cost divides it.
        # Arm 0 score ~ (1 + bonus) / 1
        # Arm 1 score ~ (1 + bonus) / 10
        # Arm 2 score ~ (0 + bonus) / 1
        
        arm = policy.select_arm()
        assert arm == 0

    def test_thompson_sampling(self):
        """Test Thompson Sampling."""
        policy = ThompsonSampling(n_arms=5, seed=42)

        for i in range(10):
            arm = policy.select_arm()
            reward = 1.0 if i % 2 == 0 else 0.0
            policy.update(arm, reward)

        assert policy.total_pulls == 10
        assert np.sum(policy.successes) > 0

    def test_budgeted_thompson_sampling(self):
        """Test Budgeted Thompson Sampling."""
        # Arm 0: High Reward (0.9), High Cost (10.0) -> Ratio 0.09
        # Arm 1: Med Reward (0.5), Low Cost (1.0) -> Ratio 0.5
        costs = np.array([10.0, 1.0])
        policy = BudgetedThompsonSampling(n_arms=2, costs=costs, seed=42)

        # Simulate some data to make posteriors concentrated
        # Arm 0: 9 successes, 1 failure
        policy.successes[0] = 90
        policy.failures[0] = 10
        # Arm 1: 5 successes, 5 failures
        policy.successes[1] = 50
        policy.failures[1] = 50

        # Posterior means: ~0.9 and ~0.5
        # Ratios: ~0.09 and ~0.5
        # Should pick arm 1 consistently
        
        counts = np.zeros(2)
        for _ in range(20):
            arm = policy.select_arm()
            counts[arm] += 1
            
        assert counts[1] > counts[0]

    def test_representation_bandit(self):
        """Test Representation Bandit."""
        # 2 Groups, 2 Arms. Arm 0 in Group 0, Arm 1 in Group 1.
        # Costs equal.
        # Target shares: 0.2 (Group 0), 0.8 (Group 1).
        # Initially, it should favor Arm 1 (Group 1) if shares are unbalanced.
        
        arm_groups = np.array([0, 1])
        target_shares = np.array([0.2, 0.8])
        costs = np.array([1.0, 1.0])
        total_budget = 100.0
        
        policy = RepresentationBandit(
            n_arms=2,
            arm_groups=arm_groups,
            target_shares=target_shares,
            total_budget=total_budget,
            costs=costs,
            seed=42
        )
        
        # Simulate equal pulls (50/50 split), which is over-representation for Group 0 (target 0.2)
        policy.counts[0] = 50
        policy.counts[1] = 50
        policy.total_pulls = 100
        policy.total_cost_incurred = 50.0 # 50% budget spent
        
        # Group 0 Share: 0.5. Target 0.2. Over by 0.3.
        # Group 1 Share: 0.5. Target 0.8. Under by 0.3.
        
        # Cost adjustment:
        # Group 0: 1.0 * (1 + 0.5 * 0.3)^2 = 1.0 * (1.15)^2 = 1.32
        # Group 1: 1.0 * (1 + 0.5 * -0.3)^2 = 1.0 * (0.85)^2 = 0.72
        
        # Ratio for Arm 0 will be penalized. Ratio for Arm 1 boosted.
        
        # Set posteriors equal
        policy.successes[:] = 10
        policy.failures[:] = 10
        
        # Should pick Arm 1
        arm = policy.select_arm()
        assert arm == 1

    def test_top_two_thompson(self):
        """Test Top-Two Thompson Sampling."""
        policy = TopTwoThompson(n_arms=5, psi=0.5, seed=42)

        # Basic selection test
        arm = policy.select_arm()
        assert 0 <= arm < 5

        # Update test
        policy.update(arm, 1.0)
        assert policy.counts[arm] == 1
        assert policy.successes[arm] == 1.0

        # State dict test
        state = policy.get_state()
        assert state["psi"] == 0.5
        assert state["max_resamples"] == 100

        # Re-initialize and restore
        new_policy = TopTwoThompson(n_arms=5)
        new_policy.set_state(state)
        assert new_policy.psi == 0.5
        assert np.array_equal(new_policy.successes, policy.successes)

    def test_lucb(self):
        """Test LUCB policy for best-arm identification."""
        policy = LUCB(n_arms=3, exploration_factor=2.0, seed=0)

        # Initial pulls should cover all arms
        pulled = []
        for _ in range(3):
            arm = policy.select_arm()
            pulled.append(arm)
            policy.update(arm, reward=0.0)
        assert set(pulled) == {0, 1, 2}

        # Set empirical means to favor arm 2 but with uncertainty on arm 1
        policy.values = np.array([0.2, 0.6, 0.8])
        policy.counts = np.array([50.0, 5.0, 50.0])
        policy.total_pulls = 105

        arm = policy.select_arm()
        # Should focus on either the leader (2) or the challenger (1)
        assert arm in (1, 2)

    def test_kasy_sautmann(self):
        """Test Kasy-Sautmann variance allocation with welfare constraint."""
        policy = KasySautmann(n_arms=2, welfare_threshold=0.75, smoothing=1e-4, seed=1)

        # After initialization, pull unobserved arm
        arm = policy.select_arm()
        assert 0 <= arm < 2

        # Set estimates: arm 0 below welfare floor, arm 1 above
        policy.values = np.array([0.2, 0.9])
        policy.counts = np.array([20.0, 20.0])
        policy.total_pulls = 40

        probs = policy._allocation_probabilities()
        assert probs[1] > probs[0]
        assert np.isclose(np.sum(probs), 1.0)

        draws = [policy.select_arm() for _ in range(5)]
        assert all(d in (0, 1) for d in draws)

    def test_linucb(self):
        """Test LinUCB with simple linear rewards."""
        contexts = np.array([[1.0, 0.0], [0.0, 1.0]])
        theta = np.array([1.0, 0.5])

        def reward_fn(arm, rng):
            return float(np.dot(contexts[arm], theta))

        env = BanditEnvironment(n_arms=2, arm_means=None, contexts=contexts, reward_fn=reward_fn)
        policy = LinUCB(n_arms=2, alpha=0.1, seed=0)
        runner = OnlineRunner()

        result = runner.run(policy, env, n_steps=50)

        counts = result.policy_state["counts"]
        assert counts[0] > counts[1]  # arm 0 has higher expected reward

        # Estimated values should reflect the linear reward ordering
        assert result.policy_state["values"][0] > result.policy_state["values"][1]

    def test_epsilon_neyman_allocation(self):
        """Test epsilon-Neyman allocation policy."""
        policy = EpsilonNeymanAllocation(
            n_arms=3, exploration_fraction=0.2, horizon=50, seed=42
        )

        # Exploration phase should return a valid arm
        arm = policy.select_arm()
        assert 0 <= arm < 3

        # Move into allocation phase with known estimates
        policy.counts = np.array([10.0, 10.0, 10.0])
        policy.values = np.array([0.1, 0.5, 0.9])
        policy.total_pulls = 30  # past exploration threshold

        probs = policy._allocation_probabilities()
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0)
        # Highest variance at p=0.5 => arm 1 should have highest allocation
        assert probs[1] > probs[0]
        assert probs[1] > probs[2]

        draws = [policy.select_arm() for _ in range(10)]
        assert all(0 <= draw < 3 for draw in draws)


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
