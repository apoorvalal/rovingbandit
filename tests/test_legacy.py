"""Tests for backward compatibility with legacy banditry.py functions."""

import numpy as np
import pytest
from rovingbandit import pick_arm


class TestLegacyFunctions:
    """Test that legacy functions still work."""

    def test_pick_arm_random(self):
        """Test legacy pick_arm with random strategy."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        arm = pick_arm(q_values, counts, "random", success, failure, costs, 0.5)
        assert 0 <= arm < 5

    def test_pick_arm_greedy(self):
        """Test legacy pick_arm with greedy strategy."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        arm = pick_arm(q_values, counts, "greedy", success, failure, costs, 0.5)
        assert arm == 4

    def test_pick_arm_ucb(self):
        """Test legacy pick_arm with UCB strategy."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        arm = pick_arm(q_values, counts, "ucb", success, failure, costs, 0.5)
        assert 0 <= arm < 5

    def test_pick_arm_thompson(self):
        """Test legacy pick_arm with Thompson sampling."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        arm = pick_arm(q_values, counts, "thompson", success, failure, costs, 0.5)
        assert 0 <= arm < 5

    def test_pick_arm_egreedy(self):
        """Test legacy pick_arm with epsilon-greedy."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        arm = pick_arm(q_values, counts, "egreedy", success, failure, costs, 0.5)
        assert 0 <= arm < 5

    def test_pick_arm_efirst(self):
        """Test legacy pick_arm with explore-first."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # In explore phase (share_elapsed < eta=0.1), should explore
        arm = pick_arm(q_values, counts, "efirst", success, failure, costs, 0.05)
        assert 0 <= arm < 5

        # In exploit phase (share_elapsed > eta=0.1), should pick best
        arm = pick_arm(q_values, counts, "efirst", success, failure, costs, 0.95)
        assert arm == 4

    def test_pick_arm_return_draws(self):
        """Test legacy pick_arm returning draws."""
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        counts = np.array([10, 10, 10, 10, 10])
        success = np.array([1, 2, 3, 4, 5])
        failure = np.array([9, 8, 7, 6, 5])
        costs = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        draws = pick_arm(
            q_values, counts, "thompson", success, failure, costs, 0.5, return_what="draws"
        )
        assert len(draws) == 5
        assert all(0 <= d <= 1 for d in draws)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
