# RovingBandit API Reference

This document details the programmatic API of `rovingbandit`.

## Core Abstractions

### `BanditEnvironment`
The environment simulates the multi-armed bandit problem.

```python
class BanditEnvironment:
    def __init__(
        self,
        n_arms: int,
        arm_means: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ): ...

    def pull(self, arm: int) -> Tuple[float, float]:
        """
        Executes an action.
        Returns: (reward, cost)
        """
        ...
```

### `Policy`
Abstract base class for all bandit algorithms.

```python
class Policy(ABC):
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """Decide which arm to pull next."""
        ...

    def update(self, arm: int, reward: float, cost: float = 0.0):
        """Update internal state with observation."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        ...
```

### `Objective`
Defines the goal of the simulation and computes metrics.

*   `RegretMinimization(optimal_reward: float)`
*   `BestArmIdentification(confidence_threshold: float, ...)`
*   `VarianceMinimization(target_shares: np.ndarray)`

### `Runner`
Orchestrates the interaction between Policy and Environment.

*   **`OnlineRunner`**: Sequential execution.
    ```python
    def run(
        self,
        policy: Policy,
        environment: BanditEnvironment,
        n_steps: int,
        objective: Objective
    ) -> Result: ...
    ```
*   **`BatchedRunner`**: Parallel execution (multiple arms pulled per "step").

---

## Policies

All policies are located in `rovingbandit.policies`.

| Class | Type | Description |
| :--- | :--- | :--- |
| `RandomPolicy` | Baseline | Uniform random selection. |
| `GreedyPolicy` | Baseline | Pure exploitation of empirical mean. |
| `EpsilonGreedy` | Regret | Random with prob $\epsilon$, greedy otherwise. |
| `ExploreFirst` | Regret | Pure exploration phase followed by exploitation. |
| `UCB1` | Regret | Optimism in face of uncertainty (Upper Confidence Bound). |
| `ThompsonSampling` | Regret | Bayesian posterior sampling (Beta-Bernoulli). |
| `TopTwoThompson` | BAI | Modified Thompson Sampling for Best-Arm Identification. |
| `BudgetedUCB` | Budget | Cost-aware UCB for knapsack-like constraints. |
| `EpsilonNeymanAllocation` | Variance | Explore uniformly then allocate by estimated standard deviation (horizon required; suited for K ≥ 3). |
| `LUCB` | BAI | Lower-Upper Confidence Bound for best-arm identification. |
| `KasySautmann` | Variance | Variance-focused allocation with welfare threshold. |
| `LinUCB` | Contextual | Linear UCB for contextual bandits. |

---

## Usage Example

```python
import numpy as np
from rovingbandit import (
    BanditEnvironment,
    ThompsonSampling,
    RegretMinimization,
    OnlineRunner
)

# 1. Setup
env = BanditEnvironment(n_arms=5, arm_means=np.array([0.1, 0.3, 0.5, 0.2, 0.4]))
policy = ThompsonSampling(n_arms=5)
objective = RegretMinimization(optimal_reward=0.5)

# 2. Run
runner = OnlineRunner()
result = runner.run(policy, env, n_steps=1000, objective=objective)

# 3. Analyze
print(f"Final Regret: {result.final_regret}")
# result.plot() # If matplotlib is installed
```
