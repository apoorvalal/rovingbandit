# RovingBandit Library Specification

## Executive Summary

This document specifies the design of RovingBandit, a Python library for multi-armed bandit algorithms supporting multiple objectives (regret minimization, best-arm identification, variance minimization) in both online and batched modes, with extensions for budget constraints and representation constraints.

## 1. Motivation

The current `banditry.py` implementation provides useful functionality but suffers from:
- Monolithic design with string-based algorithm dispatch
- Mixed concerns (simulation, selection, plotting)
- Limited extensibility and composability
- No clear separation between objectives and modes

Modern bandit applications require:
- **Multiple objectives**: Beyond regret minimization to include best-arm identification and variance-based allocation
- **Operational flexibility**: Online (sequential) and batched (parallel) decision-making
- **Real-world constraints**: Budget limits, costs per arm, representation requirements
- **Extensibility**: Easy integration of new algorithms and objectives

## 2. Design Principles

### 2.1 Core Abstractions

The library follows an object-oriented design with clear separation of concerns:

```
BanditEnvironment (data model)
    |
    v
Policy (algorithm)  -->  Objective (goal)  -->  Mode (online/batched)
    |
    v
Result (outcomes + diagnostics)
```

**Key principles**:
1. **Separation of algorithm from objective**: Same algorithm (e.g., Thompson Sampling) can serve different objectives
2. **Mode-agnostic design**: Policies work in both online and batched settings via adapters
3. **Composability**: Mix and match environments, policies, and objectives
4. **Extensibility**: New algorithms inherit from base classes
5. **Type safety**: Use type hints throughout for better IDE support and runtime checking

### 2.2 Academic Foundation

The library draws on three main problem formulations:

**Regret Minimization** (Lai & Robbins, 1985; Auer et al., 2002):
- Objective: Minimize cumulative regret = sum of (optimal reward - actual reward)
- Classic algorithms: UCB1, Thompson Sampling, epsilon-greedy
- Metric: Regret R(T) = T*mu* - sum(rewards)

**Best-Arm Identification** (Even-Dar et al., 2006; Audibert & Bubeck, 2010):
- Objective: Identify best arm with high confidence using minimal samples
- Fixed confidence: Sample until P(identified arm is best) > delta
- Fixed budget: Maximize identification probability given T samples
- Algorithms: Successive Elimination, LUCB, Top-Two Thompson Sampling

**Variance Minimization** (Kasy & Sautmann, 2021; Offer-Westort et al., 2021):
- Objective: Minimize variance of treatment effect estimates
- Applications: Experimental design, adaptive trials
- Balances exploration for precision with welfare maximization
- Related: Neyman allocation, optimal experimental design

## 3. Core Components

### 3.1 Bandit Environment

```python
class BanditEnvironment:
    """
    Represents the bandit problem structure.

    Attributes:
        n_arms: int - Number of available arms
        arm_means: Optional[np.ndarray] - True means (if known/simulated)
        costs: Optional[np.ndarray] - Cost per pull for each arm
        contexts: Optional[np.ndarray] - Contextual features (for contextual bandits)
    """

    def pull(self, arm: int) -> Tuple[float, float]:
        """Pull an arm and return (reward, cost)"""

    def get_optimal_arm(self, objective: str = "reward") -> int:
        """Return optimal arm index for given objective"""
```

### 3.2 Policy Base Class

```python
class Policy(ABC):
    """
    Abstract base class for bandit policies.

    All policies maintain:
    - counts: np.ndarray - Number of times each arm pulled
    - values: np.ndarray - Estimated value for each arm
    - successes/failures: For Bernoulli bandits (optional)
    """

    @abstractmethod
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """Select next arm to pull"""

    @abstractmethod
    def update(self, arm: int, reward: float, cost: float = 0.0):
        """Update internal state after observing reward"""

    def reset(self):
        """Reset policy to initial state"""

    def get_state(self) -> Dict:
        """Return current policy state for serialization"""
```

### 3.3 Objective Interface

```python
class Objective(ABC):
    """Defines what the bandit is trying to achieve"""

    @abstractmethod
    def compute_metric(self, history: History) -> float:
        """Compute objective-specific performance metric"""

    @abstractmethod
    def stopping_criterion(self, policy: Policy, **kwargs) -> bool:
        """Determine if objective is satisfied (for BAI)"""
```

**Concrete objectives**:
- `RegretMinimization`: Track cumulative regret
- `BestArmIdentification`: Track confidence in best arm
- `VarianceMinimization`: Track estimation variance

### 3.4 Mode: Online vs Batched

```python
class OnlineRunner:
    """Sequential decision making - one arm at a time"""

    def run(self, policy: Policy, environment: BanditEnvironment,
            n_steps: int, objective: Objective) -> Result:
        """Run online for n_steps"""

class BatchedRunner:
    """Parallel decision making - multiple arms simultaneously"""

    def run(self, policy: Policy, environment: BanditEnvironment,
            batch_size: int, n_batches: int, objective: Objective) -> Result:
        """Run in batches"""
```

## 4. Algorithm Implementations

### 4.1 Regret Minimization Policies

**UCB1** (Auer et al., 2002)
```python
class UCB1(Policy):
    """
    Upper Confidence Bound algorithm.

    Selects arm maximizing: mean_i + sqrt(2*log(t) / n_i)

    Reference: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
    Finite-time analysis of the multiarmed bandit problem.
    Machine Learning, 47(2-3), 235-256.
    """
    exploration_factor: float = 2.0
```

**Thompson Sampling** (Thompson, 1933; Chapelle & Li, 2011)
```python
class ThompsonSampling(Policy):
    """
    Bayesian approach using posterior sampling.

    For Bernoulli rewards, maintains Beta(alpha, beta) posterior.
    Samples from each posterior and picks argmax.

    Reference: Chapelle, O., & Li, L. (2011). An empirical evaluation
    of thompson sampling. NIPS.
    """
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
```

**Epsilon-Greedy** (Sutton & Barto, 2018)
```python
class EpsilonGreedy(Policy):
    """
    Explores with probability epsilon, exploits otherwise.

    Variants:
    - Fixed epsilon
    - Decaying epsilon (epsilon_t = epsilon_0 / sqrt(t))
    - Adaptive epsilon
    """
    epsilon: float = 0.1
    decay: bool = False
```

**Explore-Then-Commit** (Perchet et al., 2016)
```python
class ExploreFirst(Policy):
    """
    Pure exploration for T*eta steps, then pure exploitation.

    Simple but effective for finite horizons.
    """
    exploration_fraction: float = 0.1
```

### 4.2 Best-Arm Identification Policies

**Top-Two Thompson Sampling** (Russo, 2016)
```python
class TopTwoThompson(Policy):
    """
    Samples from posteriors, picks best arm with probability psi,
    otherwise samples again to find second-best.

    Optimized for best-arm identification.

    Reference: Russo, D. (2016). Simple Bayesian algorithms for
    best arm identification. Operations Research, 68(6), 1625-1647.
    """
    psi: float = 0.5  # Probability of pulling current best
```

**LUCB** (Kalyanakrishnan et al., 2012)
```python
class LUCB(Policy):
    """
    Lower-Upper Confidence Bound for best-arm identification.

    Pulls arms to minimize uncertainty about best arm.

    Reference: Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012).
    PAC subset selection in stochastic multi-armed bandits. ICML.
    """
```

### 4.3 Variance Minimization Policies

**Kasy-Sautmann** (Kasy & Sautmann, 2021)
```python
class KasySautmann(Policy):
    """
    Minimizes variance of treatment effect estimates while maintaining
    welfare threshold.

    Balances estimation precision with reward maximization.

    Reference: Kasy, M., & Sautmann, A. (2021). Adaptive treatment
    assignment in experiments for policy choice. Econometrica, 89(1), 113-132.
    """
    welfare_threshold: float = 0.8  # Minimum fraction of optimal welfare
```

**Neyman Allocation** (Offer-Westort et al., 2021)
```python
class NeymanAllocation(Policy):
    """
    Allocates samples to minimize variance of ATE estimate.

    For arm i with variance sigma_i^2, allocates proportional to sigma_i.

    Reference: Offer-Westort, M., Coppock, A., & Green, D. P. (2021).
    Adaptive experimental design: Prospects and applications in political science.
    American Journal of Political Science, 65(4), 826-844.
    """
```

### 4.4 Budget-Constrained Policies

**Cost-Aware UCB** (Tran-Thanh et al., 2012)
```python
class BudgetedUCB(Policy):
    """
    UCB variant that accounts for arm costs.

    Maximizes: (mean_i + exploration_bonus_i) / cost_i

    Reference: Tran-Thanh, L., Chapman, A., Rogers, A., & Jennings, N. R. (2012).
    Knapsack based optimal policies for budget-limited multi-armed bandits. AAAI.
    """
```

**Thompson Sampling with Costs** (current implementation)
```python
class ThompsonSamplingBudget(ThompsonSampling):
    """
    Thompson Sampling that divides samples by normalized costs.

    Selects arm maximizing: theta_i / cost_i
    where theta_i ~ Posterior_i
    """
```

### 4.5 Representation-Constrained Policies

**Dynamic Cost Scaling** (from rep_bandit_cost)
```python
class RepresentationBandit(Policy):
    """
    Adjusts arm costs dynamically to achieve target representation.

    When group A is over-represented, increases costs for group A arms
    to discourage selection.

    cost_t = cost_0 * (1 + progress * gap)^gamma

    where gap = current_share - target_share
    """
    target_shares: np.ndarray
    gamma: float = 2.0  # Cost scaling exponent
```

## 5. API Design

### 5.1 Basic Usage

```python
from rovingbandit import BanditEnvironment, ThompsonSampling, OnlineRunner, RegretMinimization

# Define environment
env = BanditEnvironment(
    n_arms=10,
    arm_means=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
)

# Choose policy
policy = ThompsonSampling()

# Define objective
objective = RegretMinimization()

# Run online
runner = OnlineRunner()
result = runner.run(policy, env, n_steps=1000, objective=objective)

# Analyze
print(f"Cumulative regret: {result.cumulative_regret[-1]}")
result.plot()
```

### 5.2 Best-Arm Identification

```python
from rovingbandit import TopTwoThompson, BestArmIdentification

policy = TopTwoThompson(psi=0.5)
objective = BestArmIdentification(confidence_threshold=0.95)

runner = OnlineRunner()
result = runner.run(
    policy, env,
    n_steps=10000,  # Max budget
    objective=objective,
    early_stopping=True  # Stop when confidence reached
)

print(f"Best arm identified: {result.best_arm}")
print(f"Confidence: {result.confidence}")
print(f"Samples used: {result.n_steps}")
```

### 5.3 Batched Mode

```python
from rovingbandit import BatchedRunner, EpsilonGreedy

policy = EpsilonGreedy(epsilon=0.1)
runner = BatchedRunner()

result = runner.run(
    policy, env,
    batch_size=10,  # Pull 10 arms per batch
    n_batches=100,
    objective=RegretMinimization()
)
```

### 5.4 Budget Constraints

```python
env = BanditEnvironment(
    n_arms=10,
    arm_means=means,
    costs=np.array([1.0, 2.0, 1.5, ...])  # Heterogeneous costs
)

policy = BudgetedUCB()
runner = OnlineRunner()

result = runner.run_with_budget(
    policy, env,
    budget=1000.0,
    objective=RegretMinimization(),
    pay_on_success=False  # Pay regardless of outcome
)
```

### 5.5 Variance Minimization with Representation

```python
from rovingbandit import KasySautmann, VarianceMinimization

# Environment with group structure
env = BanditEnvironment(
    n_arms=6,
    arm_means=means,
    arm_groups=np.array([0, 0, 0, 1, 1, 1])  # Two groups
)

policy = KasySautmann(welfare_threshold=0.8)
objective = VarianceMinimization(
    target_shares=np.array([0.5, 0.5])  # Equal representation
)

result = runner.run(policy, env, n_steps=1000, objective=objective)
print(f"Final shares: {result.group_shares}")
print(f"Variance: {result.estimation_variance}")
```

### 5.6 Simulation and Comparison

```python
from rovingbandit import simulate_policies

policies = {
    'UCB1': UCB1(),
    'Thompson': ThompsonSampling(),
    'EpsGreedy': EpsilonGreedy(epsilon=0.1),
    'TopTwo': TopTwoThompson()
}

results = simulate_policies(
    policies=policies,
    environment=env,
    n_steps=1000,
    n_replications=100,
    objective=RegretMinimization()
)

results.plot_comparison()
results.summary_table()
```

## 6. Advanced Features

### 6.1 Contextual Bandits

```python
class ContextualPolicy(Policy):
    """Base class for contextual policies"""

    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """Context-dependent arm selection"""

class LinUCB(ContextualPolicy):
    """
    Linear UCB for contextual bandits.

    Reference: Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
    A contextual-bandit approach to personalized news article recommendation. WWW.
    """
```

### 6.2 Non-Stationary Bandits

```python
class DiscountedThompson(ThompsonSampling):
    """
    Thompson Sampling with recency weighting for non-stationary environments.

    Reference: Raj, V., & Kalyani, S. (2017). Taming non-stationary
    bandits: A Bayesian approach. arXiv:1707.09727.
    """
    discount_factor: float = 0.99
```

### 6.3 Combinatorial Bandits

```python
class CombinatorialUCB(Policy):
    """
    Selects subset of arms rather than single arm.

    Reference: Chen, W., Wang, Y., & Yuan, Y. (2013).
    Combinatorial multi-armed bandit: General framework and applications. ICML.
    """
```

## 7. Implementation Roadmap

### Phase 1: Core Infrastructure
- [x] Base classes (Policy, Objective, BanditEnvironment)
- [x] Online and Batched runners
- [x] Result class with plotting utilities
- [x] Basic unit tests

### Phase 2: Regret Minimization
- [x] UCB1, UCB-Tuned
- [x] Thompson Sampling (Beta-Bernoulli, Gaussian)
- [x] Epsilon-Greedy variants
- [x] Explore-First
- [ ] Integration tests and benchmarks

### Phase 3: Best-Arm Identification
- [x] Top-Two Thompson Sampling
- [ ] LUCB
- [ ] Successive Elimination
- [ ] Confidence tracking utilities

### Phase 4: Variance Minimization
- [ ] Kasy-Sautmann policy
- [ ] Neyman Allocation
- [ ] Variance estimation utilities
- [ ] Representation constraints

### Phase 5: Extensions
- [x] Budget constraints
- [ ] Cost heterogeneity
- [ ] Contextual bandits (LinUCB, Neural bandits)
- [ ] Non-stationary environments
- [ ] Combinatorial actions

### Phase 6: Tooling
- [ ] Visualization dashboard
- [ ] Policy comparison utilities
- [ ] Hyperparameter tuning
- [ ] Documentation and tutorials

## 8. Testing Strategy

### Unit Tests
- Each policy tested on synthetic environments
- Edge cases: single arm, identical arms, deterministic rewards
- Numerical stability tests

### Integration Tests
- End-to-end workflows for each objective
- Batched vs online consistency checks
- Serialization/deserialization

### Benchmarks
- Reproduce results from key papers (UCB1, Thompson Sampling)
- Performance regression tests
- Scalability tests (many arms, long horizons)

## 9. Documentation Requirements

### API Documentation
- Docstrings for all public methods
- Type hints throughout
- Usage examples in docstrings

### Tutorials
- Quickstart guide
- One tutorial per objective type
- Advanced features guide
- Migration guide from current code

### Academic Context
- Bibliography of key papers
- Algorithm descriptions with mathematical formulation
- Performance guarantees and theoretical properties

## 10. Key References

### Foundational
1. Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6(1), 4-22.
2. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3), 235-256.
3. Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4), 285-294.

### Regret Minimization
4. Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. NIPS.
5. Agrawal, S., & Goyal, N. (2012). Analysis of Thompson Sampling for the multi-armed bandit problem. COLT.
6. Lattimore, T., & Szepesvári, C. (2020). Bandit Algorithms. Cambridge University Press.

### Best-Arm Identification
7. Even-Dar, E., Mannor, S., & Mansour, Y. (2006). Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems. JMLR, 7, 1079-1105.
8. Audibert, J. Y., & Bubeck, S. (2010). Best arm identification in multi-armed bandits. COLT.
9. Russo, D. (2016). Simple Bayesian algorithms for best arm identification. Operations Research, 68(6), 1625-1647.
10. Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. (2012). PAC subset selection in stochastic multi-armed bandits. ICML.

### Variance Minimization & Experimental Design
11. Kasy, M., & Sautmann, A. (2021). Adaptive treatment assignment in experiments for policy choice. Econometrica, 89(1), 113-132.
12. Offer-Westort, M., Coppock, A., & Green, D. P. (2021). Adaptive experimental design: Prospects and applications in political science. American Journal of Political Science, 65(4), 826-844.
13. Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S., & Athey, S. (2021). Confidence intervals for policy evaluation in adaptive experiments. PNAS, 118(15).

### Budget & Cost Constraints
14. Tran-Thanh, L., Chapman, A., Rogers, A., & Jennings, N. R. (2012). Knapsack based optimal policies for budget-limited multi-armed bandits. AAAI.
15. Ding, W., Qin, T., Zhang, X. D., & Liu, T. Y. (2013). Multi-armed bandit with budget constraint and variable costs. AAAI.

### Contextual Bandits
16. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. WWW.
17. Agrawal, S., & Goyal, N. (2013). Thompson Sampling for contextual bandits with linear payoffs. ICML.

### Applications & Extensions
18. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
19. Slivkins, A. (2019). Introduction to multi-armed bandits. Foundations and Trends in Machine Learning, 12(1-2), 1-286.
20. Chen, W., Wang, Y., & Yuan, Y. (2013). Combinatorial multi-armed bandit: General framework and applications. ICML.

## 11. Success Metrics

The library will be considered successful when it:
1. Reproduces all functionality from current `banditry.py`
2. Passes comprehensive test suite (>90% coverage)
3. Provides 3x-5x performance improvement via vectorization
4. Supports all objective x mode combinations (6 total)
5. Includes 10+ policy implementations with academic references
6. Has complete API documentation and 5+ tutorials
7. Successfully used in at least one research application
