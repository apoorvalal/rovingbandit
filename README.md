# RovingBandit

A flexible Python library for multi-armed bandit algorithms supporting regret minimization, best-arm identification, and variance minimization in both online and batched modes.

## Installation

### Development Installation

```bash
# Create virtual environment with uv
uv venv

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Activate the environment
source .venv/bin/activate
```

### Production Installation

```bash
uv pip install rovingbandit
```

## Quick Start

```python
import numpy as np
from rovingbandit import pick_arm, sim_runner
import matplotlib.pyplot as plt

# Define bandit arms with different success probabilities
arm_means = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1])

# Compare different bandit strategies
fig, ax = plt.subplots(figsize=(10, 6))
sim_runner(
    budget=None,
    arm_means=arm_means,
    costs=None,
    bandits=['greedy', 'egreedy', 'ucb', 'thompson'],
    ax=ax,
    number_of_arms=10,
    number_of_pulls=1000,
)
ax.legend()
plt.show()
```

## Available Algorithms

### Regret Minimization
- `random`: Random arm selection
- `greedy`: Always pick best estimated arm
- `egreedy`: Epsilon-greedy exploration
- `efirst`: Explore-first strategy
- `ucb`: Upper Confidence Bound (UCB1)
- `thompson`: Thompson Sampling (Beta-Bernoulli)

### Budget-Aware
- `fracKUBE`: Cost-aware UCB
- `thompsonBC`: Thompson Sampling with budget constraints

### Best-Arm Identification
- `thompsonTopTwo`: Top-two Thompson Sampling

### Variance Minimization
- `kasySautmann`: Variance-focused allocation

## Core Functions

### `pick_arm()`
Select a single arm based on current estimates and strategy.

```python
from rovingbandit import pick_arm

arm = pick_arm(
    q_values=q_values,      # Current value estimates
    counts=counts,          # Pull counts per arm
    strategy='thompson',    # Algorithm to use
    success=success,        # Successes per arm (for Thompson)
    failure=failure,        # Failures per arm (for Thompson)
    costs=costs,           # Cost per arm (optional)
    share_elapsed=0.5,     # Fraction of horizon elapsed
)
```

### `sim_runner()`
Run comparative simulations of multiple algorithms.

```python
from rovingbandit import sim_runner
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sim_runner(
    budget=1000,           # Total budget (or None for fixed pulls)
    arm_means=arm_means,   # True arm probabilities
    costs=costs,          # Cost per arm (required if budget set)
    bandits=['ucb', 'thompson', 'egreedy'],
    ax=ax,
    number_of_arms=10,
    number_of_pulls=5000,
)
```

### `arm_sequence()`
Track detailed arm pull sequences over time.

```python
from rovingbandit import arm_sequence

pull_mat, rewards, counts = arm_sequence(
    budget=None,
    bandit='thompson',
    arm_means=arm_means,
    number_of_pulls=1000,
)
```

### `best_arm()`
Best-arm identification with confidence thresholds.

```python
from rovingbandit import best_arm

alpha_matrix = best_arm(
    arm_means=arm_means,
    bandit='Thompson',
    M=1000,                # Monte Carlo samples
    conf_threshold=0.95,   # Confidence threshold
)
```

### Representation-Constrained Bandits

```python
from rovingbandit import rep_bandit_cost

shares, costmat, counts = rep_bandit_cost(
    bandit='thompsonBC',
    arm_means=arm_means,
    pay_levels=np.array([1.0, 2.0, 3.0]),  # Pay structure
    B=1000,                                  # Budget
    target_shares=np.array([0.5, 0.5]),     # Target representation
    gamma=2.0,                               # Cost scaling parameter
)
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format and lint
ruff check .
ruff format .

# Type checking
mypy src/
```

## Project Structure

```
rovingbandit/
├── src/
│   └── rovingbandit/
│       ├── __init__.py
│       └── banditry.py
├── tests/
├── pyproject.toml
├── SPEC.md          # Detailed specification
└── README.md
```

## Documentation

See [SPEC.md](SPEC.md) for:
- Detailed algorithm descriptions
- Academic references
- API specification for planned modular architecture
- Implementation roadmap

## Known Issues

The current implementation has some correctness issues being addressed:
1. `thompsonTopTwo` has a logic error in the while loop (line 98 in banditry.py)
2. Strategy selection uses `if` instead of `elif` (should be fixed)
3. `rep_bandit_rake` entropy calculation needs correction

See SPEC.md for full list and planned refactoring.

## License

MIT

## Contributing

This library is under active development. The current monolithic design will be refactored into a modular architecture as specified in SPEC.md.
