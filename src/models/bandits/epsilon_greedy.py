"""
Epsilon-Greedy Multi-Armed Bandit for recommendations.

This is the simplest RL approach - a great starting point to understand
the exploration-exploitation tradeoff.
"""

import numpy as np
from typing import List, Optional, Tuple


class EpsilonGreedyBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit.

    Algorithm:
    - With probability ε: Explore (choose random arm)
    - With probability 1-ε: Exploit (choose best known arm)

    Each "arm" is an item in the catalog.
    """

    def __init__(
        self,
        n_items: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Args:
            n_items: Number of items (arms) in catalog
            epsilon: Exploration probability (0 to 1)
            epsilon_decay: Multiply epsilon by this after each step (for epsilon-decay)
            min_epsilon: Minimum epsilon value
            seed: Random seed
        """
        self.n_items = n_items
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.RandomState(seed)

        # Track statistics for each item
        self.Q = np.zeros(n_items)  # Estimated reward (Q-value) for each item
        self.N = np.zeros(n_items)  # Number of times each item was selected

        # Metrics
        self.total_reward = 0.0
        self.n_steps = 0
        self.n_explorations = 0
        self.n_exploitations = 0

    def select_item(self, exclude: Optional[List[int]] = None) -> Tuple[int, bool]:
        """
        Select an item to recommend.

        Args:
            exclude: Items to exclude from selection (e.g., already shown)

        Returns:
            (item_id, is_exploration): Selected item and whether it was exploration
        """
        # Create mask for available items
        available = np.ones(self.n_items, dtype=bool)
        if exclude:
            available[exclude] = False

        available_items = np.where(available)[0]

        if len(available_items) == 0:
            raise ValueError("No available items to recommend")

        # Epsilon-greedy selection
        if self.rng.rand() < self.epsilon:
            # Explore: Random item
            item = self.rng.choice(available_items)
            is_exploration = True
            self.n_explorations += 1
        else:
            # Exploit: Best known item
            # Among available items, select the one with highest Q-value
            q_values_available = np.full(self.n_items, -np.inf)
            q_values_available[available_items] = self.Q[available_items]

            # If multiple items have same Q-value, break ties randomly
            max_q = np.max(q_values_available)
            best_items = np.where(q_values_available == max_q)[0]
            item = self.rng.choice(best_items)
            is_exploration = False
            self.n_exploitations += 1

        return item, is_exploration

    def update(self, item: int, reward: float):
        """
        Update Q-value estimates after observing reward.

        Uses incremental mean update:
        Q_new = Q_old + (1/N) * (reward - Q_old)

        Args:
            item: Item that was recommended
            reward: Observed reward (e.g., 1 for click, 0 for no click)
        """
        self.N[item] += 1
        self.Q[item] += (reward - self.Q[item]) / self.N[item]

        # Update metrics
        self.total_reward += reward
        self.n_steps += 1

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(self.n_steps, 1),
            'n_steps': self.n_steps,
            'epsilon': self.epsilon,
            'exploration_rate': self.n_explorations / max(self.n_steps, 1),
            'n_explorations': self.n_explorations,
            'n_exploitations': self.n_exploitations,
            'items_tried': np.sum(self.N > 0),
            'items_coverage': np.sum(self.N > 0) / self.n_items,
        }

    def get_top_items(self, k: int = 10) -> List[int]:
        """
        Get top-k items by estimated Q-value.

        Args:
            k: Number of items to return

        Returns:
            List of top-k item indices
        """
        return np.argsort(self.Q)[-k:][::-1].tolist()


def simulate_bandit(
    n_items: int = 100,
    n_steps: int = 1000,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    true_rewards: Optional[np.ndarray] = None,
    seed: int = 42
) -> dict:
    """
    Simulate epsilon-greedy bandit on synthetic data.

    Args:
        n_items: Number of items
        n_steps: Number of interactions to simulate
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        true_rewards: True reward probability for each item (if None, generated randomly)
        seed: Random seed

    Returns:
        Dictionary with simulation results
    """
    rng = np.random.RandomState(seed)

    # Generate true reward probabilities if not provided
    if true_rewards is None:
        # Some items are good (high reward prob), most are mediocre
        true_rewards = rng.beta(2, 5, size=n_items)  # Skewed distribution

    # Initialize bandit
    bandit = EpsilonGreedyBandit(
        n_items=n_items,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        seed=seed
    )

    # Track metrics over time
    cumulative_rewards = []
    regrets = []
    optimal_item = np.argmax(true_rewards)
    optimal_reward_prob = true_rewards[optimal_item]

    # Run simulation
    for t in range(n_steps):
        # Select item
        item, is_exploration = bandit.select_item()

        # Simulate reward (stochastic based on true reward probability)
        reward = 1.0 if rng.rand() < true_rewards[item] else 0.0

        # Update bandit
        bandit.update(item, reward)

        # Track metrics
        cumulative_rewards.append(bandit.total_reward)

        # Regret = optimal reward - actual reward
        regret = optimal_reward_prob - true_rewards[item]
        if regrets:
            regrets.append(regrets[-1] + regret)
        else:
            regrets.append(regret)

    # Results
    results = {
        'bandit': bandit,
        'true_rewards': true_rewards,
        'cumulative_rewards': cumulative_rewards,
        'regrets': regrets,
        'final_stats': bandit.get_stats(),
        'optimal_item': optimal_item,
        'times_selected_optimal': int(bandit.N[optimal_item]),
    }

    return results


if __name__ == "__main__":
    print("Testing Epsilon-Greedy Bandit\n")
    print("=" * 60)

    # Run simulation
    results = simulate_bandit(
        n_items=50,
        n_steps=1000,
        epsilon=0.2,
        epsilon_decay=0.995,
        seed=42
    )

    bandit = results['bandit']
    stats = results['final_stats']

    print(f"\nSimulation Results:")
    print(f"  Total steps: {stats['n_steps']}")
    print(f"  Total reward: {stats['total_reward']:.1f}")
    print(f"  Average reward: {stats['avg_reward']:.4f}")
    print(f"  Final epsilon: {stats['epsilon']:.4f}")
    print(f"  Exploration rate: {stats['exploration_rate']:.2%}")
    print(f"  Items tried: {stats['items_tried']} / {bandit.n_items}")
    print(f"  Coverage: {stats['items_coverage']:.2%}")

    print(f"\nOptimal Item:")
    print(f"  Item ID: {results['optimal_item']}")
    print(f"  True reward prob: {results['true_rewards'][results['optimal_item']]:.4f}")
    print(f"  Times selected: {results['times_selected_optimal']}")
    print(f"  Estimated Q-value: {bandit.Q[results['optimal_item']]:.4f}")

    print(f"\nTop 5 Items by Estimated Q-value:")
    top_items = bandit.get_top_items(k=5)
    for rank, item in enumerate(top_items, 1):
        print(f"  {rank}. Item {item}: Q={bandit.Q[item]:.4f}, "
              f"True={results['true_rewards'][item]:.4f}, "
              f"Pulls={int(bandit.N[item])}")

    print(f"\nFinal Cumulative Regret: {results['regrets'][-1]:.2f}")

    print("\n" + "=" * 60)
    print("Try experimenting with different epsilon values and decay rates!")
    print("Visualize cumulative_rewards and regrets over time to see learning.")
