# Assignment 7 - Problem 2
# Binary Bandit with Epsilon-Greedy Agent

import numpy as np

def run_binary_bandit(eps=0.1, p=(0.3, 0.7), steps=500, runs=200, seed=0):
    """Simulate a binary bandit with epsilon-greedy action selection."""
    rng = np.random.default_rng(seed)
    n_actions = 2
    all_rewards = np.zeros((runs, steps))
    for r in range(runs):
        Q = np.zeros(n_actions)
        N = np.zeros(n_actions)
        for t in range(steps):
            if rng.random() < eps:
                a = rng.integers(0, n_actions)
            else:
                a = int(np.argmax(Q))
            reward = 1.0 if rng.random() < p[a] else 0.0
            N[a] += 1
            Q[a] += (reward - Q[a]) / N[a]
            all_rewards[r, t] = reward
    return all_rewards.mean(axis=0)

if __name__ == "__main__":
    avg_rew = run_binary_bandit()
    print("Final average reward:", avg_rew[-1])
