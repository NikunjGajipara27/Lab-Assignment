# Assignment 7 - Problems 3 and 4
# 10-Armed Non-Stationary Bandit and Modified Epsilon-Greedy

import numpy as np

def run_nonstationary_bandit(steps=2000, runs=200, eps=0.1,
                             alpha=None, sigma_walk=0.01, seed=0):
    """Simulate a 10-armed non-stationary bandit.

    alpha: None -> sample-average; otherwise constant step size
    """
    rng = np.random.default_rng(seed)
    n_actions = 10
    avg_rewards = np.zeros(steps)
    optimal_action = np.zeros(steps)
    for r in range(runs):
        q_true = np.zeros(n_actions)
        Q = np.zeros(n_actions)
        N = np.zeros(n_actions)
        for t in range(steps):
            if rng.random() < eps:
                a = rng.integers(0, n_actions)
            else:
                a = int(np.argmax(Q))
            reward = rng.normal(q_true[a], 1.0)
            avg_rewards[t] += reward
            if a == int(np.argmax(q_true)):
                optimal_action[t] += 1
            if alpha is None:
                N[a] += 1
                step_size = 1.0 / N[a]
            else:
                step_size = alpha
            Q[a] += step_size * (reward - Q[a])
            q_true += rng.normal(0.0, sigma_walk, size=n_actions)
    avg_rewards /= runs
    optimal_action = optimal_action / runs * 100.0
    return avg_rewards, optimal_action

if __name__ == "__main__":
    avg_rew_std, opt_std = run_nonstationary_bandit(alpha=None)
    avg_rew_mod, opt_mod = run_nonstationary_bandit(alpha=0.1)
    print("Final avg reward (standard):", avg_rew_std[-1])
    print("Final avg reward (modified):", avg_rew_mod[-1])
