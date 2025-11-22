import numpy as np
import matplotlib.pyplot as plt

MAX_BIKES = 20
MAX_MOVE  = 5

def generate_base_policy(max_bikes=20):
    """
    Approximate optimal base policy (no free move, no parking penalty).
    Moves from 1→2 when location 1 has more bikes.
    Moves from 2→1 when location 2 has more bikes.
    """
    policy = np.zeros((max_bikes+1, max_bikes+1))

    for n1 in range(max_bikes+1):
        for n2 in range(max_bikes+1):
            diff = n1 - n2

            if diff > 1:  
                policy[n1, n2] = min(MAX_MOVE, diff // 2)
            elif diff < -1:
                policy[n1, n2] = -min(MAX_MOVE, (-diff) // 2)
            else:
                policy[n1, n2] = 0

    return policy


def plot_policy(policy, title, filename):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(policy, origin='lower', cmap='coolwarm')
    plt.colorbar(im, label='Action (bikes moved 1 → 2)')
    plt.xlabel("Bikes at Location 2")
    plt.ylabel("Bikes at Location 1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    policy = generate_base_policy(MAX_BIKES)
    plot_policy(policy,
                "Base Gbike Policy (Problem 2)",
                "base_policy_clear.png")