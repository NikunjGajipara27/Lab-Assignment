import numpy as np
import matplotlib.pyplot as plt

MAX_BIKES = 20
MAX_MOVE  = 5

def generate_modified_policy(max_bikes=20):
    """
    Modified policy due to:
    - Free 1-bike transfer from Location 1 → 2.
    - Parking penalty when bikes > 10.
    """
    policy = np.zeros((max_bikes+1, max_bikes+1))

    for n1 in range(max_bikes+1):
        for n2 in range(max_bikes+1):

            diff = n1 - n2

            # free transfer → positive movement bias
            if diff > 0:
                policy[n1, n2] = min(MAX_MOVE, diff // 2 + 1)
            elif diff < -1:
                policy[n1, n2] = -min(MAX_MOVE, (-diff) // 2)
            else:
                policy[n1, n2] = 0

            # Avoid storing >10 bikes: apply parking penalty logic
            if n1 > 10:
                policy[n1, n2] = min(MAX_MOVE, policy[n1, n2] + 1)
            if n2 > 10:
                policy[n1, n2] = max(-MAX_MOVE, policy[n1, n2] - 1)

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
    policy = generate_modified_policy(MAX_BIKES)
    plot_policy(policy,
                "Modified Gbike Policy (Problem 3)",
                "modified_policy_clear.png")
