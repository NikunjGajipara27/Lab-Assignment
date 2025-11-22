# Assignment 6 - Problem 5
# Hopfield TSP (10 cities) - Energy-based Update

import numpy as np

def hopfield_tsp_update(x: np.ndarray, D: np.ndarray,
                        A: float = 500.0, B: float = 500.0,
                        C: float = 1.0, eta: float = 0.1) -> np.ndarray:
    """One update step of Hopfield TSP continuous model.

    x: current state matrix of shape (N, N), entries in [0,1]
    D: distance matrix (N x N)
    A, B: penalty weights for constraints
    C: weight for distance cost
    eta: step size
    """ 
    N = x.shape[0]
    y = x.copy()
    for i in range(N):
        for t in range(N):
            term1 = -A * (np.sum(x[:, t]) - 1)
            term2 = -B * (np.sum(x[i, :]) - 1)
            term3 = -C * sum(
                D[i, j] * (x[j, (t + 1) % N] + x[j, (t - 1) % N])
                for j in range(N)
            )
            y[i, t] += eta * (term1 + term2 + term3)
    return y

def normalize_state(x: np.ndarray) -> np.ndarray:
    """Optional normalization / squashing of state values to [0,1]."""
    return np.clip(x, 0.0, 1.0)

def extract_tour(x: np.ndarray) -> list:
    """Extract a discrete tour by picking argmax in each column (position)."""
    N = x.shape[0]
    tour = []
    for t in range(N):
        i = int(np.argmax(x[:, t]))
        tour.append(i)
    return tour

if __name__ == "__main__":
    N = 10
    rng = np.random.default_rng(0)
    D = rng.integers(1, 20, size=(N, N))
    np.fill_diagonal(D, 0)
    x = rng.random((N, N))

    for _ in range(500):
        x = hopfield_tsp_update(x, D)
        x = normalize_state(x)

    tour = extract_tour(x)
    print("Approximate TSP tour (sequence of city indices):", tour)
