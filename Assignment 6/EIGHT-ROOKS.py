# Assignment 6 - Problem 4
# Eight-Rooks Problem using Hopfield-style Energy Minimization

import numpy as np

def eight_rooks_energy(x: np.ndarray, A: float = 2.0, B: float = 2.0) -> float:
    """Energy for Eight-Rooks constraints.

    x: binary matrix of shape (8, 8) with entries 0/1
    A, B: penalty weights for row and column constraints
    """ 
    N = x.shape[0]
    row_term = sum((np.sum(x[i, :]) - 1) ** 2 for i in range(N))
    col_term = sum((np.sum(x[:, j]) - 1) ** 2 for j in range(N))
    return float(A * row_term + B * col_term)

def eight_rooks_update(x: np.ndarray, A: float = 2.0, B: float = 2.0) -> np.ndarray:
    """Greedy coordinate-descent style update to reduce energy."""
    N = x.shape[0]
    y = x.copy()
    for i in range(N):
        for j in range(N):
            # try flipping this cell
            y[i, j] = 1 - x[i, j]
            if eight_rooks_energy(y, A, B) < eight_rooks_energy(x, A, B):
                x[i, j] = y[i, j]
            y[i, j] = x[i, j]
    return x

def solve_eight_rooks(max_iters: int = 1000, A: float = 2.0, B: float = 2.0, seed: int = 0):
    """Attempt to solve Eight-Rooks from random initialization."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(8, 8))
    for it in range(max_iters):
        E_before = eight_rooks_energy(x, A, B)
        x = eight_rooks_update(x, A, B)
        E_after = eight_rooks_energy(x, A, B)
        if E_after == 0:
            print(f"Converged to valid solution at iteration {it}")
            break
        if E_after >= E_before:
            # no improvement; might be stuck
            pass
    return x

if __name__ == "__main__":
    sol = solve_eight_rooks()
    print("Solution matrix (1 = rook):\n", sol)
