# Assignment 6 - Problem 3
# Error-correcting capability of Hopfield Network

import numpy as np

def hopfield_train(patterns: np.ndarray) -> np.ndarray:
    """Train Hopfield network using Hebbian rule.

    patterns: array of shape (P, N) with entries in {-1, +1}
    returns: weight matrix W of shape (N, N)
    """
    P, N = patterns.shape
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W / P

def hopfield_energy(W: np.ndarray, s: np.ndarray) -> float:
    """Compute Hopfield energy E = -1/2 * s^T W s."""
    return float(-0.5 * s @ W @ s)

def hopfield_recall(W: np.ndarray, pattern: np.ndarray, steps: int = 200) -> np.ndarray:
    """Asynchronous recall from an initial pattern.

    W: weight matrix
    pattern: initial state (N,) in {-1,+1}
    steps: number of asynchronous updates
    """ 
    s = pattern.copy()
    N = len(s)
    for _ in range(steps):
        i = np.random.randint(0, N)
        s[i] = np.sign(W[i] @ s)
        if s[i] == 0:  # in rare case of exact 0, choose +1
            s[i] = 1
    return s

def hopfield_recall_track(W: np.ndarray, pattern: np.ndarray, steps: int = 200):
    """Recall while tracking energy over iterations.

    returns: (final_state, energies_array)
    """ 
    s = pattern.copy()
    N = len(s)
    energies = [hopfield_energy(W, s)]
    for _ in range(steps):
        i = np.random.randint(0, N)
        s[i] = np.sign(W[i] @ s)
        if s[i] == 0:
            s[i] = 1
        energies.append(hopfield_energy(W, s))
    return s, np.array(energies)

def flip_bits(pattern: np.ndarray, k: int) -> np.ndarray:
    """Flip k random bits in a pattern (values in {-1,+1})."""
    idx = np.random.choice(len(pattern), size=k, replace=False)
    p = pattern.copy()
    p[idx] *= -1
    return p

if __name__ == "__main__":
    # Example usage / small demo
    rng = np.random.default_rng(0)
    P = 3
    N = 100
    patterns = rng.choice([-1, 1], size=(P, N))
    W = hopfield_train(patterns)
    orig = patterns[0]
    noisy = flip_bits(orig, k=20)
    final_state, energies = hopfield_recall_track(W, noisy, steps=200)
    print("Recovered correctly:", np.array_equal(final_state, orig))
