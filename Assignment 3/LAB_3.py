#!/usr/bin/env python3
"""
ksat_improved.py
Improved k-SAT experiment harness:
 - generate_k_sat(k,m,n)
 - hill_climbing, beam_search (heuristic A/B), vnd (with fixes)
 - CLI with argparse, per-run seed logging, expansion counters
 - Saves CSV with columns: m, trial, seed, method, beam_width, heuristic,
   initial, final, total, penetrance, time, expands
Usage:
    python ksat_improved.py --n 20 --k 3 --m_list 40 60 --trials 5 --seed_base 1000
"""
import random
import time
import argparse
import pandas as pd


# --------------------------
# Instance generation & evaluation
# --------------------------
def generate_k_sat(k, m, n, seed=None):
    """Generate a uniform random k-SAT instance with distinct variables per clause."""
    if seed is not None:
        random.seed(seed)
    clauses = []
    for _ in range(m):
        vars_ = random.sample(range(1, n + 1), k)
        clause = []
        for v in vars_:
            lit = v if random.random() < 0.5 else -v
            clause.append(lit)
        clauses.append(tuple(clause))
    return clauses


def evaluate_clause(clause, assignment):
    """Evaluate clause under a complete assignment (dict var->bool)."""
    for lit in clause:
        v = abs(lit)
        val = assignment[v]
        if lit < 0:
            val = not val
        if val:
            return True
    return False


def num_satisfied(clauses, assignment):
    """Count clauses satisfied by a complete assignment."""
    return sum(1 for c in clauses if evaluate_clause(c, assignment))


# --------------------------
# Partial-assignment helpers for Beam Search (safe for partials)
# --------------------------
def evaluate_clause_partial(clause, partial_assignment):
    """
    Evaluate a clause under a partial assignment.

    Returns:
      - True  : clause already satisfied
      - False : clause definitely unsatisfied (all literals assigned and none true)
      - None  : undecided (no true literal yet and some unassigned vars)
    """
    any_unassigned = False
    for lit in clause:
        v = abs(lit)
        if v not in partial_assignment:
            any_unassigned = True
            continue
        val = partial_assignment[v]
        if lit < 0:
            val = not val
        if val:
            return True
    if any_unassigned:
        return None
    return False


def partial_counts(clauses, partial_assignment):
    """
    Returns counts:
      full: clauses already satisfied (True)
      partial: clauses undecided but with at least one assigned literal (None with assigned_any)
      undecided: clauses with no assigned literals
    """
    full = 0
    partial = 0
    undecided = 0
    for c in clauses:
        r = evaluate_clause_partial(c, partial_assignment)
        if r is True:
            full += 1
        elif r is None:
            # check if any literal assigned (to count as partial)
            assigned_any = any(abs(l) in partial_assignment for l in c)
            if assigned_any:
                partial += 1
            else:
                undecided += 1
        else:
            # r is False -> all assigned and none true (counts as neither full nor partial)
            pass
    return full, partial, undecided


def heuristic_A_score(clauses, partial_assignment):
    """Baseline heuristic: full + 0.5 * partial"""
    full, partial, _ = partial_counts(clauses, partial_assignment)
    return full + 0.5 * partial


def heuristic_B_score(clauses, partial_assignment):
    """
    Optimistic heuristic:
      full + 0.3 * undecided + 0.2 * partial
    Encourages assignments that keep many undecided clauses (more potential).
    """
    full, partial, undecided = partial_counts(clauses, partial_assignment)
    return full + 0.3 * undecided + 0.2 * partial


# --------------------------
# Common utility
# --------------------------
def random_assignment(n):
    """Return a random complete assignment dict var->bool for 1..n."""
    return {i: bool(random.getrandbits(1)) for i in range(1, n + 1)}


# --------------------------
# Algorithms
# --------------------------
def hill_climbing(clauses, n, max_iters=1000, restarts=10):
    """Hill-climbing with greedy flips and restarts.
    Returns metadata including initial and final satisfied counts and expansion count.
    """
    best_score = -1
    best_assign = None
    best_init_score = None
    expands = 0
    start = time.time()

    for _ in range(restarts):
        assign = random_assignment(n)
        init_score = num_satisfied(clauses, assign)
        score = init_score
        it = 0
        improved = True
        while improved and it < max_iters:
            improved = False
            it += 1
            for v in range(1, n + 1):
                expands += 1
                assign[v] = not assign[v]
                s = num_satisfied(clauses, assign)
                if s > score:
                    score = s
                    improved = True
                    break
                assign[v] = not assign[v]
        if score > best_score:
            best_score = score
            best_assign = assign.copy()
            best_init_score = init_score
        if best_score == len(clauses):
            break

    if best_init_score is None:
        best_init_score = num_satisfied(clauses, best_assign) if best_assign else 0

    return {
        "method": "HillClimb",
        "assignment": best_assign,
        "initial_satisfied": best_init_score,
        "satisfied": best_score,
        "total": len(clauses),
        "time": round(time.time() - start, 4),
        "expands": expands
    }


def beam_search(clauses, n, beam_width=3, heuristic="A", max_expand=20000, fill_value=False):
    """
    Beam search over variables assigned in order 1..n.
    heuristic: "A" or "B"
    fill_value: value used to fill unassigned vars before final evaluation (deterministic)
    Returns metadata including expansion count.
    """
    start = time.time()
    hfun = heuristic_A_score if heuristic == "A" else heuristic_B_score

    beams = [({}, hfun(clauses, {}))]
    expands = 0
    node_expands = 0

    for var in range(1, n + 1):
        next_beams = []
        for assign, _ in beams:
            for val in (False, True):
                new_assign = assign.copy()
                new_assign[var] = val
                score_est = hfun(clauses, new_assign)
                next_beams.append((new_assign, score_est))
                expands += 1
                if expands > max_expand:
                    break
            if expands > max_expand:
                break
        next_beams.sort(key=lambda x: x[1], reverse=True)
        beams = next_beams[:beam_width]
        if not beams:
            break

    # evaluate full assignments in final beam
    best_assign = None
    best_score = -1
    initial_satisfied = partial_counts(clauses, {})[0]

    for assign, _ in beams:
        # deterministically fill unassigned variables (avoid randomness)
        if len(assign) < n:
            for v in range(1, n + 1):
                if v not in assign:
                    assign[v] = fill_value
        s = num_satisfied(clauses, assign)
        node_expands += 1
        if s > best_score:
            best_score = s
            best_assign = assign.copy()

    return {
        "method": f"Beam",
        "assignment": best_assign,
        "initial_satisfied": initial_satisfied,
        "satisfied": best_score,
        "total": len(clauses),
        "time": round(time.time() - start, 4),
        "expands": expands + node_expands,
        "beam_width": beam_width,
        "heuristic": heuristic
    }


def vnd(clauses, n, max_iters=2000, restarts=5):
    """Variable-Neighborhood-Descent with flip-1, flip-2, flip-3 neighborhoods.
    Returns metadata including initial and final satisfied counts and expansion count.
    """
    start = time.time()
    best_assign = None
    best_score = -1
    best_init_score = None
    expands = 0

    def neighborhoods(assign):
        neighs = []
        # flip-1
        for v in range(1, n + 1):
            new = assign.copy()
            new[v] = not new[v]
            neighs.append(new)
        # some flip-2
        for _ in range(min(50, n * (n - 1) // 2)):
            i, j = random.sample(range(1, n + 1), 2)
            new = assign.copy()
            new[i] = not new[i]; new[j] = not new[j]
            neighs.append(new)
        # some flip-3
        for _ in range(30):
            trio = random.sample(range(1, n + 1), 3)
            new = assign.copy()
            for v in trio:
                new[v] = not new[v]
            neighs.append(new)
        return neighs

    for _ in range(restarts):
        assign = random_assignment(n)
        init_score = num_satisfied(clauses, assign)
        score = init_score
        it = 0
        while it < max_iters:
            it += 1
            improved = False
            for neigh in neighborhoods(assign):
                expands += 1
                s = num_satisfied(clauses, neigh)
                if s > score:
                    assign = neigh
                    score = s
                    improved = True
                    break
            if not improved:
                break
        if score > best_score:
            best_score = score
            best_assign = assign.copy()
            best_init_score = init_score
        if best_score == len(clauses):
            break

    if best_init_score is None:
        best_init_score = num_satisfied(clauses, best_assign) if best_assign else 0

    return {
        "method": "VND",
        "assignment": best_assign,
        "initial_satisfied": best_init_score,
        "satisfied": best_score,
        "total": len(clauses),
        "time": round(time.time() - start, 4),
        "expands": expands
    }


# --------------------------
# Penetrance helper
# --------------------------
def compute_penetrance(initial_satisfied, final_satisfied, total):
    """Penetrance = fraction of previously unsatisfied clauses that algorithm managed to satisfy."""
    if initial_satisfied >= total:
        return 1.0
    denom = total - initial_satisfied
    if denom == 0:
        return 1.0
    return (final_satisfied - initial_satisfied) / denom


# --------------------------
# Experiment harness
# --------------------------
def run_experiment_extended(n=20, k=3, m_list=None, trials=5, seed_base=0):
    if m_list is None:
        m_list = [40, 60, 80]
    rows = []
    for m in m_list:
        for t in range(trials):
            seed = seed_base + t + m * 100
            clauses = generate_k_sat(k, m, n, seed=seed)

            # Hill Climb
            hc = hill_climbing(clauses, n, max_iters=1000, restarts=10)
            hc_pen = compute_penetrance(hc["initial_satisfied"], hc["satisfied"], hc["total"])
            rows.append({
                "m": m, "trial": t, "seed": seed, "method": hc["method"], "beam_width": None, "heuristic": None,
                "initial": hc["initial_satisfied"], "final": hc["satisfied"], "total": hc["total"],
                "penetrance": round(hc_pen, 4), "time": hc["time"], "expands": hc["expands"]
            })

            # Beam Search — two beam widths and two heuristics
            for bw in (3, 4):
                for heur in ("A", "B"):
                    bm = beam_search(clauses, n, beam_width=bw, heuristic=heur, max_expand=20000, fill_value=False)
                    bm_pen = compute_penetrance(bm["initial_satisfied"], bm["satisfied"], bm["total"])
                    rows.append({
                        "m": m, "trial": t, "seed": seed, "method": bm["method"], "beam_width": bw, "heuristic": heur,
                        "initial": bm["initial_satisfied"], "final": bm["satisfied"], "total": bm["total"],
                        "penetrance": round(bm_pen, 4), "time": bm["time"], "expands": bm["expands"]
                    })

            # VND
            vn = vnd(clauses, n, max_iters=1000, restarts=6)
            vn_pen = compute_penetrance(vn["initial_satisfied"], vn["satisfied"], vn["total"])
            rows.append({
                "m": m, "trial": t, "seed": seed, "method": vn["method"], "beam_width": None, "heuristic": None,
                "initial": vn["initial_satisfied"], "final": vn["satisfied"], "total": vn["total"],
                "penetrance": round(vn_pen, 4), "time": vn["time"], "expands": vn["expands"]
            })

    df = pd.DataFrame(rows)
    return df


def parse_args():
    p = argparse.ArgumentParser(description="k-SAT experiments harness (improved)")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--m_list", type=int, nargs="+", default=[40, 60])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=1000)
    p.add_argument("--out", type=str, default="ksat_results_improved.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Running improved k-SAT experiments...")
    df = run_experiment_extended(n=args.n, k=args.k, m_list=args.m_list, trials=args.trials, seed_base=args.seed_base)
    agg = df.groupby(["m", "method", "beam_width", "heuristic"]).agg(
        mean_final=("final", "mean"),
        mean_penetrance=("penetrance", "mean"),
        mean_time=("time", "mean"),
        mean_expands=("expands", "mean"),
        std_final=("final", "std")
    ).reset_index()
    print("\nSummary (mean final satisfied, mean penetrance, mean time, mean expands):")
    print(agg.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"\nSaved improved results to {args.out}")
