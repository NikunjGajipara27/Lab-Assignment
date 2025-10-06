#!/usr/bin/env python3
"""
lab4_submission_all.py

Combined Lab-4 submission code:
- Subcommand `jigsaw` : Simulated Annealing jigsaw solver for scrambled_lena.mat
  (supports up to 1,000,000 iterations, restarts, logging, image & perm output).

- Subcommand `tsp` : Simulated Annealing solver for TSP (In-Lab Discussion).
  Includes built-in list of 20 Rajasthan tourist locations with coordinates,
  computes great-circle distances and runs SA on tours.

Usage examples:
  # Jigsaw: run long SA (1 million iterations)
  python lab4_submission_all.py jigsaw --input /mnt/data/scrambled_lena.mat \
      --grid 4 --max_steps 1000000 --restarts 2 --seed 42 --out_prefix jigsaw_run

  # TSP: use built-in Rajasthan points
  python lab4_submission_all.py tsp --mode builtin --max_steps 200000 --seed 7 --out_prefix tsp_run

Notes:
- Jigsaw energy is L1 difference across adjacent block edges.
- For the jigsaw 1,000,000 steps can take significant time; tune parameters as needed.
"""

import argparse
import math
import random
import time
from pathlib import Path
import csv

# using non-interactive plotting backend for compatibility with headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# ----- JIGSAW SECTION -------
# ----------------------------

def load_octave_text_matrix(filename):
    """
    Robust loader for Octave/Matlab ASCII matrix exports.
    - Ignores comment lines starting with '#'.
    - Parses numeric tokens.
    - If the total token count is not a perfect square, it will try
      to remove a small number of leading or trailing tokens (<=10)
      to find a contiguous block that is a perfect square (e.g., 512*512).
    Returns a 2D uint8 numpy array.
    """
    import math
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Keep only non-comment, non-empty lines
    data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith('#')]
    if not data_lines:
        raise ValueError("No numeric data found in the file.")

    # Tokenize into numeric strings
    tokens = " ".join(data_lines).split()
    # Try to parse as integers (uint8). If parsing fails, try float -> cast.
    try:
        vals = np.array([int(t) for t in tokens], dtype=np.int64)
    except Exception:
        # fallback to float parsing then cast
        vals = np.array([float(t) for t in tokens], dtype=np.float64)

    total = vals.size

    def is_perfect_square(x):
        if x <= 0: return False
        r = int(round(math.sqrt(x)))
        return r * r == x

    # Quick accept if already perfect square
    if is_perfect_square(total):
        dim = int(round(math.sqrt(total)))
        arr = vals.astype(np.uint8).reshape((dim, dim))
        return arr

    # Try removing small numbers of leading/trailing tokens to find square block
    max_trim = 10  # search up to removing 10 tokens from front/back
    for trim_front in range(0, max_trim+1):
        for trim_back in range(0, max_trim+1):
            if trim_front + trim_back >= total:
                continue
            length = total - trim_front - trim_back
            if is_perfect_square(length):
                start = trim_front
                end = total - trim_back
                sub = vals[start:end]
                dim = int(round(math.sqrt(length)))
                try:
                    arr = sub.astype(np.uint8).reshape((dim, dim))
                    print(f"[loader] auto-trimmed front {trim_front}, back {trim_back} -> image {dim}x{dim}")
                    return arr
                except Exception:
                    # if casting or reshape fails, continue searching
                    continue

    # Last-resort: attempt to find any contiguous subsequence that is perfect-square length (costly)
    # (This is unlikely necessary; we keep it bounded.)
    max_window_checks = 2000
    for window_len in (512*512, 256*256, 128*128, 64*64):
        if window_len > total:
            continue
        # scan a limited number of windows
        checks = 0
        for start in range(0, total - window_len + 1):
            if checks >= max_window_checks:
                break
            sub = vals[start:start+window_len]
            try:
                arr = sub.astype(np.uint8).reshape((int(math.sqrt(window_len)), int(math.sqrt(window_len))))
                print(f"[loader] found window start {start} len {window_len}")
                return arr
            except Exception:
                pass
            checks += 1

    # If we reach here, give a helpful error with diagnostics
    raise ValueError(f"Data length {total} is not a perfect square and no reasonable trimming produced a square block. "
                     "First 20 tokens: " + " ".join(map(str, vals[:20])) +
                     ". Last 20 tokens: " + " ".join(map(str, vals[-20:])))



class JigsawSolver:
    def __init__(self, scrambled_image, grid=4):
        self.scrambled_image = scrambled_image
        self.grid = grid
        self.blocks, self.block_size = self._split_into_blocks()
        self.num_blocks = len(self.blocks)
        self.perm = list(range(self.num_blocks))

    def _split_into_blocks(self):
        h, w = self.scrambled_image.shape
        if h % self.grid != 0 or w % self.grid != 0:
            raise ValueError("Image dimensions not divisible by grid.")
        bh, bw = h // self.grid, w // self.grid
        blocks = []
        for i in range(self.grid):
            for j in range(self.grid):
                blocks.append(self.scrambled_image[i*bh:(i+1)*bh, j*bw:(j+1)*bw].copy())
        return blocks, (bh, bw)

    def _calculate_energy(self, perm):
        total = 0.0
        # grid of blocks
        grid_blocks = [[self.blocks[perm[i*self.grid + j]] for j in range(self.grid)] for i in range(self.grid)]
        # horizontal adjacencies
        for i in range(self.grid):
            for j in range(self.grid - 1):
                a = grid_blocks[i][j]
                b = grid_blocks[i][j+1]
                diff = a[:, -1].astype(np.int32) - b[:, 0].astype(np.int32)
                total += np.sum(np.abs(diff))
        # vertical adjacencies
        for i in range(self.grid - 1):
            for j in range(self.grid):
                a = grid_blocks[i][j]
                b = grid_blocks[i+1][j]
                diff = a[-1, :].astype(np.int32) - b[0, :].astype(np.int32)
                total += np.sum(np.abs(diff))
        return float(total)

    def stitch_image(self, perm):
        bh, bw = self.block_size
        img = np.zeros((self.grid*bh, self.grid*bw), dtype=self.blocks[0].dtype)
        idx = 0
        for i in range(self.grid):
            for j in range(self.grid):
                img[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = self.blocks[perm[idx]]
                idx += 1
        return img

    def solve_single_run(self, initial_temp=5500.0, cooling_rate=0.994,
                         iter_per_temp=250, max_steps=1000000, seed=None,
                         verbose_every=100000, allow_adj_swaps=False):
        # seed RNGs
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.perm = list(range(self.num_blocks))
        random.shuffle(self.perm)
        current_energy = self._calculate_energy(self.perm)
        best_perm = self.perm[:]
        best_energy = current_energy
        initial_energy = current_energy

        start_time = time.time()
        step = 0
        temp = float(initial_temp)
        min_temp = 1e-9

        # main SA loop
        while step < max_steps and temp > 0.1:
            for _ in range(iter_per_temp):
                if step >= max_steps:
                    break
                if allow_adj_swaps and random.random() < 0.05:
                    pos = random.randrange(self.num_blocks)
                    i = pos // self.grid
                    j = pos % self.grid
                    neighs = []
                    if j+1 < self.grid: neighs.append((i, j+1))
                    if j-1 >= 0: neighs.append((i, j-1))
                    if i+1 < self.grid: neighs.append((i+1, j))
                    if i-1 >= 0: neighs.append((i-1, j))
                    if neighs:
                        ni, nj = random.choice(neighs)
                        a = pos
                        b = ni*self.grid + nj
                    else:
                        a, b = random.sample(range(self.num_blocks), 2)
                else:
                    a, b = random.sample(range(self.num_blocks), 2)

                self.perm[a], self.perm[b] = self.perm[b], self.perm[a]
                prop_energy = self._calculate_energy(self.perm)
                dE = prop_energy - current_energy

                if dE < 0 or (temp > min_temp and random.random() < math.exp(-dE / temp)):
                    current_energy = prop_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_perm = self.perm[:]
                else:
                    self.perm[a], self.perm[b] = self.perm[b], self.perm[a]

                step += 1
                if verbose_every and step % verbose_every == 0:
                    elapsed = time.time() - start_time
                    print(f"[Jigsaw] step {step}/{max_steps} | temp {temp:.4f} | curr {current_energy:.1f} | best {best_energy:.1f} | elapsed {elapsed:.1f}s")
            temp *= cooling_rate

        total_time = time.time() - start_time
        return {
            "best_perm": best_perm,
            "best_energy": float(best_energy),
            "initial_energy": float(initial_energy),
            "steps": step,
            "time_s": total_time
        }


# ----------------------------
# ----- TSP SECTION ----------
# ----------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Built-in 20 Rajasthan tourist locations (name, lat, lon) — approximate coordinates
RAJASTHAN_20 = [
    ("Jaipur (Amber Fort)", 26.9855, 75.8513),
    ("Hawa Mahal, Jaipur", 26.9239, 75.8267),
    ("City Palace, Jaipur", 26.9258, 75.8204),
    ("Jal Mahal, Jaipur", 26.9618, 75.8315),
    ("Pushkar", 26.4930, 74.5539),
    ("Ajmer Dargah", 26.4486, 74.6399),
    ("Udaipur (City Palace)", 24.5770, 73.6806),
    ("Lake Pichola, Udaipur", 24.5740, 73.6808),
    ("Kumbhalgarh Fort", 25.1450, 73.5859),
    ("Chittorgarh Fort", 24.8896, 74.6263),
    ("Bikaner Junagarh Fort", 28.0167, 73.3160),
    ("Jaisalmer Fort", 26.9124, 70.9121),
    ("Jodhpur (Mehrangarh)", 26.2954, 73.0269),
    ("Ranakpur Temples", 25.1099, 73.3970),
    ("Sambhar Salt Lake", 26.9211, 75.0661),
    ("Bharatpur Bird Sanctuary", 27.1561, 77.5011),
    ("Mount Abu", 24.5933, 72.7126),
    ("Rajasmand Lake", 26.7881, 73.7048),
    ("Nathdwara", 24.9434, 73.7472),
    ("Alwar Bala Qila", 27.5648, 76.6110),
]

def build_distance_matrix(coords):
    n = len(coords)
    D = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = 0.0
            else:
                D[i][j] = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    return D

def tour_length(tour, D):
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i+1)%n]
        total += D[a][b]
    return total

def tsp_simulated_annealing(D, initial_temp=10000.0, cooling_rate=0.9995,
                            iter_per_temp=100, max_steps=200000,
                            seed=None, verbose_every=50000):
    """
    Simulated annealing for TSP that returns:
     - best_tour (list of indices)
     - best_cost (float)
     - initial_cost (float)
     - steps (int)
     - time_s (float)
     - history: list of (step, best_cost_so_far, current_cost)
     - accept_count, proposal_count
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(D)
    # start from a random permutation
    curr = list(range(n))
    random.shuffle(curr)
    curr_cost = tour_length(curr, D)
    initial_cost = float(curr_cost)
    best = curr[:]
    best_cost = curr_cost

    temp = initial_temp
    step = 0
    start_time = time.time()

    history = [(0, float(best_cost), float(curr_cost))]
    accept_count = 0
    proposal_count = 0

    # We'll store history every `hist_interval` proposals for plotting but keep memory small
    hist_interval = max(1, max_steps // 1000)

    while step < max_steps and temp > 1e-12:
        for _ in range(iter_per_temp):
            if step >= max_steps:
                break
            # Propose swap (2-opt style simple swap)
            i, j = random.sample(range(n), 2)
            curr[i], curr[j] = curr[j], curr[i]
            new_cost = tour_length(curr, D)
            delta = new_cost - curr_cost
            proposal_count += 1
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # accept
                curr_cost = new_cost
                accept_count += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = curr[:]
            else:
                # revert
                curr[i], curr[j] = curr[j], curr[i]
            step += 1
            if step % hist_interval == 0:
                history.append((step, float(best_cost), float(curr_cost)))
            if verbose_every and step % verbose_every == 0:
                elapsed = time.time() - start_time
                acc_rate = accept_count / proposal_count if proposal_count else 0.0
                print(f"[TSP] step {step}/{max_steps} | temp {temp:.6f} | curr {curr_cost:.2f} | best {best_cost:.2f} | acc_rate {acc_rate:.4f} | elapsed {elapsed:.1f}s")
        temp *= cooling_rate

    total_time = time.time() - start_time
    # final snapshot
    history.append((step, float(best_cost), float(curr_cost)))
    return {
        "best_tour": best,
        "best_cost": float(best_cost),
        "initial_cost": initial_cost,
        "steps": step,
        "time_s": total_time,
        "history": history,
        "accept_count": accept_count,
        "proposal_count": proposal_count
    }



# ----------------------------
# ----- CLI & Main ----------
# ----------------------------

def parse_cli():
    parser = argparse.ArgumentParser(description="Lab4 combined: jigsaw (submission) and tsp (in-lab).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # jigsaw subcommand
    p_j = sub.add_parser("jigsaw", help="Run jigsaw SA solver (submission task).")
    p_j.add_argument("--input", type=str, default="/mnt/data/scrambled_lena.mat", help="Input Octave text matrix")
    p_j.add_argument("--grid", type=int, default=4, help="Grid size (blocks per row)")
    p_j.add_argument("--initial_temp", type=float, default=5500.0)
    p_j.add_argument("--cooling_rate", type=float, default=0.994)
    p_j.add_argument("--iter_per_temp", type=int, default=250)
    p_j.add_argument("--max_steps", type=int, default=1000000, help="Total SA proposals (e.g., 1_000_000)")
    p_j.add_argument("--restarts", type=int, default=1)
    p_j.add_argument("--seed", type=int, default=42)
    p_j.add_argument("--verbose_every", type=int, default=100000)
    p_j.add_argument("--allow_adj_swaps", action="store_true")
    p_j.add_argument("--out_prefix", type=str, default="jigsaw_run")
    p_j.set_defaults(func=cmd_jigsaw)

    # tsp subcommand
    p_t = sub.add_parser("tsp", help="Run TSP SA solver (in-lab discussion).")
    p_t.add_argument("--mode", type=str, choices=["builtin", "csv"], default="builtin",
                     help="builtin uses built-in Rajasthan locations; csv expects file with name,lat,lon rows.")
    p_t.add_argument("--csv_file", type=str, default=None)
    p_t.add_argument("--initial_temp", type=float, default=10000.0)
    p_t.add_argument("--cooling_rate", type=float, default=0.9995)
    p_t.add_argument("--iter_per_temp", type=int, default=100)
    p_t.add_argument("--max_steps", type=int, default=200000)
    p_t.add_argument("--seed", type=int, default=7)
    p_t.add_argument("--verbose_every", type=int, default=50000)
    p_t.add_argument("--out_prefix", type=str, default="tsp_run")
    p_t.set_defaults(func=cmd_tsp)

    return parser.parse_args()


def cmd_jigsaw(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    img = load_octave_text_matrix(str(input_path))
    # If orientation is off, uncomment transpose line below
    # img = img.T
    print("Loaded image shape:", img.shape, "| grid:", args.grid)

    solver = JigsawSolver(img, grid=args.grid)
    best_overall = None
    best_overall_energy = float('inf')
    runs = []

    for r in range(args.restarts):
        seed_r = args.seed + r
        print(f"\n[JI] Restart {r+1}/{args.restarts} (seed={seed_r})")
        res = solver.solve_single_run(
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            iter_per_temp=args.iter_per_temp,
            max_steps=args.max_steps,
            seed=seed_r,
            verbose_every=args.verbose_every,
            allow_adj_swaps=args.allow_adj_swaps
        )
        runs.append({
            "restart": r,
            "seed": seed_r,
            "initial_energy": res["initial_energy"],
            "best_energy": res["best_energy"],
            "steps": res["steps"],
            "time_s": res["time_s"]
        })
        print(f"[JI] Completed restart {r}: initial {res['initial_energy']:.1f} -> best {res['best_energy']:.1f} in {res['steps']} steps ({res['time_s']:.1f}s)")
        if res["best_energy"] < best_overall_energy:
            best_overall_energy = res["best_energy"]
            best_overall = {
                "best_perm": res["best_perm"],
                "best_energy": res["best_energy"],
                "restart": r,
                "seed": seed_r,
                "steps": res["steps"],
                "time_s": res["time_s"]
            }

    # save run log
    csv_file = f"{args.out_prefix}_runs.csv"
    with open(csv_file, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["restart", "seed", "initial_energy", "best_energy", "steps", "time_s"])
        writer.writeheader()
        for row in runs:
            writer.writerow(row)

    perm_file = f"{args.out_prefix}_best_perm.txt"
    with open(perm_file, "w") as pf:
        pf.write(" ".join(map(str, best_overall["best_perm"])))

    final_img = solver.stitch_image(best_overall["best_perm"])
    out_image = f"{args.out_prefix}_unscrambled_energy_{int(best_overall['best_energy'])}.png"
    plt.imsave(out_image, final_img, cmap='gray')

    print(f"\n[JI] Saved run log: {csv_file}")
    print(f"[JI] Saved best permutation: {perm_file}")
    print(f"[JI] Saved best image: {out_image}")
    print(f"[JI] BEST energy: {best_overall['best_energy']:.1f} (restart {best_overall['restart']}, seed {best_overall['seed']})")


def cmd_tsp(args):
    # prepare city list
    if args.mode == "builtin":
        names_coords = [(name, (lat, lon)) for name, lat, lon in RAJASTHAN_20]
        coords = [c for _, c in names_coords]
    else:
        if not args.csv_file:
            raise ValueError("CSV file must be provided for csv mode.")
        names_coords = []
        coords = []
        with open(args.csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                name, lat, lon = row[0], float(row[1]), float(row[2])
                names_coords.append((name, (lat, lon)))
                coords.append((lat, lon))

    D = build_distance_matrix(coords)
    n = len(coords)
    print(f"[TSP] Running SA on {n} cities | mode={args.mode}")

    res = tsp_simulated_annealing(
        D,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        iter_per_temp=args.iter_per_temp,
        max_steps=args.max_steps,
        seed=args.seed,
        verbose_every=args.verbose_every
    )

    best = res["best_tour"]
    best_cost = res["best_cost"]
    steps = res["steps"]
    time_s = res["time_s"]
    initial_cost = res["initial_cost"]
    accept_count = res["accept_count"]
    proposal_count = res["proposal_count"]
    history = res["history"]

    # Print clear human-readable summary (fulfills "find a cycle ... total cost" requirement)
    print("\n=== TSP SA SUMMARY ===")
    print(f"Number of cities: {n}")
    print(f"Initial tour cost (km): {initial_cost:.2f}")
    print(f"Best tour cost (km): {best_cost:.2f}")
    print(f"Steps (proposals): {steps}")
    print(f"Time (s): {time_s:.2f}")
    acc_rate = accept_count / proposal_count if proposal_count else 0.0
    print(f"Accepted moves: {accept_count}/{proposal_count} (acceptance rate {acc_rate:.4f})")
    print("Best tour (ordered indices):")
    print(best)
    print("Best tour (ordered city names):")
    for idx in best:
        print(" -", names_coords[idx][0])

    # Save tour CSV with order and coordinates
    out_csv = f"{args.out_prefix}_tour.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "city_index", "city_name", "lat", "lon"])
        for pos, idx in enumerate(best):
            name, (lat, lon) = names_coords[idx]
            writer.writerow([pos+1, idx, name, lat, lon])
        # close tour by repeating first city
        first_idx = best[0]
        writer.writerow([len(best)+1, first_idx, names_coords[first_idx][0], names_coords[first_idx][1][0], names_coords[first_idx][1][1]])
    print(f"[TSP] Saved tour CSV: {out_csv}")

    # Save a plain text representation of the cycle (useful for automatic checking)
    out_cycle_txt = f"{args.out_prefix}_tour_cycle.txt"
    with open(out_cycle_txt, "w") as f:
        f.write(",".join(map(str, best)) + "\n")
    print(f"[TSP] Saved tour cycle indices: {out_cycle_txt}")

    # Save a small diagnostics CSV (initial, best, steps, time, accept_rate)
    diag_csv = f"{args.out_prefix}_diag.csv"
    with open(diag_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_cost_km", "best_cost_km", "steps", "time_s", "accept_count", "proposal_count", "accept_rate"])
        w.writerow([f"{initial_cost:.6f}", f"{best_cost:.6f}", steps, f"{time_s:.3f}", accept_count, proposal_count, f"{acc_rate:.6f}"])
    print(f"[TSP] Saved diagnostics CSV: {diag_csv}")

    # Plot route (longitude x latitude) and save
    xs = [names_coords[i][1][1] for i in best] + [names_coords[best[0]][1][1]]
    ys = [names_coords[i][1][0] for i in best] + [names_coords[best[0]][1][0]]
    plt.figure(figsize=(10,6))
    plt.plot(xs, ys, marker='o', linestyle='-')
    for i, idx in enumerate(best):
        name = names_coords[idx][0]
        plt.text(xs[i], ys[i], f"{i+1}:{name.split(',')[0]}", fontsize=8)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(f"TSP SA best route (cost {best_cost:.2f} km)")
    route_png = f"{args.out_prefix}_route.png"
    plt.tight_layout()
    plt.savefig(route_png, dpi=200)
    plt.close()
    print(f"[TSP] Saved route plot: {route_png}")

    # Energy (cost) history plot
    hist_steps = [h[0] for h in history]
    hist_best = [h[1] for h in history]
    hist_curr = [h[2] for h in history]
    plt.figure(figsize=(8,4))
    plt.plot(hist_steps, hist_best, label="best_cost")
    plt.plot(hist_steps, hist_curr, label="current_cost", alpha=0.6)
    plt.xlabel("Proposal step")
    plt.ylabel("Cost (km)")
    plt.legend()
    hist_png = f"{args.out_prefix}_history.png"
    plt.tight_layout()
    plt.savefig(hist_png, dpi=200)
    plt.close()
    print(f"[TSP] Saved cost history plot: {hist_png}")

    # All done: printed summary + files saved
    print("\n[TSP] Completed. You now have:")
    print(f" - final tour CSV: {out_csv}")
    print(f" - cycle txt: {out_cycle_txt}")
    print(f" - diagnostics csv: {diag_csv}")
    print(f" - route plot: {route_png}")
    print(f" - cost history plot: {hist_png}")


def main():
    args = parse_cli()
    args.func(args)


if __name__ == "__main__":
    main()
