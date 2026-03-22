"""
Parameter fitting for Astar Island simulator.

Given observations (final-state snapshots from /simulate queries),
find hidden parameter values that make our simulator produce
matching output distributions.

Approach:
1. Extract summary statistics from observations
   (settlement fraction by distance, forest survival, ruin count, etc.)
2. For candidate parameter values, run simulator and extract same statistics
3. Minimize difference between observed and simulated statistics
4. Return best-fit parameters

This can be validated on completed rounds where we have ground truth.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, fields, asdict
import sys
sys.path.insert(0, '.')

from fast_sim import Params, FastSimulator, TERRAIN_TO_CLASS, \
    OCEAN, PLAINS, EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN


def dist_bin(d):
    if d <= 1: return 1
    if d <= 3: return 3
    if d <= 6: return 6
    if d <= 10: return 10
    return 99


def extract_statistics(grids, initial_grid, settlements_data):
    """
    Extract summary statistics from observed final-state grids.

    grids: list of 2D arrays (observed final terrain grids, can be partial viewports)
    initial_grid: the initial terrain (known, full map)
    settlements_data: initial settlement positions

    Returns dict of statistics that characterize the simulation outcome.
    """
    ig = np.array(initial_grid)
    H, W = ig.shape
    sett_pos = [(sd["x"], sd["y"]) for sd in settlements_data]

    # Compute distance map
    dist_map = np.full((H, W), 999, dtype=int)
    for sx, sy in sett_pos:
        for y in range(H):
            for x in range(W):
                d = abs(sx - x) + abs(sy - y)
                dist_map[y][x] = min(dist_map[y][x], d)

    # Coastal map
    coastal = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and ig[ny][nx] == OCEAN:
                        coastal[y][x] = True

    # Accumulate terrain class counts per feature bucket from observations
    # Each observation is one stochastic run's result for a viewport
    bucket_counts = defaultdict(lambda: np.zeros(6))  # (init_terrain, dist_bin) -> class counts
    bucket_total = defaultdict(int)

    for obs_grid, vx, vy in grids:
        obs = np.array(obs_grid)
        vh, vw = obs.shape
        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if y >= H or x >= W:
                    continue
                code = ig[y][x]
                if code in (OCEAN, MOUNTAIN):
                    continue
                cls = TERRAIN_TO_CLASS.get(int(obs[dy][dx]), 0)
                db = dist_bin(dist_map[y][x])
                key = (int(code), db)
                bucket_counts[key][cls] += 1
                bucket_total[key] += 1

    # Convert to frequency distributions
    stats = {}
    for key in bucket_counts:
        if bucket_total[key] > 0:
            stats[key] = bucket_counts[key] / bucket_total[key]

    return stats


def simulate_statistics(grid, settlements_data, params, n_runs=50, seed=None):
    """
    Run simulator with given params and extract same summary statistics.
    Returns dict of (init_terrain, dist_bin) -> class frequency distribution.
    """
    ig = np.array(grid)
    H, W = ig.shape
    sett_pos = [(sd["x"], sd["y"]) for sd in settlements_data]

    dist_map = np.full((H, W), 999, dtype=int)
    for sx, sy in sett_pos:
        for y in range(H):
            for x in range(W):
                d = abs(sx - x) + abs(sy - y)
                dist_map[y][x] = min(dist_map[y][x], d)

    bucket_counts = defaultdict(lambda: np.zeros(6))
    bucket_total = defaultdict(int)

    base_rng = np.random.default_rng(seed)
    for _ in range(n_runs):
        rng = np.random.default_rng(base_rng.integers(0, 2**31))
        sim = FastSimulator(grid, settlements_data, params, rng=rng)
        final = sim.run(years=50)

        for y in range(H):
            for x in range(W):
                code = ig[y][x]
                if code in (OCEAN, MOUNTAIN):
                    continue
                cls = TERRAIN_TO_CLASS.get(int(final[y][x]), 0)
                db = dist_bin(dist_map[y][x])
                key = (int(code), db)
                bucket_counts[key][cls] += 1
                bucket_total[key] += 1

    stats = {}
    for key in bucket_counts:
        if bucket_total[key] > 0:
            stats[key] = bucket_counts[key] / bucket_total[key]

    return stats


def compare_statistics(obs_stats, sim_stats):
    """
    Compare observed vs simulated statistics.
    Returns a loss value (lower = better match).
    Uses symmetric KL divergence weighted by bucket importance.
    """
    total_loss = 0.0
    total_weight = 0.0

    for key in obs_stats:
        if key not in sim_stats:
            continue

        obs = obs_stats[key]
        sim = sim_stats[key]
        eps = 0.01

        # Clamp both to avoid log(0)
        obs_c = np.maximum(obs, eps)
        sim_c = np.maximum(sim, eps)
        obs_c /= obs_c.sum()
        sim_c /= sim_c.sum()

        # Symmetric KL
        kl_fwd = np.sum(obs_c * np.log(obs_c / sim_c))
        kl_rev = np.sum(sim_c * np.log(sim_c / obs_c))
        loss = (kl_fwd + kl_rev) / 2

        # Weight by entropy of observed distribution (dynamic buckets matter more)
        entropy = -np.sum(obs_c * np.log(obs_c))
        weight = max(entropy, 0.01)

        total_loss += loss * weight
        total_weight += weight

    if total_weight == 0:
        return 999.0
    return total_loss / total_weight


def fit_parameters(grid, settlements_data, observations, n_sim_runs=30, seed=42):
    """
    Fit hidden parameters by comparing observed statistics to simulated statistics.

    observations: list of (obs_grid_2d, viewport_x, viewport_y) from /simulate queries
    Returns: best-fit Params object
    """
    # Extract observed statistics
    obs_stats = extract_statistics(observations, grid, settlements_data)

    print(f"  Observed stats from {len(observations)} observations:")
    terrain_names = {PLAINS: 'Plains', FOREST: 'Forest', SETTLEMENT: 'Settl', EMPTY: 'Empty'}
    labels = ['E', 'S', 'P', 'R', 'F', 'M']
    for key in sorted(obs_stats.keys()):
        code, db = key
        name = terrain_names.get(code, str(code))
        freq = obs_stats[key]
        print(f"    {name:8s} d<={db:2d}: {' '.join(f'{labels[i]}={freq[i]:.3f}' for i in range(6))}")

    # Parameter search space
    # Key parameters to fit based on what matters most
    param_grid = {
        'expand_prob': [0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035],
        'winter_severity': [0.20, 0.30, 0.40, 0.50],
        'collapse_prob': [0.10, 0.20, 0.30],
        'food_forest': [0.08, 0.12, 0.18],
    }

    # Coarse grid search
    best_loss = 999
    best_params = Params()
    n_configs = 1
    for v in param_grid.values():
        n_configs *= len(v)
    print(f"\n  Coarse search: {n_configs} configs × {n_sim_runs} sim runs each")

    from itertools import product as iterproduct
    keys = list(param_grid.keys())
    for values in iterproduct(*param_grid.values()):
        overrides = dict(zip(keys, values))
        p = Params(**overrides)

        sim_stats = simulate_statistics(grid, settlements_data, p,
                                         n_runs=n_sim_runs, seed=seed)
        loss = compare_statistics(obs_stats, sim_stats)

        if loss < best_loss:
            best_loss = loss
            best_params = p
            print(f"    NEW BEST loss={loss:.4f}: "
                  f"expand={p.expand_prob} winter={p.winter_severity} "
                  f"collapse={p.collapse_prob} food_f={p.food_forest}")

    # Fine search around best
    print(f"\n  Fine search around best params...")
    best_dict = asdict(best_params)
    fine_grid = {}
    for k in keys:
        v = best_dict[k]
        fine_grid[k] = [v * 0.7, v * 0.85, v, v * 1.15, v * 1.3]

    n_fine = 1
    for v in fine_grid.values():
        n_fine *= len(v)
    print(f"    {n_fine} configs")

    for values in iterproduct(*fine_grid.values()):
        overrides = dict(zip(keys, values))
        p = Params(**overrides)
        sim_stats = simulate_statistics(grid, settlements_data, p,
                                         n_runs=n_sim_runs, seed=seed)
        loss = compare_statistics(obs_stats, sim_stats)
        if loss < best_loss:
            best_loss = loss
            best_params = p
            print(f"    NEW BEST loss={loss:.4f}: "
                  f"expand={p.expand_prob:.4f} winter={p.winter_severity:.3f} "
                  f"collapse={p.collapse_prob:.3f} food_f={p.food_forest:.3f}")

    print(f"\n  Final best loss: {best_loss:.4f}")
    return best_params


if __name__ == "__main__":
    """
    Validation: use a completed round's observations to fit parameters,
    then check if the fitted params produce good predictions vs ground truth.
    """
    import requests
    import time
    from pathlib import Path
    from fast_sim import run_monte_carlo

    with open(Path(__file__).parent.parent / ".env") as f:
        token = next(l.strip().split("=", 1)[1] for l in f if l.startswith("NMAI_TOKEN="))

    BASE = "https://api.ainm.no"
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    def score_pred(pred, gt):
        eps = 1e-10
        kl = np.sum(gt * np.log((gt + eps) / (pred + eps)), axis=-1)
        ent = -np.sum(gt * np.log(gt + eps), axis=-1)
        d = ent > 0.01
        if d.sum() == 0: return 100.0
        return max(0, min(100, 100 * np.exp(-3 * np.sum(ent[d] * kl[d]) / np.sum(ent[d]))))

    # Load a completed round
    my_rounds = session.get(f"{BASE}/astar-island/my-rounds").json()

    for r in my_rounds:
        if r["round_number"] == 2 and r["status"] == "completed":
            rid = r["id"]
            detail = session.get(f"{BASE}/astar-island/rounds/{rid}").json()
            H, W = detail["map_height"], detail["map_width"]

            print(f"=== Round {r['round_number']} ===")

            # Simulate observations: sample from GT as if we queried the simulator
            # In practice these would come from actual /simulate calls
            seed_idx = 0
            state = detail["initial_states"][seed_idx]
            grid = state["grid"]
            sett = state["settlements"]

            analysis = session.get(f"{BASE}/astar-island/analysis/{rid}/{seed_idx}").json()
            gt = np.array(analysis["ground_truth"])

            # Create fake observations by sampling from ground truth
            # (simulates what we'd get from /simulate queries)
            print("Generating synthetic observations from GT...")
            rng = np.random.default_rng(123)
            observations = []
            # 9 viewports for full coverage, each observed once
            for vy in range(0, H, 13):
                for vx in range(0, W, 13):
                    vw = min(15, W - vx)
                    vh = min(15, H - vy)
                    # Sample one outcome per cell from GT distribution
                    obs = np.zeros((vh, vw), dtype=int)
                    for dy in range(vh):
                        for dx in range(vw):
                            y, x = vy + dy, vx + dx
                            cls = rng.choice(6, p=gt[y][x])
                            # Map class back to terrain code
                            code_map = {0: PLAINS, 1: SETTLEMENT, 2: PORT,
                                       3: RUIN, 4: FOREST, 5: MOUNTAIN}
                            obs[dy][dx] = code_map[cls]
                    observations.append((obs, vx, vy))

            print(f"Created {len(observations)} synthetic observations")

            # Fit parameters
            t0 = time.time()
            best_params = fit_parameters(grid, sett, observations, n_sim_runs=30, seed=42)
            fit_time = time.time() - t0
            print(f"Fitting took {fit_time:.1f}s")

            # Generate predictions with fitted parameters
            print("\nGenerating predictions with fitted params (500 MC runs)...")
            t0 = time.time()
            pred = run_monte_carlo(grid, sett, best_params, n_runs=500, seed=42)
            mc_time = time.time() - t0
            print(f"MC took {mc_time:.1f}s")

            score = score_pred(pred, gt)
            print(f"\nScore with fitted params: {score:.2f}")
            print(f"GT settlements: {gt[:,:,1].sum():.0f}")
            print(f"Predicted settlements: {pred[:,:,1].sum():.0f}")

            # Compare to calibration-only baseline
            print(f"\n(For reference, calibration-only baseline scored ~87 on this round)")
            break
