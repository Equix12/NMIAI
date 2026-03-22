#!/usr/bin/env python3
"""
Astar Island Solver v2 - NM i AI 2026

Improvements over v1:
- Calibrated priors from historical ground truth analysis
- Bayesian combination of observations with priors
- Smarter query allocation: more observations on high-entropy areas
- Uses settlement stats from observations to refine predictions
"""

import requests
import numpy as np
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
with open(Path(__file__).parent.parent / ".env") as f:
    ACCESS_TOKEN = next(l.strip().split("=", 1)[1] for l in f if l.startswith("NMAI_TOKEN="))

BASE = "https://api.ainm.no"
PROB_FLOOR = 0.01
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

session = requests.Session()
session.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

# ── Calibrated priors from historical ground truth ──────────────────────────
# Key: (initial_terrain_code, distance_bin, coastal) -> probability distribution
# Distance bins: 1, 3, 6, 10, 99
CALIBRATION = {
    # Settlement (code=1)
    (1, 1, False): [0.377, 0.415, 0.000, 0.036, 0.171, 0.000],
    (1, 1, True):  [0.388, 0.322, 0.087, 0.032, 0.171, 0.000],
    (1, 3, False): [0.377, 0.415, 0.000, 0.036, 0.171, 0.000],  # same as dist<=1
    (1, 3, True):  [0.388, 0.322, 0.087, 0.032, 0.171, 0.000],
    (1, 6, False): [0.377, 0.415, 0.000, 0.036, 0.171, 0.000],
    (1, 6, True):  [0.388, 0.322, 0.087, 0.032, 0.171, 0.000],
    (1, 10, False): [0.377, 0.415, 0.000, 0.036, 0.171, 0.000],
    (1, 10, True):  [0.388, 0.322, 0.087, 0.032, 0.171, 0.000],
    (1, 99, False): [0.377, 0.415, 0.000, 0.036, 0.171, 0.000],
    (1, 99, True):  [0.388, 0.322, 0.087, 0.032, 0.171, 0.000],
    # Port (code=2) - rare, use what we have
    (2, 3, True):  [0.303, 0.162, 0.352, 0.027, 0.157, 0.000],
    (2, 6, True):  [0.387, 0.125, 0.280, 0.032, 0.176, 0.000],
    (2, 10, True): [0.330, 0.159, 0.341, 0.034, 0.136, 0.000],
    # Forest (code=4)
    (4, 1, False): [0.132, 0.266, 0.000, 0.024, 0.577, 0.000],
    (4, 1, True):  [0.102, 0.157, 0.155, 0.018, 0.569, 0.000],
    (4, 3, False): [0.139, 0.241, 0.000, 0.026, 0.595, 0.000],
    (4, 3, True):  [0.096, 0.152, 0.133, 0.021, 0.598, 0.000],
    (4, 6, False): [0.099, 0.191, 0.000, 0.021, 0.689, 0.000],
    (4, 6, True):  [0.075, 0.117, 0.086, 0.016, 0.706, 0.000],
    (4, 10, False): [0.023, 0.088, 0.000, 0.008, 0.881, 0.000],
    (4, 10, True):  [0.034, 0.081, 0.051, 0.009, 0.825, 0.000],
    (4, 99, False): [0.011, 0.066, 0.000, 0.001, 0.923, 0.000],
    (4, 99, True):  [0.004, 0.014, 0.007, 0.001, 0.975, 0.000],
    # Plains (code=11)
    (11, 1, False): [0.661, 0.258, 0.000, 0.024, 0.057, 0.000],
    (11, 1, True):  [0.646, 0.156, 0.132, 0.021, 0.045, 0.000],
    (11, 3, False): [0.677, 0.237, 0.000, 0.025, 0.061, 0.000],
    (11, 3, True):  [0.670, 0.143, 0.125, 0.020, 0.043, 0.000],
    (11, 6, False): [0.745, 0.190, 0.000, 0.021, 0.044, 0.000],
    (11, 6, True):  [0.748, 0.113, 0.090, 0.016, 0.033, 0.000],
    (11, 10, False): [0.893, 0.089, 0.000, 0.008, 0.010, 0.000],
    (11, 10, True):  [0.857, 0.074, 0.046, 0.009, 0.014, 0.000],
    (11, 99, False): [0.977, 0.021, 0.000, 0.001, 0.001, 0.000],
    (11, 99, True):  [0.936, 0.035, 0.022, 0.004, 0.003, 0.000],
    # Empty (code=0) - treat like distant plains
    (0, 1, False): [0.661, 0.258, 0.000, 0.024, 0.057, 0.000],
    (0, 1, True):  [0.646, 0.156, 0.132, 0.021, 0.045, 0.000],
    (0, 3, False): [0.677, 0.237, 0.000, 0.025, 0.061, 0.000],
    (0, 3, True):  [0.670, 0.143, 0.125, 0.020, 0.043, 0.000],
    (0, 6, False): [0.745, 0.190, 0.000, 0.021, 0.044, 0.000],
    (0, 6, True):  [0.748, 0.113, 0.090, 0.016, 0.033, 0.000],
    (0, 10, False): [0.893, 0.089, 0.000, 0.008, 0.010, 0.000],
    (0, 10, True):  [0.857, 0.074, 0.046, 0.009, 0.014, 0.000],
    (0, 99, False): [0.977, 0.021, 0.000, 0.001, 0.001, 0.000],
    (0, 99, True):  [0.936, 0.035, 0.022, 0.004, 0.003, 0.000],
}


def dist_bin(d):
    if d <= 1: return 1
    if d <= 3: return 3
    if d <= 6: return 6
    if d <= 10: return 10
    return 99


def get_calibrated_prior(code, min_dist, coastal):
    """Get calibrated prior distribution for a cell based on initial terrain and features."""
    db = dist_bin(min_dist)
    key = (code, db, coastal)
    if key in CALIBRATION:
        return np.array(CALIBRATION[key])
    # Fallback: try without coastal
    key2 = (code, db, not coastal)
    if key2 in CALIBRATION:
        return np.array(CALIBRATION[key2])
    # Generic fallback
    return np.array([0.70, 0.10, 0.05, 0.05, 0.05, 0.05])


def get_active_round():
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = [r for r in rounds if r["status"] == "active"]
    if not active:
        print("No active round!")
        return None
    return active[0]


def get_round_details(round_id):
    return session.get(f"{BASE}/astar-island/rounds/{round_id}").json()


def simulate(round_id, seed_index, vx, vy, vw, vh):
    resp = session.post(f"{BASE}/astar-island/simulate", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx, "viewport_y": vy,
        "viewport_w": vw, "viewport_h": vh,
    })
    resp.raise_for_status()
    return resp.json()


def precompute_features(grid, settlements, height, width):
    """Precompute distance-to-settlement and coastal status for every cell."""
    # Distance map using BFS from all settlements
    dist_map = np.full((height, width), 999, dtype=int)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        if 0 <= sy < height and 0 <= sx < width:
            dist_map[sy][sx] = 0

    # Simple Manhattan distance
    for y in range(height):
        for x in range(width):
            for s in settlements:
                d = abs(s["x"] - x) + abs(s["y"] - y)
                dist_map[y][x] = min(dist_map[y][x], d)

    # Coastal map
    coastal_map = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == 10:
                        coastal_map[y][x] = True

    return dist_map, coastal_map


def generate_viewports(width, height, max_vp=15):
    """Generate viewports for full map coverage."""
    viewports = []
    step = max_vp - 2  # 13
    for vy in range(0, height, step):
        for vx in range(0, width, step):
            x = min(vx, max(0, width - max_vp))
            y = min(vy, max(0, height - max_vp))
            w = min(max_vp, width - x)
            h = min(max_vp, height - y)
            vp = (x, y, w, h)
            if vp not in viewports:
                viewports.append(vp)
    return viewports


def plan_queries(detail, total_budget=50):
    """
    Smart query allocation:
    - Full coverage for each seed (9 viewports × 5 seeds = 45)
    - Extra queries on viewports with most settlements for better frequency estimates
    """
    seeds_count = detail["seeds_count"]
    width, height = detail["map_width"], detail["map_height"]
    base_viewports = generate_viewports(width, height)
    print(f"  {len(base_viewports)} viewports for full coverage")

    queries = []
    for sid in range(seeds_count):
        for vx, vy, vw, vh in base_viewports:
            queries.append({"seed_index": sid, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "coverage"})

    remaining = total_budget - len(queries)
    if remaining > 0:
        # Add repeats focusing on settlement-heavy viewports
        for sid in range(seeds_count):
            if remaining <= 0:
                break
            state = detail["initial_states"][sid]
            spos = [(s["x"], s["y"]) for s in state["settlements"]]
            scored = []
            for vx, vy, vw, vh in base_viewports:
                count = sum(1 for sx, sy in spos if vx <= sx < vx + vw and vy <= sy < vy + vh)
                if count > 0:
                    scored.append((count, vx, vy, vw, vh))
            scored.sort(reverse=True)
            for count, vx, vy, vw, vh in scored[:1]:  # 1 repeat per seed
                if remaining <= 0:
                    break
                queries.append({"seed_index": sid, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "repeat"})
                remaining -= 1

    queries = queries[:total_budget]
    print(f"  Planned {len(queries)} queries")
    for sid in range(seeds_count):
        n = sum(1 for q in queries if q["seed_index"] == sid)
        print(f"    Seed {sid}: {n} queries")
    return queries


def build_prediction(grid, settlements, observations, height, width):
    """
    Build probability prediction combining calibrated priors with observations.

    For cells with observations: Bayesian update of prior with observed frequencies.
    For cells without observations: use calibrated prior directly.
    Static cells (ocean, mountain): deterministic.
    """
    dist_map, coastal_map = precompute_features(grid, settlements, height, width)

    # Count observations per cell
    obs_counts = np.zeros((height, width, NUM_CLASSES))
    obs_total = np.zeros((height, width), dtype=int)

    for obs_grid, obs_settlements, vx, vy in observations:
        vh = len(obs_grid)
        vw = len(obs_grid[0]) if vh > 0 else 0
        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    cls = TERRAIN_TO_CLASS.get(obs_grid[dy][dx], 0)
                    obs_counts[y][x][cls] += 1
                    obs_total[y][x] += 1

    prediction = np.zeros((height, width, NUM_CLASSES))

    for y in range(height):
        for x in range(width):
            code = grid[y][x]

            # Static cells
            if code == 10:  # Ocean
                prediction[y][x][0] = 1.0
                continue
            if code == 5:  # Mountain
                prediction[y][x][5] = 1.0
                continue

            # Get calibrated prior
            prior = get_calibrated_prior(code, dist_map[y][x], bool(coastal_map[y][x]))

            # Check if this cell has a port initially
            if code == 1:
                has_port = any(
                    s["x"] == x and s["y"] == y and s.get("has_port", False)
                    for s in settlements
                )
                if has_port:
                    # Ports: use port-specific calibration if available
                    port_prior = get_calibrated_prior(2, dist_map[y][x], True)
                    prior = port_prior

            if obs_total[y][x] > 0:
                # Bayesian-ish combination: weight prior vs observations
                # With few observations (1-2), prior should still matter a lot
                # pseudo_count controls how much we trust the prior vs observations
                pseudo_count = 3  # equivalent to 3 prior "observations"
                combined = prior * pseudo_count + obs_counts[y][x]
                prediction[y][x] = combined / combined.sum()
            else:
                prediction[y][x] = prior

    # Apply floor and renormalize
    prediction = np.maximum(prediction, PROB_FLOOR)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)
    return prediction


def submit_prediction(round_id, seed_idx, prediction):
    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    return resp


def main():
    print("=" * 60)
    print("ASTAR ISLAND SOLVER v2")
    print("=" * 60)

    round_info = get_active_round()
    if not round_info:
        return
    round_id = round_info["id"]
    print(f"\nRound #{round_info['round_number']} (weight: {round_info['round_weight']})")
    print(f"  Closes at: {round_info['closes_at']}")

    detail = get_round_details(round_id)
    width, height = detail["map_width"], detail["map_height"]
    seeds_count = detail["seeds_count"]
    print(f"  Map: {width}x{height}, Seeds: {seeds_count}")

    # Check budget
    budget = session.get(f"{BASE}/astar-island/budget").json()
    queries_remaining = budget["queries_max"] - budget["queries_used"]
    print(f"  Budget: {budget['queries_used']}/{budget['queries_max']} used ({queries_remaining} remaining)")

    # Prepare seed data
    seed_data = []
    for i in range(seeds_count):
        state = detail["initial_states"][i]
        grid = state["grid"]
        sett = state["settlements"]
        alive = sum(1 for s in sett if s.get("alive", True))
        ports = sum(1 for s in sett if s.get("has_port", False))
        print(f"  Seed {i}: {alive} settlements ({ports} ports)")
        seed_data.append({
            "grid": grid,
            "settlements": sett,
            "observations": [],  # (obs_grid, obs_settlements, vx, vy)
        })

    # Run queries if budget available
    if queries_remaining > 0:
        queries = plan_queries(detail, total_budget=queries_remaining)
        print(f"\nQuerying simulator ({queries_remaining} queries available)...")
        for qi, q in enumerate(queries):
            sid = q["seed_index"]
            vp = f"({q['vx']},{q['vy']},{q['vw']},{q['vh']})"
            try:
                result = simulate(round_id, sid, q["vx"], q["vy"], q["vw"], q["vh"])
                seed_data[sid]["observations"].append(
                    (result["grid"], result.get("settlements", []), q["vx"], q["vy"])
                )
                flat = [TERRAIN_TO_CLASS.get(c, 0) for row in result["grid"] for c in row]
                dist = Counter(flat)
                labels = {0: "E", 1: "S", 2: "P", 3: "R", 4: "F", 5: "M"}
                summary = " ".join(f"{labels[k]}:{v}" for k, v in sorted(dist.items()))
                print(f"  [{qi+1:2d}/{len(queries)}] seed={sid} {vp:20s} {q['type']:8s} → {summary}")
            except Exception as e:
                print(f"  [{qi+1:2d}/{len(queries)}] seed={sid} {vp:20s} ERROR: {e}")
            time.sleep(0.15)
    else:
        print("\n  No queries remaining - using calibrated priors only")

    # Build and submit predictions
    print(f"\nSubmitting predictions...")
    for sid in range(seeds_count):
        sd = seed_data[sid]
        prediction = build_prediction(
            sd["grid"], sd["settlements"], sd["observations"], height, width,
        )
        resp = submit_prediction(round_id, sid, prediction)
        n_obs = len(sd["observations"])
        result = resp.json()
        print(f"  Seed {sid} ({n_obs} obs): {resp.status_code} - {result.get('status', str(result)[:100])}")

    print("\nDone!")


if __name__ == "__main__":
    main()
