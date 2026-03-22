#!/usr/bin/env python3
"""
Astar Island - NM i AI 2026
Observes the simulation and predicts terrain probabilities.

Strategy:
- 50 queries total across 5 seeds (10 per seed)
- Each query: max 15x15 viewport on a 40x40 map
- Need ~9 viewports per seed for full coverage (3x3 grid)
- With 10 queries/seed: cover entire map + 1 duplicate for frequency data
- Use initial state to fill in static cells (ocean, mountains)
- Use observation frequency counts for dynamic cells

Usage:
    source /home/haava/NMAI/.venv/bin/activate
    python solve.py
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


def classify_initial_cells(grid):
    h, w = len(grid), len(grid[0])
    initial_classes = np.zeros((h, w), dtype=int)
    static = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            code = grid[y][x]
            initial_classes[y][x] = TERRAIN_TO_CLASS.get(code, 0)
            if code in (10, 5):  # Ocean, Mountain
                static[y][x] = True
    return initial_classes, static


def generate_viewports(width, height, max_vp=15):
    """Generate non-overlapping viewports covering the full map."""
    viewports = []
    # Step by ~13 to get slight overlap for better coverage
    step = max_vp - 2  # 13
    for vy in range(0, height, step):
        for vx in range(0, width, step):
            x = min(vx, max(0, width - max_vp))
            y = min(vy, max(0, height - max_vp))
            w = min(max_vp, width - x)
            h = min(max_vp, height - y)
            # Deduplicate
            vp = (x, y, w, h)
            if vp not in viewports:
                viewports.append(vp)
    return viewports


def plan_queries(detail, total_budget=50):
    """
    Allocate 50 queries across 5 seeds.
    Each seed gets ~10 queries covering the full map (9 viewports needed for 40x40).
    Extra queries go to seeds with more settlements for frequency data.
    """
    seeds_count = detail["seeds_count"]
    width, height = detail["map_width"], detail["map_height"]

    # Base allocation: 9 viewports per seed for full coverage = 45
    # Remaining 5 go to seeds with most settlements for repeat observations
    base_viewports = generate_viewports(width, height)
    print(f"  {len(base_viewports)} viewports needed for full map coverage")

    queries = []

    # Full coverage for each seed
    for sid in range(seeds_count):
        for vx, vy, vw, vh in base_viewports:
            queries.append({"seed_index": sid, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "coverage"})

    # If under budget, add repeat queries on high-settlement viewports
    remaining = total_budget - len(queries)
    if remaining > 0:
        # Score seeds by settlement count
        seed_settlements = []
        for sid in range(seeds_count):
            state = detail["initial_states"][sid]
            n = len(state["settlements"])
            seed_settlements.append((n, sid))
        seed_settlements.sort(reverse=True)

        # Add repeats for top seeds on viewports with most settlements
        for _, sid in seed_settlements:
            if remaining <= 0:
                break
            state = detail["initial_states"][sid]
            spos = [(s["x"], s["y"]) for s in state["settlements"]]
            # Score viewports by settlement coverage
            scored = []
            for vx, vy, vw, vh in base_viewports:
                count = sum(1 for sx, sy in spos if vx <= sx < vx + vw and vy <= sy < vy + vh)
                if count > 0:
                    scored.append((count, vx, vy, vw, vh))
            scored.sort(reverse=True)
            for count, vx, vy, vw, vh in scored[:2]:  # Max 2 repeats per seed
                if remaining <= 0:
                    break
                queries.append({"seed_index": sid, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "repeat"})
                remaining -= 1

    # Truncate to budget
    queries = queries[:total_budget]

    print(f"  Planned {len(queries)} queries")
    for sid in range(seeds_count):
        n = sum(1 for q in queries if q["seed_index"] == sid)
        print(f"    Seed {sid}: {n} queries")

    return queries


def build_prediction(initial_classes, static, observations, grid, settlements, height, width):
    """Build probability prediction from initial state + observations."""
    counts = np.zeros((height, width, NUM_CLASSES))
    obs_count = np.zeros((height, width), dtype=int)

    for obs_grid, vx, vy in observations:
        vh = len(obs_grid)
        vw = len(obs_grid[0]) if vh > 0 else 0
        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    cls = TERRAIN_TO_CLASS.get(obs_grid[dy][dx], 0)
                    counts[y][x][cls] += 1
                    obs_count[y][x] += 1

    prediction = np.zeros((height, width, NUM_CLASSES))

    for y in range(height):
        for x in range(width):
            if static[y][x]:
                cls = initial_classes[y][x]
                prediction[y][x][cls] = 1.0
            elif obs_count[y][x] > 0:
                prediction[y][x] = counts[y][x] / obs_count[y][x]
            else:
                # Unobserved cell - use heuristics based on initial state
                code = grid[y][x]
                min_dist = min((abs(s["x"] - x) + abs(s["y"] - y) for s in settlements), default=999)
                coastal = any(
                    0 <= y+dy < height and 0 <= x+dx < width and grid[y+dy][x+dx] == 10
                    for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                )

                if code == 1:  # Settlement
                    if coastal:
                        prediction[y][x] = [0.05, 0.25, 0.25, 0.25, 0.10, 0.10]
                    else:
                        prediction[y][x] = [0.05, 0.40, 0.10, 0.30, 0.10, 0.05]
                elif code == 4:  # Forest
                    if min_dist <= 2:
                        prediction[y][x] = [0.10, 0.15, 0.05, 0.05, 0.60, 0.05]
                    else:
                        prediction[y][x] = [0.05, 0.02, 0.01, 0.02, 0.85, 0.05]
                elif code == 11:  # Plains
                    if min_dist <= 2:
                        prediction[y][x] = [0.40, 0.25, 0.10, 0.10, 0.10, 0.05]
                    elif min_dist <= 5:
                        prediction[y][x] = [0.55, 0.15, 0.05, 0.10, 0.10, 0.05]
                    else:
                        prediction[y][x] = [0.75, 0.05, 0.03, 0.05, 0.07, 0.05]
                else:
                    prediction[y][x] = [0.70, 0.08, 0.04, 0.06, 0.07, 0.05]

    # Apply floor and renormalize
    prediction = np.maximum(prediction, PROB_FLOOR)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)
    return prediction


def main():
    print("=" * 60)
    print("ASTAR ISLAND SOLVER")
    print("=" * 60)

    # Get active round
    round_info = get_active_round()
    if not round_info:
        return
    round_id = round_info["id"]
    print(f"\nRound #{round_info['round_number']} (weight: {round_info['round_weight']})")
    print(f"  Closes at: {round_info['closes_at']}")

    # Get details
    detail = get_round_details(round_id)
    width, height = detail["map_width"], detail["map_height"]
    seeds_count = detail["seeds_count"]
    print(f"  Map: {width}x{height}, Seeds: {seeds_count}")

    # Analyze initial states
    seed_data = []
    for i in range(seeds_count):
        state = detail["initial_states"][i]
        initial_classes, static = classify_initial_cells(state["grid"])
        sett = state["settlements"]
        alive = sum(1 for s in sett if s.get("alive", True))
        ports = sum(1 for s in sett if s.get("has_port", False))
        print(f"  Seed {i}: {alive} settlements ({ports} ports), "
              f"{static.sum()} static / {(~static).sum()} dynamic cells")
        seed_data.append({
            "initial_classes": initial_classes,
            "static": static,
            "grid": state["grid"],
            "settlements": sett,
            "observations": [],
        })

    # Plan and execute queries
    queries = plan_queries(detail, total_budget=50)

    print(f"\nQuerying simulator...")
    for qi, q in enumerate(queries):
        sid = q["seed_index"]
        vp = f"({q['vx']},{q['vy']},{q['vw']},{q['vh']})"
        try:
            result = simulate(round_id, sid, q["vx"], q["vy"], q["vw"], q["vh"])
            seed_data[sid]["observations"].append((result["grid"], q["vx"], q["vy"]))
            flat = [TERRAIN_TO_CLASS.get(c, 0) for row in result["grid"] for c in row]
            dist = Counter(flat)
            labels = {0: "E", 1: "S", 2: "P", 3: "R", 4: "F", 5: "M"}
            summary = " ".join(f"{labels[k]}:{v}" for k, v in sorted(dist.items()))
            print(f"  [{qi+1:2d}/50] seed={sid} {vp:20s} {q['type']:8s} → {summary}")
        except Exception as e:
            print(f"  [{qi+1:2d}/50] seed={sid} {vp:20s} ERROR: {e}")
        time.sleep(0.15)

    # Build and submit predictions
    print(f"\nSubmitting predictions...")
    for sid in range(seeds_count):
        sd = seed_data[sid]
        prediction = build_prediction(
            sd["initial_classes"], sd["static"], sd["observations"],
            sd["grid"], sd["settlements"], height, width,
        )
        resp = session.post(f"{BASE}/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": sid,
            "prediction": prediction.tolist(),
        })
        n_obs = len(sd["observations"])
        print(f"  Seed {sid} ({n_obs} obs): {resp.status_code} - {resp.json().get('status', resp.text[:100])}")

    print("\nDone!")


if __name__ == "__main__":
    main()
