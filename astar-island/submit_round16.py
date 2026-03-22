#!/usr/bin/env python3
"""Quick submit for round 16 using observation data already collected."""

import requests
import numpy as np
from pathlib import Path

TOKEN = Path(__file__).parent.parent / ".env"
with open(TOKEN) as f:
    for line in f:
        if line.startswith("NMAI_TOKEN="):
            ACCESS_TOKEN = line.strip().split("=", 1)[1]
            break

BASE = "https://api.ainm.no"
PROB_FLOOR = 0.01
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

session = requests.Session()
session.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

# Get round details
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next(r for r in rounds if r["status"] == "active")
round_id = active["id"]
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
width, height = detail["map_width"], detail["map_height"]

print(f"Round {active['round_number']}, closes: {active['closes_at']}")

# We can't re-query, but we can build predictions from initial state knowledge
# and submit better-than-uniform predictions

for seed_idx in range(detail["seeds_count"]):
    state = detail["initial_states"][seed_idx]
    grid = state["grid"]
    settlements = state["settlements"]

    prediction = np.zeros((height, width, NUM_CLASSES))

    for y in range(height):
        for x in range(width):
            code = grid[y][x]
            cls = TERRAIN_TO_CLASS.get(code, 0)

            if code == 10:  # Ocean - never changes
                prediction[y][x] = [1.0, 0, 0, 0, 0, 0]
            elif code == 5:  # Mountain - never changes
                prediction[y][x] = [0, 0, 0, 0, 0, 1.0]
            elif code == 4:  # Forest - mostly static
                # Check if near a settlement (could be reclaimed)
                near_settlement = False
                for s in settlements:
                    if abs(s["x"] - x) <= 2 and abs(s["y"] - y) <= 2:
                        near_settlement = True
                        break
                if near_settlement:
                    prediction[y][x] = [0.10, 0.15, 0.05, 0.05, 0.60, 0.05]
                else:
                    prediction[y][x] = [0.05, 0.02, 0.01, 0.02, 0.85, 0.05]
            elif code == 1:  # Settlement
                has_port = any(s["x"] == x and s["y"] == y and s.get("has_port", False) for s in settlements)
                if has_port:
                    prediction[y][x] = [0.05, 0.15, 0.45, 0.25, 0.05, 0.05]
                else:
                    # Check coastal
                    coastal = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == 10:
                                coastal = True
                    if coastal:
                        prediction[y][x] = [0.05, 0.25, 0.25, 0.25, 0.10, 0.10]
                    else:
                        prediction[y][x] = [0.05, 0.40, 0.10, 0.30, 0.10, 0.05]
            elif code == 11:  # Plains
                # Check distance to nearest settlement
                min_dist = 999
                for s in settlements:
                    d = abs(s["x"] - x) + abs(s["y"] - y)
                    min_dist = min(min_dist, d)
                if min_dist <= 2:
                    prediction[y][x] = [0.40, 0.25, 0.10, 0.10, 0.10, 0.05]
                elif min_dist <= 5:
                    prediction[y][x] = [0.55, 0.15, 0.05, 0.10, 0.10, 0.05]
                else:
                    prediction[y][x] = [0.75, 0.05, 0.03, 0.05, 0.07, 0.05]
            else:  # Empty/other
                prediction[y][x] = [0.70, 0.08, 0.04, 0.06, 0.07, 0.05]

    # Apply floor and renormalize
    prediction = np.maximum(prediction, PROB_FLOOR)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code} - {resp.text[:200]}")

print("\nDone!")
