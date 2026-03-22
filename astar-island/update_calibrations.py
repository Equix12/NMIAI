#!/usr/bin/env python3
"""Update round_calibrations.json with any new completed rounds."""
import requests, numpy as np, json, time
from collections import defaultdict
from pathlib import Path

with open(Path(__file__).parent.parent / ".env") as f:
    ACCESS_TOKEN = next(l.strip().split("=", 1)[1] for l in f if l.startswith("NMAI_TOKEN="))

BASE = "https://api.ainm.no"
session = requests.Session()
session.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

def db(d):
    if d <= 1: return 1
    if d <= 3: return 3
    if d <= 6: return 6
    if d <= 10: return 10
    return 99

cal_path = Path(__file__).parent / "round_calibrations.json"
with open(cal_path) as f:
    cals = json.load(f)

my = session.get(f"{BASE}/astar-island/my-rounds").json()
updated = False

for r in my:
    rn_str = str(r["round_number"])
    if rn_str in cals or r["status"] != "completed":
        continue

    rid = r["id"]
    detail = session.get(f"{BASE}/astar-island/rounds/{rid}").json()
    bins = defaultdict(lambda: [0.0] * 6)
    bins_n = defaultdict(int)
    exps = []
    rrs = []

    for seed in range(5):
        try:
            a = session.get(f"{BASE}/astar-island/analysis/{rid}/{seed}").json()
            gt = np.array(a["ground_truth"])
            ig = np.array(a.get("initial_grid", []))
            h, w = ig.shape
            sett = [(x, y) for y in range(h) for x in range(w) if ig[y][x] == 1]
            exps.append(float(gt[:, :, 1].sum() / max(len(sett), 1)))
            rrs.append(float(gt[:, :, 3].sum() / max(gt[:, :, 1].sum(), 0.01)))

            for y in range(h):
                for x in range(w):
                    code = int(ig[y][x])
                    if code in (10, 5):
                        continue
                    md = min((abs(sx - x) + abs(sy - y) for sx, sy in sett), default=99)
                    co = 1 if any(0 <= y + dy < h and 0 <= x + dx < w and ig[y + dy][x + dx] == 10
                                  for dy in [-1, 0, 1] for dx in [-1, 0, 1]) else 0
                    k = f"{code}_{db(md)}_{co}"
                    for c in range(6):
                        bins[k][c] += gt[y][x][c]
                    bins_n[k] += 1
        except Exception as e:
            print(f"  R{rn_str} seed {seed}: {e}")
        time.sleep(0.1)

    if exps:
        cal = {k: [round(bins[k][c] / bins_n[k], 5) for c in range(6)]
               for k in bins if bins_n[k] >= 10}
        cals[rn_str] = {
            "expansion": round(float(np.mean(exps)), 3),
            "ruin_rate": round(float(np.mean(rrs)), 4),
            "cal": cal,
        }
        updated = True
        print(f"Added R{rn_str}: exp={cals[rn_str]['expansion']}x, ruin={cals[rn_str]['ruin_rate']}")

if updated:
    with open(cal_path, "w") as f:
        json.dump(cals, f)
    print(f"Saved {cal_path}")
else:
    print("No new rounds to add")
