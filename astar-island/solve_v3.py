#!/usr/bin/env python3
"""
Astar Island Solver v3 - NM i AI 2026

Strategy (validated by backtesting):
1. Full coverage: 9 viewports × 5 seeds = 45 queries (1 obs per cell per seed)
2. Pool ALL observations into round-specific calibration buckets
   (same hidden params across seeds → 5x more calibration data)
3. Blend round-specific calibration with historical priors (weight=0.5)
4. Do NOT use per-cell observations directly (too noisy with 1 sample)

Backtested scores:
- Round 1: 82.64 (vs 79.75 hist-only)
- Round 2: 87.07 (vs 86.58 hist-only)
- Round 6: 80.74 (vs 73.30 hist-only)
"""

import requests
import numpy as np
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
with open(Path(__file__).parent.parent / ".env") as f:
    ACCESS_TOKEN = next(l.strip().split("=", 1)[1] for l in f if l.startswith("NMAI_TOKEN="))

BASE = "https://api.ainm.no"
PROB_FLOOR = 0.003  # base floor; smart_floor uses per-class floors
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {0: 0, 10: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
HIST_BLEND_WEIGHT = 0.2  # how much to weight historical vs round-specific cal (lower = trust round more)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

# ── Per-round calibration data (loaded from all 16 completed rounds) ───────
import json as _json
_cal_path = Path(__file__).parent / "round_calibrations.json"
with open(_cal_path) as _f:
    ROUND_CALIBRATIONS = _json.load(_f)  # {round_number_str: {expansion, cal: {key_str: [6 probs]}}}


def build_hist_prior(expansion_rate, ruin_rate=None, top_n=3):
    """
    Build historical calibration weighted toward rounds with similar expansion rate
    and ruin rate. Returns a dict of (code, dist_bin, coastal) -> np.array([6 probs])
    """
    # Rank rounds by similarity to observed expansion + ruin rate
    similarities = []
    for rn_str, rd in ROUND_CALIBRATIONS.items():
        rd_exp = rd["expansion"]
        d_exp = abs(np.log(max(rd_exp, 0.01)) - np.log(max(expansion_rate, 0.01)))
        dist = d_exp
        if ruin_rate is not None and "ruin_rate" in rd:
            rd_ruin = rd["ruin_rate"]
            d_ruin = abs(np.log(max(rd_ruin, 0.001)) - np.log(max(ruin_rate, 0.001)))
            dist += d_ruin * 0.5  # ruin gets half weight
        sim = 1.0 / (1.0 + dist)
        similarities.append((sim, rn_str))
    similarities.sort(reverse=True)

    # Use top_n most similar rounds with soft similarity weighting
    selected = [(sim, rn) for sim, rn in similarities[:top_n]]
    selected_names = [f"R{rn}({ROUND_CALIBRATIONS[rn]['expansion']:.1f}x)" for _, rn in selected[:5]]
    print(f"  Historical prior from {top_n} similar rounds: {', '.join(selected_names)}{'...' if top_n > 5 else ''}")

    # Merge their calibration tables, weighted by similarity
    merged = {}
    total_weight = 0.0
    for sim, rn_str in selected:
        cal = ROUND_CALIBRATIONS[rn_str]["cal"]
        for key_str, probs in cal.items():
            if key_str not in merged:
                merged[key_str] = np.zeros(6)
            merged[key_str] += np.array(probs) * sim
        total_weight += sim

    # Normalize
    hist_cal = {}
    for key_str, total in merged.items():
        parts = key_str.split("_")
        code, dbin, coastal = int(parts[0]), int(parts[1]), bool(int(parts[2]))
        hist_cal[(code, dbin, coastal)] = total / total_weight

    # Fill missing entries with fallbacks
    for code in [0, 1, 2, 4, 11]:
        for dbin in [1, 3, 6, 10, 99]:
            for coastal in [True, False]:
                key = (code, dbin, coastal)
                if key not in hist_cal:
                    alt = (code, dbin, not coastal)
                    if alt in hist_cal:
                        hist_cal[key] = hist_cal[alt].copy()
                    else:
                        for fdb in [3, 6, 1, 10, 99]:
                            fb = (code, fdb, coastal)
                            if fb in hist_cal:
                                hist_cal[key] = hist_cal[fb].copy()
                                break
                if key not in hist_cal:
                    hist_cal[key] = np.array([0.70, 0.10, 0.05, 0.05, 0.05, 0.05])

    return hist_cal


# Default: uniform average of all rounds (used when no observations available)
HIST_CAL = build_hist_prior(expansion_rate=4.5, top_n=16)  # median expansion as default


def dist_bin(d):
    if d <= 1: return 1
    if d <= 3: return 3
    if d <= 6: return 6
    if d <= 10: return 10
    return 99


def get_hist_prior(code, min_dist, coastal):
    key = (code, dist_bin(min_dist), coastal)
    if key in HIST_CAL:
        return np.array(HIST_CAL[key])
    return np.array([0.70, 0.10, 0.05, 0.05, 0.05, 0.05])


# ── API helpers ─────────────────────────────────────────────────────────────

def get_active_round():
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = [r for r in rounds if r["status"] == "active"]
    if not active:
        print("No active round!")
        return None
    return active[0]


def get_round_details(round_id):
    return session.get(f"{BASE}/astar-island/rounds/{round_id}").json()


def get_budget():
    return session.get(f"{BASE}/astar-island/budget").json()


def simulate(round_id, seed_index, vx, vy, vw, vh):
    resp = session.post(f"{BASE}/astar-island/simulate", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx, "viewport_y": vy,
        "viewport_w": vw, "viewport_h": vh,
    })
    resp.raise_for_status()
    return resp.json()


# ── Map analysis ────────────────────────────────────────────────────────────

def precompute_features(grid, settlements, height, width):
    """Compute all per-cell features for calibration and prediction."""
    sett_pos = [(s["x"], s["y"]) for s in settlements]
    port_pos = [(s["x"], s["y"]) for s in settlements if s.get("has_port", False)]

    dist_map = np.full((height, width), 999, dtype=int)
    for sx, sy in sett_pos:
        for y in range(height):
            for x in range(width):
                d = abs(sx - x) + abs(sy - y)
                if d < dist_map[y][x]:
                    dist_map[y][x] = d

    coastal_map = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and grid[ny][nx] == 10:
                        coastal_map[y][x] = True

    # Settlement density: count settlements within radius 5
    density_map = np.zeros((height, width), dtype=int)
    for sx, sy in sett_pos:
        for y in range(max(0, sy - 5), min(height, sy + 6)):
            for x in range(max(0, sx - 5), min(width, sx + 6)):
                if abs(sx - x) + abs(sy - y) <= 5:
                    density_map[y][x] += 1

    # Port proximity: count ports within radius 8 (trade range)
    port_prox_map = np.zeros((height, width), dtype=int)
    for px, py in port_pos:
        for y in range(max(0, py - 8), min(height, py + 9)):
            for x in range(max(0, px - 8), min(width, px + 9)):
                if abs(px - x) + abs(py - y) <= 8:
                    port_prox_map[y][x] += 1

    # Initial port positions
    init_port_set = set((s["x"], s["y"]) for s in settlements if s.get("has_port", False))

    return dist_map, coastal_map, density_map, port_prox_map, init_port_set


def cell_feature_key(code, dist_map, coastal_map, density_map, port_prox_map, init_port_set, y, x):
    """Compute extended feature key for a cell."""
    d = dist_map[y][x]
    coastal = bool(coastal_map[y][x])
    density_bin = min(int(density_map[y][x]), 3)     # 0, 1, 2, 3+
    port_bin = min(int(port_prox_map[y][x]), 2) if coastal else 0  # 0, 1, 2+
    is_init_port = (x, y) in init_port_set
    return (int(code), dist_bin(d), coastal, density_bin, port_bin, is_init_port)


def cell_base_key(code, dist_map, coastal_map, y, x):
    """Compute base feature key (fallback)."""
    return (int(code), dist_bin(dist_map[y][x]), bool(coastal_map[y][x]))


def generate_viewports(width, height, max_vp=15):
    """Generate non-overlapping viewports covering the full map."""
    viewports = []
    step = max_vp - 2  # 13, slight overlap
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


# ── Query planning ──────────────────────────────────────────────────────────

def plan_queries(detail, total_budget=50):
    """
    Three-phase query strategy:
    Phase 1 (9q): Full coverage on BEST seed → estimate expansion, score viewports
    Phase 2 (~31q): Deep repeats on best seed's highest-entropy viewports
    Phase 3 (~10q): Diversity queries on seeds 1-4 → fill rare calibration buckets

    Best seed = highest (n_ports * 50 + n_settlements + n_dynamic_cells)
    to maximize both bucket diversity (ports/coastal) and scoring data.
    """
    seeds_count = detail["seeds_count"]
    width, height = detail["map_width"], detail["map_height"]
    base_viewports = generate_viewports(width, height)
    n_vp = len(base_viewports)

    # Pick the best seed for deep observation
    # Prefer seeds with more ports (for coastal/port bucket coverage) and more settlements
    seed_scores = []
    for sid in range(seeds_count):
        sett_i = detail["initial_states"][sid]["settlements"]
        ig_i = detail["initial_states"][sid]["grid"]
        n_sett = len(sett_i)
        n_ports = sum(1 for ss in sett_i if ss.get("has_port", False))
        n_dynamic = sum(1 for row in ig_i for c in row if c not in (10, 5))
        score = n_ports * 50 + n_sett + n_dynamic * 0.01
        seed_scores.append((score, sid))
    seed_scores.sort(reverse=True)
    deep_seed = seed_scores[0][1]
    print(f"  Deep observation seed: {deep_seed} "
          f"(ports={sum(1 for ss in detail['initial_states'][deep_seed]['settlements'] if ss.get('has_port',False))}, "
          f"sett={len(detail['initial_states'][deep_seed]['settlements'])})")

    ig_deep = detail["initial_states"][deep_seed]["grid"]
    sett_deep = detail["initial_states"][deep_seed]["settlements"]
    sett_pos = [(ss["x"], ss["y"]) for ss in sett_deep]

    # Score viewports by dynamic cell count near settlements (d<=6)
    # These are the highest-entropy areas
    vp_scores = []
    for vx, vy, vw, vh in base_viewports:
        n_near_sett = 0
        n_dynamic = 0
        for dy in range(vh):
            for dx in range(vw):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    code = ig_deep[y][x]
                    if code in (10, 5):
                        continue
                    n_dynamic += 1
                    md = min((abs(sx - x) + abs(sy - y) for sx, sy in sett_pos), default=99)
                    if md <= 6:
                        n_near_sett += 1
        vp_scores.append((n_near_sett, n_dynamic, vx, vy, vw, vh))
    vp_scores.sort(reverse=True)

    # Budget allocation
    phase1_budget = n_vp
    phase3_budget = min(10, total_budget - phase1_budget - 20)
    phase3_budget = max(phase3_budget, 0)
    phase2_budget = total_budget - phase1_budget - phase3_budget

    queries = []

    # Phase 1: Full coverage on deep seed
    for vx, vy, vw, vh in base_viewports:
        queries.append({"seed_index": deep_seed, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "scout"})

    # Phase 2: Repeats on deep seed, prioritizing high-entropy viewports
    # Skip the 2 lowest-scoring viewports, give their budget to the top ones
    top_vps = [(vx, vy, vw, vh) for _, _, vx, vy, vw, vh in vp_scores[:-2]]
    p2_remaining = phase2_budget
    while p2_remaining > 0:
        for vx, vy, vw, vh in top_vps:
            if p2_remaining <= 0:
                break
            queries.append({"seed_index": deep_seed, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "deep"})
            p2_remaining -= 1

    # Phase 3: Diversity on OTHER seeds
    # Pick viewports with most ports/coastal settlements for bucket diversity
    p3_remaining = phase3_budget
    other_seeds = [sid for sid in range(seeds_count) if sid != deep_seed]
    n_remaining_seeds = len(other_seeds)
    for sid in other_seeds:
        if p3_remaining <= 0:
            break
        sett_i = detail["initial_states"][sid]["settlements"]
        port_pos = [(ss["x"], ss["y"]) for ss in sett_i if ss.get("has_port", False)]
        # Score viewports: ports > settlements > dynamic
        scored = []
        for vx, vy, vw, vh in base_viewports:
            n_port = sum(1 for px, py in port_pos if vx <= px < vx + vw and vy <= py < vy + vh)
            n_sett = sum(1 for ss in sett_i if vx <= ss["x"] < vx + vw and vy <= ss["y"] < vy + vh)
            scored.append((n_port * 10 + n_sett, vx, vy, vw, vh))
        scored.sort(reverse=True)
        per_seed = max(1, p3_remaining // n_remaining_seeds)
        for _, vx, vy, vw, vh in scored[:per_seed]:
            if p3_remaining <= 0:
                break
            queries.append({"seed_index": sid, "vx": vx, "vy": vy, "vw": vw, "vh": vh, "type": "diverse"})
            p3_remaining -= 1
        n_remaining_seeds -= 1

    queries = queries[:total_budget]

    # Summary
    print(f"  {n_vp} viewports, Phase 2 repeats on top {len(top_vps)} viewports:")
    for sid in range(seeds_count):
        n = sum(1 for q in queries if q["seed_index"] == sid)
        types = Counter(q["type"] for q in queries if q["seed_index"] == sid)
        type_str = " + ".join(f"{v} {k}" for k, v in types.items())
        if n > 0:
            marker = " ← DEEP" if sid == deep_seed else ""
            print(f"    Seed {sid}: {n} queries ({type_str}){marker}")

    return queries, deep_seed


# ── Prediction building ────────────────────────────────────────────────────

def build_round_calibration(all_observations, all_grids, all_settlements, height, width):
    """
    Pool observations from ALL seeds into round-specific calibration buckets.
    Key insight: hidden parameters are shared across seeds, so pooling gives 5x data.

    Also tracks observed settlement positions per seed for improved distance features,
    and computes adaptive blend weight based on round-vs-historical divergence.
    """
    round_cal = defaultdict(lambda: np.zeros(NUM_CLASSES))
    round_cal_n = defaultdict(int)
    round_cal_base = defaultdict(lambda: np.zeros(NUM_CLASSES))  # base-key fallback
    round_cal_base_n = defaultdict(int)
    # Track observed settlement positions per seed
    obs_sett_positions = [[] for _ in range(len(all_grids))]

    for sid, (observations, grid, settlements) in enumerate(zip(all_observations, all_grids, all_settlements)):
        features = precompute_features(grid, settlements, height, width)
        dist_map, coastal_map, density_map, port_prox_map, init_port_set = features

        for obs_grid, vx, vy in observations:
            vh = len(obs_grid)
            vw = len(obs_grid[0]) if vh > 0 else 0
            for dy in range(vh):
                for dx in range(vw):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < height and 0 <= x < width:
                        code = grid[y][x]
                        if code in (10, 5):
                            continue
                        obs_code = obs_grid[dy][dx]
                        cls = TERRAIN_TO_CLASS.get(obs_code, 0)

                        # Track where we observed alive settlements
                        if obs_code in (1, 2):
                            obs_sett_positions[sid].append((x, y))

                        key = cell_feature_key(code, dist_map, coastal_map,
                                               density_map, port_prox_map, init_port_set, y, x)
                        round_cal[key][cls] += 1
                        round_cal_n[key] += 1
                        bkey = cell_base_key(code, dist_map, coastal_map, y, x)
                        round_cal_base[bkey][cls] += 1
                        round_cal_base_n[bkey] += 1

    # Estimate expansion rate from observations
    # Count observed settlements vs initial settlements across all observed cells
    obs_sett_count = 0
    init_sett_count = 0
    total_dynamic_obs = 0
    for sid, (observations, grid, settlements) in enumerate(zip(all_observations, all_grids, all_settlements)):
        ig = np.array(grid)
        for obs_grid, vx, vy in observations:
            vh_obs = len(obs_grid)
            vw_obs = len(obs_grid[0]) if vh_obs > 0 else 0
            for dy in range(vh_obs):
                for dx in range(vw_obs):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < height and 0 <= x < width:
                        init_code = ig[y][x]
                        if init_code in (10, 5):
                            continue
                        total_dynamic_obs += 1
                        obs_code = obs_grid[dy][dx]
                        if obs_code in (1, 2):
                            obs_sett_count += 1
                        if init_code == 1:
                            init_sett_count += 1

    if init_sett_count > 0:
        observed_expansion = obs_sett_count / init_sett_count
    else:
        observed_expansion = 4.5  # default

    # Estimate ruin rate: ruins per settlement observed
    obs_ruin_count = 0
    for sid, (observations, grid, settlements) in enumerate(zip(all_observations, all_grids, all_settlements)):
        for obs_grid, vx, vy in observations:
            vh_obs = len(obs_grid)
            vw_obs = len(obs_grid[0]) if vh_obs > 0 else 0
            for dy in range(vh_obs):
                for dx in range(vw_obs):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < height and 0 <= x < width:
                        if obs_grid[dy][dx] == 3:  # Ruin
                            obs_ruin_count += 1

    observed_ruin_rate = obs_ruin_count / max(obs_sett_count, 1)
    print(f"\n  Observed expansion: {observed_expansion:.1f}x, ruin_rate: {observed_ruin_rate:.3f} "
          f"(sett={obs_sett_count}, ruin={obs_ruin_count}, init_sett={init_sett_count})")

    # Build similarity-weighted historical prior
    global HIST_CAL
    HIST_CAL = build_hist_prior(observed_expansion, ruin_rate=observed_ruin_rate, top_n=5)

    # Compute adaptive blend weight
    divergence = 0.0
    div_count = 0
    for key in round_cal_base:
        if round_cal_base_n[key] < 20 or key not in HIST_CAL:
            continue
        rf = round_cal_base[key] / round_cal_base_n[key]
        hf = np.array(HIST_CAL[key])
        rf_c = np.maximum(rf, 0.01); hf_c = np.maximum(hf, 0.01)
        rf_c /= rf_c.sum(); hf_c /= hf_c.sum()
        kl = float(np.sum(rf_c * np.log(rf_c / hf_c)))
        divergence += kl
        div_count += 1

    avg_divergence = divergence / max(div_count, 1)
    adaptive_weight = max(0.1, HIST_BLEND_WEIGHT - avg_divergence * 2)

    print(f"  Round-vs-historical divergence: {avg_divergence:.3f} → blend_weight={adaptive_weight:.2f}")
    # Print base-key calibration summary (more readable)
    labels = ['Empty', 'Settl', 'Port', 'Ruin', 'Forest', 'Mount']
    terrain_names = {0: 'Empty', 11: 'Plains', 1: 'Settl', 4: 'Forest', 2: 'Port'}
    for key in sorted(round_cal_base.keys()):
        code, db_val, coastal = key
        n = round_cal_base_n[key]
        if n < 20:
            continue
        freq = round_cal_base[key] / n
        name = terrain_names.get(code, str(code))
        c_str = 'coast' if coastal else 'inlnd'
        print(f"    {name:6s} d<={db_val:2d} {c_str} n={n:5d}  "
              f"{' '.join(f'{labels[i]}={freq[i]:.3f}' for i in range(6))}")

    return round_cal, round_cal_n, round_cal_base, round_cal_base_n, obs_sett_positions, adaptive_weight


def build_prediction(grid, settlements, height, width, round_cal, round_cal_n,
                     round_cal_base=None, round_cal_base_n=None,
                     obs_sett_pos=None, blend_weight=HIST_BLEND_WEIGHT,
                     per_cell_obs=None,
                     strength_cal=None, strength_cal_n=None, strength_grid=None):
    """
    Build prediction blending calibration priors with per-cell observations.

    For deeply-observed seeds (per_cell_obs provided):
      Bayesian update of bucket prior with per-cell observation counts.
    For lightly-observed seeds:
      Bucket calibration + historical prior blend.
    """
    features = precompute_features(grid, settlements, height, width)
    dist_map, coastal_map, density_map, port_prox_map, init_port_set = features

    # Observed settlement distance
    obs_dist_map = None
    if obs_sett_pos and len(obs_sett_pos) > 0:
        obs_dist_map = np.full((height, width), 999, dtype=int)
        for sx, sy in obs_sett_pos:
            for y in range(height):
                for x in range(width):
                    d = abs(sx - x) + abs(sy - y)
                    if d < obs_dist_map[y][x]:
                        obs_dist_map[y][x] = d

    prediction = np.zeros((height, width, NUM_CLASSES))

    for y in range(height):
        for x in range(width):
            code = grid[y][x]

            if code == 10:
                prediction[y][x][0] = 1.0
                continue
            if code == 5:
                prediction[y][x][5] = 1.0
                continue

            # Extended feature key
            key = cell_feature_key(code, dist_map, coastal_map,
                                   density_map, port_prox_map, init_port_set, y, x)

            # Historical prior
            hist_prior = get_hist_prior(key[0], dist_map[y][x], bool(coastal_map[y][x]))

            # Round-specific bucket calibration
            # Priority: strength-aware → extended features → base features
            round_freq = None
            n_round = 0

            # Try strength-aware calibration first (only for deeply-observed seed)
            if strength_cal is not None and strength_grid is not None:
                ss = strength_grid[y][x]
                sb = 0 if ss < 0.1 else (1 if ss < 0.25 else 2)
                base_key = cell_base_key(code, dist_map, coastal_map, y, x)
                sk = (base_key[0], base_key[1], base_key[2], sb)
                if sk in strength_cal and strength_cal_n[sk] >= 10:
                    round_freq = strength_cal[sk] / strength_cal_n[sk]
                    n_round = strength_cal_n[sk]

            # Fallback: extended features → base features
            if round_freq is None:
                if key in round_cal and round_cal_n[key] >= 5:
                    round_freq = round_cal[key] / round_cal_n[key]
                    n_round = round_cal_n[key]
                elif round_cal_base is not None:
                    base_key = cell_base_key(code, dist_map, coastal_map, y, x)
                    if base_key in round_cal_base and round_cal_base_n[base_key] > 0:
                        round_freq = round_cal_base[base_key] / round_cal_base_n[base_key]
                        n_round = round_cal_base_n[base_key]

            # Build bucket-level prior (blend round cal + historical)
            # Blend weight scales with sample size: trust round data more when n is high
            if round_freq is not None:
                # For small buckets (n<50), increase hist weight for stability
                effective_blend = blend_weight * max(1.0, 50.0 / max(n_round, 1))
                effective_blend = min(effective_blend, 2.0)  # cap at 2x
                hist_effective_n = n_round * effective_blend
                bucket_prior = (round_freq * n_round + hist_prior * hist_effective_n) / (n_round + hist_effective_n)
            else:
                bucket_prior = hist_prior

            # Per-cell Bayesian update (for deeply observed seeds)
            if per_cell_obs is not None and (y, x) in per_cell_obs:
                cell_counts = per_cell_obs[(y, x)]
                n_obs = cell_counts.sum()
                if n_obs >= 2:
                    # Use bucket prior as Bayesian prior, update with cell observations
                    # High pseudo_count (30) = trust bucket prior, gently refine with cell data
                    pseudo_count = 30.0
                    combined = bucket_prior * pseudo_count + cell_counts
                    prediction[y][x] = combined / combined.sum()
                else:
                    prediction[y][x] = bucket_prior
            else:
                prediction[y][x] = bucket_prior

    # Smart floor
    for y in range(height):
        for x in range(width):
            code = grid[y][x]
            if code in (10, 5):
                continue
            p = prediction[y][x]
            is_coastal = bool(coastal_map[y][x])
            p[0] = max(p[0], 0.003)
            p[1] = max(p[1], 0.003)
            p[2] = max(p[2], 0.003 if is_coastal else 0.0005)
            p[3] = max(p[3], 0.002)
            p[4] = max(p[4], 0.003)
            p[5] = max(p[5], 0.0005)
            prediction[y][x] = p / p.sum()

    return prediction


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ASTAR ISLAND SOLVER v3")
    print("=" * 60)

    round_info = get_active_round()
    if not round_info:
        return
    round_id = round_info["id"]
    print(f"\nRound #{round_info['round_number']} (weight: {round_info['round_weight']:.3f})")
    print(f"  Closes at: {round_info['closes_at']}")

    detail = get_round_details(round_id)
    width, height = detail["map_width"], detail["map_height"]
    seeds_count = detail["seeds_count"]
    print(f"  Map: {width}x{height}, Seeds: {seeds_count}")

    budget = get_budget()
    queries_remaining = budget["queries_max"] - budget["queries_used"]
    print(f"  Budget: {budget['queries_used']}/{budget['queries_max']} ({queries_remaining} remaining)")

    # Prepare seed data
    all_grids = []
    all_settlements = []
    all_observations = []  # list of list of (obs_grid, vx, vy)

    for i in range(seeds_count):
        state = detail["initial_states"][i]
        grid = state["grid"]
        sett = state["settlements"]
        alive = sum(1 for s in sett if s.get("alive", True))
        ports = sum(1 for s in sett if s.get("has_port", False))
        print(f"  Seed {i}: {alive} settlements ({ports} ports)")
        all_grids.append(grid)
        all_settlements.append(sett)
        all_observations.append([])

    # Execute queries
    if queries_remaining > 0:
        queries, deep_seed = plan_queries(detail, total_budget=queries_remaining)
        print(f"\nQuerying simulator...")
        for qi, q in enumerate(queries):
            sid = q["seed_index"]
            vp = f"({q['vx']},{q['vy']},{q['vw']},{q['vh']})"
            try:
                result = simulate(round_id, sid, q["vx"], q["vy"], q["vw"], q["vh"])
                all_observations[sid].append((result["grid"], q["vx"], q["vy"]))
                flat = [TERRAIN_TO_CLASS.get(c, 0) for row in result["grid"] for c in row]
                dist = Counter(flat)
                labels = {0: "E", 1: "S", 2: "P", 3: "R", 4: "F", 5: "M"}
                summary = " ".join(f"{labels[k]}:{v}" for k, v in sorted(dist.items()))
                # Log settlement stats for deep seed
                if sid == deep_seed and result.get("settlements"):
                    alive_sett = [ss for ss in result["settlements"] if ss.get("alive", True)]
                    if alive_sett:
                        avg_food = np.mean([ss["food"] for ss in alive_sett])
                        avg_pop = np.mean([ss["population"] for ss in alive_sett])
                        summary += f" food={avg_food:.1f} pop={avg_pop:.1f}"
                print(f"  [{qi+1:2d}/{len(queries)}] seed={sid} {vp:20s} {q['type']:8s} → {summary}")
            except Exception as e:
                print(f"  [{qi+1:2d}/{len(queries)}] seed={sid} {vp:20s} ERROR: {e}")
            time.sleep(0.2)

        # Build round-specific calibration from ALL observations
        round_cal, round_cal_n, round_cal_base, round_cal_base_n, obs_sett_positions, adaptive_weight = build_round_calibration(
            all_observations, all_grids, all_settlements, height, width
        )
    else:
        print("\n  No queries remaining - skipping (keeping existing submission)")
        print("Done!")
        return

    # Build per-cell observation counts and settlement strength for deep seed
    per_cell_obs_s0 = defaultdict(lambda: np.zeros(NUM_CLASSES))
    ig0 = np.array(all_grids[deep_seed])

    for obs_grid, vx, vy in all_observations[deep_seed]:
        vh_obs = len(obs_grid)
        vw_obs = len(obs_grid[0]) if vh_obs > 0 else 0
        for dy in range(vh_obs):
            for dx in range(vw_obs):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    cls = TERRAIN_TO_CLASS.get(obs_grid[dy][dx], 0)
                    per_cell_obs_s0[(y, x)][cls] += 1

    n_deep_cells = sum(1 for k, v in per_cell_obs_s0.items() if v.sum() >= 3)
    n_obs_per_cell = np.zeros((height, width))
    for (y, x), counts in per_cell_obs_s0.items():
        n_obs_per_cell[y][x] = counts.sum()
    avg_obs = n_obs_per_cell[n_obs_per_cell > 0].mean() if (n_obs_per_cell > 0).any() else 0
    print(f"\n  Seed {deep_seed} per-cell: {len(per_cell_obs_s0)} cells, {n_deep_cells} with 3+ obs, avg={avg_obs:.1f} obs/cell")

    # Compute settlement strength grid for deep seed:
    # Use per-cell observation counts to compute fraction of nearby settlements
    # This avoids viewport edge bias by using aggregated per-cell data
    sett_strength_s0 = np.zeros((height, width))
    if len(per_cell_obs_s0) > 0:
        # Build settlement fraction map from per-cell observations
        sett_frac = np.zeros((height, width))
        obs_count_map = np.zeros((height, width))
        for (y, x), counts in per_cell_obs_s0.items():
            n = counts.sum()
            if n > 0:
                sett_frac[y][x] = counts[1] / n  # fraction of obs that were Settlement
                obs_count_map[y][x] = n

        # For each cell, compute average settlement fraction in radius 3
        for y in range(height):
            for x in range(width):
                if ig0[y][x] in (10, 5):
                    continue
                total_weight = 0.0
                weighted_sum = 0.0
                for dy2 in range(-3, 4):
                    for dx2 in range(-3, 4):
                        if abs(dy2) + abs(dx2) > 3:
                            continue
                        ny, nx = y + dy2, x + dx2
                        if 0 <= ny < height and 0 <= nx < width and obs_count_map[ny][nx] > 0:
                            w = obs_count_map[ny][nx]
                            weighted_sum += sett_frac[ny][nx] * w
                            total_weight += w
                if total_weight > 0:
                    sett_strength_s0[y][x] = weighted_sum / total_weight

    print(f"  Seed {deep_seed} settlement strength: computed from per-cell data")

    # Build strength-aware calibration for deep seed
    round_cal_str = defaultdict(lambda: np.zeros(NUM_CLASSES))
    round_cal_str_n = defaultdict(int)
    dm0, cm0 = precompute_features(all_grids[deep_seed], all_settlements[deep_seed], height, width)[:2]
    for obs_grid, vx, vy in all_observations[deep_seed]:
        vh_obs = len(obs_grid)
        vw_obs = len(obs_grid[0]) if vh_obs > 0 else 0
        for dy in range(vh_obs):
            for dx in range(vw_obs):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    code = ig0[y][x]
                    if code in (10, 5): continue
                    cls = TERRAIN_TO_CLASS.get(obs_grid[dy][dx], 0)
                    co = bool(cm0[y][x])
                    ss = sett_strength_s0[y][x]
                    sb = 0 if ss < 0.1 else (1 if ss < 0.25 else 2)
                    sk = (int(code), dist_bin(dm0[y][x]), co, sb)
                    round_cal_str[sk][cls] += 1
                    round_cal_str_n[sk] += 1

    # Build and submit predictions
    print(f"\nSubmitting predictions...")
    for sid in range(seeds_count):
        # Deep seed: per-cell + strength-aware calibration
        # Other seeds: base calibration only
        cell_obs = per_cell_obs_s0 if sid == deep_seed else None
        str_cal = round_cal_str if sid == deep_seed else None
        str_cal_n = round_cal_str_n if sid == deep_seed else None
        str_grid = sett_strength_s0 if sid == deep_seed else None

        prediction = build_prediction(
            all_grids[sid], all_settlements[sid], height, width,
            round_cal, round_cal_n,
            round_cal_base=round_cal_base, round_cal_base_n=round_cal_base_n,
            obs_sett_pos=obs_sett_positions[sid] if sid == deep_seed else None,
            blend_weight=adaptive_weight,
            per_cell_obs=cell_obs,
            strength_cal=str_cal, strength_cal_n=str_cal_n, strength_grid=str_grid,
        )
        resp = session.post(f"{BASE}/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": sid,
            "prediction": prediction.tolist(),
        })
        n_obs = len(all_observations[sid])
        result = resp.json()
        status = result.get("status", str(result)[:100])
        print(f"  Seed {sid} ({n_obs} obs): {resp.status_code} - {status}")

    print("\nDone!")


if __name__ == "__main__":
    main()
