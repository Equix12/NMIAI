"""
Astar Island Simulator v2 - Reverse-engineered from ground truth analysis.

Key insights from data:
- Settlements expand slowly (~1 cell/year outward over 50 years, reaching ~10 cells)
- ~55% of initial settlements collapse and get reclaimed
- Forests AND plains can become settlements
- Ruins get reclaimed quickly (→ forest or empty within a few years)
- Expansion probability decays gradually with distance from settlements

Phases per year: Growth → Conflict → Trade → Winter → Environment
"""

import numpy as np
from dataclasses import dataclass
from typing import List

# Terrain codes
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

TERRAIN_TO_CLASS = {OCEAN: 0, PLAINS: 0, EMPTY: 0, SETTLEMENT: 1, PORT: 2, RUIN: 3, FOREST: 4, MOUNTAIN: 5}


@dataclass
class HiddenParams:
    """Hidden parameters controlling simulation behavior."""
    # Growth & Expansion
    food_per_forest: float = 0.25       # food from each adjacent forest
    food_per_plains: float = 0.05       # food from each adjacent plains
    base_food_production: float = 0.15  # base food per settlement
    growth_rate: float = 0.08           # population growth multiplier
    expansion_threshold: float = 2.5    # min population to attempt expansion
    expansion_prob: float = 0.015       # per-candidate-cell probability per year
    expand_into_forest: bool = True     # can settlements replace forests?
    port_develop_prob: float = 0.08     # prob coastal settlement → port
    longship_prob: float = 0.04         # prob port builds longship

    # Conflict
    raid_range: float = 5.0
    raid_prob: float = 0.08
    desperate_raid_mult: float = 2.5
    raid_damage: float = 0.25
    raid_loot: float = 0.15
    conquest_prob: float = 0.08
    longship_range_bonus: float = 8.0

    # Trade
    trade_range: float = 8.0
    trade_food: float = 0.08
    trade_wealth: float = 0.05

    # Winter
    winter_severity: float = 0.35
    winter_variance: float = 0.15
    collapse_threshold: float = -0.3
    collapse_prob: float = 0.25

    # Environment
    ruin_forest_prob: float = 0.12      # ruins → forest per year
    ruin_rebuild_range: float = 3.0
    ruin_rebuild_prob: float = 0.04
    ruin_to_plains_prob: float = 0.08   # ruins → empty/plains


class AstarSimulator:
    """Fast Norse world simulator using numpy arrays for state."""

    def __init__(self, grid, settlements_data, params: HiddenParams, rng=None):
        self.H = len(grid)
        self.W = len(grid[0])
        self.p = params
        self.rng = rng or np.random.default_rng()

        # Grid state
        self.grid = np.array(grid, dtype=np.int8)

        # Settlement state as parallel arrays for speed
        max_sett = len(settlements_data) * 15  # allow for expansion
        self.n_sett = len(settlements_data)
        self.sx = np.zeros(max_sett, dtype=np.int16)
        self.sy = np.zeros(max_sett, dtype=np.int16)
        self.pop = np.zeros(max_sett, dtype=np.float32)
        self.food = np.zeros(max_sett, dtype=np.float32)
        self.wealth = np.zeros(max_sett, dtype=np.float32)
        self.defense = np.zeros(max_sett, dtype=np.float32)
        self.is_port = np.zeros(max_sett, dtype=bool)
        self.has_ship = np.zeros(max_sett, dtype=bool)
        self.alive = np.zeros(max_sett, dtype=bool)
        self.owner = np.zeros(max_sett, dtype=np.int16)

        for i, sd in enumerate(settlements_data):
            self.sx[i] = sd["x"]
            self.sy[i] = sd["y"]
            self.pop[i] = 1.0 + self.rng.random() * 0.5
            self.food[i] = 0.5 + self.rng.random() * 0.5
            self.wealth[i] = self.rng.random() * 0.2
            self.defense[i] = 0.3 + self.rng.random() * 0.3
            self.is_port[i] = sd.get("has_port", False)
            self.alive[i] = sd.get("alive", True)
            self.owner[i] = i

        # Precompute adjacency: for each cell, count adjacent terrain types
        self._precompute_coastal()

    def _precompute_coastal(self):
        """Precompute which cells are coastal (adjacent to ocean)."""
        self.coastal = np.zeros((self.H, self.W), dtype=bool)
        for y in range(self.H):
            for x in range(self.W):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.H and 0 <= nx < self.W:
                            if self.grid[ny][nx] == OCEAN:
                                self.coastal[y][x] = True

    def _count_adjacent_terrain(self, x, y):
        """Count adjacent terrain types for food calculation."""
        n_forest = 0
        n_plains = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    t = self.grid[ny][nx]
                    if t == FOREST:
                        n_forest += 1
                    elif t == PLAINS or t == EMPTY:
                        n_plains += 1
        return n_forest, n_plains

    def _find_expansion_candidates(self, x, y):
        """Find cells this settlement could expand to."""
        candidates = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                if abs(dx) + abs(dy) > 2:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    t = self.grid[ny][nx]
                    if t == PLAINS or t == EMPTY:
                        # Check no settlement already there
                        occupied = False
                        for j in range(self.n_sett):
                            if self.alive[j] and self.sx[j] == nx and self.sy[j] == ny:
                                occupied = True
                                break
                        if not occupied:
                            candidates.append((nx, ny))
                    elif t == FOREST and self.p.expand_into_forest:
                        occupied = False
                        for j in range(self.n_sett):
                            if self.alive[j] and self.sx[j] == nx and self.sy[j] == ny:
                                occupied = True
                                break
                        if not occupied:
                            candidates.append((nx, ny))
        return candidates

    def _add_settlement(self, x, y, parent_idx):
        """Add a new settlement, growing the arrays if needed."""
        idx = self.n_sett
        if idx >= len(self.sx):
            # Grow arrays
            n = len(self.sx)
            for attr in ['sx', 'sy', 'pop', 'food', 'wealth', 'defense', 'owner']:
                old = getattr(self, attr)
                new = np.zeros(n * 2, dtype=old.dtype)
                new[:n] = old
                setattr(self, attr, new)
            for attr in ['is_port', 'has_ship', 'alive']:
                old = getattr(self, attr)
                new = np.zeros(n * 2, dtype=old.dtype)
                new[:n] = old
                setattr(self, attr, new)

        self.sx[idx] = x
        self.sy[idx] = y
        self.pop[idx] = self.pop[parent_idx] * 0.25
        self.food[idx] = self.food[parent_idx] * 0.15
        self.wealth[idx] = 0.05
        self.defense[idx] = 0.2
        self.is_port[idx] = self.coastal[y][x] and self.rng.random() < 0.2
        self.has_ship[idx] = False
        self.alive[idx] = True
        self.owner[idx] = self.owner[parent_idx]
        self.n_sett += 1

        self.grid[y][x] = SETTLEMENT
        self.pop[parent_idx] *= 0.8
        self.food[parent_idx] *= 0.85

    def phase_growth(self):
        p = self.p
        alive_idx = [i for i in range(self.n_sett) if self.alive[i]]

        for i in alive_idx:
            # Food production
            n_f, n_p = self._count_adjacent_terrain(self.sx[i], self.sy[i])
            food_gain = p.base_food_production + n_f * p.food_per_forest + n_p * p.food_per_plains
            self.food[i] += food_gain

            # Population growth
            if self.food[i] > 0:
                self.pop[i] += p.growth_rate * self.pop[i] * min(self.food[i], 1.0)
                self.defense[i] = min(self.defense[i] + 0.01, 1.0)

            # Port development
            if not self.is_port[i] and self.coastal[self.sy[i]][self.sx[i]]:
                if self.pop[i] > 1.5 and self.rng.random() < p.port_develop_prob:
                    self.is_port[i] = True

            # Longship
            if self.is_port[i] and not self.has_ship[i]:
                if self.wealth[i] > 0.2 and self.rng.random() < p.longship_prob:
                    self.has_ship[i] = True

        # Expansion
        for i in alive_idx:
            if self.pop[i] < p.expansion_threshold:
                continue

            candidates = self._find_expansion_candidates(self.sx[i], self.sy[i])
            if not candidates:
                continue

            for cx, cy in candidates:
                if self.rng.random() < p.expansion_prob:
                    self._add_settlement(cx, cy, i)
                    break  # max one expansion per settlement per year

    def phase_conflict(self):
        p = self.p
        alive_idx = [i for i in range(self.n_sett) if self.alive[i]]

        for i in alive_idx:
            raid_p = p.raid_prob
            if self.food[i] < 0:
                raid_p *= p.desperate_raid_mult

            if self.rng.random() > raid_p:
                continue

            max_range = p.raid_range
            if self.has_ship[i]:
                max_range += p.longship_range_bonus

            # Find targets
            best_j = -1
            best_dist = 999
            for j in alive_idx:
                if self.owner[j] == self.owner[i]:
                    continue
                d = abs(self.sx[i] - self.sx[j]) + abs(self.sy[i] - self.sy[j])
                if d <= max_range and d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j < 0:
                continue

            j = best_j
            attack = self.pop[i] * (1 + 0.1)
            defend = self.pop[j] * self.defense[j]

            if attack > defend * 0.5:
                loot = min(self.food[j] * p.raid_loot, 0.5)
                self.food[i] += loot
                self.wealth[i] += self.wealth[j] * p.raid_loot
                self.food[j] -= loot
                self.defense[j] -= p.raid_damage
                self.pop[j] *= (1 - p.raid_damage * 0.3)

                if self.defense[j] < 0.1 and self.rng.random() < p.conquest_prob:
                    self.owner[j] = self.owner[i]

    def phase_trade(self):
        p = self.p
        ports = [i for i in range(self.n_sett) if self.alive[i] and self.is_port[i]]

        for ii, i in enumerate(ports):
            for j in ports[ii+1:]:
                d = abs(self.sx[i] - self.sx[j]) + abs(self.sy[i] - self.sy[j])
                if d <= p.trade_range:
                    self.food[i] += p.trade_food
                    self.food[j] += p.trade_food
                    self.wealth[i] += p.trade_wealth
                    self.wealth[j] += p.trade_wealth

    def phase_winter(self):
        p = self.p
        severity = max(0.05, p.winter_severity + self.rng.normal(0, p.winter_variance))

        for i in range(self.n_sett):
            if not self.alive[i]:
                continue

            self.food[i] -= severity
            self.wealth[i] = max(0, self.wealth[i] - severity * 0.05)

            if self.food[i] < p.collapse_threshold:
                if self.rng.random() < p.collapse_prob:
                    self.alive[i] = False
                    self.grid[self.sy[i]][self.sx[i]] = RUIN

                    # Disperse population
                    for j in range(self.n_sett):
                        if self.alive[j] and self.owner[j] == self.owner[i]:
                            d = abs(self.sx[i]-self.sx[j]) + abs(self.sy[i]-self.sy[j])
                            if d <= 5:
                                self.pop[j] += self.pop[i] * 0.15
                                break

    def phase_environment(self):
        p = self.p

        for i in range(self.n_sett):
            if self.alive[i]:
                continue
            x, y = self.sx[i], self.sy[i]
            if self.grid[y][x] != RUIN:
                continue

            # Try rebuild by nearby settlement
            rebuilt = False
            for j in range(self.n_sett):
                if not self.alive[j]:
                    continue
                d = abs(x - self.sx[j]) + abs(y - self.sy[j])
                if d <= p.ruin_rebuild_range and self.pop[j] > 1.5:
                    if self.rng.random() < p.ruin_rebuild_prob:
                        self._add_settlement(x, y, j)
                        rebuilt = True
                        break

            if not rebuilt:
                if self.rng.random() < p.ruin_forest_prob:
                    self.grid[y][x] = FOREST
                elif self.rng.random() < p.ruin_to_plains_prob:
                    self.grid[y][x] = PLAINS

    def run(self, years=50):
        for _ in range(years):
            self.phase_growth()
            self.phase_conflict()
            self.phase_trade()
            self.phase_winter()
            self.phase_environment()

        # Finalize: update grid for alive settlements
        for i in range(self.n_sett):
            if self.alive[i]:
                if self.is_port[i]:
                    self.grid[self.sy[i]][self.sx[i]] = PORT
                else:
                    self.grid[self.sy[i]][self.sx[i]] = SETTLEMENT

        return self.grid

    def get_class_grid(self):
        cg = np.zeros((self.H, self.W), dtype=np.int8)
        for y in range(self.H):
            for x in range(self.W):
                cg[y][x] = TERRAIN_TO_CLASS.get(int(self.grid[y][x]), 0)
        return cg


def run_monte_carlo(grid, settlements_data, params, n_runs=500, years=50, seed=None):
    """Run Monte Carlo and return (H, W, 6) probability tensor."""
    H = len(grid)
    W = len(grid[0])
    counts = np.zeros((H, W, 6), dtype=np.int32)

    base_rng = np.random.default_rng(seed)

    for i in range(n_runs):
        rng = np.random.default_rng(base_rng.integers(0, 2**31))
        sim = AstarSimulator(grid, settlements_data, params, rng=rng)
        sim.run(years=years)
        cg = sim.get_class_grid()
        for y in range(H):
            for x in range(W):
                counts[y][x][cg[y][x]] += 1

    probs = counts.astype(np.float64) / n_runs
    probs = np.maximum(probs, 0.01)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


if __name__ == "__main__":
    import time
    grid = [[OCEAN]*10 for _ in range(10)]
    for y in range(1, 9):
        for x in range(1, 9):
            grid[y][x] = PLAINS
    grid[3][3] = FOREST; grid[3][4] = FOREST; grid[4][3] = FOREST
    settlements = [{"x": 5, "y": 5, "has_port": False, "alive": True},
                   {"x": 2, "y": 2, "has_port": False, "alive": True}]
    params = HiddenParams()
    print("Running 100 Monte Carlo runs...")
    t0 = time.time()
    probs = run_monte_carlo(grid, settlements, params, n_runs=100)
    print(f"Done in {time.time()-t0:.1f}s")
    labels = ['E','S','P','R','F','M']
    for y in range(10):
        for x in range(10):
            if grid[y][x] == OCEAN:
                print("~~ ", end="")
            else:
                tc = np.argmax(probs[y][x])
                print(f"{labels[tc]}{int(probs[y][x][tc]*9)} ", end="")
        print()
