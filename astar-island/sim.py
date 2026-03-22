"""
Astar Island Simulator - Agent-based, matching documented mechanics.

Each settlement is an agent with: position, population, food, wealth,
defense, tech_level, has_port, has_longship, alive, owner_id.

Phases per year (50 years total):
1. Growth: food from terrain → population growth → port/longship dev → expansion
2. Conflict: raids between factions, looting, conquest
3. Trade: ports exchange food/wealth, tech diffusion
4. Winter: food loss, collapse → ruins, population dispersal
5. Environment: ruins → rebuilt/forest/plains

Hidden parameters (to be fitted from observations):
- Food production rates
- Expansion threshold & probability
- Raid range, probability, damage
- Trade range, food/wealth generation
- Winter severity
- Collapse threshold & probability
- Ruin reclamation rates
"""

import numpy as np
from dataclasses import dataclass

OCEAN = 10; PLAINS = 11; EMPTY = 0; SETTLEMENT = 1
PORT = 2; RUIN = 3; FOREST = 4; MOUNTAIN = 5
TERRAIN_TO_CLASS = {OCEAN:0, PLAINS:0, EMPTY:0, SETTLEMENT:1, PORT:2, RUIN:3, FOREST:4, MOUNTAIN:5}


@dataclass
class Params:
    # Growth
    food_per_forest: float = 0.20
    food_per_plains: float = 0.03
    food_base: float = 0.10
    pop_growth: float = 0.08
    expand_pop_thresh: float = 2.0    # min population to attempt expansion
    expand_prob: float = 0.10         # prob of founding per candidate cell
    expand_max_dist: int = 2          # max Manhattan dist for founding
    port_prob: float = 0.08           # coastal settlement → port
    longship_prob: float = 0.05       # port → longship (needs wealth)

    # Conflict
    raid_range: int = 5
    longship_range_bonus: int = 8
    raid_prob: float = 0.10           # base yearly raid probability
    desperate_mult: float = 2.5       # multiplier when food < 0
    raid_damage: float = 0.25         # defense/pop damage to defender
    loot_frac: float = 0.20           # fraction of food/wealth looted
    conquest_prob: float = 0.10       # faction change after successful raid

    # Trade
    trade_range: int = 8
    trade_food: float = 0.08
    trade_wealth: float = 0.05

    # Winter
    winter_base: float = 0.30
    winter_var: float = 0.12
    collapse_food: float = -0.3       # food threshold for collapse risk
    collapse_prob: float = 0.25

    # Environment
    ruin_rebuild_range: int = 3
    ruin_rebuild_prob: float = 0.04
    ruin_to_forest: float = 0.10
    ruin_to_plains: float = 0.06


class Sim:
    __slots__ = ('H','W','grid','p','rng','n',
                 'sx','sy','pop','food','wealth','defense','tech',
                 'port','ship','alive','owner','_adj_cache','occ')

    def __init__(self, grid_data, sett_data, params, rng=None):
        self.H = len(grid_data)
        self.W = len(grid_data[0])
        self.p = params
        self.rng = rng or np.random.default_rng()
        self.grid = [row[:] for row in grid_data]  # mutable copy

        N = len(sett_data)
        cap = N * 12  # room for expansion
        self.n = N
        self.sx = np.zeros(cap, dtype=np.int16)
        self.sy = np.zeros(cap, dtype=np.int16)
        self.pop = np.zeros(cap, dtype=np.float32)
        self.food = np.zeros(cap, dtype=np.float32)
        self.wealth = np.zeros(cap, dtype=np.float32)
        self.defense = np.zeros(cap, dtype=np.float32)
        self.tech = np.zeros(cap, dtype=np.float32)
        self.port = np.zeros(cap, dtype=np.bool_)
        self.ship = np.zeros(cap, dtype=np.bool_)
        self.alive = np.zeros(cap, dtype=np.bool_)
        self.owner = np.zeros(cap, dtype=np.int16)

        r = self.rng
        for i, sd in enumerate(sett_data):
            self.sx[i] = sd['x']; self.sy[i] = sd['y']
            self.pop[i] = 1.0 + r.random()*0.5
            self.food[i] = 0.5 + r.random()*0.5
            self.wealth[i] = r.random()*0.2
            self.defense[i] = 0.3 + r.random()*0.3
            self.port[i] = sd.get('has_port', False)
            self.alive[i] = True
            self.owner[i] = i

        # Occupancy grid: occ[y][x] = settlement index or -1
        self.occ = np.full((self.H, self.W), -1, dtype=np.int16)
        for i in range(N):
            self.occ[self.sy[i]][self.sx[i]] = i

        # Precompute adjacency info
        self._adj_cache = {}

    def _adj(self, x, y):
        """8-connected neighbors."""
        key = (x, y)
        if key in self._adj_cache:
            return self._adj_cache[key]
        cells = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    cells.append((nx, ny))
        self._adj_cache[key] = cells
        return cells

    def _sett_at(self, x, y):
        """Find alive settlement index at position, or -1. O(1) via occ grid."""
        i = self.occ[y][x]
        if i >= 0 and self.alive[i]:
            return i
        return -1

    def _is_coastal(self, x, y):
        for nx, ny in self._adj(x, y):
            if self.grid[ny][nx] == OCEAN:
                return True
        return False

    def _grow(self, cap):
        """Double capacity of settlement arrays."""
        for attr in ('sx','sy','pop','food','wealth','defense','tech','owner'):
            old = getattr(self, attr)
            new = np.zeros(cap*2, dtype=old.dtype)
            new[:cap] = old
            setattr(self, attr, new)
        for attr in ('port','ship','alive'):
            old = getattr(self, attr)
            new = np.zeros(cap*2, dtype=old.dtype)
            new[:cap] = old
            setattr(self, attr, new)

    def _add_sett(self, x, y, parent):
        """Spawn new settlement from parent."""
        i = self.n
        if i >= len(self.sx):
            self._grow(len(self.sx))
        self.sx[i] = x; self.sy[i] = y
        self.pop[i] = self.pop[parent] * 0.25
        self.food[i] = self.food[parent] * 0.15
        self.wealth[i] = 0.05
        self.defense[i] = 0.2
        self.tech[i] = self.tech[parent] * 0.5
        self.port[i] = self._is_coastal(x, y) and self.rng.random() < 0.2
        self.ship[i] = False
        self.alive[i] = True
        self.owner[i] = self.owner[parent]
        self.n += 1
        self.grid[y][x] = SETTLEMENT
        self.occ[y][x] = i
        # Cost to parent
        self.pop[parent] *= 0.75
        self.food[parent] *= 0.8

    def phase_growth(self):
        p = self.p
        rng = self.rng
        g = self.grid

        alive_ids = [i for i in range(self.n) if self.alive[i]]

        for i in alive_ids:
            x, y = int(self.sx[i]), int(self.sy[i])
            # Food from adjacent terrain
            fg = p.food_base
            for nx, ny in self._adj(x, y):
                t = g[ny][nx]
                if t == FOREST: fg += p.food_per_forest
                elif t in (PLAINS, EMPTY): fg += p.food_per_plains
            self.food[i] += fg

            # Population growth
            if self.food[i] > 0:
                rate = p.pop_growth * min(self.food[i], 1.0)
                self.pop[i] += rate * self.pop[i]
                self.defense[i] = min(self.defense[i] + 0.01, 1.0)

            # Port development
            if not self.port[i] and self._is_coastal(x, y):
                if self.pop[i] > 1.5 and rng.random() < p.port_prob:
                    self.port[i] = True

            # Longship
            if self.port[i] and not self.ship[i]:
                if self.wealth[i] > 0.3 and rng.random() < p.longship_prob:
                    self.ship[i] = True

        # Expansion
        for i in alive_ids:
            if self.pop[i] < p.expand_pop_thresh:
                continue
            x, y = int(self.sx[i]), int(self.sy[i])

            # Find candidate cells within expand_max_dist
            candidates = []
            md = p.expand_max_dist
            for dy in range(-md, md+1):
                for dx in range(-md, md+1):
                    d = abs(dx)+abs(dy)
                    if d == 0 or d > md: continue
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        t = g[ny][nx]
                        if t in (PLAINS, EMPTY, FOREST):
                            if self._sett_at(nx, ny) < 0:
                                candidates.append((nx, ny))

            if not candidates:
                continue

            # Try to expand (at most one per settlement per year)
            rng.shuffle(candidates)
            for cx, cy in candidates:
                if rng.random() < p.expand_prob:
                    self._add_sett(cx, cy, i)
                    break

    def phase_conflict(self):
        p = self.p
        rng = self.rng
        alive_ids = [i for i in range(self.n) if self.alive[i]]

        for i in alive_ids:
            rp = p.raid_prob
            if self.food[i] < 0:
                rp *= p.desperate_mult
            if rng.random() > rp:
                continue

            max_r = p.raid_range
            if self.ship[i]: max_r += p.longship_range_bonus

            # Find enemy targets in range
            best_j = -1; best_d = 999
            ix, iy = int(self.sx[i]), int(self.sy[i])
            for j in alive_ids:
                if self.owner[j] == self.owner[i]: continue
                d = abs(ix-int(self.sx[j])) + abs(iy-int(self.sy[j]))
                if d <= max_r and d < best_d:
                    best_d = d; best_j = j

            if best_j < 0: continue
            j = best_j

            # Raid outcome
            atk = self.pop[i] * (1 + self.tech[i]*0.1)
            dfn = self.pop[j] * max(self.defense[j], 0.1)

            if atk > dfn * 0.5:
                loot_f = min(self.food[j] * p.loot_frac, self.pop[i]*0.5)
                loot_w = self.wealth[j] * p.loot_frac
                self.food[i] += loot_f; self.wealth[i] += loot_w
                self.food[j] -= loot_f; self.wealth[j] -= loot_w
                self.defense[j] -= p.raid_damage
                self.pop[j] *= (1 - p.raid_damage*0.3)
                if self.defense[j] < 0.1 and rng.random() < p.conquest_prob:
                    self.owner[j] = self.owner[i]

    def phase_trade(self):
        p = self.p
        ports = [i for i in range(self.n) if self.alive[i] and self.port[i]]
        for ii, i in enumerate(ports):
            for j in ports[ii+1:]:
                d = abs(int(self.sx[i])-int(self.sx[j])) + abs(int(self.sy[i])-int(self.sy[j]))
                if d <= p.trade_range:
                    self.food[i] += p.trade_food; self.food[j] += p.trade_food
                    self.wealth[i] += p.trade_wealth; self.wealth[j] += p.trade_wealth
                    avg_t = (self.tech[i]+self.tech[j])/2
                    self.tech[i] = self.tech[i]*0.9 + avg_t*0.1
                    self.tech[j] = self.tech[j]*0.9 + avg_t*0.1

    def phase_winter(self):
        p = self.p
        sev = max(0.05, p.winter_base + self.rng.normal(0, p.winter_var))
        for i in range(self.n):
            if not self.alive[i]: continue
            self.food[i] -= sev
            self.wealth[i] = max(0, self.wealth[i] - sev*0.05)

            if self.food[i] < p.collapse_food:
                if self.rng.random() < p.collapse_prob:
                    self.alive[i] = False
                    self.grid[int(self.sy[i])][int(self.sx[i])] = RUIN
                    self.occ[int(self.sy[i])][int(self.sx[i])] = -1
                    # Disperse to nearby friendly
                    for j in range(self.n):
                        if not self.alive[j] or self.owner[j] != self.owner[i]: continue
                        d = abs(int(self.sx[i])-int(self.sx[j])) + abs(int(self.sy[i])-int(self.sy[j]))
                        if d <= 5:
                            self.pop[j] += self.pop[i]*0.15
                            break

    def phase_env(self):
        p = self.p
        rng = self.rng
        for i in range(self.n):
            if self.alive[i]: continue
            x, y = int(self.sx[i]), int(self.sy[i])
            if self.grid[y][x] != RUIN: continue

            # Rebuild by nearby thriving settlement?
            rebuilt = False
            for j in range(self.n):
                if not self.alive[j]: continue
                d = abs(x-int(self.sx[j])) + abs(y-int(self.sy[j]))
                if d <= p.ruin_rebuild_range and self.pop[j] > 1.5:
                    if rng.random() < p.ruin_rebuild_prob:
                        self._add_sett(x, y, j)
                        rebuilt = True; break

            if not rebuilt:
                if rng.random() < p.ruin_to_forest:
                    self.grid[y][x] = FOREST
                elif rng.random() < p.ruin_to_plains:
                    self.grid[y][x] = PLAINS

    def run(self, years=50):
        for _ in range(years):
            self.phase_growth()
            self.phase_conflict()
            self.phase_trade()
            self.phase_winter()
            self.phase_env()

        # Finalize grid
        for i in range(self.n):
            if self.alive[i]:
                x, y = int(self.sx[i]), int(self.sy[i])
                self.grid[y][x] = PORT if self.port[i] else SETTLEMENT
        return self.grid

    def get_class_grid(self):
        cg = np.zeros((self.H, self.W), dtype=np.int8)
        for y in range(self.H):
            for x in range(self.W):
                cg[y][x] = TERRAIN_TO_CLASS.get(self.grid[y][x], 0)
        return cg

    def get_alive_settlements(self):
        """Return list of alive settlement dicts (like API response)."""
        result = []
        for i in range(self.n):
            if self.alive[i]:
                result.append({
                    'x': int(self.sx[i]), 'y': int(self.sy[i]),
                    'population': float(self.pop[i]),
                    'food': float(self.food[i]),
                    'wealth': float(self.wealth[i]),
                    'defense': float(self.defense[i]),
                    'has_port': bool(self.port[i]),
                    'alive': True,
                    'owner_id': int(self.owner[i]),
                })
        return result


def monte_carlo(grid, sett_data, params, n_runs=500, years=50, seed=None):
    """Run MC simulations, return (H,W,6) probability tensor."""
    H, W = len(grid), len(grid[0])
    counts = np.zeros((H, W, 6), dtype=np.int32)
    base_rng = np.random.default_rng(seed)

    for _ in range(n_runs):
        rng = np.random.default_rng(base_rng.integers(0, 2**31))
        s = Sim(grid, sett_data, params, rng)
        s.run(years)
        cg = s.get_class_grid()
        for c in range(6):
            counts[:,:,c] += (cg == c)

    probs = counts.astype(np.float64) / n_runs
    probs = np.maximum(probs, 0.01)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


if __name__ == '__main__':
    import time
    # Benchmark
    grid = [[OCEAN]*20 for _ in range(20)]
    for y in range(2,18):
        for x in range(2,18):
            grid[y][x] = PLAINS
    for y in range(5,10):
        for x in range(5,10):
            grid[y][x] = FOREST
    sett = [{'x':10,'y':10,'has_port':False},
            {'x':5,'y':15,'has_port':False},
            {'x':15,'y':5,'has_port':False}]

    p = Params()
    t0 = time.time()
    probs = monte_carlo(grid, sett, p, n_runs=200, seed=42)
    elapsed = time.time()-t0
    print(f'200 runs on 20x20: {elapsed:.1f}s ({200/elapsed:.0f} runs/sec)')
