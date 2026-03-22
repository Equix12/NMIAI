"""
Fast Astar Island Simulator using vectorized numpy operations.

Models the world as a cellular automaton where each cell has a state
and transitions depend on neighborhood features.

Key dynamics:
- Settlements produce food based on adjacent terrain
- Prosperous settlements expand to nearby cells (plains, forest, empty)
- Settlements can collapse due to food shortage / raids / winter
- Ruins get reclaimed by forest or rebuilt by nearby settlements
- Ports develop on coastal settlements, enable trade

All operations are vectorized with numpy for speed.
Target: 1000 Monte Carlo runs in <30 seconds on 40x40 grid.
"""

import numpy as np
from scipy.signal import convolve2d
from dataclasses import dataclass

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

# 8-connected neighborhood kernel
KERNEL_8 = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]], dtype=np.float32)

# 4-connected (cardinal) kernel
KERNEL_4 = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]], dtype=np.float32)


@dataclass
class Params:
    """Hidden parameters for the simulation."""
    # Food & Growth
    food_forest: float = 0.12        # food contribution per adjacent forest cell
    food_plains: float = 0.02        # food contribution per adjacent plains
    food_base: float = 0.08          # base food production
    food_port_bonus: float = 0.05    # extra food for ports (trade)

    # Expansion - uses wider kernel for gradual wave-like spread
    expand_prob: float = 0.005       # per-candidate-cell expansion probability per year
    expand_food_thresh: float = 0.3  # min accumulated food to expand
    expand_into_forest: float = 0.8  # relative probability of expanding into forest vs plains
    expand_radius: int = 5           # radius of expansion influence kernel

    # Collapse
    winter_severity: float = 0.30    # food consumed per winter
    winter_var: float = 0.10         # year-to-year variance
    collapse_thresh: float = -0.2    # food level below which collapse is possible
    collapse_prob: float = 0.20      # probability of collapse when below threshold
    raid_collapse: float = 0.02      # additional collapse probability from raids

    # Environment
    ruin_to_forest: float = 0.15     # probability ruin → forest per year (increased)
    ruin_to_empty: float = 0.10      # probability ruin → empty per year (increased)
    ruin_rebuild: float = 0.04       # probability ruin gets rebuilt by nearby settlement

    # Port
    port_prob: float = 0.06          # probability coastal settlement → port per year


class FastSimulator:
    def __init__(self, grid, settlements_data, params: Params, rng=None):
        self.H = len(grid)
        self.W = len(grid[0])
        self.p = params
        self.rng = rng or np.random.default_rng()

        # Store initial terrain (immutable: ocean, mountain)
        self.initial = np.array(grid, dtype=np.int8)

        # Current grid state
        self.grid = self.initial.copy()

        # Food accumulator per cell (only meaningful for settlements)
        self.food = np.zeros((self.H, self.W), dtype=np.float32)

        # Initialize food for existing settlements
        for sd in settlements_data:
            x, y = sd["x"], sd["y"]
            self.food[y][x] = 0.5 + self.rng.random() * 0.5

        # Precompute static masks
        self.ocean_mask = (self.initial == OCEAN)
        self.mountain_mask = (self.initial == MOUNTAIN)
        self.static_mask = self.ocean_mask | self.mountain_mask

        # Coastal mask (adjacent to ocean)
        self.coastal = convolve2d(self.ocean_mask.astype(np.float32),
                                   KERNEL_8, mode='same', boundary='fill') > 0

    def step(self):
        """Run one year of simulation."""
        p = self.p
        g = self.grid
        rng = self.rng

        # ── Masks for current state ──
        is_settlement = (g == SETTLEMENT) | (g == PORT)
        is_port = (g == PORT)
        is_forest = (g == FOREST)
        is_plains = (g == PLAINS) | (g == EMPTY)
        is_ruin = (g == RUIN)
        is_expandable = is_plains | (is_forest if p.expand_into_forest > 0 else np.zeros_like(is_forest))

        # ── Food production ──
        adj_forest = convolve2d(is_forest.astype(np.float32), KERNEL_8,
                                 mode='same', boundary='fill')
        adj_plains = convolve2d(is_plains.astype(np.float32), KERNEL_8,
                                 mode='same', boundary='fill')
        food_production = (p.food_base + adj_forest * p.food_forest +
                          adj_plains * p.food_plains)
        food_production += is_port * p.food_port_bonus

        self.food += food_production * is_settlement

        # ── Expansion ──
        # Chain-reaction model: settlements expand to adjacent cells (8-connected)
        # This creates a natural wave front that propagates outward over 50 years
        # The expand_prob controls how fast the front moves
        prosperous = is_settlement & (self.food > p.expand_food_thresh)

        # Count prosperous neighbors (expansion source strength)
        adj_prosperous = convolve2d(prosperous.astype(np.float32), KERNEL_8,
                                     mode='same', boundary='fill')

        # Expansion candidates: expandable non-static non-settlement cells
        can_expand = is_expandable & ~self.static_mask & ~is_settlement

        # Forest has lower expansion probability
        forest_penalty = np.where(is_forest, p.expand_into_forest, 1.0)

        # Expansion probability scales with number of prosperous neighbors
        expand_chance = 1.0 - (1.0 - p.expand_prob * forest_penalty) ** adj_prosperous
        expand_roll = rng.random((self.H, self.W)) < expand_chance
        new_settlements = can_expand & expand_roll

        # Apply expansion
        g[new_settlements] = SETTLEMENT
        self.food[new_settlements] = 0.3

        # Cost to parent settlements
        if new_settlements.any():
            adj_new = convolve2d(new_settlements.astype(np.float32), KERNEL_8,
                                  mode='same', boundary='fill')
            self.food -= adj_new * 0.1 * is_settlement

        # ── Port development ──
        coastal_settlement = is_settlement & ~is_port & self.coastal
        new_ports = coastal_settlement & (rng.random((self.H, self.W)) < p.port_prob)
        new_ports &= (self.food > 0.3)
        g[new_ports] = PORT

        # ── Conflict (simplified) ──
        # Settlements near many other settlements face raid pressure
        settlement_density = convolve2d(is_settlement.astype(np.float32),
                                         np.ones((5, 5)) / 25, mode='same', boundary='fill')
        raid_pressure = settlement_density * p.raid_collapse
        self.food -= raid_pressure * is_settlement

        # ── Winter ──
        severity = max(0.05, p.winter_severity + rng.normal(0, p.winter_var))
        self.food -= severity * is_settlement

        # ── Collapse ──
        can_collapse = is_settlement & (self.food < p.collapse_thresh)
        collapse_roll = rng.random((self.H, self.W)) < p.collapse_prob
        collapsed = can_collapse & collapse_roll

        g[collapsed] = RUIN
        self.food[collapsed] = 0

        # ── Environment: ruin reclamation ──
        is_ruin = (g == RUIN)

        # Rebuild by nearby settlements
        adj_alive = convolve2d(((g == SETTLEMENT) | (g == PORT)).astype(np.float32),
                                KERNEL_8, mode='same', boundary='fill')
        can_rebuild = is_ruin & (adj_alive > 0)
        rebuilt = can_rebuild & (rng.random((self.H, self.W)) < p.ruin_rebuild)
        g[rebuilt] = SETTLEMENT
        self.food[rebuilt] = 0.2

        # Forest reclamation
        remaining_ruin = (g == RUIN)
        to_forest = remaining_ruin & (rng.random((self.H, self.W)) < p.ruin_to_forest)
        g[to_forest] = FOREST

        # Plains reclamation
        remaining_ruin = (g == RUIN)
        to_empty = remaining_ruin & (rng.random((self.H, self.W)) < p.ruin_to_empty)
        g[to_empty] = PLAINS

        self.grid = g

    def run(self, years=50):
        for _ in range(years):
            self.step()
        return self.grid

    def get_class_grid(self):
        cg = np.zeros((self.H, self.W), dtype=np.int8)
        for code, cls in TERRAIN_TO_CLASS.items():
            cg[self.grid == code] = cls
        return cg


def run_monte_carlo(grid, settlements_data, params, n_runs=500, years=50, seed=None):
    """Run Monte Carlo simulations and return (H, W, 6) probability tensor."""
    H = len(grid)
    W = len(grid[0])
    counts = np.zeros((H, W, 6), dtype=np.int32)
    base_rng = np.random.default_rng(seed)

    for i in range(n_runs):
        rng = np.random.default_rng(base_rng.integers(0, 2**31))
        sim = FastSimulator(grid, settlements_data, params, rng=rng)
        sim.run(years=years)
        cg = sim.get_class_grid()
        for c in range(6):
            counts[:, :, c] += (cg == c)

    probs = counts.astype(np.float64) / n_runs
    probs = np.maximum(probs, 0.01)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


if __name__ == "__main__":
    import time

    # Quick benchmark
    grid = [[OCEAN]*20 for _ in range(20)]
    for y in range(2, 18):
        for x in range(2, 18):
            grid[y][x] = PLAINS
    for y in range(5, 10):
        for x in range(5, 10):
            grid[y][x] = FOREST

    settlements = [
        {"x": 10, "y": 10, "has_port": False, "alive": True},
        {"x": 5, "y": 15, "has_port": False, "alive": True},
        {"x": 15, "y": 5, "has_port": False, "alive": True},
    ]

    params = Params()
    print("Benchmarking 500 runs on 20x20...")
    t0 = time.time()
    probs = run_monte_carlo(grid, settlements, params, n_runs=500, seed=42)
    print(f"Done in {time.time()-t0:.1f}s ({500/(time.time()-t0):.0f} runs/sec)")
