"""
Grocery Bot for NM i AI 2026
Greedy nearest-item strategy with proper pathfinding.
Every round: pick the single best action to minimize total wasted rounds.
"""

import asyncio
import json
import sys
from collections import deque

WS_URL = "wss://game.ainm.no/ws?token=YOUR_TOKEN"
DEBUG = "--debug" in sys.argv


def log(*args):
    if DEBUG:
        print(*args)


# ── Pathfinding ──────────────────────────────────────────────

def bfs_path(start, goals, blocked, grid_w, grid_h):
    """BFS shortest path. Returns (actions, dest) or (None, None)."""
    sx, sy = start
    goal_set = set(goals)
    if (sx, sy) in goal_set:
        return [], (sx, sy)
    queue = deque([(sx, sy, [])])
    visited = {(sx, sy)}
    while queue:
        x, y, path = queue.popleft()
        for dx, dy, action in [(0, -1, "move_up"), (0, 1, "move_down"),
                                (-1, 0, "move_left"), (1, 0, "move_right")]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in visited:
                new_path = path + [action]
                if (nx, ny) in goal_set:
                    return new_path, (nx, ny)
                if (nx, ny) not in blocked:
                    visited.add((nx, ny))
                    queue.append((nx, ny, new_path))
    return None, None


def get_adjacent_walkable(pos, blocked, grid_w, grid_h):
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
            if 0 <= x + dx < grid_w and 0 <= y + dy < grid_h and (x + dx, y + dy) not in blocked]


def dist_to_item(pos, item_pos, blocked, grid_w, grid_h):
    """Steps to reach a cell adjacent to item. Returns (dist, dest) or (inf, None)."""
    adj = get_adjacent_walkable(item_pos, blocked, grid_w, grid_h)
    if not adj:
        return float('inf'), None
    path, dest = bfs_path(pos, adj, blocked, grid_w, grid_h)
    if path is None:
        return float('inf'), None
    return len(path), dest


def dist_to_dropoff(pos, drop_off_zones, blocked, grid_w, grid_h):
    path, _ = bfs_path(pos, drop_off_zones, blocked, grid_w, grid_h)
    return len(path) if path else float('inf')


# ── Decision Logic ───────────────────────────────────────────

def decide_actions(state):
    grid_w = state["grid"]["width"]
    grid_h = state["grid"]["height"]
    walls_set = set(map(tuple, state["grid"]["walls"]))
    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off_zones = [tuple(z) for z in state.get("drop_off_zones", [state["drop_off"]])]
    drop_off_set = set(drop_off_zones)

    item_positions = set(tuple(i["position"]) for i in items)
    blocked = walls_set | item_positions

    active_order = next((o for o in orders if o["status"] == "active"), None)
    preview_order = next((o for o in orders if o["status"] == "preview"), None)

    needed_active = []
    if active_order:
        needed_active = list(active_order["items_required"])
        for d in active_order["items_delivered"]:
            if d in needed_active:
                needed_active.remove(d)

    needed_preview = []
    if preview_order:
        needed_preview = list(preview_order["items_required"])
        for d in preview_order["items_delivered"]:
            if d in needed_preview:
                needed_preview.remove(d)

    # What all bots carry that counts toward active
    still_need_from_shelf = list(needed_active)
    for bot in bots:
        for t in bot["inventory"]:
            if t in still_need_from_shelf:
                still_need_from_shelf.remove(t)

    log(f"  Active: {needed_active}, shelf: {still_need_from_shelf}, preview: {needed_preview}")

    assigned_item_ids = set()
    actions = []

    for bot in bots:
        bx, by = bot["position"]
        bid = bot["id"]
        inv = bot["inventory"]
        bot_pos = (bx, by)
        has_active = any(t in needed_active for t in inv)

        log(f"  Bot {bid} at ({bx},{by}) inv={inv}")

        # ── 1. On drop-off with deliverable items -> drop off ──
        if bot_pos in drop_off_set and has_active:
            log(f"    -> DROP OFF")
            actions.append({"bot": bid, "action": "drop_off"})
            continue

        # ── 2. Adjacent to needed active item -> pick up (always worth 1 action) ──
        picked = False
        if len(inv) < 3 and still_need_from_shelf:
            for item in items:
                if item["id"] in assigned_item_ids:
                    continue
                if item["type"] not in still_need_from_shelf:
                    continue
                ix, iy = item["position"]
                if abs(ix - bx) + abs(iy - by) == 1:
                    log(f"    -> PICK UP {item['type']} (adjacent)")
                    actions.append({"bot": bid, "action": "pick_up", "item_id": item["id"]})
                    assigned_item_ids.add(item["id"])
                    still_need_from_shelf.remove(item["type"])
                    picked = True
                    break
        if picked:
            continue

        # ── 3. Decide: pick more or deliver? ──
        # Calculate cost of continuing to pick vs delivering now
        dist_drop = dist_to_dropoff(bot_pos, drop_off_zones, blocked, grid_w, grid_h)

        # Should we deliver?
        should_deliver = False

        if len(inv) >= 3:
            should_deliver = True
        elif has_active and not still_need_from_shelf:
            # We carry all remaining active items — deliver now
            should_deliver = True
        elif has_active and len(inv) >= 2:
            # We have 2 items, 1 slot left. Is the closest needed item worth the detour?
            best_item_dist = float('inf')
            for item in items:
                if item["id"] in assigned_item_ids:
                    continue
                if item["type"] not in still_need_from_shelf:
                    continue
                d, _ = dist_to_item(bot_pos, tuple(item["position"]), blocked, grid_w, grid_h)
                if d < best_item_dist:
                    best_item_dist = d

            if best_item_dist == float('inf'):
                should_deliver = True
            else:
                # Pick up if the item is roughly on the way (within 3 extra actions vs going straight to drop)
                # Cost to pick + deliver from item vs deliver now + come back later
                # Picking now saves a future trip to drop-off (~dist_drop actions saved)
                # But costs best_item_dist extra actions now
                # Worth it if best_item_dist <= dist_drop (picking now is "free" compared to a return trip)
                if best_item_dist > dist_drop + 3:
                    should_deliver = True

        if should_deliver:
            if bot_pos in drop_off_set:
                log(f"    -> DROP OFF (on zone, delivering)")
                actions.append({"bot": bid, "action": "drop_off"})
                continue
            path, _ = bfs_path(bot_pos, drop_off_zones, blocked, grid_w, grid_h)
            if path is not None and len(path) > 0:
                log(f"    -> DELIVER (dist {len(path)})")
                actions.append({"bot": bid, "action": path[0]})
                continue

        # ── 4. Pick up closest needed item ──
        # Priority: active items first, then preview to fill slots
        best_item = None
        best_dist = float('inf')
        best_type = None

        # First pass: active items
        if still_need_from_shelf and len(inv) < 3:
            for item in items:
                if item["id"] in assigned_item_ids:
                    continue
                if item["type"] not in still_need_from_shelf:
                    continue
                d, _ = dist_to_item(bot_pos, tuple(item["position"]), blocked, grid_w, grid_h)
                if d < best_dist:
                    best_dist = d
                    best_item = item
                    best_type = "active"

        # Second pass: preview items (only if we have room AND active items are far or done)
        if len(inv) < 3 and not still_need_from_shelf and needed_preview:
            for item in items:
                if item["id"] in assigned_item_ids:
                    continue
                if item["type"] not in needed_preview:
                    continue
                d, _ = dist_to_item(bot_pos, tuple(item["position"]), blocked, grid_w, grid_h)
                if d < best_dist:
                    best_dist = d
                    best_item = item
                    best_type = "preview"

        if best_item:
            assigned_item_ids.add(best_item["id"])
            adj = get_adjacent_walkable(tuple(best_item["position"]), blocked, grid_w, grid_h)
            if adj:
                path, _ = bfs_path(bot_pos, adj, blocked, grid_w, grid_h)
                if path is not None:
                    if len(path) == 0:
                        log(f"    -> PICK UP {best_item['type']} (adjacent, {best_type})")
                        actions.append({"bot": bid, "action": "pick_up", "item_id": best_item["id"]})
                    else:
                        log(f"    -> MOVE toward {best_item['type']} (dist {len(path)}, {best_type})")
                        actions.append({"bot": bid, "action": path[0]})
                    continue

        # ── 5. Fallback: deliver whatever we have ──
        if inv:
            if bot_pos in drop_off_set:
                log(f"    -> DROP OFF fallback")
                actions.append({"bot": bid, "action": "drop_off"})
                continue
            path, _ = bfs_path(bot_pos, drop_off_zones, blocked, grid_w, grid_h)
            if path is not None and len(path) > 0:
                log(f"    -> DELIVER fallback (dist {len(path)})")
                actions.append({"bot": bid, "action": path[0]})
                continue

        log(f"    -> WAIT")
        actions.append({"bot": bid, "action": "wait"})

    return actions


# ── WebSocket Client ─────────────────────────────────────────

async def play():
    import websockets

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    token = args[0] if args else WS_URL.split("token=")[1]
    if token.startswith("wss://"):
        token = token.split("token=")[1]
    url = f"wss://game.ainm.no/ws?token={token}"

    print("Connecting to game server...")
    async with websockets.connect(url) as ws:
        print("Connected!")

        first_state = True
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg["type"] == "game_over":
                print(f"\nGame over!")
                print(f"  Score: {msg.get('score', '?')}")
                print(f"  Items delivered: {msg.get('items_delivered', '?')}")
                print(f"  Orders completed: {msg.get('orders_completed', '?')}")
                break

            state = msg

            if first_state:
                gw = state["grid"]["width"]
                gh = state["grid"]["height"]
                print(f"Map: {gw}x{gh}, {len(state['grid']['walls'])} walls, "
                      f"{len(state['bots'])} bots, {len(state['items'])} items")

                wall_set = set(map(tuple, state["grid"]["walls"]))
                ipos = {tuple(i["position"]): i["type"][0].upper() for i in state["items"]}
                bpos = {tuple(b["position"]): str(b["id"]) for b in state["bots"]}
                dzones = set(map(tuple, state.get("drop_off_zones", [state["drop_off"]])))

                print("\nMap:")
                for y in range(gh):
                    row = ""
                    for x in range(gw):
                        if (x, y) in bpos: row += bpos[(x, y)]
                        elif (x, y) in dzones: row += "D"
                        elif (x, y) in ipos: row += ipos[(x, y)]
                        elif (x, y) in wall_set: row += "#"
                        else: row += "."
                    print(f"  {row}")
                print()
                first_state = False

            log(f"\n── Round {state['round']} ──")
            actions = decide_actions(state)

            if state["round"] % 10 == 0:
                active = next((o for o in state["orders"] if o["status"] == "active"), None)
                inv_summary = ", ".join(f"B{b['id']}:{b['inventory']}" for b in state["bots"])
                if active:
                    print(f"R{state['round']}/{state['max_rounds']} | "
                          f"Score: {state['score']} | "
                          f"{active['id']} {len(active['items_delivered'])}/{len(active['items_required'])} | "
                          f"[{inv_summary}]")
                else:
                    print(f"R{state['round']}/{state['max_rounds']} | Score: {state['score']}")

            await ws.send(json.dumps({"actions": actions}))


if __name__ == "__main__":
    asyncio.run(play())
