"""
run_demo.py — Standalone Mid-Review Visual Simulation Demo (2 Robots + Decentralized Bidding)

Runs a 2-robot Pygame visual simulation showing:
  - Heterogeneous Robots (Robot 0: FAST_LIGHT, Robot 1: HEAVY_LOAD)
  - Neural Network Task Bidding
  - A* Path Navigation & Shelf Pickups, Delivery & Returns
  - Collision Priority Yielding & Dynamic Task Release
  - Live HUD with step count, battery levels, and event log

Controls:
  SPACE      : Pause / Resume simulation
  RIGHT ARROW: Advance single step (when paused)
  Q or ESC   : Exit simulation
"""

import os
import sys
import torch
import pygame
import numpy as np

# Ensure mid_review_demo directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_env import SimpleWarehouseEnv
from simple_bidding import SimpleBiddingNetwork, evaluate_auction
from simple_planner import build_walkable_grid_simple, astar, path_to_action_queue
from rware.warehouse import Action, Direction

# ── Colors ─────────────────────────────────────────────────────────────────────
BG_COLOR       = (240, 243, 246)
GRID_COLOR     = (210, 215, 220)
ROBOT_0_COLOR  = (41, 128, 185)    # Blue: FAST_LIGHT
ROBOT_1_COLOR  = (211, 84, 0)      # Orange: HEAVY_LOAD
SHELF_COLOR    = (241, 196, 15)    # Yellow: Requested Shelf
DELIVERY_COLOR = (26, 188, 156)    # Teal: Goal Delivery Zone
TEXT_DARK      = (44, 62, 80)
PANEL_BG       = (255, 255, 255)

class RobotPhase:
    IDLE           = 0
    MOVING_TO_SHELF = 1
    PICKING_UP     = 2
    MOVING_TO_GOAL = 3
    RETURNING_SHELF = 4
    DROPPING_OFF   = 5


# ── Path helpers ───────────────────────────────────────────────────────────────

def plan_safe(start, goal, base_env, other_pos=None):
    """
    A* path from start→goal, optionally treating other_pos as a temporary obstacle.
    Falls back to ignoring the obstacle if no path found.
    """
    walkable = build_walkable_grid_simple(base_env)
    # Temporarily block other robot's cell (unless it IS the goal or start)
    if other_pos and other_pos != goal and other_pos != start:
        ox, oy = other_pos
        walkable[oy, ox] = False
    path = astar(walkable, start, goal)
    if path:
        return path
    # Retry without blocking other robot (fallback)
    walkable2 = build_walkable_grid_simple(base_env)
    path2 = astar(walkable2, start, goal)
    return path2 if path2 else [start]


def make_actions(path, current_dir):
    """Wrapper: convert waypoint path → action list."""
    if not path or len(path) <= 1:
        return []
    return path_to_action_queue(path, current_dir)


def other_robot_pos(env, i):
    """Return (x,y) of the OTHER robot (not robot i)."""
    j = 1 - i
    a = env.base_env.agents[j]
    return (a.x, a.y)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.font.init()

    font_large = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)

    env = SimpleWarehouseEnv(n_agents=2)
    obs, info = env.reset(seed=42)

    bidding_net = SimpleBiddingNetwork(obs_dim=80)
    bidding_net.eval()

    CELL_SIZE   = 48
    grid_width, grid_height = env.base_env.grid_size
    sidebar_width = 320

    screen_width  = grid_width * CELL_SIZE + sidebar_width
    screen_height = max(grid_height * CELL_SIZE + 60, 450)

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mid-Review Demo — Multi-Robot Warehouse Coordination")
    clock = pygame.time.Clock()

    paused = False
    step_count = 0
    event_logs = ["Simulation Started.", "R0: FAST_LIGHT  R1: HEAVY_LOAD"]

    # Per-robot state
    phases              = [RobotPhase.IDLE, RobotPhase.IDLE]
    action_queues       = [[], []]
    assigned_shelves    = [None, None]
    shelf_orig_positions = [None, None]
    deliveries_count    = 0
    pickup_attempts     = [0, 0]   # timeout for TOGGLE_LOAD attempts

    # Stuck-detection
    last_positions  = [(env.base_env.agents[i].x, env.base_env.agents[i].y) for i in range(2)]
    stuck_counters  = [0, 0]
    STUCK_THRESHOLD = 8   # steps without moving before forced replan

    running = True
    while running:
        # ── Event handling ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    event_logs.append(f"[{step_count}] {'PAUSED' if paused else 'RESUMED'}")

        if not paused:
            # ── 0. Update stuck counters ────────────────────────────────────────
            for i in range(2):
                cur = (env.base_env.agents[i].x, env.base_env.agents[i].y)
                if cur == last_positions[i] and phases[i] != RobotPhase.IDLE:
                    stuck_counters[i] += 1
                else:
                    stuck_counters[i] = 0
                last_positions[i] = cur

                if stuck_counters[i] >= STUCK_THRESHOLD:
                    # Force a replan — clear the queue so state machine replans next tick
                    action_queues[i] = []
                    stuck_counters[i] = 0
                    event_logs.append(f"[{step_count}] R{i} stuck → replan")

            # ── 1. Auction for IDLE robots ──────────────────────────────────────
            requests = env.unassigned_requests
            assigned_shelf_objs = set(s for s in assigned_shelves if s is not None)
            unassigned = [r for r in requests if r not in assigned_shelf_objs]

            for req in unassigned:
                idle_robots = [i for i in range(2)
                               if phases[i] == RobotPhase.IDLE and assigned_shelves[i] is None]
                if not idle_robots:
                    break

                with torch.no_grad():
                    bids = []
                    for i in range(env.n_agents):
                        ob = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                        bid = bidding_net(ob).item() + float(np.random.uniform(0, 0.15))
                        bids.append(bid)

                winner = evaluate_auction(bids, env.robot_type_names, env.battery)
                if winner not in idle_robots:
                    winner = idle_robots[0]

                assigned_shelves[winner]     = req
                shelf_orig_positions[winner] = (int(req.x), int(req.y))
                phases[winner]               = RobotPhase.MOVING_TO_SHELF
                pickup_attempts[winner]      = 0

                event_logs.append(
                    f"[{step_count}] Auction→R{winner} "
                    f"(R0={bids[0]:.2f}, R1={bids[1]:.2f})"
                )

                agent = env.base_env.agents[winner]
                path  = plan_safe((agent.x, agent.y),
                                  (int(req.x), int(req.y)),
                                  env.base_env,
                                  other_pos=other_robot_pos(env, winner))
                action_queues[winner] = make_actions(path, agent.dir)

            # ── 2. State machine ────────────────────────────────────────────────
            goal_positions = set((int(g[0]), int(g[1])) for g in env.base_env.goals)
            actions = [Action.NOOP.value, Action.NOOP.value]

            for i in range(env.n_agents):
                agent  = env.base_env.agents[i]
                other  = other_robot_pos(env, i)

                # ── MOVING_TO_SHELF ──────────────────────────────────────────
                if phases[i] == RobotPhase.MOVING_TO_SHELF:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif assigned_shelves[i] and \
                         (agent.x, agent.y) == (int(assigned_shelves[i].x), int(assigned_shelves[i].y)):
                        phases[i] = RobotPhase.PICKING_UP
                        pickup_attempts[i] = 0
                        event_logs.append(f"[{step_count}] R{i} at shelf → TOGGLE")
                        actions[i] = Action.TOGGLE_LOAD.value
                    elif assigned_shelves[i]:
                        # Replan avoiding other robot
                        path = plan_safe((agent.x, agent.y),
                                         (int(assigned_shelves[i].x), int(assigned_shelves[i].y)),
                                         env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value

                # ── PICKING_UP ───────────────────────────────────────────────
                elif phases[i] == RobotPhase.PICKING_UP:
                    if agent.carrying_shelf is not None:
                        # Cancel any competing assignment to the same shelf
                        for j in range(2):
                            if j != i and assigned_shelves[j] == assigned_shelves[i]:
                                assigned_shelves[j] = None
                                phases[j]           = RobotPhase.IDLE
                                action_queues[j]    = []

                        nearest_goal = min(env.base_env.goals,
                                           key=lambda g: abs(agent.x - int(g[0])) + abs(agent.y - int(g[1])))
                        gx, gy = int(nearest_goal[0]), int(nearest_goal[1])
                        path = plan_safe((agent.x, agent.y), (gx, gy), env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        phases[i]        = RobotPhase.MOVING_TO_GOAL
                        event_logs.append(f"[{step_count}] R{i} picked up → Goal")
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                    else:
                        pickup_attempts[i] += 1
                        if pickup_attempts[i] > 6:
                            # Give up — shelf may have moved; go back IDLE
                            phases[i]            = RobotPhase.IDLE
                            assigned_shelves[i]  = None
                            shelf_orig_positions[i] = None
                            action_queues[i]     = []
                            pickup_attempts[i]   = 0
                            event_logs.append(f"[{step_count}] R{i} pickup failed → IDLE")
                        else:
                            actions[i] = Action.TOGGLE_LOAD.value

                # ── MOVING_TO_GOAL ───────────────────────────────────────────
                elif phases[i] == RobotPhase.MOVING_TO_GOAL:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif (agent.x, agent.y) in goal_positions:
                        # Successfully at goal — plan return
                        orig = shelf_orig_positions[i]
                        path = plan_safe((agent.x, agent.y), orig, env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        phases[i]        = RobotPhase.RETURNING_SHELF
                        deliveries_count += 1
                        event_logs.append(f"[{step_count}] R{i} DELIVERED! → returning")
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                    else:
                        # Not at goal yet — replan
                        nearest_goal = min(env.base_env.goals,
                                           key=lambda g: abs(agent.x - int(g[0])) + abs(agent.y - int(g[1])))
                        gx, gy = int(nearest_goal[0]), int(nearest_goal[1])
                        path = plan_safe((agent.x, agent.y), (gx, gy), env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value

                # ── RETURNING_SHELF ──────────────────────────────────────────
                elif phases[i] == RobotPhase.RETURNING_SHELF:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif (agent.x, agent.y) == shelf_orig_positions[i]:
                        phases[i]  = RobotPhase.DROPPING_OFF
                        actions[i] = Action.TOGGLE_LOAD.value
                    elif shelf_orig_positions[i]:
                        path = plan_safe((agent.x, agent.y), shelf_orig_positions[i],
                                         env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value

                # ── DROPPING_OFF ─────────────────────────────────────────────
                elif phases[i] == RobotPhase.DROPPING_OFF:
                    if agent.carrying_shelf is None:
                        phases[i]              = RobotPhase.IDLE
                        assigned_shelves[i]    = None
                        shelf_orig_positions[i] = None
                        event_logs.append(f"[{step_count}] R{i} shelf returned → IDLE")
                        actions[i] = Action.NOOP.value
                    else:
                        actions[i] = Action.TOGGLE_LOAD.value

            # ── 3. Smarter collision resolution ────────────────────────────────
            a0, a1 = env.base_env.agents[0], env.base_env.agents[1]
            dist = abs(a0.x - a1.x) + abs(a0.y - a1.y)
            if dist <= 1:
                loaded0, loaded1 = a0.carrying_shelf is not None, a1.carrying_shelf is not None
                if loaded0 and not loaded1:
                    # R1 yields: try to step sideways/forward instead of pure NOOP
                    actions[1] = _yield_action(a1, a0)
                elif loaded1 and not loaded0:
                    actions[0] = _yield_action(a0, a1)
                # If both free or both loaded, let RWARE handle it naturally

            # ── 4. Step ────────────────────────────────────────────────────────
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1

            if len(event_logs) > 8:
                event_logs.pop(0)

            # ── 5. Check if both batteries depleted → end simulation ────────────
            if env.battery[0] <= 0 and env.battery[1] <= 0:
                # Render a final "Simulation Over" overlay for 3 seconds
                overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, 0))
                cx, cy = screen_width // 2, screen_height // 2
                font_xl = pygame.font.SysFont("Arial", 32, bold=True)
                font_md = pygame.font.SysFont("Arial", 20)
                screen.blit(font_xl.render("⚡ Batteries Depleted", True, (241, 196, 15)),
                            font_xl.size("⚡ Batteries Depleted")[0] // 2 * 0
                            or (cx - font_xl.size("⚡ Batteries Depleted")[0] // 2, cy - 60))
                screen.blit(font_md.render("Simulation Over", True, (255, 255, 255)),
                            (cx - font_md.size("Simulation Over")[0] // 2, cy - 10))
                screen.blit(font_md.render(f"Total Steps: {step_count}", True, (200, 200, 200)),
                            (cx - font_md.size(f"Total Steps: {step_count}")[0] // 2, cy + 25))
                screen.blit(font_md.render(f"Deliveries:  {deliveries_count}", True, (200, 200, 200)),
                            (cx - font_md.size(f"Deliveries:  {deliveries_count}")[0] // 2, cy + 55))
                pygame.display.flip()
                pygame.time.wait(3000)   # show for 3 seconds
                running = False
                continue

        # ── Rendering ──────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        for x in range(grid_width):
            for y in range(grid_height):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        # Goals
        for gx, gy in env.base_env.goals:
            rect = pygame.Rect(gx * CELL_SIZE + 2, gy * CELL_SIZE + 52, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(screen, DELIVERY_COLOR, rect, 0, border_radius=4)
            lbl = font_small.render("GOAL", True, (255, 255, 255))
            screen.blit(lbl, (gx * CELL_SIZE + 6, gy * CELL_SIZE + 66))

        # Shelves
        for shelf in env.base_env.shelfs:
            sx, sy = shelf.x, shelf.y
            rect = pygame.Rect(sx * CELL_SIZE + 6, sy * CELL_SIZE + 56, CELL_SIZE - 12, CELL_SIZE - 12)
            color = SHELF_COLOR if shelf in env.base_env.request_queue else (180, 190, 200)
            pygame.draw.rect(screen, color, rect, 0, border_radius=6)

        # Robots
        colors     = [ROBOT_0_COLOR, ROBOT_1_COLOR]
        phase_names = {0: "IDLE", 1: "TO_SHELF", 2: "PICKING",
                       3: "TO_GOAL", 4: "RETURNING", 5: "DROPPING"}
        for i, agent in enumerate(env.base_env.agents):
            rx = agent.x * CELL_SIZE + CELL_SIZE // 2
            ry = agent.y * CELL_SIZE + 50 + CELL_SIZE // 2
            pygame.draw.circle(screen, colors[i], (rx, ry), CELL_SIZE // 3)
            txt = font_small.render(f"R{i}", True, (255, 255, 255))
            screen.blit(txt, (rx - 8, ry - 8))
            batt_pct = env.battery[i] / 100.0
            bar_w    = int((CELL_SIZE - 8) * batt_pct)
            batt_col = (46, 204, 113) if batt_pct > 0.4 else (231, 76, 60)
            pygame.draw.rect(screen, (200, 200, 200),
                             (rx - CELL_SIZE // 2 + 4, ry + 16, CELL_SIZE - 8, 4))
            pygame.draw.rect(screen, batt_col,
                             (rx - CELL_SIZE // 2 + 4, ry + 16, bar_w, 4))

        # HUD
        pygame.draw.rect(screen, PANEL_BG, (0, 0, screen_width, 45))
        pygame.draw.line(screen, GRID_COLOR, (0, 45), (screen_width, 45), 2)
        screen.blit(font_large.render("Mid-Review Multi-Robot Warehouse Simulation",
                                       True, TEXT_DARK), (12, 10))
        status_str = "PAUSED" if paused else "RUNNING"
        screen.blit(font_small.render(
            f"Step: {step_count}  |  {status_str}  |  Deliveries: {deliveries_count}",
            True, TEXT_DARK), (screen_width - 340, 14))

        # Sidebar
        sb_x = grid_width * CELL_SIZE + 10
        sb_panel = pygame.Rect(sb_x, 50, sidebar_width - 20, screen_height - 60)
        pygame.draw.rect(screen, PANEL_BG, sb_panel, 0, border_radius=8)
        pygame.draw.rect(screen, GRID_COLOR, sb_panel, 2, border_radius=8)
        screen.blit(font_large.render("Robot Status & Logs", True, TEXT_DARK), (sb_x + 12, 62))

        for i in range(2):
            cy = 95 + i * 80
            screen.blit(font_small.render(
                f"Robot {i} ({env.robot_type_names[i]}) [{phase_names.get(phases[i], '?')}]",
                True, colors[i]), (sb_x + 12, cy))
            screen.blit(font_small.render(
                f"Battery: {env.battery[i]:.1f}%  Stuck: {stuck_counters[i]}",
                True, TEXT_DARK), (sb_x + 12, cy + 18))

        screen.blit(font_large.render("Live Event Log:", True, TEXT_DARK), (sb_x + 12, 265))
        for idx, entry in enumerate(event_logs):
            screen.blit(font_small.render(entry, True, (80, 90, 100)),
                        (sb_x + 12, 290 + idx * 20))

        pygame.display.flip()
        clock.tick(5 if not paused else 15)

    pygame.quit()
    sys.exit(0)


def _yield_action(yielder, priority_agent):
    """
    Generate a move action for 'yielder' that steps AWAY from priority_agent
    rather than simply doing NOOP (to avoid perpetual freeze).
    Uses a simple perpendicular-step heuristic.
    """
    dx = yielder.x - priority_agent.x
    dy = yielder.y - priority_agent.y
    # Step perpendicular to the blocker direction: swap axes, pick first valid
    candidates = [
        Action.RIGHT.value,   # turn clockwise
        Action.LEFT.value,    # turn counter-clockwise
        Action.NOOP.value,    # fallback
    ]
    # If difference is horizontal → try to face perpendicular (vertical move)
    # Simple heuristic: just issue a turn so next step they won't be head-to-head
    if abs(dx) >= abs(dy):
        return Action.LEFT.value   # turn away
    else:
        return Action.RIGHT.value


if __name__ == "__main__":
    main()
