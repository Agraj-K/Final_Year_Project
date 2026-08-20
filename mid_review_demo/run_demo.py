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
import time
import torch
import pygame
import numpy as np

# Ensure mid_review_demo directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_env import SimpleWarehouseEnv
from simple_bidding import SimpleBiddingNetwork, evaluate_auction
from simple_planner import plan_path_a_star, waypoints_to_actions
from rware.warehouse import Action, Direction

# Colors for Pygame Visualizer
BG_COLOR = (240, 243, 246)
GRID_COLOR = (210, 215, 220)
ROBOT_0_COLOR = (41, 128, 185)   # Blue: FAST_LIGHT
ROBOT_1_COLOR = (211, 84, 0)     # Orange: HEAVY_LOAD
SHELF_COLOR = (241, 196, 15)     # Yellow: Requested Shelf
CARRIED_COLOR = (46, 204, 113)   # Green: Carried Shelf
DELIVERY_COLOR = (26, 188, 156)  # Teal: Goal Delivery Zone
TEXT_DARK = (44, 62, 80)
PANEL_BG = (255, 255, 255)

class RobotPhase:
    IDLE = 0
    MOVING_TO_SHELF = 1
    PICKING_UP = 2
    MOVING_TO_GOAL = 3
    RETURNING_SHELF = 4
    DROPPING_OFF = 5

def main():
    pygame.init()
    pygame.font.init()
    
    font_large = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)

    # Initialize Environment & Bidding Model for 2 Robots
    env = SimpleWarehouseEnv(n_agents=2)
    obs, info = env.reset(seed=42)
    
    bidding_net = SimpleBiddingNetwork(obs_dim=80)
    bidding_net.eval()
    
    CELL_SIZE = 48
    grid_width, grid_height = env.base_env.grid_size
    sidebar_width = 320
    
    screen_width = grid_width * CELL_SIZE + sidebar_width
    screen_height = max(grid_height * CELL_SIZE + 60, 450)
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mid-Review Demo — Multi-Robot Warehouse Coordination")
    clock = pygame.time.Clock()
    
    paused = False
    step_count = 0
    event_logs = ["Simulation Started.", "Robots initialized (R0: Fast, R1: Heavy)."]
    
    # State tracking per robot
    phases = [RobotPhase.IDLE, RobotPhase.IDLE]
    action_queues = [[], []]
    assigned_shelves = [None, None]
    shelf_orig_positions = [None, None]
    deliveries_count = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    event_logs.append(f"[Step {step_count}] Simulation {status}")

        if not paused:
            # 1. Bidding / Auction for Idle Robots
            requests = env.unassigned_requests
            if requests:
                for req in requests:
                    # Check which robots are IDLE and available for bidding
                    idle_robots = [i for i in range(2) if phases[i] == RobotPhase.IDLE and assigned_shelves[i] is None]
                    if idle_robots:
                        with torch.no_grad():
                            bids = []
                            for i in range(env.n_agents):
                                ob_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                                bid_val = bidding_net(ob_tensor).item()
                                bids.append(bid_val)
                                
                        # Pick winner among available/idle robots
                        winner = evaluate_auction(bids, env.robot_type_names, env.battery)
                        if phases[winner] != RobotPhase.IDLE or assigned_shelves[winner] is not None:
                            winner = idle_robots[0]
                            
                        assigned_shelves[winner] = req
                        shelf_orig_positions[winner] = (req.x, req.y)
                        phases[winner] = RobotPhase.MOVING_TO_SHELF
                        
                        r0_bid = f"{bids[0]:.2f}"
                        r1_bid = f"{bids[1]:.2f}"
                        event_logs.append(f"[Auction] Shelf assigned to R{winner} (Bids: R0={r0_bid}, R1={r1_bid})")
                        
                        # Plan A* path to shelf
                        agent = env.base_env.agents[winner]
                        path = plan_path_a_star((agent.x, agent.y), (req.x, req.y), (grid_width, grid_height), base_env=env.base_env)
                        action_queues[winner] = waypoints_to_actions(path, agent.dir)

            # 2. State Machine for Robot Action Generation
            actions = [Action.NOOP.value, Action.NOOP.value]
            for i in range(env.n_agents):
                agent = env.base_env.agents[i]
                
                if phases[i] == RobotPhase.MOVING_TO_SHELF:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif assigned_shelves[i] and (agent.x, agent.y) == (assigned_shelves[i].x, assigned_shelves[i].y):
                        phases[i] = RobotPhase.PICKING_UP
                        event_logs.append(f"[Step {step_count}] Robot {i} arrived at shelf, picking up")
                        actions[i] = Action.TOGGLE_LOAD.value
                    else:
                        path = plan_path_a_star((agent.x, agent.y), (assigned_shelves[i].x, assigned_shelves[i].y), (grid_width, grid_height), base_env=env.base_env)
                        action_queues[i] = waypoints_to_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                        
                elif phases[i] == RobotPhase.PICKING_UP:
                    if agent.carrying_shelf is not None:
                        # Cancel task for any other robot targeting this same shelf
                        for other in range(2):
                            if other != i and assigned_shelves[other] == assigned_shelves[i]:
                                assigned_shelves[other] = None
                                phases[other] = RobotPhase.IDLE
                                action_queues[other] = []
                                
                        goal_x, goal_y = env.base_env.goals[0]
                        path = plan_path_a_star((agent.x, agent.y), (goal_x, goal_y), (grid_width, grid_height), base_env=env.base_env)
                        action_queues[i] = waypoints_to_actions(path, agent.dir)
                        phases[i] = RobotPhase.MOVING_TO_GOAL
                        event_logs.append(f"[Step {step_count}] Robot {i} picked up shelf, heading to Goal")
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                    else:
                        actions[i] = Action.TOGGLE_LOAD.value
                        
                elif phases[i] == RobotPhase.MOVING_TO_GOAL:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    else:
                        orig_pos = shelf_orig_positions[i]
                        path = plan_path_a_star((agent.x, agent.y), orig_pos, (grid_width, grid_height), base_env=env.base_env)
                        action_queues[i] = waypoints_to_actions(path, agent.dir)
                        phases[i] = RobotPhase.RETURNING_SHELF
                        deliveries_count += 1
                        event_logs.append(f"[Step {step_count}] Robot {i} delivered at Goal! Returning shelf")
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                        
                elif phases[i] == RobotPhase.RETURNING_SHELF:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif (agent.x, agent.y) == shelf_orig_positions[i]:
                        phases[i] = RobotPhase.DROPPING_OFF
                        actions[i] = Action.TOGGLE_LOAD.value
                        
                elif phases[i] == RobotPhase.DROPPING_OFF:
                    if agent.carrying_shelf is None:
                        phases[i] = RobotPhase.IDLE
                        assigned_shelves[i] = None
                        shelf_orig_positions[i] = None
                        event_logs.append(f"[Step {step_count}] Robot {i} returned shelf. Going IDLE.")
                        actions[i] = Action.NOOP.value
                    else:
                        actions[i] = Action.TOGGLE_LOAD.value

            # Collision Priority Yielding: Non-carrying robot yields if adjacent to carrying robot
            if env.base_env.agents[0].carrying_shelf is not None and env.base_env.agents[1].carrying_shelf is None:
                if abs(env.base_env.agents[0].x - env.base_env.agents[1].x) + abs(env.base_env.agents[0].y - env.base_env.agents[1].y) <= 1:
                    actions[1] = Action.NOOP.value
            elif env.base_env.agents[1].carrying_shelf is not None and env.base_env.agents[0].carrying_shelf is None:
                if abs(env.base_env.agents[0].x - env.base_env.agents[1].x) + abs(env.base_env.agents[0].y - env.base_env.agents[1].y) <= 1:
                    actions[0] = Action.NOOP.value

            # 3. Step Environment
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1
            
            if len(event_logs) > 7:
                event_logs.pop(0)

        # ── Rendering ──────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)
        
        # Draw Grid Cells & Shelves
        for x in range(grid_width):
            for y in range(grid_height):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        # Draw Goal/Delivery Slot
        for goal_x, goal_y in env.base_env.goals:
            rect = pygame.Rect(goal_x * CELL_SIZE + 2, goal_y * CELL_SIZE + 52, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(screen, DELIVERY_COLOR, rect, 0, border_radius=4)
            lbl = font_small.render("GOAL", True, (255, 255, 255))
            screen.blit(lbl, (goal_x * CELL_SIZE + 6, goal_y * CELL_SIZE + 66))

        # Draw Shelves
        for shelf in env.base_env.shelfs:
            sx, sy = shelf.x, shelf.y
            rect = pygame.Rect(sx * CELL_SIZE + 6, sy * CELL_SIZE + 56, CELL_SIZE - 12, CELL_SIZE - 12)
            color = SHELF_COLOR if shelf in env.base_env.request_queue else (180, 190, 200)
            pygame.draw.rect(screen, color, rect, 0, border_radius=6)

        # Draw Robots & Battery Bars
        colors = [ROBOT_0_COLOR, ROBOT_1_COLOR]
        phase_names = {0:"IDLE", 1:"TO_SHELF", 2:"PICKING", 3:"TO_GOAL", 4:"RETURNING", 5:"DROPPING"}
        for i, agent in enumerate(env.base_env.agents):
            rx = agent.x * CELL_SIZE + CELL_SIZE // 2
            ry = agent.y * CELL_SIZE + 50 + CELL_SIZE // 2
            
            # Robot body circle
            pygame.draw.circle(screen, colors[i], (rx, ry), CELL_SIZE // 3)
            
            # Label
            txt = font_small.render(f"R{i}", True, (255, 255, 255))
            screen.blit(txt, (rx - 8, ry - 8))
            
            # Battery bar under robot
            batt_pct = env.battery[i] / 100.0
            bar_w = int((CELL_SIZE - 8) * batt_pct)
            batt_color = (46, 204, 113) if batt_pct > 0.4 else (231, 76, 60)
            pygame.draw.rect(screen, (200, 200, 200), (rx - CELL_SIZE//2 + 4, ry + 16, CELL_SIZE - 8, 4))
            pygame.draw.rect(screen, batt_color, (rx - CELL_SIZE//2 + 4, ry + 16, bar_w, 4))

        # Render Top HUD
        hud_bg = pygame.Rect(0, 0, screen_width, 45)
        pygame.draw.rect(screen, PANEL_BG, hud_bg)
        pygame.draw.line(screen, GRID_COLOR, (0, 45), (screen_width, 45), 2)
        
        title_txt = font_large.render("Mid-Review Multi-Robot Warehouse Simulation", True, TEXT_DARK)
        screen.blit(title_txt, (12, 10))
        
        status_str = "PAUSED" if paused else "RUNNING"
        status_txt = font_small.render(f"Step: {step_count} | Status: {status_str} | Deliveries: {deliveries_count}", True, TEXT_DARK)
        screen.blit(status_txt, (screen_width - 310, 14))

        # Render Sidebar (Stats & Event Log)
        sb_x = grid_width * CELL_SIZE + 10
        sb_panel = pygame.Rect(sb_x, 50, sidebar_width - 20, screen_height - 60)
        pygame.draw.rect(screen, PANEL_BG, sb_panel, 0, border_radius=8)
        pygame.draw.rect(screen, GRID_COLOR, sb_panel, 2, border_radius=8)
        
        # Sidebar Header
        sb_title = font_large.render("Robot Status & Logs", True, TEXT_DARK)
        screen.blit(sb_title, (sb_x + 12, 62))
        
        # Robot Cards
        for i in range(2):
            card_y = 95 + i * 75
            r_name = env.robot_type_names[i]
            r_batt = f"{env.battery[i]:.1f}%"
            p_str = phase_names.get(phases[i], "IDLE")
            txt_card = font_small.render(f"Robot {i} ({r_name}) — [{p_str}]", True, colors[i])
            txt_batt = font_small.render(f"Battery: {r_batt}", True, TEXT_DARK)
            screen.blit(txt_card, (sb_x + 12, card_y))
            screen.blit(txt_batt, (sb_x + 12, card_y + 18))

        # Event Log Window
        log_lbl = font_large.render("Live Event Log:", True, TEXT_DARK)
        screen.blit(log_lbl, (sb_x + 12, 250))
        
        for idx, log_entry in enumerate(event_logs):
            entry_txt = font_small.render(log_entry, True, (80, 90, 100))
            screen.blit(entry_txt, (sb_x + 12, 275 + idx * 20))

        pygame.display.flip()
        clock.tick(5 if not paused else 15)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
