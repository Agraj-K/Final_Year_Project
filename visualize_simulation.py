"""
visualize_simulation.py — Visual Pygame Renderer for the Warehouse Simulation

A completely standalone visual layer on top of main_simulation.py.
Does NOT modify any existing file. Just imports and drives the same pipeline.

Controls:
    SPACE       — Pause / Resume
    RIGHT ARROW — Step one frame (while paused)
    + / -       — Speed up / slow down
    F           — Toggle fault injection label
    ESC / Q     — Quit

Usage:
    py -3.12 visualize_simulation.py
    py -3.12 visualize_simulation.py --total-steps 300 --fail-robot 0 --fail-step 50
    py -3.12 visualize_simulation.py --speed 5   # steps per second
"""

import argparse
import glob
import heapq
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch

from rware.warehouse import Warehouse, RewardType, Direction, Action
from hetero_wrapper import (
    HeterogeneousWarehouse,
    RobotType,
    TaskType,
    COMPATIBILITY_MAP,
    DEFAULT_TASK_WEIGHTS,
)
from network import AgentNetwork

# ── Re-import all logic from main_simulation without modification ────────────
# We import all the non-UI functions directly.
# This avoids touching main_simulation.py at all.
from main_simulation import (
    RobotPhase,
    RobotState,
    HEARTBEAT_MISS_THRESHOLD,
    build_walkable_grid,
    astar,
    path_to_action_queue,
    assign_task_to_robot,
    plan_path_to_goal,
    get_next_action,
    resolve_collision,
    run_auction,
    attempt_communication,
    update_heartbeats,
    check_for_failures,
    _get_task_priority,
)

import logging
logging.disable(logging.CRITICAL)   # silence main_simulation log output in visual mode


# ═══════════════════════════════════════════════════════════════════════════════
# Colour Palette
# ═══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":          (15,  17,  26),   # dark navy background
    "grid_line":   (35,  40,  58),   # subtle grid lines
    "shelf":       (52,  73, 102),   # unselected shelf — steel blue
    "shelf_req":   (255, 200,  50),   # requested shelf — amber
    "shelf_deliv": ( 80, 220, 120),   # shelf being delivered (carried) — green
    "goal":        ( 60, 200, 130),   # goal zone — teal
    "highway":     (22,  25,  38),   # highway corridor — slightly different bg

    # Robot colours per index
    "r0":          ( 80, 160, 255),   # robot 0 — blue
    "r0_body":     ( 30,  90, 200),
    "r1":          (255, 120,  80),   # robot 1 — orange
    "r1_body":     (200,  70,  30),
    "r_failed":    (120,  30,  30),   # failed robot — dark red

    "text":        (220, 225, 240),
    "text_dim":    (100, 110, 140),
    "text_warn":   (255, 160,  60),
    "text_ok":     ( 80, 220, 120),
    "text_bad":    (220,  70,  70),
    "panel_bg":    ( 22,  25,  40),
    "panel_border":(50,  60,  90),

    # Event flash colours
    "flash_drop":  (255,  80,  80),
    "flash_deliv": ( 80, 240, 120),
    "flash_fault": (255,  60,  60),
    "flash_bid":   (180, 120, 255),
    "flash_coll":  (255, 200,  40),
}

ROBOT_COLORS = [
    (C["r0"], C["r0_body"]),
    (C["r1"], C["r1_body"]),
]

PHASE_SHORT = {
    RobotPhase.IDLE:            "IDLE",
    RobotPhase.MOVING_TO_SHELF: "→SHELF",
    RobotPhase.PICKING_UP:      "PICKUP",
    RobotPhase.MOVING_TO_GOAL:  "→GOAL",
    RobotPhase.RETURNING_SHELF: "RETURN",
    RobotPhase.DROPPING_OFF:    "DROPOFF",
    RobotPhase.FAILED:          "FAILED",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Event Log (last N events shown in sidebar)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogEvent:
    step: int
    text: str
    color: tuple
    ttl: int = 60   # frames to display (for on-grid flashes)


class EventLog:
    def __init__(self, maxlen=14):
        self.entries: List[LogEvent] = []
        self.maxlen = maxlen
        self.flashes: List[LogEvent] = []   # short-lived grid-overlay messages

    def add(self, step, text, color=None, flash=False):
        c = color or C["text"]
        ev = LogEvent(step, text, c)
        self.entries.append(ev)
        if len(self.entries) > self.maxlen:
            self.entries.pop(0)
        if flash:
            self.flashes.append(LogEvent(step, text, c, ttl=90))

    def tick(self):
        self.flashes = [f for f in self.flashes if f.ttl > 0]
        for f in self.flashes:
            f.ttl -= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════════════════════

class WarehouseRenderer:
    CELL = 54           # px per grid cell
    PANEL_W = 320       # right sidebar width
    HUD_H = 110         # top HUD strip height

    def __init__(self, grid_rows, grid_cols, n_agents=2):
        self.rows = grid_rows
        self.cols = grid_cols
        self.n_agents = n_agents

        self.grid_w = grid_cols * self.CELL
        self.grid_h = grid_rows * self.CELL
        self.win_w  = self.grid_w + self.PANEL_W
        self.win_h  = self.grid_h + self.HUD_H

        pygame.init()
        pygame.display.set_caption("Warehouse Robot Coordination — Visual Simulation")
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock  = pygame.time.Clock()

        self.font_sm  = pygame.font.SysFont("Consolas", 12)
        self.font_md  = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_lg  = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_xl  = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_hd  = pygame.font.SysFont("Consolas", 11)

    def _cell_rect(self, x, y):
        """Return pygame.Rect for grid cell (x,y). y is row from top."""
        return pygame.Rect(
            x * self.CELL,
            self.HUD_H + y * self.CELL,
            self.CELL,
            self.CELL
        )

    def _draw_grid(self, env, robots, assigned_ids):
        base = env.base_env
        grid_surf = self.screen

        # Highway / floor
        for gy in range(self.rows):
            for gx in range(self.cols):
                r = self._cell_rect(gx, gy)
                is_hw = base.highways[gy, gx]
                pygame.draw.rect(grid_surf, C["highway"] if is_hw else C["bg"], r)
                pygame.draw.rect(grid_surf, C["grid_line"], r, 1)

        # Goals
        for gx, gy in base.goals:
            r = self._cell_rect(gx, gy)
            pygame.draw.rect(grid_surf, C["goal"], r)
            lbl = self.font_sm.render("GOAL", True, (10, 40, 30))
            grid_surf.blit(lbl, (r.x + 8, r.y + r.h // 2 - 6))

        # Shelves
        shelf_layer = base.grid[1]
        for gy in range(self.rows):
            for gx in range(self.cols):
                sid = shelf_layer[gy, gx]
                if sid == 0:
                    continue
                shelf = base.shelfs[sid - 1]
                in_queue = shelf in base.request_queue
                # Is a robot carrying it?
                carried = any(
                    base.agents[i].carrying_shelf is not None and
                    base.agents[i].carrying_shelf.id == sid
                    for i in range(self.n_agents)
                )

                if carried:
                    col = C["shelf_deliv"]
                elif in_queue:
                    col = C["shelf_req"]
                else:
                    col = C["shelf"]

                r = self._cell_rect(gx, gy)
                inner = r.inflate(-10, -10)
                pygame.draw.rect(grid_surf, col, inner, border_radius=4)

                # Task type label if in queue
                if in_queue:
                    task_type = env.task_type_map.get(sid)
                    short = {
                        TaskType.URGENT_LIGHT:      "UL",
                        TaskType.HEAVY_SHELF:       "HS",
                        TaskType.STANDARD_DELIVERY: "SD",
                    }.get(task_type, "?")
                    lbl = self.font_hd.render(short, True, (20, 20, 20))
                    grid_surf.blit(lbl, (r.x + r.w // 2 - lbl.get_width() // 2,
                                         r.y + r.h // 2 - lbl.get_height() // 2))

        # Robots
        for i, robot in enumerate(robots):
            agent = base.agents[i]
            r = self._cell_rect(agent.x, agent.y)
            cx, cy = r.centerx, r.centery

            if robot.failed:
                body_col = C["r_failed"]
                ring_col = (180, 40, 40)
            else:
                ring_col, body_col = ROBOT_COLORS[i]

            # Body circle
            pygame.draw.circle(grid_surf, body_col, (cx, cy), self.CELL // 2 - 6)
            pygame.draw.circle(grid_surf, ring_col, (cx, cy), self.CELL // 2 - 6, 3)

            # Direction arrow
            if not robot.failed:
                dx, dy = {
                    Direction.UP:    (0, -1),
                    Direction.DOWN:  (0,  1),
                    Direction.LEFT:  (-1, 0),
                    Direction.RIGHT: (1,  0),
                }[agent.dir]
                arr_end = (cx + dx * 14, cy + dy * 14)
                pygame.draw.line(grid_surf, ring_col, (cx, cy), arr_end, 3)
                # arrowhead
                pygame.draw.circle(grid_surf, ring_col, arr_end, 4)

            # Label: R0 / R1
            lbl = self.font_md.render(f"R{i}", True, C["text"])
            grid_surf.blit(lbl, (cx - lbl.get_width() // 2,
                                  cy - lbl.get_height() // 2))

            # Battery bar underneath robot
            bat_pct = env.battery[i] / 100.0
            bar_w = self.CELL - 10
            bar_h = 5
            bar_x = r.x + 5
            bar_y = r.y + r.h - 9
            pygame.draw.rect(grid_surf, (50, 50, 70), (bar_x, bar_y, bar_w, bar_h))
            bat_col = (
                (80, 220, 80) if bat_pct > 0.5 else
                (255, 200, 0) if bat_pct > 0.2 else
                (220, 60, 60)
            )
            pygame.draw.rect(grid_surf, bat_col, (bar_x, bar_y, int(bar_w * bat_pct), bar_h))

    def _draw_hud(self, step, total_steps, paused, speed, stats, robots, env):
        hud = pygame.Rect(0, 0, self.win_w, self.HUD_H)
        pygame.draw.rect(self.screen, C["panel_bg"], hud)
        pygame.draw.line(self.screen, C["panel_border"],
                         (0, self.HUD_H - 1), (self.win_w, self.HUD_H - 1), 2)

        # Title
        title = self.font_lg.render("Warehouse Robot Coordination", True, C["text"])
        self.screen.blit(title, (14, 10))

        # Step / speed
        step_lbl = self.font_md.render(
            f"Step {step:04d}/{total_steps}   {'[PAUSED]' if paused else f'{speed} fps'}",
            True, C["text_warn"] if paused else C["text_dim"])
        self.screen.blit(step_lbl, (14, 36))

        # Controls hint
        hint = self.font_hd.render(
            "SPACE=pause  →=step  +/-=speed  ESC/Q=quit", True, C["text_dim"])
        self.screen.blit(hint, (14, 56))

        # Stats row
        stat_txt = (
            f"Auctions: {stats['auctions']}   "
            f"Deliveries: {stats['deliveries']}   "
            f"Comm Drops: {stats['comm_drops']}   "
            f"Collisions Resolved: {stats['collisions']}"
        )
        st = self.font_md.render(stat_txt, True, C["text"])
        self.screen.blit(st, (14, 80))

        # Robot type badges
        for i in range(2):
            ring_col, _ = ROBOT_COLORS[i]
            rtype = env.robot_types[i].name if not robots[i].failed else "FAILED"
            badge = self.font_sm.render(f"R{i}: {rtype}", True, ring_col)
            self.screen.blit(badge, (self.grid_w - 260 + i * 170, 10))

    def _draw_panel(self, robots, env, log: EventLog, step, fault_info):
        px = self.grid_w
        panel = pygame.Rect(px, 0, self.PANEL_W, self.win_h)
        pygame.draw.rect(self.screen, C["panel_bg"], panel)
        pygame.draw.line(self.screen, C["panel_border"],
                         (px, 0), (px, self.win_h), 2)

        y = 14
        title = self.font_lg.render("Robot Status", True, C["text"])
        self.screen.blit(title, (px + 14, y)); y += 30

        # Per-robot status cards
        for i, robot in enumerate(robots):
            ring_col, body_col = ROBOT_COLORS[i]
            agent = env.base_env.agents[i]
            bat = env.battery[i]

            card = pygame.Rect(px + 8, y, self.PANEL_W - 16, 110)
            if robot.failed:
                pygame.draw.rect(self.screen, (50, 20, 20), card, border_radius=6)
                pygame.draw.rect(self.screen, C["r_failed"], card, 2, border_radius=6)
            else:
                pygame.draw.rect(self.screen, (28, 32, 52), card, border_radius=6)
                pygame.draw.rect(self.screen, ring_col, card, 2, border_radius=6)

            cx, cy2 = px + 18, y + 12
            label = f"Robot {i} — {env.robot_types[i].name}"
            if robot.failed:
                label += "  ✗ FAILED"
            lbl = self.font_md.render(label, True, C["r_failed"] if robot.failed else ring_col)
            self.screen.blit(lbl, (cx, cy2)); cy2 += 20

            phase_col = C["text_bad"] if robot.failed else C["text_ok"]
            phase_lbl = self.font_sm.render(
                f"Phase : {PHASE_SHORT[robot.phase]}", True, phase_col)
            self.screen.blit(phase_lbl, (cx, cy2)); cy2 += 16

            shelf_lbl = self.font_sm.render(
                f"Task  : shelf {robot.assigned_shelf_id}" if robot.assigned_shelf_id else "Task  : none",
                True, C["text"])
            self.screen.blit(shelf_lbl, (cx, cy2)); cy2 += 16

            pos_lbl = self.font_sm.render(
                f"Pos   : ({agent.x}, {agent.y})  Dir: {agent.dir.name}",
                True, C["text_dim"])
            self.screen.blit(pos_lbl, (cx, cy2)); cy2 += 16

            # Battery bar
            bat_pct = bat / 100.0
            bw = card.width - 20
            bh = 8
            bx = cx; by2 = cy2
            pygame.draw.rect(self.screen, (40, 40, 60), (bx, by2, bw, bh), border_radius=3)
            bat_col = (
                (80, 220, 80) if bat_pct > 0.5 else
                (255, 200, 0) if bat_pct > 0.2 else
                (220, 60, 60)
            )
            pygame.draw.rect(self.screen, bat_col,
                             (bx, by2, int(bw * bat_pct), bh), border_radius=3)
            bat_text = self.font_sm.render(f"Bat: {bat:.1f}%", True, C["text_dim"])
            self.screen.blit(bat_text, (bx + bw - 60, by2 - 14))

            y += 120

        # Fault injection info
        if fault_info:
            y += 8
            fi_lbl = self.font_md.render("Fault Injection", True, C["text_warn"])
            self.screen.blit(fi_lbl, (px + 14, y)); y += 20
            fi_txt = self.font_sm.render(fault_info, True, C["text_warn"])
            self.screen.blit(fi_txt, (px + 14, y)); y += 22

        # Event log
        y += 8
        pygame.draw.line(self.screen, C["panel_border"],
                         (px + 8, y), (px + self.PANEL_W - 8, y), 1)
        y += 8
        log_title = self.font_md.render("Event Log", True, C["text_dim"])
        self.screen.blit(log_title, (px + 14, y)); y += 18

        for ev in log.entries[-12:]:
            # Truncate long messages
            text = ev.text
            while self.font_hd.size(text)[0] > self.PANEL_W - 24:
                text = text[:-1]
            if text != ev.text:
                text = text[:-1] + "…"
            lbl = self.font_hd.render(text, True, ev.color)
            self.screen.blit(lbl, (px + 14, y))
            y += 14
            if y > self.win_h - 20:
                break

    def _draw_legend(self):
        """Small legend in the bottom-left of the grid."""
        items = [
            (C["shelf_req"],   "Requested shelf"),
            (C["shelf_deliv"], "Being delivered"),
            (C["shelf"],       "Empty shelf"),
            (C["goal"],        "Delivery goal"),
        ]
        bx = 6
        by = self.HUD_H + self.grid_h - len(items) * 18 - 6
        for col, label in items:
            pygame.draw.rect(self.screen, col, (bx, by + 3, 12, 12), border_radius=2)
            lbl = self.font_hd.render(label, True, C["text_dim"])
            self.screen.blit(lbl, (bx + 16, by))
            by += 16

    def render(self, env, robots, assigned_ids, step, total_steps,
               paused, speed, stats, log, fault_info):
        self.screen.fill(C["bg"])
        self._draw_grid(env, robots, assigned_ids)
        self._draw_hud(step, total_steps, paused, speed, stats, robots, env)
        self._draw_panel(robots, env, log, step, fault_info)
        self._draw_legend()

        # Draw flash overlays (comm drops, deliveries, etc.)
        for flash in log.flashes:
            alpha = min(255, int(255 * flash.ttl / 90))
            surf = self.font_md.render(flash.text, True, flash.color)
            surf.set_alpha(alpha)
            self.screen.blit(surf, (self.grid_w // 2 - surf.get_width() // 2,
                                     self.HUD_H + self.grid_h // 2 - 20))

        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Driver
# ═══════════════════════════════════════════════════════════════════════════════

def run_visual_simulation(args):
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")

    # Build environment
    base_env = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=2,
        max_inactivity_steps=None,
        max_steps=args.total_steps + 100,
        reward_type=RewardType.INDIVIDUAL,
    )
    env = HeterogeneousWarehouse(
        base_env,
        comm_range=args.comm_range,
        comm_drop_prob=args.comm_drop_prob,
    )
    obs, _ = env.reset(seed=args.seed)

    rows, cols = env.base_env.grid_size

    # Network
    net = AgentNetwork(obs_dim=80).to(device)
    model_files = glob.glob("models/ppo_hetero_*.pt")
    if model_files:
        latest = max(model_files, key=os.path.getctime)
        net.load_state_dict(torch.load(latest, map_location=device, weights_only=True))
    net.eval()

    # Simulation state
    robots = [RobotState(idx=0), RobotState(idx=1)]
    heartbeat_log: Dict[int, int] = {0: 0, 1: 0}
    prev_heartbeats: Dict[int, int] = {0: 0, 1: 0}
    assigned_shelf_ids: set = set()

    stats = {"auctions": 0, "deliveries": 0, "comm_drops": 0, "collisions": 0}
    event_log = EventLog(maxlen=40)

    fault_info = (
        f"Robot {args.fail_robot} fails at step {args.fail_step}"
        if args.fail_robot is not None else None
    )

    # Renderer
    renderer = WarehouseRenderer(rows, cols, n_agents=2)

    # Control state
    paused = False
    speed = args.speed     # target sim steps per real second
    manual_step = False    # step one frame while paused
    step = 0

    running = True
    while running and step < args.total_steps:
        # ── Pygame events ─────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    manual_step = True
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    speed = min(60, speed + 1)
                elif event.key == pygame.K_MINUS:
                    speed = max(1, speed - 1)

        if paused and not manual_step:
            renderer.render(env, robots, assigned_shelf_ids, step, args.total_steps,
                            paused, speed, stats, event_log, fault_info)
            renderer.tick(30)
            continue
        manual_step = False

        # ══════════════════════════════════════════════════════════════════
        # ONE SIMULATION STEP  (mirrors main_simulation.py exactly)
        # ══════════════════════════════════════════════════════════════════

        # ── Fault injection ─────────────────────────────────────────────
        if (args.fail_robot is not None and
            step == args.fail_step and
            not robots[args.fail_robot].failed):
            fidx = args.fail_robot
            robots[fidx].failed = True
            robots[fidx].phase = RobotPhase.FAILED
            event_log.add(step, f"FAULT: Robot {fidx} FAILED!",
                          C["flash_fault"], flash=True)
            if robots[fidx].assigned_shelf_id is not None:
                released = robots[fidx].assigned_shelf_id
                assigned_shelf_ids.discard(released)
                event_log.add(step, f"  Shelf {released} released", C["text_warn"])
                robots[fidx].assigned_shelf_id = None
                robots[fidx].assigned_shelf_pos = None
                robots[fidx].action_queue = []

        # ── Heartbeats ──────────────────────────────────────────────────
        prev_heartbeats = dict(heartbeat_log)
        update_heartbeats(robots, heartbeat_log)
        newly_failed = check_for_failures(robots, heartbeat_log, prev_heartbeats)
        for fi in newly_failed:
            event_log.add(step, f"DETECTED: Robot {fi} failed (3 missed HBs)",
                          C["flash_fault"], flash=True)
            if robots[fi].assigned_shelf_id is not None:
                assigned_shelf_ids.discard(robots[fi].assigned_shelf_id)
                event_log.add(step,
                    f"  Shelf {robots[fi].assigned_shelf_id} → re-auction",
                    C["text_warn"])
                robots[fi].assigned_shelf_id = None
                robots[fi].assigned_shelf_pos = None
                robots[fi].action_queue = []

        # ── Network bids & comm gates ───────────────────────────────────
        bids: Dict[int, float] = {}
        comm_gates: List[bool] = [False, False]
        for i in range(2):
            if robots[i].failed:
                bids[i] = 0.0
                comm_gates[i] = False
                continue
            with torch.no_grad():
                t = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                _, ca, bid, _, _, _ = net.get_action_and_value(t)
                bids[i] = bid.item()
                comm_gates[i] = ca.item() > 0.5

        # ── Communication ───────────────────────────────────────────────
        env._update_comm_positions()
        comm_success = attempt_communication(
            robots, env, comm_gates, rng, args.comm_drop_prob)
        for (s, r_), ok in comm_success.items():
            if not ok:
                stats["comm_drops"] += 1
                event_log.add(step, f"COMM DROP: R{s}→R{r_}",
                              C["flash_drop"])

        # ── Auctions ────────────────────────────────────────────────────
        for shelf in env.base_env.request_queue:
            if shelf.id in assigned_shelf_ids:
                continue
            idle = [r for r in robots if not r.failed and r.phase == RobotPhase.IDLE]
            if not idle:
                continue
            task_type = env.task_type_map.get(shelf.id)
            if task_type is None:
                continue

            stats["auctions"] += 1
            winner = run_auction(shelf.id, (int(shelf.x), int(shelf.y)),
                                 task_type, robots, bids, comm_success,
                                 env.robot_types, env.battery)
            if winner is not None:
                assigned_shelf_ids.add(shelf.id)
                assign_task_to_robot(robots[winner], shelf.id,
                                     (int(shelf.x), int(shelf.y)), env)
                loser = 1 - winner
                event_log.add(step,
                    f"AUCTION S{shelf.id}({task_type.name[:2]}): R{winner} wins "
                    f"(bid {bids[winner]:.3f} vs {bids[loser]:.3f})",
                    C["flash_bid"])

        # ── Actions & collision ─────────────────────────────────────────
        proposed = [get_next_action(robots[i], env) for i in range(2)]
        orig = list(proposed)
        resolved = resolve_collision(proposed, robots, env)
        if orig != resolved:
            stats["collisions"] += 1
            event_log.add(step, f"COLLISION resolved at step {step}",
                          C["flash_coll"])

        env_actions = [[resolved[i], int(comm_gates[i])] for i in range(2)]

        old_q = set(s.id for s in env.base_env.request_queue)
        obs, rewards, done, _, _ = env.step(env_actions)
        new_q = set(s.id for s in env.base_env.request_queue)

        # Delivery detection
        for sid in (old_q - new_q):
            stats["deliveries"] += 1
            event_log.add(step, f"DELIVERED shelf {sid}!",
                          C["flash_deliv"], flash=True)
            assigned_shelf_ids.discard(sid)

        # Stale assignment cleanup
        for i in range(2):
            if (robots[i].assigned_shelf_id is not None and
                robots[i].phase in (RobotPhase.MOVING_TO_SHELF,
                                     RobotPhase.PICKING_UP)):
                still = any(s.id == robots[i].assigned_shelf_id
                            for s in env.base_env.request_queue)
                if not still:
                    assigned_shelf_ids.discard(robots[i].assigned_shelf_id)
                    robots[i].assigned_shelf_id = None
                    robots[i].assigned_shelf_pos = None
                    robots[i].phase = RobotPhase.IDLE
                    robots[i].action_queue = []

        if done:
            obs, _ = env.reset(seed=args.seed + step)
            for r in robots:
                if not r.failed:
                    r.phase = RobotPhase.IDLE
                    r.assigned_shelf_id = None
                    r.assigned_shelf_pos = None
                    r.action_queue = []
            assigned_shelf_ids.clear()

        event_log.tick()
        step += 1

        # ── Render ──────────────────────────────────────────────────────
        renderer.render(env, robots, assigned_shelf_ids, step, args.total_steps,
                        paused, speed, stats, event_log, fault_info)
        renderer.tick(speed)

    # Final frame — hold until user closes
    if running:
        event_log.add(step, "SIMULATION COMPLETE", C["text_ok"])
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_ESCAPE, pygame.K_q):
                    break
            else:
                renderer.render(env, robots, assigned_shelf_ids,
                                step, args.total_steps,
                                True, speed, stats, event_log, fault_info)
                renderer.tick(30)
                continue
            break

    renderer.quit()

    # Print final summary to terminal
    print()
    print("=" * 60)
    print("  VISUAL SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Total steps:       {step}")
    print(f"  Auctions:          {stats['auctions']}")
    print(f"  Deliveries:        {stats['deliveries']}")
    print(f"  Comm drops:        {stats['comm_drops']}")
    print(f"  Collisions solved: {stats['collisions']}")
    print(f"  Robot 0 state:     {robots[0].phase.name}")
    print(f"  Robot 1 state:     {robots[1].phase.name}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Warehouse Simulation")
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--fail-robot", type=int, default=None,
                        help="Robot to fail (0 or 1). Omit to disable.")
    parser.add_argument("--fail-step", type=int, default=50,
                        help="Step at which to inject failure.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comm-drop-prob", type=float, default=0.05)
    parser.add_argument("--comm-range", type=int, default=20)
    parser.add_argument("--speed", type=int, default=6,
                        help="Simulation steps per second (default: 6)")
    args = parser.parse_args()

    run_visual_simulation(args)
