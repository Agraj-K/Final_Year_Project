"""
main_simulation.py — 2-Robot End-to-End Warehouse Simulation

Runs the full pipeline for 2 robots:
  1. Decision network (reuses AgentNetwork from network.py)
  2. Decentralized communication with 5% drop probability
  3. Decentralized auction (bid comparison + tie-breaking)
  4. A* path planning (static grid, replans on assignment change)
  5. Collision manager (per-step conflict resolution)
  6. Fault detection (heartbeat tracking, configurable failure injection)

Usage:
    python main_simulation.py --total-steps 200 --fail-robot 0 --fail-step 50
    python main_simulation.py --total-steps 300   # no fault injection
"""

import argparse
import heapq
import logging
import os
import glob
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
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


# ═══════════════════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="[Step %(step)04d] %(message)s",
)
logger = logging.getLogger("simulation")


class StepAdapter(logging.LoggerAdapter):
    """Logger adapter that auto-injects the current step number."""

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})["step"] = self.extra.get("step", 0)
        return msg, kwargs


log = StepAdapter(logger, {"step": 0})


def set_step(step: int):
    log.extra["step"] = step


# ═══════════════════════════════════════════════════════════════════════════════
# Direction Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_DIR_DELTAS: Dict[Direction, Tuple[int, int]] = {
    Direction.UP:    (0, -1),   # (dx, dy)
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}


def direction_for_delta(dx: int, dy: int) -> Optional[Direction]:
    """Get the RWARE Direction for a given (dx, dy) step."""
    for d, (ddx, ddy) in _DIR_DELTAS.items():
        if ddx == dx and ddy == dy:
            return d
    return None


def turn_actions_to_face(current_dir: Direction, target_dir: Direction) -> List[int]:
    """
    Return a list of turn actions (LEFT=2, RIGHT=3) to rotate from
    current_dir to target_dir. Returns [] if already facing correctly.
    """
    if current_dir == target_dir:
        return []

    # Clockwise order: UP -> RIGHT -> DOWN -> LEFT -> UP
    cw_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    ci = cw_order.index(current_dir)
    ti = cw_order.index(target_dir)

    # Number of clockwise turns
    cw_turns = (ti - ci) % 4
    # Number of counter-clockwise turns
    ccw_turns = (ci - ti) % 4

    if cw_turns <= ccw_turns:
        return [Action.RIGHT.value] * cw_turns  # RIGHT = clockwise turn
    else:
        return [Action.LEFT.value] * ccw_turns  # LEFT = counter-clockwise turn


# ═══════════════════════════════════════════════════════════════════════════════
# A* Path Planner
# ═══════════════════════════════════════════════════════════════════════════════

def build_walkable_grid(env: HeterogeneousWarehouse) -> np.ndarray:
    """
    Build a 2D boolean grid where True = walkable cell.
    Shelf cells are NOT walkable unless they are the start/goal of the path.
    """
    base = env.base_env
    rows, cols = base.grid_size  # (height, width)
    walkable = np.ones((rows, cols), dtype=bool)

    # Shelf cells are NOT walkable (unless specifically target or starting cell)
    shelf_layer = base.grid[1]  # _LAYER_SHELFS
    walkable[shelf_layer > 0] = False

    return walkable


def astar(
    walkable: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on a 2D grid.
    """
    sx, sy = start
    gx, gy = goal
    rows, cols = walkable.shape

    def heuristic(x, y):
        return abs(x - gx) + abs(y - gy)

    # Priority queue: (f_score, counter, x, y)
    counter = 0
    open_set = [(heuristic(sx, sy), counter, sx, sy)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0}

    while open_set:
        _, _, cx, cy = heapq.heappop(open_set)

        if (cx, cy) == (gx, gy):
            path = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < cols and 0 <= ny_ < rows:
                # Allow walking into the goal/start cell even if it has a shelf
                if not walkable[ny_, nx_] and (nx_, ny_) != (gx, gy) and (nx_, ny_) != (sx, sy):
                    continue
                new_g = g_score[(cx, cy)] + 1
                if new_g < g_score.get((nx_, ny_), float("inf")):
                    g_score[(nx_, ny_)] = new_g
                    f = new_g + heuristic(nx_, ny_)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nx_, ny_))
                    came_from[(nx_, ny_)] = (cx, cy)

    return None  # No path found


def path_to_action_queue(
    path: List[Tuple[int, int]],
    start_dir: Direction,
) -> List[int]:
    """Convert waypoints into movement actions."""
    actions = []
    current_dir = start_dir

    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx_, ny_ = path[i + 1]
        dx = nx_ - cx
        dy = ny_ - cy

        target_dir = direction_for_delta(dx, dy)
        if target_dir is None:
            continue

        turns = turn_actions_to_face(current_dir, target_dir)
        actions.extend(turns)
        actions.append(Action.FORWARD.value)
        current_dir = target_dir

    return actions


# ═══════════════════════════════════════════════════════════════════════════════
# Robot State Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class RobotPhase(IntEnum):
    IDLE = 0
    MOVING_TO_SHELF = 1
    PICKING_UP = 2
    MOVING_TO_GOAL = 3
    RETURNING_SHELF = 4
    DROPPING_OFF = 5
    FAILED = 6


@dataclass
class RobotState:
    idx: int
    assigned_shelf_id: Optional[int] = None
    assigned_shelf_pos: Optional[Tuple[int, int]] = None
    phase: RobotPhase = RobotPhase.IDLE
    action_queue: List[int] = field(default_factory=list)
    heartbeat_counter: int = 0
    failed: bool = False
    missed_heartbeats: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Decentralized Auction
# ═══════════════════════════════════════════════════════════════════════════════

def run_auction(
    shelf_id: int,
    shelf_pos: Tuple[int, int],
    task_type: TaskType,
    robots: List[RobotState],
    bids: Dict[int, float],
    comm_success: Dict[Tuple[int, int], bool],
    robot_types: List[RobotType],
    battery: np.ndarray,
) -> Optional[int]:
    """Decentralized auction with capability and battery tie-breaks."""
    eligible = [
        r.idx for r in robots
        if not r.failed and r.phase == RobotPhase.IDLE and battery[r.idx] > 0
    ]

    if not eligible:
        return None

    if len(eligible) == 1:
        winner = eligible[0]
        log.info(
            f"AUCTION: Shelf {shelf_id} at {shelf_pos} ({task_type.name}) -> "
            f"Robot {winner} wins (only eligible bidder, bid={bids.get(winner, 0):.3f})"
        )
        return winner

    r0, r1 = eligible[0], eligible[1]
    b0, b1 = bids.get(r0, 0.0), bids.get(r1, 0.0)

    can_compare = (
        comm_success.get((r0, r1), False) or
        comm_success.get((r1, r0), False)
    )

    if can_compare:
        log.info(
            f"AUCTION: Shelf {shelf_id} ({task_type.name}) - "
            f"Robot {r0} bid={b0:.3f}, Robot {r1} bid={b1:.3f} (comm OK)"
        )

        if abs(b0 - b1) < 1e-6:
            match0 = COMPATIBILITY_MAP[robot_types[r0]] == task_type
            match1 = COMPATIBILITY_MAP[robot_types[r1]] == task_type
            if match0 and not match1:
                winner = r0
                reason = "capability match"
            elif match1 and not match0:
                winner = r1
                reason = "capability match"
            elif battery[r0] >= battery[r1]:
                winner = r0
                reason = "higher battery"
            else:
                winner = r1
                reason = "higher battery"
            log.info(f"  -> TIE BROKEN: Robot {winner} wins ({reason})")
        elif b0 > b1:
            winner = r0
            log.info(f"  -> Robot {r0} wins (higher bid)")
        else:
            winner = r1
            log.info(f"  -> Robot {r1} wins (higher bid)")
    else:
        log.info(
            f"AUCTION: Shelf {shelf_id} ({task_type.name}) - "
            f"Robot {r0} bid={b0:.3f}, Robot {r1} bid={b1:.3f} (COMM FAILED)"
        )
        if b0 >= b1:
            winner = r0
        else:
            winner = r1
        log.info(f"  -> Robot {winner} assigned (no comm, resolved by bid)")

    return winner


# ═══════════════════════════════════════════════════════════════════════════════
# Collision Manager
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_collision(
    proposed_actions: List[int],
    robots: List[RobotState],
    env: HeterogeneousWarehouse,
) -> List[int]:
    """Collision avoidance using priority rules: loaded > priority > battery."""
    if len(robots) != 2:
        return proposed_actions

    base = env.base_env
    agents = base.agents
    actions = list(proposed_actions)

    # Compute target position for each agent
    next_pos = []
    for i in range(2):
        agent = agents[i]
        if actions[i] == Action.FORWARD.value:
            dx, dy = _DIR_DELTAS[agent.dir]
            nx_ = max(0, min(base.grid_size[1] - 1, agent.x + dx))
            ny_ = max(0, min(base.grid_size[0] - 1, agent.y + dy))
            next_pos.append((nx_, ny_))
        else:
            next_pos.append((agent.x, agent.y))

    # Same target cell collision
    if next_pos[0] == next_pos[1] and actions[0] == Action.FORWARD.value and actions[1] == Action.FORWARD.value:
        loaded_0 = agents[0].carrying_shelf is not None
        loaded_1 = agents[1].carrying_shelf is not None

        if loaded_0 and not loaded_1:
            loser = 1
            reason = "Robot 0 is loaded"
        elif loaded_1 and not loaded_0:
            loser = 0
            reason = "Robot 1 is loaded"
        else:
            task_priority_0 = _get_task_priority(robots[0], env)
            task_priority_1 = _get_task_priority(robots[1], env)

            if task_priority_0 > task_priority_1:
                loser = 1
                reason = f"Robot 0 higher task priority ({task_priority_0:.1f} > {task_priority_1:.1f})"
            elif task_priority_1 > task_priority_0:
                loser = 0
                reason = f"Robot 1 higher task priority ({task_priority_1:.1f} > {task_priority_0:.1f})"
            else:
                if env.battery[0] <= env.battery[1]:
                    loser = 1
                    reason = f"Robot 0 lower battery ({env.battery[0]:.1f} <= {env.battery[1]:.1f})"
                else:
                    loser = 0
                    reason = f"Robot 1 lower battery ({env.battery[1]:.1f} < {env.battery[0]:.1f})"

        actions[loser] = Action.NOOP.value
        log.info(
            f"COLLISION: Both robots want cell {next_pos[0]} -> "
            f"Robot {loser} yields ({reason})"
        )

    # Position swapping collision
    elif (next_pos[0] == (agents[1].x, agents[1].y) and
          next_pos[1] == (agents[0].x, agents[0].y)):
        actions[1] = Action.NOOP.value
        log.info(
            f"COLLISION: Head-on swap detected, Robot 1 yields"
        )

    return actions


def _get_task_priority(robot: RobotState, env: HeterogeneousWarehouse) -> float:
    if robot.assigned_shelf_id is None:
        return 0.0
    task_type = env.task_type_map.get(robot.assigned_shelf_id)
    if task_type is None:
        return 0.0
    return DEFAULT_TASK_WEIGHTS.get(task_type, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Communication Layer
# ═══════════════════════════════════════════════════════════════════════════════

def attempt_communication(
    robots: List[RobotState],
    env: HeterogeneousWarehouse,
    comm_gates: List[bool],
    rng: np.random.Generator,
    drop_prob: float = 0.05,
) -> Dict[Tuple[int, int], bool]:
    results: Dict[Tuple[int, int], bool] = {}

    for i in range(len(robots)):
        if robots[i].failed:
            continue
        if not comm_gates[i]:
            continue

        neighbors = env.comm.get_neighbors(i)
        for j in neighbors:
            if robots[j].failed:
                continue
            dropped = rng.random() < drop_prob
            results[(i, j)] = not dropped
            if dropped:
                log.info(
                    f"COMM: Robot {i} -> Robot {j}: MESSAGE DROPPED (5% chance)"
                )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Fault Detection
# ═══════════════════════════════════════════════════════════════════════════════

HEARTBEAT_MISS_THRESHOLD = 3


def update_heartbeats(
    robots: List[RobotState],
    heartbeat_log: Dict[int, int],
):
    for r in robots:
        if not r.failed:
            r.heartbeat_counter += 1
            heartbeat_log[r.idx] = r.heartbeat_counter


def check_for_failures(
    robots: List[RobotState],
    heartbeat_log: Dict[int, int],
    previous_heartbeats: Dict[int, int],
) -> List[int]:
    newly_detected = []

    for r in robots:
        if r.failed:
            continue
        for other in robots:
            if other.idx == r.idx:
                continue
            if other.failed and other.missed_heartbeats >= HEARTBEAT_MISS_THRESHOLD:
                continue

            current_hb = heartbeat_log.get(other.idx, 0)
            prev_hb = previous_heartbeats.get(other.idx, 0)

            if current_hb == prev_hb:
                other.missed_heartbeats += 1
                if other.missed_heartbeats >= HEARTBEAT_MISS_THRESHOLD and other.idx not in newly_detected:
                    newly_detected.append(other.idx)
                    log.info(
                        f"FAULT: Robot {r.idx} detected Robot {other.idx} has FAILED "
                        f"({other.missed_heartbeats} missed heartbeats)"
                    )
            else:
                other.missed_heartbeats = 0

    return newly_detected


# ═══════════════════════════════════════════════════════════════════════════════
# Robot Navigation Loop
# ═══════════════════════════════════════════════════════════════════════════════

def assign_task_to_robot(
    robot: RobotState,
    shelf_id: int,
    shelf_pos: Tuple[int, int],
    env: HeterogeneousWarehouse,
):
    """Plan a path to the shelf and set the robot's action queue."""
    agent = env.base_env.agents[robot.idx]
    walkable = build_walkable_grid(env)

    path = astar(walkable, (agent.x, agent.y), shelf_pos)
    if path is None:
        log.info(
            f"PATH: Robot {robot.idx} cannot find path to shelf {shelf_id} "
            f"at {shelf_pos} — staying IDLE"
        )
        return

    robot.assigned_shelf_id = shelf_id
    robot.assigned_shelf_pos = shelf_pos
    robot.phase = RobotPhase.MOVING_TO_SHELF
    robot.action_queue = path_to_action_queue(path, agent.dir)

    log.info(
        f"PATH: Robot {robot.idx} planned path to shelf {shelf_id} at {shelf_pos} "
        f"({len(robot.action_queue)} actions)"
    )


def plan_path_to_goal(robot: RobotState, env: HeterogeneousWarehouse):
    agent = env.base_env.agents[robot.idx]
    walkable = build_walkable_grid(env)

    best_goal = None
    best_dist = float("inf")
    for gx, gy in env.base_env.goals:
        d = abs(agent.x - gx) + abs(agent.y - gy)
        if d < best_dist:
            best_dist = d
            best_goal = (gx, gy)

    if best_goal is None:
        return

    path = astar(walkable, (agent.x, agent.y), best_goal)
    if path is None:
        return

    robot.phase = RobotPhase.MOVING_TO_GOAL
    robot.action_queue = path_to_action_queue(path, agent.dir)
    log.info(
        f"PATH: Robot {robot.idx} planned path to goal {best_goal} "
        f"({len(robot.action_queue)} actions)"
    )


def plan_path_to_return(robot: RobotState, env: HeterogeneousWarehouse):
    agent = env.base_env.agents[robot.idx]
    walkable = build_walkable_grid(env)

    if robot.assigned_shelf_pos is None:
        return

    path = astar(walkable, (agent.x, agent.y), robot.assigned_shelf_pos)
    if path is None:
        return

    robot.phase = RobotPhase.RETURNING_SHELF
    robot.action_queue = path_to_action_queue(path, agent.dir)
    log.info(
        f"PATH: Robot {robot.idx} planned path back to original shelf spot {robot.assigned_shelf_pos} "
        f"({len(robot.action_queue)} actions)"
    )


def get_next_action(robot: RobotState, env: HeterogeneousWarehouse) -> int:
    if robot.failed:
        return Action.NOOP.value

    agent = env.base_env.agents[robot.idx]

    if robot.phase == RobotPhase.IDLE:
        return Action.NOOP.value

    if robot.phase == RobotPhase.MOVING_TO_SHELF:
        if robot.action_queue:
            return robot.action_queue.pop(0)
        if robot.assigned_shelf_pos and (agent.x, agent.y) == robot.assigned_shelf_pos:
            robot.phase = RobotPhase.PICKING_UP
            log.info(f"Robot {robot.idx} arrived at shelf {robot.assigned_shelf_id}, picking up")
            return Action.TOGGLE_LOAD.value
        else:
            # Replan if off-track
            if robot.assigned_shelf_pos:
                walkable = build_walkable_grid(env)
                path = astar(walkable, (agent.x, agent.y), robot.assigned_shelf_pos)
                if path and len(path) > 1:
                    robot.action_queue = path_to_action_queue(path, agent.dir)
                    if robot.action_queue:
                        return robot.action_queue.pop(0)
            return Action.NOOP.value

    if robot.phase == RobotPhase.PICKING_UP:
        if agent.carrying_shelf is not None:
            log.info(f"Robot {robot.idx} picked up shelf, heading to goal")
            plan_path_to_goal(robot, env)
            if robot.action_queue:
                return robot.action_queue.pop(0)
            return Action.NOOP.value
        else:
            return Action.TOGGLE_LOAD.value

    if robot.phase == RobotPhase.MOVING_TO_GOAL:
        if robot.action_queue:
            return robot.action_queue.pop(0)
        # Reached goal! RWARE handles delivery immediately when we step on goal.
        # Now plan to return the shelf to its original slot.
        plan_path_to_return(robot, env)
        if robot.action_queue:
            return robot.action_queue.pop(0)
        return Action.NOOP.value

    if robot.phase == RobotPhase.RETURNING_SHELF:
        if robot.action_queue:
            return robot.action_queue.pop(0)
        if robot.assigned_shelf_pos and (agent.x, agent.y) == robot.assigned_shelf_pos:
            robot.phase = RobotPhase.DROPPING_OFF
            log.info(f"Robot {robot.idx} returned shelf to spot, dropping off")
            return Action.TOGGLE_LOAD.value
        else:
            plan_path_to_return(robot, env)
            if robot.action_queue:
                return robot.action_queue.pop(0)
            return Action.NOOP.value

    if robot.phase == RobotPhase.DROPPING_OFF:
        if agent.carrying_shelf is None:
            log.info(f"Robot {robot.idx} successfully dropped off shelf. Going IDLE.")
            robot.assigned_shelf_id = None
            robot.assigned_shelf_pos = None
            robot.phase = RobotPhase.IDLE
            return Action.NOOP.value
        else:
            return Action.TOGGLE_LOAD.value

    return Action.NOOP.value


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2-Robot Warehouse Simulation")
    parser.add_argument("--total-steps", type=int, default=200,
                        help="Total simulation steps (default: 200)")
    parser.add_argument("--fail-robot", type=int, default=None,
                        help="Robot index to fail (0 or 1). Omit for no failure.")
    parser.add_argument("--fail-step", type=int, default=50,
                        help="Step at which to inject failure (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--comm-drop-prob", type=float, default=0.05,
                        help="Communication drop probability (default: 0.05)")
    parser.add_argument("--comm-range", type=int, default=20,
                        help="Communication range in Manhattan distance (default: 20)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu")

    print("=" * 70)
    print("  2-Robot Warehouse Simulation")
    print("=" * 70)
    print(f"  Steps: {args.total_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Comm drop prob: {args.comm_drop_prob}")
    if args.fail_robot is not None:
        print(f"  Fault injection: Robot {args.fail_robot} fails at step {args.fail_step}")
    else:
        print(f"  Fault injection: DISABLED")
    print("=" * 70)
    print()

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
    obs, info = env.reset(seed=args.seed)

    print(f"Robot 0: type={env.robot_types[0].name}, battery={env.battery[0]:.0f}")
    print(f"Robot 1: type={env.robot_types[1].name}, battery={env.battery[1]:.0f}")
    print(f"Grid size: {env.base_env.grid_size}")
    print(f"Goals: {env.base_env.goals}")
    print(f"Initial request queue: {[(s.id, int(s.x), int(s.y)) for s in env.base_env.request_queue]}")
    print()

    net = AgentNetwork(obs_dim=80).to(device)

    model_files = glob.glob("models/ppo_hetero_*.pt")
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        print(f"Loading trained model: {latest_model}")
        net.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
    else:
        print("No trained model found — using randomly initialized network")
    net.eval()
    print()

    robots = [RobotState(idx=0), RobotState(idx=1)]
    heartbeat_log: Dict[int, int] = {0: 0, 1: 0}
    previous_heartbeats: Dict[int, int] = {0: 0, 1: 0}
    assigned_shelf_ids: set = set()

    total_deliveries = 0
    total_comm_drops = 0
    total_auctions = 0
    total_collisions_resolved = 0

    for step in range(args.total_steps):
        set_step(step)

        # ── Fault injection ──────────────────────────────────────────
        if (args.fail_robot is not None and
            step == args.fail_step and
            not robots[args.fail_robot].failed):
            fail_idx = args.fail_robot
            robots[fail_idx].failed = True
            robots[fail_idx].phase = RobotPhase.FAILED
            log.info(
                f"*** FAULT INJECTED: Robot {fail_idx} has FAILED ***"
            )
            if robots[fail_idx].assigned_shelf_id is not None:
                released_id = robots[fail_idx].assigned_shelf_id
                assigned_shelf_ids.discard(released_id)
                log.info(
                    f"  Task (shelf {released_id}) released back to pool"
                )
                robots[fail_idx].assigned_shelf_id = None
                robots[fail_idx].assigned_shelf_pos = None
                robots[fail_idx].action_queue = []

        # ── Heartbeats & Failures ────────────────────────────────────
        previous_heartbeats = dict(heartbeat_log)
        update_heartbeats(robots, heartbeat_log)
        newly_failed = check_for_failures(robots, heartbeat_log, previous_heartbeats)

        for failed_idx in newly_failed:
            if robots[failed_idx].assigned_shelf_id is not None:
                released_id = robots[failed_idx].assigned_shelf_id
                assigned_shelf_ids.discard(released_id)
                log.info(
                    f"  Task (shelf {released_id}) released for re-auction"
                )
                robots[failed_idx].assigned_shelf_id = None
                robots[failed_idx].assigned_shelf_pos = None
                robots[failed_idx].action_queue = []

        # ── Network Bidding & Comm Gates ─────────────────────────────
        bids: Dict[int, float] = {}
        comm_gates: List[bool] = [False, False]

        for i in range(2):
            if robots[i].failed:
                bids[i] = 0.0
                comm_gates[i] = False
                continue

            with torch.no_grad():
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).to(device)
                _, comm_action, bid_value, _, _, _ = net.get_action_and_value(obs_tensor)
                bids[i] = bid_value.item()
                comm_gates[i] = comm_action.item() > 0.5

        # ── Communication Success Check ──────────────────────────────
        env._update_comm_positions()
        comm_success = attempt_communication(
            robots, env, comm_gates, rng, args.comm_drop_prob
        )
        for success in comm_success.values():
            if not success:
                total_comm_drops += 1

        # ── Auctions ─────────────────────────────────────────────────
        for shelf in env.base_env.request_queue:
            if shelf.id in assigned_shelf_ids:
                continue

            idle_robots = [r for r in robots if not r.failed and r.phase == RobotPhase.IDLE]
            if not idle_robots:
                continue

            task_type = env.task_type_map.get(shelf.id)
            if task_type is None:
                continue

            total_auctions += 1
            winner = run_auction(
                shelf.id,
                (int(shelf.x), int(shelf.y)),
                task_type,
                robots,
                bids,
                comm_success,
                env.robot_types,
                env.battery,
            )

            if winner is not None:
                assigned_shelf_ids.add(shelf.id)
                assign_task_to_robot(
                    robots[winner],
                    shelf.id,
                    (int(shelf.x), int(shelf.y)),
                    env,
                )

        # ── Action Generation & Collision Resolution ─────────────────
        proposed_actions = [
            get_next_action(robots[i], env)
            for i in range(2)
        ]

        original_actions = list(proposed_actions)
        resolved_actions = resolve_collision(proposed_actions, robots, env)

        if original_actions != resolved_actions:
            total_collisions_resolved += 1

        # ── Step Environment ─────────────────────────────────────────
        env_actions = [
            [resolved_actions[i], int(comm_gates[i])]
            for i in range(2)
        ]

        old_queue_ids = set(s.id for s in env.base_env.request_queue)
        obs, rewards, done, truncated, info = env.step(env_actions)
        new_queue_ids = set(s.id for s in env.base_env.request_queue)

        # Detect delivery events (RWARE delivers shelf when it enters the goal)
        delivered_ids = old_queue_ids - new_queue_ids
        for shelf_id in delivered_ids:
            total_deliveries += 1
            log.info(f"DELIVERY: Shelf {shelf_id} delivered successfully!")
            # Find which robot had this shelf and put it in returning mode
            for r in robots:
                if r.assigned_shelf_id == shelf_id:
                    # Clear task registration from queue tracker so it can be re-assigned as new task
                    assigned_shelf_ids.discard(shelf_id)

        # ── Handle Environment Done/Reset ────────────────────────────
        if done:
            log.info("Environment episode done, resetting")
            obs, info = env.reset(seed=args.seed + step)
            for r in robots:
                if not r.failed:
                    r.phase = RobotPhase.IDLE
                    r.assigned_shelf_id = None
                    r.assigned_shelf_pos = None
                    r.action_queue = []
            assigned_shelf_ids.clear()

        # ── Periodic Status Logs ─────────────────────────────────────
        if step % 25 == 0 and step > 0:
            log.info(
                f"--- STATUS: R0={robots[0].phase.name} "
                f"(shelf={robots[0].assigned_shelf_id}, bat={env.battery[0]:.1f}), "
                f"R1={robots[1].phase.name} "
                f"(shelf={robots[1].assigned_shelf_id}, bat={env.battery[1]:.1f}) ---"
            )

    print()
    print("=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Total steps:              {args.total_steps}")
    print(f"  Total auctions:           {total_auctions}")
    print(f"  Total deliveries:         {total_deliveries}")
    print(f"  Total comm drops:         {total_comm_drops}")
    print(f"  Total collisions resolved:{total_collisions_resolved}")
    print(f"  Robot 0 final state:      {robots[0].phase.name} (battery={env.battery[0]:.1f})")
    print(f"  Robot 1 final state:      {robots[1].phase.name} (battery={env.battery[1]:.1f})")
    if args.fail_robot is not None:
        print(f"  Fault injected:           Robot {args.fail_robot} at step {args.fail_step}")
    print("=" * 70)


if __name__ == "__main__":
    main()
