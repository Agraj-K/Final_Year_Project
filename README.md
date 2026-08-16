# Warehouse Robot Coordination Simulation

A complete, minimal end-to-end multi-robot warehouse coordination simulation built on top of [RWARE (Robotic Warehouse)](https://github.com/semitable/robotic-warehouse).

The system demonstrates a full pipeline — from heterogeneous robot environments and neural-network-driven bidding, all the way to A\* path planning, collision management, and fault-tolerant task reassignment — running on a real Gymnasium grid world.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [File Structure](#3-file-structure)
4. [Component Breakdown](#4-component-breakdown)
   - [Environment Layer](#41-environment-layer--hetero_wrapperpy)
   - [Decision Layer](#42-decision-layer--networkpy--eval_bidspy)
   - [Simulation Control Layer](#43-simulation-control-layer--main_simulationpy)
   - [Visual Simulation](#44-visual-simulation--visualize_simulationpy)
5. [Step-by-Step Execution Flow](#5-step-by-step-execution-flow)
6. [How to Run](#6-how-to-run)
7. [Controls (Visual Mode)](#7-controls-visual-mode)

---

## 1. Project Overview

The project simulates 2 heterogeneous warehouse robots coordinating to pick, carry, and deliver shelves. Each robot has a distinct archetype, battery constraints, and task compatibility. A central thesis of the system is that coordination can be **decentralised** — robots communicate and bid for tasks locally rather than relying on a central dispatcher.

**Key Features:**
- Heterogeneous robots (`FAST_LIGHT`, `HEAVY_LOAD`, `BALANCED`) with unique battery drain and speed probabilities
- Three task types (`URGENT_LIGHT`, `HEAVY_SHELF`, `STANDARD_DELIVERY`) with priority weights and robot compatibility penalties
- Decentralised auction with neural-network bids and capability/battery tie-breaking
- 5% communication drop probability on all inter-robot messages
- A\* path planning with action-queue conversion (RWARE turn/forward/toggle actions)
- Per-step collision resolution (loaded > task priority > battery)
- Heartbeat-based fault detection — Robot detects peer failure in ≤3 steps and reclaims the task
- Full pygame visual simulation with live event log, battery indicators, and keyboard controls
- Verified end-to-end: 30/30 unit tests passing, stability tests passing across 2, 4, and 12 robots

---

## 2. System Architecture

The system is built as three cooperating layers:

```
┌──────────────────────────────────────────────────────┐
│         Environment Layer                            │
│   RWARE Base Env + HeterogeneousWarehouse Wrapper    │
│   → 80-value observation per robot                   │
└───────────────────┬──────────────────────────────────┘
                    │ obs[i]  (80-dim)
                    ▼
┌──────────────────────────────────────────────────────┐
│         Decision Layer                               │
│   AgentNetwork (PyTorch)  →  bid value + comm gate   │
│   Decentralised Auction   →  task assignment         │
└───────────────────┬──────────────────────────────────┘
                    │ assigned task
                    ▼
┌──────────────────────────────────────────────────────┐
│         Navigation & Safety Layer                    │
│   A* Planner  →  action queue                        │
│   Collision Manager  →  NOOP for loser               │
│   Fault Detector  →  heartbeat / reassignment        │
└───────────────────┬──────────────────────────────────┘
                    │ gym actions
                    ▼
             env.step(actions)
```

---

## 3. File Structure

```
Final Year Project/
│
├── hetero_wrapper.py        # HeterogeneousWarehouse wrapper on RWARE
│                            # Adds: robot types, battery, task types,
│                            # speed effects, mismatch penalties,
│                            # CommunicationChannel with drop_prob
│
├── network.py               # AgentNetwork (PyTorch)
│                            # 80→128→128→(move | comm | bid | value)
│
├── eval_bids.py             # Bid evaluation utilities
│
├── train_ppo.py             # IPPO training script (CleanRL-style)
│
├── main_simulation.py       # Full 2-robot simulation pipeline
│                            # A*, Collision Manager, Auction, Fault Detector
│
├── visualize_simulation.py  # Pygame visual simulation (new file)
│                            # Imports all logic from main_simulation.py
│                            # Does NOT modify any existing file
│
├── test_hetero.py           # 30 unit tests for the wrapper
├── test_stability.py        # Long-running episode stability tests
│
└── README.md                # This file
```

---

## 4. Component Breakdown

### 4.1 Environment Layer — `hetero_wrapper.py`

Wraps the base RWARE `Warehouse` environment with heterogeneous multi-agent features:

**Robot Types:**

| Type | Battery Drain | Speed Effect |
|---|---|---|
| `FAST_LIGHT` | 0.1/step | 20% chance of extra move |
| `HEAVY_LOAD` | 0.4/step | 15% chance of skipped move |
| `BALANCED` | 0.2/step | No speed effect |

**Task Types & Compatibility:**

| Task Type | Compatible Robot | Priority Weight |
|---|---|---|
| `URGENT_LIGHT` | `FAST_LIGHT` | 1.5 |
| `HEAVY_SHELF` | `HEAVY_LOAD` | 2.0 |
| `STANDARD_DELIVERY` | `BALANCED` | 1.0 |

Delivering a task with an incompatible robot applies a configurable reward penalty.

**80-Dimensional Observation Vector:**

| Indices | Content |
|---|---|
| `[0:71]` | Flat RWARE sensor observations (direction, sensors, location, etc.) |
| `[71:74]` | Robot type one-hot encoding |
| `[74]` | Normalised battery level (0.0–1.0) |
| `[75:80]` | Nearest task: type one-hot (3), weight (1), normalised distance (1) |

**`CommunicationChannel` (patched):**
- Maintains neighbour lookup by Manhattan distance
- Added `drop_prob` parameter — each message has a configurable chance (default 5%) of being dropped before delivery

---

### 4.2 Decision Layer — `network.py` & `eval_bids.py`

**`AgentNetwork` (PyTorch):**

```
Input (80)
    │
    ├─ Actor backbone: Linear(80→128) → Tanh → Linear(128→128) → Tanh
    │       ├─ move_action:  Categorical(5)      [NOOP/FORWARD/LEFT/RIGHT/TOGGLE]
    │       ├─ comm_action:  Bernoulli(1)         [open/close comm gate]
    │       └─ bid_value:    Linear(128→1)+Sigmoid [continuous 0–1 bid]
    │
    └─ Critic: Linear(80→128) → Tanh → Linear(128→1) [value estimate]
```

Uses orthogonal weight initialisation and is shared across all robots (parameter sharing IPPO).

---

### 4.3 Simulation Control Layer — `main_simulation.py`

This file contains and orchestrates the full pipeline. **Not modified during visual work.**

#### A\* Path Planner
- Builds a walkable boolean grid from RWARE's shelf layer
- Runs standard A\* (Manhattan heuristic) from agent position to target
- Converts coordinate waypoints to a queue of RWARE actions:
  - Turn left/right to face each waypoint direction
  - FORWARD to move
  - TOGGLE_LOAD at shelf (pickup) and at return slot (drop-off)

#### Collision Manager
Runs every step, before `env.step()`. If both robots target the same cell:

```
Priority:
  1. Loaded robot (carrying a shelf) has right of way
  2. Higher task priority weight wins
  3. Lower battery robot goes first (needs to finish sooner)
Loser → NOOP (waits one step)
```

Also handles head-on position swaps (both robots trying to swap cells).

#### Decentralised Auction
When an unassigned task appears in the request queue:
- Each robot's network outputs a bid value
- If communication succeeded (in range + not dropped): bids are compared, higher wins
- If communication failed: highest bid still wins (coordinator resolves to prevent double-allocation)
- **Tie-breaking** (in order): capability match → higher battery

#### Fault Detector
- Each alive robot increments a `heartbeat_counter` every step
- Peers monitor each other's last-seen heartbeat value
- After **3 consecutive unchanged values** → failure declared
- On failure: task released to pool, survivor re-auctions it next step

**Configurable CLI options:**

```
--total-steps   Total simulation steps          (default: 200)
--fail-robot    Robot index to fail (0 or 1)    (omit = no fault)
--fail-step     Step at which failure occurs     (default: 50)
--seed          Random seed                      (default: 42)
--comm-drop-prob  Message drop probability       (default: 0.05)
--comm-range    Manhattan comm range             (default: 20)
```

---

### 4.4 Visual Simulation — `visualize_simulation.py`

A completely standalone pygame renderer. **Does not touch any existing file.** Imports all simulation logic directly from `main_simulation.py`.

**What is rendered:**

| Visual Element | Meaning |
|---|---|
| Yellow shelf | Requested — currently up for auction |
| Green shelf | Being carried and delivered |
| Steel-blue shelf | Idle (not in request queue) |
| Teal cell | Delivery goal zone |
| Blue / Orange circle | Robot 0 / Robot 1 with direction arrow |
| Dark red circle | Failed robot |
| Bar under robot | Battery level (green → yellow → red) |

**Right sidebar shows:**
- Per-robot card: type, phase, task, position, battery bar
- Fault injection label (if active)
- Live event log: auctions, comm drops, deliveries, fault detections

**Top HUD shows:**
- Current step / total steps
- Running speed (steps/sec)
- Cumulative stats: auctions, deliveries, comm drops, collisions resolved

---

## 5. Step-by-Step Execution Flow

Every simulation step runs the following phases in order:

```
1. FAULT INJECTION      → (if step == fail_step) mark robot as FAILED, release task
2. HEARTBEATS           → increment live counters; detect misses ≥ 3 → declare failure
3. NETWORK FORWARD PASS → compute bid value and comm gate for each robot
4. COMMUNICATION        → check range, apply drop_prob, record success/fail per pair
5. TASK AUCTION         → for each unassigned queue entry: bid comparison + assign
6. A* ACTION QUEUES     → pop next action from each robot's planned queue
7. COLLISION RESOLUTION → if both robots target same cell → lower priority yields
8. env.step(actions)    → RWARE physics: move, battery drain, delivery detection
9. DELIVERY DETECTION   → compare old vs new request queue; count new deliveries
10. RENDER              → (visual mode only) draw frame
```

---

## 6. How to Run

> **Requirement:** Use **Python 3.12** for all commands (PyTorch is not yet available for Python 3.14 on Windows).

### Install dependencies (first time only)

```bash
py -3.12 -m pip install torch gymnasium rware pygame pytest
```

### Run unit tests (30 tests)
```bash
py -3.12 -m pytest test_hetero.py -v
```

### Run stability tests
```bash
py -3.12 test_stability.py --steps 300 --episodes 2 --robots 2,4
```

### Run headless simulation (terminal output only)
```bash
# Normal — no fault injection
py -3.12 main_simulation.py --total-steps 200

# With fault injection — Robot 0 fails at step 50
py -3.12 main_simulation.py --total-steps 200 --fail-robot 0 --fail-step 50
```

### Run visual simulation (pygame window)
```bash
# Normal run
py -3.12 visualize_simulation.py

# Slow speed for easier watching
py -3.12 visualize_simulation.py --speed 3

# With fault injection
py -3.12 visualize_simulation.py --fail-robot 0 --fail-step 50 --speed 4
```

---

## 7. Controls (Visual Mode)

| Key | Action |
|---|---|
| `SPACE` | Pause / Resume |
| `→` (Right Arrow) | Step one frame (when paused) |
| `+` / `-` | Increase / decrease simulation speed |
| `ESC` or `Q` | Quit |

---

## Verified Output (Fault Injection Run)

```
[Step 0000] AUCTION: Shelf 27 (HEAVY_SHELF) - R0 bid=0.500, R1 bid=0.501 (comm OK)
[Step 0000]   -> Robot 1 wins (higher bid)
[Step 0006] Robot 1 arrived at shelf 27, picking up
[Step 0015] DELIVERY: Shelf 27 delivered successfully!
[Step 0050] *** FAULT INJECTED: Robot 0 has FAILED ***
[Step 0050]   Task (shelf 3) released back to pool
[Step 0052] FAULT: Robot 1 detected Robot 0 has FAILED (3 missed heartbeats)
[Step 0062] DELIVERY: Shelf 8 delivered successfully!
...
  Total deliveries:         4
  Fault injected:           Robot 0 at step 50
```
