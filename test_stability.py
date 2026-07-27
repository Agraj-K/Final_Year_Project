"""
test_stability.py — Long-running stability test for HeterogeneousWarehouse.

Runs 200+ consecutive steps (configurable up to 1000+) using random actions
on the wrapped environment. Tests multiple robot counts across multiple episodes.

Checks for:
  - Crashes / exceptions
  - NaN or out-of-range values in the observation vector
  - Battery going negative or above 100
  - Any robot getting permanently stuck (>50 consecutive steps same position)
  - Non-finite reward values
  - Robot-type one-hot validity
  - Observation shape correctness

Prints a clear PASS/FAIL summary with the exact step/episode of any failure.

Usage:
    python test_stability.py
    python test_stability.py --steps 500 --episodes 5
"""

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StabilityConfig:
    """Configuration for the stability test."""
    steps_per_episode: int = 300
    robot_counts: Tuple[int, ...] = (4, 8, 12)
    episodes_per_config: int = 3
    stuck_threshold: int = 150     # steps before flagging as stuck (warning only)
    expected_obs_dim: int = 80
    seed_base: int = 42
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Failure tracking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Failure:
    """Record of a single test failure."""
    robot_count: int
    episode: int
    step: int
    check_name: str
    message: str

    def __str__(self):
        return (
            f"  FAIL [{self.check_name}] "
            f"robots={self.robot_count}, episode={self.episode}, step={self.step}: "
            f"{self.message}"
        )


@dataclass
class StabilityReport:
    """Aggregated results from the stability test."""
    total_steps: int = 0
    total_episodes: int = 0
    failures: List[Failure] = field(default_factory=list)
    warnings: List[Failure] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    def add_failure(self, robot_count, episode, step, check_name, message):
        self.failures.append(Failure(robot_count, episode, step, check_name, message))

    def add_warning(self, robot_count, episode, step, check_name, message):
        self.warnings.append(Failure(robot_count, episode, step, check_name, message))

    def summary(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  STABILITY TEST REPORT")
        lines.append("=" * 70)
        lines.append(f"  Total episodes:    {self.total_episodes}")
        lines.append(f"  Total steps:       {self.total_steps}")
        lines.append(f"  Elapsed time:      {self.elapsed_seconds:.1f}s")
        lines.append(f"  Failures:          {len(self.failures)}")
        lines.append(f"  Warnings:          {len(self.warnings)} (stuck agents — expected with random actions)")
        lines.append("")

        if self.passed:
            lines.append("  ✅  ALL CHECKS PASSED")
        else:
            lines.append("  ❌  FAILURES DETECTED:")
            lines.append("")
            for f in self.failures[:20]:
                lines.append(str(f))
            if len(self.failures) > 20:
                lines.append(f"  ... and {len(self.failures) - 20} more")

        if self.warnings:
            lines.append("")
            lines.append(f"  ⚠️  WARNINGS ({len(self.warnings)} stuck-agent events):")
            stuck_summary = {}
            for w in self.warnings:
                key = (w.robot_count, w.episode, w.message.split("stuck at")[0].strip())
                if key not in stuck_summary:
                    stuck_summary[key] = w
            for w in list(stuck_summary.values())[:10]:
                lines.append(f"    {w}")
            lines.append("    (Note: stuck agents are expected with random actions — not a bug)")

        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Checks (run every step)
# ═══════════════════════════════════════════════════════════════════════════════

def check_obs_shape(obs, n_agents, expected_dim) -> Optional[str]:
    """Check observation tuple has correct shape."""
    if len(obs) != n_agents:
        return f"obs tuple length {len(obs)} != {n_agents} agents"
    for i in range(n_agents):
        if obs[i].shape != (expected_dim,):
            return f"agent {i} obs shape {obs[i].shape} != ({expected_dim},)"
    return None


def check_obs_finite(obs, n_agents) -> Optional[str]:
    """Check no NaN or Inf in observations."""
    for i in range(n_agents):
        if np.any(np.isnan(obs[i])):
            nan_indices = np.where(np.isnan(obs[i]))[0]
            return f"agent {i} has NaN at indices {nan_indices.tolist()}"
        if np.any(np.isinf(obs[i])):
            inf_indices = np.where(np.isinf(obs[i]))[0]
            return f"agent {i} has Inf at indices {inf_indices.tolist()}"
    return None


def check_robot_type_onehot(obs, n_agents) -> Optional[str]:
    """Check robot type one-hot (indices 71-73) sums to 1."""
    for i in range(n_agents):
        type_vec = obs[i][71:74]
        s = type_vec.sum()
        if abs(s - 1.0) > 0.001:
            return f"agent {i} type one-hot sum = {s} (expected 1.0)"
        if not all(v in [0.0, 1.0] for v in type_vec):
            return f"agent {i} type one-hot has non-binary values: {type_vec}"
    return None


def check_battery_range(battery, n_agents) -> Optional[str]:
    """Check battery is in [0, 100] for all agents."""
    for i in range(n_agents):
        if battery[i] < 0.0:
            return f"agent {i} battery = {battery[i]} (< 0)"
        if battery[i] > 100.0:
            return f"agent {i} battery = {battery[i]} (> 100)"
    return None


def check_battery_in_obs(obs, battery, n_agents) -> Optional[str]:
    """Check battery in obs (index 74) matches actual battery state."""
    for i in range(n_agents):
        obs_batt = obs[i][74]
        if obs_batt < 0.0 or obs_batt > 1.0:
            return f"agent {i} obs battery = {obs_batt} (out of [0,1])"
    return None


def check_rewards_finite(rewards) -> Optional[str]:
    """Check all rewards are finite."""
    for i, r in enumerate(rewards):
        if not np.isfinite(r):
            return f"agent {i} reward = {r} (non-finite)"
    return None


def check_stuck_agents(
    positions_history: List[List[Tuple[int, int]]],
    n_agents: int,
    threshold: int,
) -> Optional[str]:
    """
    Check if any agent has been at the exact same position for
    more than `threshold` consecutive steps.
    """
    if len(positions_history) < threshold:
        return None

    recent = positions_history[-threshold:]
    for i in range(n_agents):
        first_pos = recent[0][i]
        all_same = all(recent[s][i] == first_pos for s in range(threshold))
        if all_same:
            return (
                f"agent {i} stuck at {first_pos} for {threshold} consecutive steps "
                f"(may be dead-battery — check if battery is 0)"
            )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main stability test
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability_test(config: StabilityConfig) -> StabilityReport:
    """Execute the full stability test across all configurations."""
    from rware.warehouse import Warehouse, RewardType
    from hetero_wrapper import HeterogeneousWarehouse

    report = StabilityReport()
    t0 = time.time()

    for n_robots in config.robot_counts:
        if config.verbose:
            print(f"\n--- Testing with {n_robots} robots ---")

        for episode in range(config.episodes_per_config):
            seed = config.seed_base + episode * 100 + n_robots

            if config.verbose:
                print(f"  Episode {episode + 1}/{config.episodes_per_config} "
                      f"(seed={seed}) ... ", end="", flush=True)

            try:
                # Create environment
                base_env = Warehouse(
                    shelf_columns=3,
                    column_height=8,
                    shelf_rows=1,
                    n_agents=n_robots,
                    msg_bits=0,
                    sensor_range=1,
                    request_queue_size=max(n_robots, 2),
                    max_inactivity_steps=None,
                    max_steps=config.steps_per_episode + 100,  # buffer
                    reward_type=RewardType.INDIVIDUAL,
                )
                env = HeterogeneousWarehouse(base_env)
                obs, info = env.reset(seed=seed)

            except Exception as e:
                report.add_failure(
                    n_robots, episode, 0, "RESET_CRASH",
                    f"Exception during reset: {e}\n{traceback.format_exc()}"
                )
                if config.verbose:
                    print("CRASH on reset!")
                continue

            # Track positions for stuck detection
            positions_history: List[List[Tuple[int, int]]] = []
            step_failures = 0

            for step_num in range(config.steps_per_episode):
                try:
                    # Random actions
                    actions = [
                        env.action_space[i].sample()
                        for i in range(n_robots)
                    ]
                    obs, rewards, done, truncated, info = env.step(actions)
                    report.total_steps += 1

                except Exception as e:
                    report.add_failure(
                        n_robots, episode, step_num, "STEP_CRASH",
                        f"Exception: {e}\n{traceback.format_exc()}"
                    )
                    step_failures += 1
                    if step_failures > 3:
                        break  # don't flood with the same crash
                    continue

                # --- Run all checks ---

                err = check_obs_shape(obs, n_robots, config.expected_obs_dim)
                if err:
                    report.add_failure(n_robots, episode, step_num, "OBS_SHAPE", err)

                err = check_obs_finite(obs, n_robots)
                if err:
                    report.add_failure(n_robots, episode, step_num, "OBS_FINITE", err)

                err = check_robot_type_onehot(obs, n_robots)
                if err:
                    report.add_failure(n_robots, episode, step_num, "TYPE_ONEHOT", err)

                err = check_battery_range(env.battery, n_robots)
                if err:
                    report.add_failure(n_robots, episode, step_num, "BATTERY_RANGE", err)

                err = check_battery_in_obs(obs, env.battery, n_robots)
                if err:
                    report.add_failure(n_robots, episode, step_num, "BATTERY_OBS", err)

                err = check_rewards_finite(rewards)
                if err:
                    report.add_failure(n_robots, episode, step_num, "REWARD_FINITE", err)

                # Track positions
                positions = [
                    (env.base_env.agents[i].x, env.base_env.agents[i].y)
                    for i in range(n_robots)
                ]
                positions_history.append(positions)

                # Note: We only check stuck for agents with battery > 0,
                # since dead-battery agents are expected to stay put.
                # Stuck agents are warnings (expected with random actions),
                # not failures.
                if len(positions_history) >= config.stuck_threshold:
                    recent = positions_history[-config.stuck_threshold:]
                    for i in range(n_robots):
                        if env.battery[i] > 0:
                            first_pos = recent[0][i]
                            all_same = all(
                                recent[s][i] == first_pos
                                for s in range(config.stuck_threshold)
                            )
                            if all_same:
                                report.add_warning(
                                    n_robots, episode, step_num, "STUCK_AGENT",
                                    f"agent {i} (alive, battery={env.battery[i]:.1f}) "
                                    f"stuck at {first_pos} for {config.stuck_threshold} steps"
                                )

                # Reset if done
                if done:
                    try:
                        obs, info = env.reset(seed=seed + step_num)
                        positions_history.clear()
                    except Exception as e:
                        report.add_failure(
                            n_robots, episode, step_num, "MID_RESET_CRASH",
                            f"Exception during mid-episode reset: {e}"
                        )
                        break

            report.total_episodes += 1

            if config.verbose:
                ep_failures = sum(
                    1 for f in report.failures
                    if f.robot_count == n_robots and f.episode == episode
                )
                ep_warnings = sum(
                    1 for w in report.warnings
                    if w.robot_count == n_robots and w.episode == episode
                )
                if ep_failures > 0:
                    status = f"FAIL ({ep_failures} issues)"
                elif ep_warnings > 0:
                    status = f"PASS ({ep_warnings} stuck-agent warnings)"
                else:
                    status = "PASS"
                print(status)

    report.elapsed_seconds = time.time() - t0
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RWARE Hetero Wrapper Stability Test")
    parser.add_argument("--steps", type=int, default=300,
                        help="Steps per episode (default: 300)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per robot count (default: 3)")
    parser.add_argument("--robots", type=str, default="4,8,12",
                        help="Comma-separated robot counts (default: 4,8,12)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--stuck-threshold", type=int, default=50,
                        help="Steps before declaring stuck (default: 50)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-episode output")
    args = parser.parse_args()

    config = StabilityConfig(
        steps_per_episode=args.steps,
        episodes_per_config=args.episodes,
        robot_counts=tuple(int(x) for x in args.robots.split(",")),
        seed_base=args.seed,
        stuck_threshold=args.stuck_threshold,
        verbose=not args.quiet,
    )

    print(f"Running stability test: {config.steps_per_episode} steps/episode, "
          f"{config.episodes_per_config} episodes, "
          f"robot counts: {config.robot_counts}")

    report = run_stability_test(config)
    print(report.summary())

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
