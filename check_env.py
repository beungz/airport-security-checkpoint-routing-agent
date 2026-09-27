"""
Sanity checks for AirportSecurityEnv.

    python check_env.py

1. Prints the passenger-type mix implied by the A320 seat layout, and
   checks that every generated episode has < 25% priority passengers + crew.
2. Verifies that minute-by-minute reward accrual and lump-sum-at-completion
   give the same episode totals for both reward versions.
3. Prints lane-load KPIs for the baselines in each evaluation scenario.
"""

import numpy as np

from airport_security_env import (SCENARIOS, TYPE_MIX, AirportSecurityEnv, PassengerType,
                                  generate_passengers)
from baselines import BASELINES


def print_mix():
    """Print the expected passenger-type mix from the A320 layout."""
    parts = ", ".join(f"{t.name.lower()} {100 * TYPE_MIX[t]:.1f}%" for t in PassengerType)
    print(f"A320 layout: priority {100 * (1 - TYPE_MIX[0]):.1f}%  ({parts})")


def check_priority_share(n_episodes=3000):
    """Generate many episodes per scenario and assert priority + crew stays below 25%."""
    for name, sc in SCENARIOS.items():
        rng = np.random.default_rng(0)
        shares = np.array([generate_passengers(rng, sc)[1]["priority_share"] for _ in range(n_episodes)])
        print(f"{name:17s} priority+crew share per episode: mean {100 * shares.mean():.1f}%  "
              f"p99 {100 * np.percentile(shares, 99):.1f}%  max {100 * shares.max():.1f}%")
        assert shares.max() < sc.max_priority_realized


def run(policy_name, scenario, seed, **env_kwargs):
    """Play one episode with a baseline policy and return its episode KPIs."""
    env = AirportSecurityEnv(scenario=scenario, **env_kwargs)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    policy = BASELINES[policy_name]
    done = False
    while not done:
        action = policy(env, rng) if policy_name == "random" else policy(env)
        _, _, done, _, info = env.step(action)
    return info["episode_kpis"]


def check_reward_timing():
    """Assert that accrual and completion timing give the same episode reward totals."""
    worst = 0.0
    for policy_name in ("random", "urgent_jump", "slack_aware"):
        for scenario in ("normal", "rush_hour", "tight_connections"):
            for seed in range(5):
                a = run(policy_name, scenario, seed, reward_timing="accrual")
                c = run(policy_name, scenario, seed, reward_timing="completion")
                for key in ("reward_v1", "reward_v2"):
                    worst = max(worst, abs(a[key] - c[key]) / abs(c[key]))
    print(f"max relative difference, accrual vs completion totals: {worst:.2e}")
    assert worst < 1e-6


def print_loads(n_episodes=40):
    """Print average queue and wait KPIs of the baselines in each evaluation scenario."""
    keys = ["reward_v1", "missed_total", "missed_economy_relaxed", "mean_wait_economy",
            "p95_wait_economy", "max_wait_economy", "long_wait_count", "mean_queue_regular",
            "peak_queue_regular", "mean_wait_sla", "n_passengers"]
    for scenario in ("normal", "rush_hour", "tight_connections"):
        for policy_name in ("shortest_queue", "urgent_jump", "slack_aware"):
            k = [run(policy_name, scenario, 10_000 + s) for s in range(n_episodes)]
            vals = "  ".join(f"{key}={np.mean([x[key] for x in k]):.2f}" for key in keys)
            print(f"[{scenario:17s}] {policy_name:14s} {vals}")


if __name__ == "__main__":
    print_mix()
    check_priority_share()
    check_reward_timing()
    print_loads()
