"""
Evaluate trained PPO agents and hand-written baselines on held-out seeds.

    python evaluate.py --runs v1 v2 --episodes 100

Every policy sees exactly the same passenger streams (same seeds), in each
scenario. Both reward versions are reported for every policy, next to the
task-level KPIs, so you can see when a policy scores well on its reward while
doing worse on what the airport actually cares about.
Outputs: results/eval_episodes.csv, results/eval_summary.csv,
results/kpi_comparison.png
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from airport_security_env import AirportSecurityEnv
from baselines import BASELINES

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
EVAL_SEED_OFFSET = 100_000          # training never uses these seeds
EVAL_SCENARIOS = ["normal", "rush_hour", "tight_connections"]

SUMMARY_KPIS = [
    "reward_v1", "reward_v2", "missed_total", "missed_economy_urgent", "missed_economy_relaxed",
    "missed_disabled", "mean_wait_economy", "p95_wait_economy", "max_wait_economy", "long_wait_count",
    "mean_wait_sla", "sla_breach_rate", "jump_rate", "special_override_rate",
    "mean_overtaken_economy", "max_overtaken",
]


class PPOPolicy:
    """Wraps a saved PPO run as a `policy(env) -> action` callable."""

    def __init__(self, run_dir: Path):
        """Load the model, its observation normalizer and its env settings from run_dir."""
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / "model", device="cpu")
        with open(run_dir / "vecnormalize.pkl", "rb") as f:
            self.vecnorm = pickle.load(f)
        self.env_kwargs = json.loads((run_dir / "config.json").read_text())["env_kwargs"]

    def __call__(self, env: AirportSecurityEnv) -> int:
        """Deterministic action for the environment's current passenger."""
        obs = self.vecnorm.normalize_obs(env._get_obs()[None])
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action[0])


def load_policies(run_names):
    """Baselines plus any PPO runs found under runs/. Returns name -> (policy, env_kwargs)."""
    policies = {name: (fn, {}) for name, fn in BASELINES.items()}
    for name in run_names:
        run_dir = RUNS_DIR / name
        if (run_dir / "model.zip").exists():
            pol = PPOPolicy(run_dir)
            env_kwargs = {k: v for k, v in pol.env_kwargs.items() if k != "scenario"}
            policies[f"ppo_{name}"] = (pol, env_kwargs)
        else:
            print(f"skipping {name}: no model at {run_dir}")
    return policies


def run_episode(policy, scenario: str, seed: int, env_kwargs=None) -> dict:
    """Play one full episode with `policy` and return its episode KPIs."""
    env = AirportSecurityEnv(scenario=scenario, **(env_kwargs or {}))
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, _, info = env.step(policy(env))
    return info["episode_kpis"]


def evaluate(policies, scenarios=EVAL_SCENARIOS, n_episodes=100) -> pd.DataFrame:
    """Run every policy on the same held-out seeds in each scenario.
    Returns one row of KPIs per (policy, scenario, episode)."""
    rows = []
    for scenario in scenarios:
        for name, (policy, env_kwargs) in policies.items():
            for i in range(n_episodes):
                kpis = run_episode(policy, scenario, EVAL_SEED_OFFSET + i, env_kwargs)
                rows.append({"policy": name, "scenario": scenario, "seed": i, **kpis})
            print(f"  done {scenario:18s} {name}")
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, path: Path):
    """Grouped bar chart of mean KPIs (with SEM error bars) per scenario and policy."""
    panels = [
        ("reward_v1", "Reward v1 (higher = better)"),
        ("reward_v2", "Reward v2 (higher = better)"),
        ("missed_total", "Missed flights / episode"),
        ("missed_economy_relaxed", "Missed: economy who arrived with >15 min"),
        ("mean_wait_economy", "Economy mean wait (min)"),
        ("max_wait_economy", "Longest economy wait (min)"),
        ("long_wait_count", "Passengers waiting > 15 min / episode"),
        ("max_overtaken", "Most times one passenger was overtaken"),
        ("sla_breach_rate", "Business/VIP/crew SLA breach rate"),
        ("mean_wait_sla", "Business/VIP/crew mean wait (min)"),
        ("special_override_rate", "Economy sent to special lane (share)"),
    ]
    policies = [p for p in summary.index.get_level_values("policy").unique() if p != "random"]
    scenarios = list(summary.index.get_level_values("scenario").unique())
    x = np.arange(len(scenarios))
    width = 0.8 / len(policies)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (col, title) in zip(axes.flat, panels):
        for j, pol in enumerate(policies):
            vals = [summary.loc[(sc, pol), (col, "mean")] for sc in scenarios]
            errs = [summary.loc[(sc, pol), (col, "sem")] for sc in scenarios]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width, yerr=errs,
                   label=pol, color=colors[j % 10], capsize=2)
        ax.set_xticks(x, scenarios, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Held-out evaluation (random policy omitted for scale; error bars = SEM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    """Parse arguments, evaluate all policies, and save the CSVs and plot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["v1", "v2"])
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    policies = load_policies(args.runs)
    df = evaluate(policies, n_episodes=args.episodes)
    df.to_csv(RESULTS_DIR / "eval_episodes.csv", index=False)

    summary = df.groupby(["scenario", "policy"])[SUMMARY_KPIS].agg(["mean", "sem"])
    summary.to_csv(RESULTS_DIR / "eval_summary.csv")
    plot_comparison(summary, RESULTS_DIR / "kpi_comparison.png")

    means = df.groupby(["scenario", "policy"])[SUMMARY_KPIS].mean().round(2)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(means)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
