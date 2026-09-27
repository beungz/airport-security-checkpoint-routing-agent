"""
Train a PPO agent on AirportSecurityEnv.

    python train.py --version v1 --timesteps 1000000
    python train.py --version v2 --timesteps 1000000

Outputs go to runs/<run-name>/: model.zip, vecnormalize.pkl, config.json,
train_log.csv (one row per finished episode: reward + task KPIs) and
learning_curves.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from airport_security_env import AirportSecurityEnv

RUNS_DIR = Path(__file__).parent / "runs"
# ~7 routing decisions per simulated minute; a wait's consequences can play out
# over 15+ minutes, so the discount horizon has to span ~200 decisions.
GAMMA = 0.995


class EpisodeKPICallback(BaseCallback):
    """Records the task KPIs of every episode finished during training."""

    def __init__(self):
        """Start with an empty log; one row is appended per finished episode."""
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        """Collect episode_kpis from any environment that just finished an episode."""
        for info in self.locals["infos"]:
            if "episode_kpis" in info:
                self.rows.append({"timesteps": self.num_timesteps, **info["episode_kpis"]})
        return True


def plot_learning_curves(df: pd.DataFrame, version: str, path: Path):
    """Plot reward and key KPIs over training (raw and rolling mean) to `path`."""
    panels = [
        (f"reward_{version}", f"Episode reward ({version}, the one being optimized)"),
        ("missed_total", "Missed flights per episode"),
        ("missed_economy_relaxed", "Missed flights: economy who arrived with >15 min slack"),
        ("sla_breach_rate", "Business/VIP/crew SLA breach rate (wait > 8 min)"),
        ("long_wait_count", "Passengers waiting > 15 min per episode"),
        ("jump_rate", "Share of economy passengers queue-jumped"),
    ]
    window = max(1, len(df) // 25)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(df["timesteps"], df[col], alpha=0.2, color="#4C72B0")
        ax.plot(df["timesteps"], df[col].rolling(window, min_periods=1).mean(), color="#4C72B0", lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("timesteps")
        ax.grid(alpha=0.3)
    fig.suptitle(f"PPO training -- reward {version}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    """Parse arguments, train PPO, and save the model, logs and plots."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--max-overtakes", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    # Run folder and the environment settings evaluation will need to reload
    run_dir = RUNS_DIR / (args.run_name or args.version)
    run_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(scenario="mixed", reward_version=args.version, reward_timing="accrual",
                      max_overtakes=args.max_overtakes)
    (run_dir / "config.json").write_text(json.dumps({"env_kwargs": env_kwargs, **vars(args)}, indent=2))

    # Parallel environments with running normalization of observations and rewards
    vec_env = make_vec_env(AirportSecurityEnv, n_envs=args.n_envs, seed=args.seed,
                           env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=GAMMA)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=GAMMA,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=0,
    )
    cb = EpisodeKPICallback()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=True)

    # Save everything and print how the KPIs moved from early to late training
    model.save(run_dir / "model")
    vec_env.save(str(run_dir / "vecnormalize.pkl"))
    df = pd.DataFrame(cb.rows)
    df.to_csv(run_dir / "train_log.csv", index=False)
    plot_learning_curves(df, args.version, run_dir / "learning_curves.png")

    n = max(1, len(df) // 10)
    print(f"\n{len(df)} episodes. First 10% vs last 10%:")
    for col in ["reward_v1", "reward_v2", "missed_total", "missed_economy_relaxed", "long_wait_count",
                "sla_breach_rate", "jump_rate", "special_override_rate"]:
        print(f"  {col:24s} {df[col][:n].mean():10.2f} -> {df[col][-n:].mean():10.2f}")
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
