"""
Render an MP4 demo of routing policies side by side on the same scenarios.

    python demo_video.py                                  # ppo_v1 vs ppo_v2
    python demo_video.py --policies slack_aware ppo_v1 ppo_v2 --scenarios rush_hour

Each frame is one routing decision. All panels face the exact same passenger
stream (same seed), so differences come only from the policies.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from airport_security_env import AirportSecurityEnv, PassengerType, REWARD_V1, SLA_TYPES
from evaluate import EVAL_SEED_OFFSET, RESULTS_DIR, load_policies

TYPE_COLORS = {
    PassengerType.ECONOMY: "#9fb4c7",
    PassengerType.BUSINESS: "#e3b448",
    PassengerType.DISABLED: "#9467bd",
    PassengerType.VIP: "#d62728",
    PassengerType.CREW: "#2ca02c",
}
SCENARIO_BLURB = {
    "normal": "Normal traffic: no rush hour, no tight-connection wave.",
    "rush_hour": "Rush hour: a wave of up to +65% extra passengers mid-episode.",
    "tight_connections": "Flight bank: for 15-30 min, 30-50% of arrivals have tight connections.",
    "mixed": "Training distribution: random mix of rush hours and flight banks.",
}
MAX_SHOWN = 40


def _traffic_curve(env: AirportSecurityEnv):
    """Arrival rate over the episode (passengers / hour) for the timeline plot."""
    meta, sc = env.scenario_meta, env.scenario
    t = np.linspace(0, sc.horizon, 300)
    rate = np.full_like(t, meta["rate"])
    if meta["rush"]:
        r = meta["rush"]
        rate *= 1 + r["amp"] * np.exp(-0.5 * ((t - r["center"]) / r["width"]) ** 2)
    return t, rate * 60


def _running_stats(env: AirportSecurityEnv):
    """Counts so far in the episode (missed flights, long waits, SLA breaches,
    jumps, special-lane overrides) and both running rewards."""
    done = env._completed
    missed = sum(p.completion_time - p.arrival_time > p.time_budget for p in done)
    sla = [p for p in done if p.ptype in SLA_TYPES]
    breaches = sum(p.service_start - p.arrival_time > REWARD_V1.sla_max_wait for p in sla)
    long_waits = sum(p.service_start - p.arrival_time > REWARD_V1.max_wait for p in done)
    eco = [p for p in env.passengers[:env._next_idx] if p.ptype == PassengerType.ECONOMY]
    jumps = sum(p.jumped for p in eco)
    overrides = sum(p.lane == env.special_lane_idx for p in eco)
    return dict(missed=missed, breaches=breaches, n_sla=len(sla), jumps=jumps, long_waits=long_waits,
                overrides=overrides, n_eco=len(eco), r1=env._episode_reward["v1"],
                r2=env._episode_reward["v2"])


def _draw_lanes(ax, env: AirportSecurityEnv, title: str, last_action):
    """Draw one policy's panel: every lane's queue as colored squares, the
    passenger just routed, and running stats in the title."""
    # Axes and lane labels
    ax.clear()
    ax.set_xlim(-2.2, MAX_SHOWN + 2.5)
    ax.set_ylim(-0.8, env.n_lanes - 0.2)
    ax.set_yticks(range(env.n_lanes),
                  [f"Regular {i + 1}" for i in range(env.n_regular_lanes)] + ["SPECIAL"])
    ax.invert_yaxis()
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    # Queues: fill = passenger type, outline = projected risk of missing the flight
    placed = last_action[0] if last_action else None
    for li, lane in enumerate(env.lanes):
        ax.add_patch(plt.Rectangle((-1.9, li - 0.35), 1.2, 0.7, color="#444", zorder=1))
        slacks = env._projected_slacks(lane)
        xs, fc, ec, lw = [], [], [], []
        for k, (p, s) in enumerate(zip(list(lane.queue)[:MAX_SHOWN], slacks)):
            xs.append(k)
            fc.append(TYPE_COLORS[p.ptype])
            if s < 0:
                ec.append("#d00000"); lw.append(2.5)
            elif s < 10:
                ec.append("#ff9f1c"); lw.append(2.0)
            else:
                ec.append("white"); lw.append(0.5)
            if p is placed:
                ax.scatter([k], [li], s=300, facecolors="none", edgecolors="black", linewidths=2.5, zorder=4)
        if xs:
            ax.scatter(xs, [li] * len(xs), s=110, marker="s", c=fc, edgecolors=ec, linewidths=lw, zorder=3)
        if len(lane.queue) > MAX_SHOWN:
            ax.text(MAX_SHOWN + 0.3, li, f"+{len(lane.queue) - MAX_SHOWN}", va="center", fontsize=9)
        ax.text(MAX_SHOWN + 2.4, li, f"~{env.expected_wait(lane):4.1f} min", va="center", ha="right",
                fontsize=8, color="#555")

    # Title with running stats, and a caption describing the latest decision
    st = _running_stats(env)
    if placed is not None:
        _, lane_idx, jump = last_action
        lane_name = "SPECIAL" if lane_idx == env.special_lane_idx else f"Regular {lane_idx + 1}"
        decision = (f"Economy pax, {placed.time_budget:.0f} min to spare -> {lane_name}"
                    f"{'  (JUMP to front)' if jump else ''}")
    else:
        decision = "Draining remaining queues"
    ax.set_title(
        f"{title}\n"
        f"missed flights: {st['missed']}   waits >{REWARD_V1.max_wait:.0f} min: {st['long_waits']}   "
        f"SLA breaches: {st['breaches']}/{st['n_sla']}\n"
        f"jumps: {st['jumps']}/{st['n_eco']}   special overrides: {st['overrides']}   "
        f"reward so far  v1: {st['r1']:,.0f}   v2: {st['r2']:,.0f}",
        fontsize=10, loc="left",
    )
    ax.text(0, env.n_lanes - 0.3, decision, fontsize=9, style="italic", va="top")


def _fig_to_array(fig):
    """Render a matplotlib figure to an RGB image array (one video frame)."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _title_card(fig, lines):
    """Frame with a large first line and smaller lines below (scenario intro)."""
    fig.clf()
    fig.text(0.5, 0.6, lines[0], ha="center", fontsize=24, weight="bold")
    for i, line in enumerate(lines[1:]):
        fig.text(0.5, 0.45 - 0.07 * i, line, ha="center", fontsize=14)
    return _fig_to_array(fig)


def make_demo_video(policies: dict, scenarios=("normal", "rush_hour", "tight_connections"),
                    seed: int = EVAL_SEED_OFFSET + 3, out_path=RESULTS_DIR / "demo.mp4",
                    fps: int = 8, frame_skip: int = 6, title_seconds: float = 2.5):
    """
    policies  : name -> (policy_callable, env_kwargs), e.g. from evaluate.load_policies
    scenarios : scenario names from airport_security_env.SCENARIOS, one clip each
    seed      : passenger-stream seed (same for every policy within a scenario)
    frame_skip: render every n-th decision (1 = every decision)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(policies)
    fig = plt.figure(figsize=(7.2 * len(names), 6.2), dpi=90)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", macro_block_size=1)

    for scenario in scenarios:
        # One environment per policy, all on the same seed
        envs = []
        for name in names:
            env = AirportSecurityEnv(scenario=scenario, **policies[name][1])
            env.reset(seed=seed)
            envs.append(env)

        card = _title_card(fig, [f"Scenario: {scenario}", SCENARIO_BLURB.get(scenario, ""),
                                 "Panels: " + "  |  ".join(names)])
        for _ in range(int(fps * title_seconds)):
            writer.append_data(card)

        # Layout: traffic timeline on top, one lane panel per policy below
        fig.clf()
        gs = fig.add_gridspec(2, len(names), height_ratios=[1, 5], hspace=0.55,
                              left=0.07, right=0.98, top=0.93, bottom=0.1)
        ax_t = fig.add_subplot(gs[0, :])
        lane_axes = [fig.add_subplot(gs[1, i]) for i in range(len(names))]
        t, rate = _traffic_curve(envs[0])
        bank = envs[0].scenario_meta["bank"]
        legend = [Patch(color=c, label=pt.name.title()) for pt, c in TYPE_COLORS.items()]
        legend += [Line2D([], [], marker="s", ls="", mfc="white", mec="#ff9f1c", mew=2, label="at risk (<10 min)"),
                   Line2D([], [], marker="s", ls="", mfc="white", mec="#d00000", mew=2, label="projected to miss"),
                   Line2D([], [], marker="o", ls="", mfc="none", mec="black", mew=2, label="just routed")]
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=9, frameon=False)

        # Step all policies in lockstep; draw a frame every `frame_skip` decisions
        dones = [False] * len(envs)
        step = 0
        while not all(dones):
            for i, (name, env) in enumerate(zip(names, envs)):
                if not dones[i]:
                    _, _, dones[i], _, _ = env.step(policies[name][0](env))
            step += 1
            if step % frame_skip and not all(dones):
                continue
            ax_t.clear()
            ax_t.plot(t, rate, color="#4C72B0")
            if bank:
                ax_t.axvspan(bank["start"], bank["end"], color="#ff9f1c", alpha=0.25, label="tight-connection wave")
                ax_t.legend(fontsize=8, loc="upper right")
            ax_t.axvline(envs[0].clock, color="black", lw=1.5)
            ax_t.set_xlim(0, t[-1])
            ax_t.set_ylim(0, rate.max() * 1.25)
            ax_t.set_ylabel("pax / hour", fontsize=8)
            ax_t.set_title(f"{scenario}   clock {envs[0].clock:5.1f} min", fontsize=10, loc="left")
            ax_t.tick_params(labelsize=8)
            for ax, name, env in zip(lane_axes, names, envs):
                _draw_lanes(ax, env, name, env._last_action if env._current is not None else None)
            writer.append_data(_fig_to_array(fig))

        # Hold the final state on screen for a moment
        frame = _fig_to_array(fig)
        for _ in range(int(fps * title_seconds)):
            writer.append_data(frame)

    writer.close()
    plt.close(fig)
    print(f"Saved demo video to {out_path}")
    return out_path


def main():
    """Parse arguments, load the requested policies, and render the video."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=["ppo_v1", "ppo_v2"],
                    help="baseline names and/or ppo_<run-name>")
    ap.add_argument("--scenarios", nargs="*", default=["normal", "rush_hour", "tight_connections"])
    ap.add_argument("--seed", type=int, default=EVAL_SEED_OFFSET + 3)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--frame-skip", type=int, default=6)
    ap.add_argument("--out", default=str(RESULTS_DIR / "demo.mp4"))
    args = ap.parse_args()

    runs = [p.removeprefix("ppo_") for p in args.policies if p.startswith("ppo_")]
    available = load_policies(runs)
    missing = [p for p in args.policies if p not in available]
    if missing:
        raise SystemExit(f"unknown/missing policies: {missing}. Available: {list(available)}")
    make_demo_video({p: available[p] for p in args.policies}, args.scenarios, seed=args.seed,
                    out_path=args.out, fps=args.fps, frame_skip=args.frame_skip)


if __name__ == "__main__":
    main()
