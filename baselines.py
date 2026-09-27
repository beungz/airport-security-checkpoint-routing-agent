"""
Hand-written routing policies for economy passengers, used as reference
points for the PPO agents. Each policy is a callable `policy(env) -> action`
that reads the (unwrapped) environment state directly.
"""

import numpy as np

from airport_security_env import AirportSecurityEnv, REWARD_V1

REGULAR, SPECIAL = 0, 1


def _encode(target: int, jump: bool) -> int:
    """Turn (REGULAR/SPECIAL, jump) into the environment's Discrete(4) action."""
    return target * 2 + int(jump)


def random_policy(env: AirportSecurityEnv, rng=np.random.default_rng(0)) -> int:
    """Uniformly random action (lane type and jump)."""
    return int(rng.integers(env.action_space.n))


def shortest_queue(env: AirportSecurityEnv) -> int:
    """Back of the shortest regular lane; never jumps, never uses the special lane."""
    return _encode(REGULAR, False)


def urgent_jump(env: AirportSecurityEnv) -> int:
    """Shortest regular lane; urgent passengers (budget <= threshold) jump."""
    return _encode(REGULAR, env._current.time_budget <= REWARD_V1.urgency_threshold)


def slack_aware(env: AirportSecurityEnv) -> int:
    """A careful human supervisor.

    - Send to the back of the shortest regular lane if the passenger still
      makes it with a margin.
    - Otherwise let them jump in that lane, as long as nobody waiting there
      would then be projected to miss their own flight.
    - Only if that still is not enough, use the special lane (front), and
      only while the special lane's expected wait leaves room under the SLA.
    """
    p = env._current
    lane = env.lanes[env.shortest_regular_lane()]
    margin = 3.0

    if p.time_budget - env.expected_wait(lane) - p.expected_service > margin:
        return _encode(REGULAR, False)

    slacks = env._projected_slacks(lane)[1:]
    jump_ok = all(s - p.expected_service > 0 for s in slacks)
    if jump_ok and p.time_budget - env.expected_wait(lane, front=True) - p.expected_service > 0:
        return _encode(REGULAR, True)

    special = env.lanes[env.special_lane_idx]
    if env.expected_wait(special) + p.expected_service < REWARD_V1.sla_max_wait - 2.0:
        return _encode(SPECIAL, True)
    return _encode(REGULAR, jump_ok)


BASELINES = {
    "random": random_policy,
    "shortest_queue": shortest_queue,
    "urgent_jump": urgent_jump,
    "slack_aware": slack_aware,
}
