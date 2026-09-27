"""
airport_security_env.py

Discrete-event, stochastic, procedurally generated Gymnasium environment for
routing ECONOMY passengers at an airport security checkpoint.

SETTING
A scanner reads every passenger's boarding pass on arrival. Passengers
eligible for the special lane (business, VIP, crew, disabled/elderly) are
ALWAYS sent to the back of the special lane -- that is a hard airport rule,
not an agent decision. The agent is only queried for ECONOMY passengers and
chooses:

    (a) which lane:  regular (staff pick the shortest regular lane), or the
                     special lane (override)
    (b) where:       back of the queue, or front of the waiting line (jump)

Because boarding passes are scanned, the agent can see who is waiting in
every lane (how many are at risk of missing their flight, tightest slack).

PROCEDURAL GENERATION
Every episode is a freshly generated scenario: the full passenger stream
(arrival times, types, time budgets, screening times) is sampled at reset()
from randomized scenario parameters - base traffic level, passenger-type mix
(based on A320 seat layout), an optional rush-hour wave, and an
optional "flight bank" window in which many passengers arrive with tight
connections. Because the stream is fixed at
reset, two policies run on the same seed face exactly the same passengers.

REWARD VERSIONS
Each passenger's cost is a function of their wait, lateness and type. By
default it is charged minute by minute while they are in the system
(reward_timing="accrual"); reward_timing="completion" charges the same total
in one lump when they finish screening. The episode drains all lanes at the
end so nobody escapes scoring. Two versions are computed for
every passenger; `reward_version` selects which one is returned as the step
reward, the other is still reported in `info` for evaluation.

  v1 -- the original design (see _reward_v1)
  v2 -- the mitigated design (see _reward_v2)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Passenger model

class PassengerType(IntEnum):
    """Passenger categories, read from the boarding pass. Everyone except
    ECONOMY is eligible for (and always sent to) the special lane."""

    ECONOMY = 0
    BUSINESS = 1
    DISABLED = 2   # disabled / elderly
    VIP = 3
    CREW = 4


N_TYPES = len(PassengerType)
PRIORITY_TYPES = (PassengerType.BUSINESS, PassengerType.DISABLED,
                  PassengerType.VIP, PassengerType.CREW)
SLA_TYPES = (PassengerType.BUSINESS, PassengerType.VIP, PassengerType.CREW)

# Head-counts per departing flight on a typical two-class A320: 12 business +
# 150 economy, 2 pilots + 4 cabin crew. Disabled/elderly passengers are assumed
# to occupy the front economy row; no first class, so no VIPs.
A320_LAYOUT = {PassengerType.ECONOMY: 150 - 6, PassengerType.BUSINESS: 12,
               PassengerType.DISABLED: 6, PassengerType.VIP: 0, PassengerType.CREW: 6}
_counts = np.array([A320_LAYOUT[t] for t in PassengerType], dtype=float)
TYPE_MIX = _counts / _counts.sum()   # passenger-type probabilities

# Expected screening time (minutes) per type - what the scanner "knows".
# 0.40 min per economy passenger : approx 150 passengers / hour / lane.
MEAN_SERVICE = {
    PassengerType.ECONOMY: 0.40,
    PassengerType.BUSINESS: 0.32,
    PassengerType.DISABLED: 0.48,
    PassengerType.VIP: 0.32,
    PassengerType.CREW: 0.28,
}
SERVICE_CV = 0.4                     # coefficient of variation of screening time
MIN_SERVICE = 0.1

GATE_WALK_MIN = np.array([3.0, 10.0, 18.0])   # near / far / APM-required
GATE_WALK_P = np.array([0.5, 0.3, 0.2])
MOBILITY_EXTRA_MIN = 10.0                     # disabled/elderly walk slower


@dataclass
class Passenger:
    """One passenger: fixed attributes sampled at reset, plus timing and
    routing fields filled in by the simulation as they move through."""

    id: int
    ptype: PassengerType
    arrival_time: float
    time_budget: float        # minutes (from arrival) until screening must be DONE
    service_time: float       # actual screening time (hidden from the agent)
    gate_walk: float
    service_start: float = -1.0
    completion_time: float = -1.0
    lane: int = -1
    jumped: bool = False
    n_overtaken: int = 0

    @property
    def is_priority(self) -> bool:
        """True if this passenger must use the special lane."""
        return self.ptype in PRIORITY_TYPES

    @property
    def expected_service(self) -> float:
        """Mean screening time for this passenger's type (the actual time is hidden)."""
        return MEAN_SERVICE[self.ptype]


@dataclass
class Lane:
    """A security lane: a queue whose head is being screened."""

    is_special: bool
    queue: deque = field(default_factory=deque)   # index 0 = in service
    remaining_service: float = 0.0


# Scenario generation

@dataclass
class ScenarioConfig:
    """Parameters from which each episode's traffic is randomly drawn.
    Tuples are (low, high) ranges sampled uniformly per episode."""

    horizon: float = 120.0          # minutes of arrivals, then drain
    base_rate: float = 8.5          # passengers / minute (~510 / hour, ~97% of regular-lane capacity)
    rate_jitter: float = 0.05       # base rate multiplied by U(1-j, 1+j)
    p_rush: float = 0.5             # chance of a rush-hour wave
    rush_amp: tuple = (0.35, 0.65)  # peak extra traffic (fraction of base)
    rush_width: tuple = (10.0, 18.0)
    p_bank: float = 0.3             # chance of a tight-connection flight bank
    bank_tight_share: tuple = (0.3, 0.5)
    bank_duration: tuple = (15.0, 30.0)
    mix_concentration: float = 300.0
    # With 4 lanes, the special lane only has value if priority + crew make up
    # less than a quarter of all passengers.
    max_priority_mix: float = 0.22      # cap on the expected share
    max_priority_realized: float = 0.25  # hard bound on each episode's actual share


SCENARIOS = {
    "mixed": ScenarioConfig(),                                   # training distribution
    "normal": ScenarioConfig(p_rush=0.0, p_bank=0.0),
    "rush_hour": ScenarioConfig(p_rush=1.0, p_bank=0.0),
    "tight_connections": ScenarioConfig(p_rush=0.0, p_bank=1.0),
}


def generate_passengers(rng: np.random.Generator, sc: ScenarioConfig):
    """Sample one episode's full passenger stream (non-homogeneous Poisson).

    Returns (passengers sorted by arrival time, meta dict describing the
    sampled scenario: base rate, rush wave, flight bank, type mix and the
    realized priority share).
    """
    # Scenario-level randomness: traffic level, rush wave, flight bank
    rate = sc.base_rate * rng.uniform(1 - sc.rate_jitter, 1 + sc.rate_jitter)

    rush = None
    if rng.random() < sc.p_rush:
        rush = dict(center=rng.uniform(25, sc.horizon - 25),
                    width=rng.uniform(*sc.rush_width),
                    amp=rng.uniform(*sc.rush_amp))
    bank = None
    if rng.random() < sc.p_bank:
        dur = rng.uniform(*sc.bank_duration)
        start = rng.uniform(10, sc.horizon - dur - 10)
        bank = dict(start=start, end=start + dur, share=rng.uniform(*sc.bank_tight_share))

    # Passenger-type mix: jittered around the A320 layout, priority share capped
    rng.random()   # unused draw; keeps each seed's passenger stream identical to the saved results
    mix = rng.dirichlet(TYPE_MIX * sc.mix_concentration)
    priority = 1.0 - mix[PassengerType.ECONOMY]
    if priority > sc.max_priority_mix:
        mix[1:] *= sc.max_priority_mix / priority
        mix[PassengerType.ECONOMY] = 1.0 - sc.max_priority_mix

    def rate_at(t):
        """Arrival rate (passengers / minute) at time t, including any rush wave."""
        if rush is None:
            return rate
        return rate * (1 + rush["amp"] * np.exp(-0.5 * ((t - rush["center"]) / rush["width"]) ** 2))

    # Arrivals by thinning: propose at the peak rate, keep with prob rate_at(t) / peak
    lam_max = rate * (1 + (rush["amp"] if rush else 0.0))
    k = 1.0 / SERVICE_CV ** 2
    while True:   # resample until this episode's actual priority share is below the bound
        passengers, t = [], 0.0
        while True:
            t += rng.exponential(1.0 / lam_max)
            if t > sc.horizon:
                break
            if rng.random() > rate_at(t) / lam_max:
                continue
            # Per-passenger attributes: type, time budget, screening time
            ptype = PassengerType(rng.choice(N_TYPES, p=mix))
            in_bank = bank is not None and bank["start"] <= t <= bank["end"] and rng.random() < bank["share"]
            minutes_to_gate_close = rng.uniform(15, 35) if in_bank else rng.uniform(15, 100)
            walk = rng.choice(GATE_WALK_MIN, p=GATE_WALK_P)
            mobility = MOBILITY_EXTRA_MIN if ptype == PassengerType.DISABLED else 0.0
            budget = max(2.0, minutes_to_gate_close - walk - mobility)
            service = max(MIN_SERVICE, rng.gamma(k, MEAN_SERVICE[ptype] / k))
            passengers.append(Passenger(id=len(passengers), ptype=ptype, arrival_time=t,
                                        time_budget=budget, service_time=service, gate_walk=walk))
        realized = np.mean([p.is_priority for p in passengers])
        if realized < sc.max_priority_realized:
            break

    meta = dict(rate=rate, rush=rush, bank=bank, mix=mix, priority_share=float(realized))
    return passengers, meta


# Reward configuration

@dataclass
class RewardConfig:
    """Weights of the per-passenger cost terms. v1 uses the defaults; v2
    (REWARD_V2 below) switches on the extra terms marked "v2 only"."""

    w1_base: float = 1.0                  # term 1: every minute of wait
    urgency_threshold: float = 15.0       # term 2: "soon departure" if slack <= this
    urgency_cap: float = 4.0              #         max urgency multiplier
    w2_urgency: float = 2.5
    w2_missed_flight: float = 40.0        #         one-off penalty for a missed flight
    w2_late_per_min: float = 0.0          #         v2 only: keeps charging after the miss
    w3_disabled: float = 2.0              # term 3: extra weight on disabled/elderly wait
    sla_max_wait: float = 8.0             # term 4: service-level cap for business/VIP/crew
    w4_sla_overage: float = 3.0           #         per minute over the cap
    w4_sla_breach: float = 0.0            #         v2 only: one-off per breach
    max_wait: float = 15.0                # term 5 (v2 only): wait limit for EVERY passenger
    w5_long_wait_overage: float = 0.0     #         per minute over the limit
    w5_long_wait_breach: float = 0.0      #         one-off per passenger over the limit


REWARD_V1 = RewardConfig()
REWARD_V2 = replace(REWARD_V1, w2_late_per_min=5.0, w4_sla_overage=10.0, w4_sla_breach=20.0,
                    w5_long_wait_overage=5.0, w5_long_wait_breach=10.0)

BREAKDOWN_KEYS = ("base", "urgency", "missed_flight", "disabled", "sla", "long_wait")


# Environment

class AirportSecurityEnv(gym.Env):
    """
    Action space: Discrete(4).
        target, jump = divmod(action, 2); target 0 = regular lanes, 1 = special
        lane. jump == 1 -> insert at the front of the waiting line (behind
        whoever is currently being screened).
        Which regular lane is not the agent's choice: staff send the passenger
        to the regular lane with the shortest expected wait. When all regular
        lanes are busy, total wait is nearly the same whichever lane a
        passenger joins, so the reward gives almost no signal to balance lanes
        and a learned lane choice ends up close to random.
        With max_overtakes=K, a jump is turned into a back-of-queue insert
        if anyone waiting in that lane has already been overtaken K times,
        unless the jump is what saves the jumper's flight.

    Observation (float32 vector):
        per lane   : queue length, expected wait at back, #waiting at risk
                     (projected slack < 10 min), #waiting projected to miss,
                     tightest projected slack (clipped)
        special    : longest current wait among waiting business/VIP/crew,
                     projected wait at back minus SLA cap
        arriving   : time budget, urgent flag, projected slack if sent to the
                     back / front of the shortest regular lane and of the
                     special lane
        global     : arrivals in the last 10 min (per min), normalized clock
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_regular_lanes: int = 3,
        scenario: str | ScenarioConfig = "mixed",
        reward_version: str = "v1",
        reward_timing: str = "accrual",
        max_overtakes: int | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        """
        n_regular_lanes : number of regular lanes (plus one special lane)
        scenario        : a key of SCENARIOS or a custom ScenarioConfig
        reward_version  : "v1" or "v2" -- which reward is returned by step()
        reward_timing   : "accrual" (charged minute by minute) or "completion"
                          (charged in one lump when screening ends)
        max_overtakes   : optional overtake limit K (None = no limit)
        seed            : seed for the passenger-stream generator
        """
        super().__init__()
        self.n_regular_lanes = n_regular_lanes
        self.n_lanes = n_regular_lanes + 1
        self.special_lane_idx = n_regular_lanes
        self.scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.scenario_name = scenario if isinstance(scenario, str) else "custom"
        assert reward_version in ("v1", "v2")
        self.reward_version = reward_version
        assert reward_timing in ("accrual", "completion")
        self.reward_timing = reward_timing
        self.max_overtakes = max_overtakes
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        obs_dim = self.n_lanes * 5 + 2 + 6 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)

    # Episode lifecycle
    def reset(self, *, seed=None, options=None):
        """Generate a new passenger stream and run until the first economy
        passenger arrives. Returns (observation, info)."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.passengers, self.scenario_meta = generate_passengers(self._rng, self.scenario)
        self.clock = 0.0
        self.lanes = [Lane(is_special=False) for _ in range(self.n_regular_lanes)]
        self.lanes.append(Lane(is_special=True))
        self._next_idx = 0
        self._arrival_times = deque()
        self._completed: list[Passenger] = []
        self._pending = self._empty_breakdown()
        self._episode_reward = {"v1": 0.0, "v2": 0.0}
        self._last_action = None
        self._n_decisions = 0
        self._queue_sum = 0.0
        self._queue_peak = 0
        self._current: Passenger | None = None

        done = self._advance_until_decision()
        assert not done, "scenario produced no economy passengers"
        return self._get_obs(), {}

    def step(self, action: int):
        """Route the current economy passenger, then simulate until the next
        one arrives (or all lanes drain). Returns the usual Gymnasium tuple;
        info carries both reward versions and, at the end, episode_kpis."""
        # Apply the routing decision
        target, jump = divmod(int(action), 2)
        lane_idx = self.special_lane_idx if target == 1 else self.shortest_regular_lane()
        p = self._current

        if jump and self.max_overtakes is not None:
            lane = self.lanes[lane_idx]
            protected = any(q.n_overtaken >= self.max_overtakes for q in list(lane.queue)[1:])
            if protected and not self._jump_saves_flight(p, lane):
                jump = 0
        self._enqueue(lane_idx, p, bool(jump))
        self._last_action = (p, lane_idx, bool(jump))
        self._n_decisions += 1
        regular = [len(l.queue) for l in self.lanes[:self.n_regular_lanes]]
        self._queue_sum += sum(regular) / len(regular)
        self._queue_peak = max(self._queue_peak, max(regular))

        # Simulate to the next decision and collect the reward accrued meanwhile
        done = self._advance_until_decision()
        bd, self._pending = self._pending, self._empty_breakdown()
        reward = sum(bd[self.reward_version][k] for k in BREAKDOWN_KEYS)
        info = {
            "reward_breakdown": dict(bd[self.reward_version]),
            "reward_v1": sum(bd["v1"][k] for k in BREAKDOWN_KEYS),
            "reward_v2": sum(bd["v2"][k] for k in BREAKDOWN_KEYS),
        }
        self._episode_reward["v1"] += info["reward_v1"]
        self._episode_reward["v2"] += info["reward_v2"]
        if done:
            self._current = None
            info["episode_kpis"] = self.episode_kpis()
            return self._get_obs(), reward, True, False, info
        return self._get_obs(), reward, False, False, info

    # Simulation
    def _advance_until_decision(self) -> bool:
        """Run the sim to the next economy arrival (auto-routing priority
        passengers on the way). Returns True once the episode is drained."""
        while self._next_idx < len(self.passengers):
            p = self.passengers[self._next_idx]
            self._run_until(p.arrival_time)
            self._next_idx += 1
            self._arrival_times.append(p.arrival_time)
            if p.is_priority:
                self._enqueue(self.special_lane_idx, p, jump=False)
                continue
            self._current = p
            return False
        self._run_until(None)
        return True

    def _run_until(self, t_target: float | None):
        """Advance the clock to t_target (None = drain every lane)."""
        while True:
            best, best_dt = None, np.inf
            for i, lane in enumerate(self.lanes):
                if lane.queue and lane.remaining_service < best_dt:
                    best, best_dt = i, lane.remaining_service
            if best is None:
                if t_target is not None:
                    self.clock = t_target
                return
            if t_target is not None and self.clock + best_dt > t_target:
                self._tick(t_target - self.clock)
                return
            self._tick(best_dt)
            self._complete_head(best)

    def _tick(self, dt: float):
        """Advance the clock by dt minutes with no arrivals or completions in between."""
        if self.reward_timing == "accrual" and dt > 0:
            self._accrue(dt)
        for lane in self.lanes:
            if lane.queue:
                lane.remaining_service -= dt
        self.clock += dt

    def _enqueue(self, lane_idx: int, p: Passenger, jump: bool):
        """Add a passenger to a lane: at the back, or (jump) right behind the
        person being screened, counting an overtake for everyone pushed back."""
        lane = self.lanes[lane_idx]
        p.lane = lane_idx
        if not lane.queue:
            lane.queue.append(p)
            self._start_service(lane)
        elif jump and len(lane.queue) > 1:
            lane.queue.insert(1, p)
            p.jumped = True
            for q in list(lane.queue)[2:]:
                q.n_overtaken += 1
        else:
            lane.queue.append(p)

    def shortest_regular_lane(self) -> int:
        """Index of the regular lane with the shortest expected wait (staff's choice)."""
        return min(range(self.n_regular_lanes), key=lambda i: self.expected_wait(self.lanes[i]))

    def _jump_saves_flight(self, p: Passenger, lane: Lane) -> bool:
        """Emergency exception to the overtake limit: the passenger would miss
        their flight from the back of this lane but make it from the front."""
        slack_back = p.time_budget - self.expected_wait(lane) - p.expected_service
        slack_front = p.time_budget - self.expected_wait(lane, front=True) - p.expected_service
        return slack_back < 0 <= slack_front

    def _start_service(self, lane: Lane):
        """Begin screening the passenger at the head of the lane."""
        head = lane.queue[0]
        head.service_start = self.clock
        lane.remaining_service = head.service_time

    def _complete_head(self, lane_idx: int):
        """Finish screening the lane's head passenger and start the next one.
        With reward_timing="completion", charge that passenger's full cost now."""
        lane = self.lanes[lane_idx]
        p = lane.queue.popleft()
        p.completion_time = self.clock
        self._completed.append(p)
        if self.reward_timing == "completion":
            for version, fn in (("v1", self._reward_v1), ("v2", self._reward_v2)):
                for k, v in fn(p).items():
                    self._pending[version][k] += v
        if lane.queue:
            self._start_service(lane)
        else:
            lane.remaining_service = 0.0

    @staticmethod
    def _empty_breakdown():
        """Zeroed reward breakdown, per term, for both reward versions."""
        return {v: {k: 0.0 for k in BREAKDOWN_KEYS} for v in ("v1", "v2")}

    # REWARD FUNCTIONS -- all terms are penalties (<= 0). These define each
    # passenger's total cost; with reward_timing="completion" it is paid in
    # one lump when they finish screening, with "accrual" (default) the same
    # total is paid out minute by minute by _accrue().
    #   wait    = time from joining the lane until screening starts
    #   sojourn = time from joining the lane until screening ends
    #   missed  = sojourn > time_budget
    def _reward_v1(self, p: Passenger) -> dict:
        """Original design.

        term 1  -w1 * wait
        term 2  if time_budget <= threshold (judged ON ARRIVAL):
                    -w2 * clip(threshold / time_budget, 1, cap) * wait
                if missed: -w2_missed_flight (one-off, no further cost)
        term 3  disabled/elderly: -w3 * wait
        term 4  business/VIP/crew: -w4 * max(0, wait - sla_max_wait)
        """
        return self._reward(p, REWARD_V1, dynamic_urgency=False)

    def _reward_v2(self, p: Passenger) -> dict:
        """Mitigated design.

        term 2  urgency is judged on the passenger's REMAINING slack while
                they wait, not on a label fixed at arrival:
                    -w2 * integral over the wait of u(tau),
                    u(tau) = clip(threshold / slack(tau), 1, cap) when
                    slack(tau) = time_budget - tau <= threshold, else 0
                so a passenger who arrived relaxed but is kept waiting until
                their deadline is as costly as one who arrived urgent.
                Missing a flight costs the one-off penalty PLUS
                w2_late_per_min per minute late (no "write-off" once missed).
        term 4  SLA cap enforced harder: one-off breach penalty plus a
                steeper per-minute overage.
        term 5  NEW: a wait limit for every passenger (max_wait). Terms 1-4
                are sums of waits, which are nearly blind to WHO waits --
                reshuffling a queue barely changes the total -- so v1
                happily lets a few passengers be overtaken again and again.
                Past the limit, each minute costs much more:
                    -w5_breach - w5_overage * max(0, wait - max_wait)
        terms 1, 3 unchanged.
        """
        return self._reward(p, REWARD_V2, dynamic_urgency=True)

    def _reward(self, p: Passenger, w: RewardConfig, dynamic_urgency: bool) -> dict:
        """A finished passenger's total cost, split by term (see _reward_v1/_reward_v2)."""
        wait = p.service_start - p.arrival_time
        sojourn = p.completion_time - p.arrival_time
        b = {k: 0.0 for k in BREAKDOWN_KEYS}

        b["base"] = -w.w1_base * wait

        if dynamic_urgency:
            b["urgency"] = -w.w2_urgency * self._urgency_integral(p.time_budget, wait, w)
        elif p.time_budget <= w.urgency_threshold:
            u = np.clip(w.urgency_threshold / max(p.time_budget, 1.0), 1.0, w.urgency_cap)
            b["urgency"] = -w.w2_urgency * u * wait

        if sojourn > p.time_budget:
            b["missed_flight"] = -w.w2_missed_flight - w.w2_late_per_min * (sojourn - p.time_budget)

        if p.ptype == PassengerType.DISABLED:
            b["disabled"] = -w.w3_disabled * wait

        if p.ptype in SLA_TYPES and wait > w.sla_max_wait:
            b["sla"] = -w.w4_sla_breach - w.w4_sla_overage * (wait - w.sla_max_wait)

        if wait > w.max_wait:
            b["long_wait"] = -w.w5_long_wait_breach - w.w5_long_wait_overage * (wait - w.max_wait)
        return b

    @staticmethod
    def _urgency_antiderivative(slack: float, w: RewardConfig) -> float:
        """H(s) with dH/ds = u(s): cap for s <= thr/cap, thr/s up to thr, 0 above."""
        c = w.urgency_threshold / w.urgency_cap
        if slack <= c:
            return w.urgency_cap * slack
        return w.urgency_cap * c + w.urgency_threshold * math.log(min(slack, w.urgency_threshold) / c)

    @classmethod
    def _urgency_integral(cls, budget: float, wait: float, w: RewardConfig) -> float:
        """Integral of u(budget - tau) for tau in [0, wait]."""
        return cls._urgency_antiderivative(budget, w) - cls._urgency_antiderivative(budget - wait, w)

    def _accrue(self, dt: float):
        """Charge the same per-passenger costs as _reward(), but continuously
        as time passes instead of in one lump at completion. Episode totals are
        identical; the agent just gets feedback closer to the decision that
        caused it."""
        t0 = self.clock
        for lane in self.lanes:
            for i, q in enumerate(lane.queue):
                s0 = t0 - q.arrival_time
                s1 = s0 + dt
                is_sla = q.ptype in SLA_TYPES
                is_disabled = q.ptype == PassengerType.DISABLED
                for version, w, dynamic in (("v1", REWARD_V1, False), ("v2", REWARD_V2, True)):
                    bd = self._pending[version]
                    if i > 0:   # still waiting: wait-based terms
                        bd["base"] -= w.w1_base * dt
                        if dynamic:
                            h = self._urgency_antiderivative
                            bd["urgency"] -= w.w2_urgency * (h(q.time_budget - s0, w) - h(q.time_budget - s1, w))
                        elif q.time_budget <= w.urgency_threshold:
                            u = min(max(w.urgency_threshold / max(q.time_budget, 1.0), 1.0), w.urgency_cap)
                            bd["urgency"] -= w.w2_urgency * u * dt
                        if is_disabled:
                            bd["disabled"] -= w.w3_disabled * dt
                        if is_sla and s1 > w.sla_max_wait:
                            if s0 <= w.sla_max_wait:
                                bd["sla"] -= w.w4_sla_breach
                            bd["sla"] -= w.w4_sla_overage * (s1 - max(s0, w.sla_max_wait))
                        if s1 > w.max_wait:
                            if s0 <= w.max_wait:
                                bd["long_wait"] -= w.w5_long_wait_breach
                            bd["long_wait"] -= w.w5_long_wait_overage * (s1 - max(s0, w.max_wait))
                    # waiting or being screened: lateness terms
                    if s1 > q.time_budget:
                        if s0 <= q.time_budget:
                            bd["missed_flight"] -= w.w2_missed_flight
                        bd["missed_flight"] -= w.w2_late_per_min * (s1 - max(s0, q.time_budget))

    # Observation helpers (everything here is knowable from boarding passes)
    def _projected_slacks(self, lane: Lane):
        """Projected slack at completion for everyone in the lane, in order."""
        out, t_done = [], self.clock
        for i, q in enumerate(lane.queue):
            t_done += lane.remaining_service if i == 0 else q.expected_service
            out.append(q.time_budget - (t_done - q.arrival_time))
        return out

    def expected_wait(self, lane: Lane, front: bool = False) -> float:
        """Expected minutes until a newcomer would start screening in this lane,
        joining at the back (default) or at the front of the waiting line."""
        if not lane.queue:
            return 0.0
        if front:
            return max(lane.remaining_service, 0.0)
        return max(lane.remaining_service, 0.0) + sum(q.expected_service for q in list(lane.queue)[1:])

    def _get_obs(self):
        """Build the observation vector (layout described in the class docstring)."""
        f = []
        # Per-lane features
        for lane in self.lanes:
            slacks = self._projected_slacks(lane)
            f += [
                len(lane.queue) / 40.0,
                self.expected_wait(lane) / 10.0,
                sum(s < 10.0 for s in slacks) / 20.0,
                sum(s < 0.0 for s in slacks) / 20.0,
                np.clip(min(slacks) if slacks else 60.0, -30.0, 60.0) / 30.0,
            ]
        # Special-lane service level
        special = self.lanes[self.special_lane_idx]
        sla_waits = [self.clock - q.arrival_time for q in list(special.queue)[1:] if q.ptype in SLA_TYPES]
        f.append((max(sla_waits) if sla_waits else 0.0) / REWARD_V1.sla_max_wait)
        f.append((self.expected_wait(special) - REWARD_V1.sla_max_wait) / REWARD_V1.sla_max_wait)

        # The arriving passenger (zeros once the episode is over)
        p = self._current
        if p is None:
            f += [0.0] * 6
        else:
            f.append(p.time_budget / 60.0)
            f.append(1.0 if p.time_budget <= REWARD_V1.urgency_threshold else 0.0)
            for lane in (self.lanes[self.shortest_regular_lane()], special):
                for front in (False, True):
                    slack = p.time_budget - self.expected_wait(lane, front) - p.expected_service
                    f.append(np.clip(slack, -30.0, 90.0) / 30.0)

        # Global: recent arrival rate and time of day
        while self._arrival_times and self._arrival_times[0] < self.clock - 10.0:
            self._arrival_times.popleft()
        f.append(len(self._arrival_times) / 10.0 / self.scenario.base_rate)
        f.append(self.clock / self.scenario.horizon)
        return np.asarray(f, dtype=np.float32)

    # Task-level KPIs (what an airport would actually judge the policy on)
    def episode_kpis(self) -> dict:
        """Episode totals: both rewards, missed flights, waits, queue lengths,
        SLA breaches, jumps and overtakes. Called once the lanes have drained."""
        ps = self._completed
        eco = [p for p in ps if p.ptype == PassengerType.ECONOMY]
        sla = [p for p in ps if p.ptype in SLA_TYPES]
        dis = [p for p in ps if p.ptype == PassengerType.DISABLED]
        thr = REWARD_V1.urgency_threshold

        def wait(p):
            """Minutes from joining the lane until screening started."""
            return p.service_start - p.arrival_time

        def missed(p):
            """True if queueing plus screening took longer than the time budget."""
            return p.completion_time - p.arrival_time > p.time_budget

        def mean(xs):
            """Mean that returns 0 for an empty list."""
            return float(np.mean(xs)) if xs else 0.0

        eco_waits = [wait(p) for p in eco]
        relaxed = [p for p in eco if p.time_budget > thr]
        urgent = [p for p in eco if p.time_budget <= thr]
        return {
            "reward_v1": self._episode_reward["v1"],
            "reward_v2": self._episode_reward["v2"],
            "n_passengers": len(ps),
            "missed_total": sum(missed(p) for p in ps),
            "missed_rate": mean([missed(p) for p in ps]),
            "missed_economy_urgent": sum(missed(p) for p in urgent),
            "missed_economy_relaxed": sum(missed(p) for p in relaxed),
            "missed_disabled": sum(missed(p) for p in dis),
            "mean_wait_all": mean([wait(p) for p in ps]),
            "mean_wait_economy": mean(eco_waits),
            "p95_wait_economy": float(np.percentile(eco_waits, 95)) if eco_waits else 0.0,
            "max_wait_economy": max(eco_waits) if eco_waits else 0.0,
            "long_wait_count": sum(wait(p) > REWARD_V1.max_wait for p in ps),
            "mean_queue_regular": self._queue_sum / max(self._n_decisions, 1),
            "peak_queue_regular": self._queue_peak,
            "mean_wait_urgent_economy": mean([wait(p) for p in urgent]),
            "mean_wait_disabled": mean([wait(p) for p in dis]),
            "mean_wait_sla": mean([wait(p) for p in sla]),
            "sla_breach_rate": mean([wait(p) > REWARD_V1.sla_max_wait for p in sla]),
            "jump_rate": mean([p.jumped for p in eco]),
            "special_override_rate": mean([p.lane == self.special_lane_idx for p in eco]),
            "mean_overtaken_economy": mean([p.n_overtaken for p in eco]),
            "max_overtaken": max((p.n_overtaken for p in ps), default=0),
        }


if __name__ == "__main__":
    env = AirportSecurityEnv(seed=0)
    obs, _ = env.reset()
    done, total = False, 0.0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        done = term or trunc
    print(f"random policy: reward={total:.1f}  decisions={env._n_decisions}")
    for k, v in info["episode_kpis"].items():
        print(f"  {k:26s} {v:.3f}" if isinstance(v, float) else f"  {k:26s} {v}")
