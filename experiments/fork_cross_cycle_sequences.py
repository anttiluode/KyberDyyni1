from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAU = 2.0 * np.pi
PHI = (1.0 + np.sqrt(5.0)) / 2.0


def circ(x):
    x = np.asarray(x, dtype=float)
    y = (x + np.pi) % TAU - np.pi
    return float(y) if y.ndim == 0 else y


def value_at(probe: float, target: float) -> float:
    return 0.5 + 0.5 * np.cos(circ(probe - target))


def confidence_from_sigma(sigma):
    if sigma is None:
        return 0.0
    return float(1.0 / (1.0 + (sigma / 0.25) ** 2))


def make_target(seed: int, horizon: int):
    rng = np.random.default_rng(seed + 600)
    x = np.zeros(horizon + 300)
    x[0] = float(rng.uniform(-2.0, 2.0))
    vel = float(rng.choice([-1.0, 1.0]) * 0.0018)
    for t in range(1, len(x)):
        if t % 900 == 0:
            vel = float(np.clip(0.5 * vel + rng.normal(0.0, 0.0015), -0.004, 0.004))
        x[t] = circ(x[t - 1] + vel)
    return x


def linear_path(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, n)


class CrossCycleSampler:
    MODES = {
        "cross_cycle_shuttle",
        "cross_cycle_sine",
        "closed_bilateral",
        "closed_one_side",
        "golden_endpoint",
        "vdc_endpoint",
        "smooth_random_walk",
        "iid_random_unmatched",
        "axis_only",
    }

    def __init__(self, mode: str, seed: int, period: int = 100):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.rng = np.random.default_rng(seed + 51000)
        self.period = period
        self.axis = 0.0
        self.confidence = 0.0
        self.offset = 0.0
        self.side = 1.0
        self.cycle = 0
        self.prev_probe = None
        self.travel = 0.0
        self.wall = 0
        self.path = []
        self.radius = []

    def schedule(self, radius: float) -> np.ndarray:
        n = self.period

        if self.mode == "cross_cycle_shuttle":
            # State persists. Traverse monotonically from wherever the previous
            # cycle ended to the opposite uncertainty boundary.
            target = self.side * radius
            x = linear_path(self.offset, target, n)
            self.offset = float(target)
            self.side *= -1.0
            return x

        if self.mode == "cross_cycle_sine":
            # Half a cosine wave per cycle. Endpoints alternate +/- radius and
            # the derivative naturally vanishes at the boundaries.
            start = self.offset
            target = self.side * radius
            u = np.linspace(0.0, 1.0, n)
            x = start + (target - start) * 0.5 * (1.0 - np.cos(np.pi * u))
            self.offset = float(target)
            self.side *= -1.0
            return x

        if self.mode == "closed_bilateral":
            # Same total path as a full shuttle but wastes some distance
            # returning to the middle each cycle.
            a = radius / 2.0
            xp = np.array([0.0, 0.25, 0.75, 1.0])
            yp = np.array([0.0, -a, a, 0.0])
            self.offset = 0.0
            return np.interp(np.linspace(0.0, 1.0, n), xp, yp)

        if self.mode == "closed_one_side":
            u = np.linspace(0.0, 1.0, n)
            x = self.side * radius * (1.0 - np.abs(2.0 * u - 1.0))
            self.offset = 0.0
            self.side *= -1.0
            return x

        if self.mode == "golden_endpoint":
            # Persistent low-discrepancy endpoints. Movement is direct and
            # continuous; it is allowed to use less than the shuttle's maximum
            # path budget but never to teleport.
            q = 2.0 * ((self.cycle / PHI) % 1.0) - 1.0
            target = q * radius
            x = linear_path(self.offset, target, n)
            self.offset = float(target)
            return x

        if self.mode == "vdc_endpoint":
            q = 2.0 * vdc(self.cycle + 1) - 1.0
            target = q * radius
            x = linear_path(self.offset, target, n)
            self.offset = float(target)
            return x

        if self.mode == "smooth_random_walk":
            x = np.zeros(n)
            x[0] = self.offset
            # Step scale chosen so total travel is in the same ballpark as
            # structured samplers; reflect at current uncertainty bounds.
            for i in range(1, n):
                x[i] = x[i - 1] + self.rng.normal(0.0, 0.012)
                if x[i] > radius:
                    x[i] = radius - (x[i] - radius)
                if x[i] < -radius:
                    x[i] = -radius - (x[i] + radius)
            self.offset = float(x[-1])
            return x

        if self.mode == "iid_random_unmatched":
            x = self.rng.uniform(-radius, radius, size=n)
            self.offset = float(x[-1])
            return x

        if self.mode == "axis_only":
            self.offset = 0.0
            return np.zeros(n)

        raise ValueError(self.mode)

    def run_cycle(self, target_fn, start_ms: int, cue, cue_confidence: float):
        c = float(np.clip(cue_confidence, 0.0, 1.0))
        self.confidence = 0.78 * self.confidence + 0.22 * c
        if cue is not None:
            gain = 0.78 * max(0.12, c)
            self.axis = circ(self.axis + gain * circ(float(cue) - self.axis))

        radius = 0.72 * (1.0 - 0.42 * self.confidence)
        offsets = self.schedule(radius)
        vals = []
        best = 0.0
        hit = 0.0
        axis_error = []

        cycle_travel = 0.0
        for j, off in enumerate(offsets):
            target = float(target_fn(start_ms + j))
            probe = circ(self.axis + float(off))
            val = float(value_at(probe, target))
            vals.append(val)
            best = max(best, val)
            hit = max(hit, float(abs(circ(probe - target)) < 0.20))
            axis_error.append(abs(circ(self.axis - target)))
            if self.prev_probe is not None:
                d = abs(circ(probe - self.prev_probe))
                self.travel += d
                cycle_travel += d
            self.prev_probe = probe

        self.cycle += 1
        self.wall += self.period
        self.path.append(cycle_travel)
        self.radius.append(radius)
        return {
            "best_value": best,
            "mean_value": float(np.mean(vals)),
            "hit": hit,
            "axis_error": float(np.mean(axis_error)),
        }

    @property
    def travel_per_ms(self):
        return self.travel / max(1, self.wall - 1)

    @property
    def mean_path(self):
        return float(np.mean(self.path))

    @property
    def mean_radius(self):
        return float(np.mean(self.radius))


def vdc(n: int, base: int = 2) -> float:
    out, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        out += rem / denom
    return out


MODES = [
    "cross_cycle_shuttle",
    "cross_cycle_sine",
    "closed_bilateral",
    "closed_one_side",
    "golden_endpoint",
    "vdc_endpoint",
    "smooth_random_walk",
    "iid_random_unmatched",
    "axis_only",
]


def run_world(seed: int, mode: str, world: str):
    rng = np.random.default_rng(seed + 700)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    s = CrossCycleSampler(mode, seed)
    t = 0
    bests, means, hits, axis_errors = [], [], [], []
    reacq = []
    return_start = None
    found = None

    while t < horizon:
        if world == "reliable":
            sigma, bias = 0.12, 0.0
        elif world == "mixed":
            sigma, bias = [0.12, 0.65, 0.28][(t // 1000) % 3], 0.0
        elif world == "biased":
            sigma = 0.12
            bias = 0.34 if t < horizon // 2 else -0.34
        elif world == "loss":
            local = t % 1800
            sigma = None if 900 <= local < 1300 else 0.12
            bias = 0.0
            if local >= 1300 and return_start is None:
                return_start, found = t, None
            if local < 900 and return_start is not None:
                reacq.append(500.0 if found is None else float(found))
                return_start = None
        else:
            raise ValueError(world)

        conf = confidence_from_sigma(sigma)
        cue = None if sigma is None else circ(target_fn(t) + bias + rng.normal(0.0, sigma))
        row = s.run_cycle(target_fn, t, cue, conf)
        bests.append(row["best_value"])
        means.append(row["mean_value"])
        hits.append(row["hit"])
        axis_errors.append(row["axis_error"])

        if world == "loss" and return_start is not None and found is None and row["hit"] > 0.5:
            found = t - return_start

        t += s.period

    if world == "loss" and return_start is not None:
        reacq.append(500.0 if found is None else float(found))

    out = {
        "mean_cycle_best_value": float(np.mean(bests)),
        "mean_within_sweep_value": float(np.mean(means)),
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_axis_error_rad": float(np.mean(axis_errors)),
        "probe_travel_rad_per_ms": s.travel_per_ms,
        "mean_path_per_cycle_rad": s.mean_path,
        "mean_uncertainty_radius_rad": s.mean_radius,
    }
    if world == "loss":
        out["reacquisition_ms"] = float(np.mean(reacq))
    return out


WORLDS = ["reliable", "mixed", "biased", "loss"]


def summarize(rows):
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main():
    n_seeds = 12
    result = {
        world: {
            mode: summarize([run_world(seed, mode, world) for seed in range(n_seeds)])
            for mode in MODES
        }
        for world in WORLDS
    }
    result["settings"] = {
        "n_seeds": n_seeds,
        "period_ms": 100,
        "slow_weight_changes": 0,
        "key_change": (
            "sampling offset persists across cycle boundaries; cross-cycle "
            "samplers do not pay an artificial return-to-center cost"
        ),
    }
    result["question"] = (
        "Is alternating left-right traversal useful because a continuous "
        "sampler can cover the full uncertainty interval by carrying its end "
        "state into the next cycle, rather than closing every sweep at center?"
    )
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_cross_cycle_sequences.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
