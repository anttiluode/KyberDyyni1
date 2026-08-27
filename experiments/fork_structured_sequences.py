from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAU = 2.0 * np.pi
PHI = (1.0 + np.sqrt(5.0)) / 2.0


def circ(x: float | np.ndarray):
    x = np.asarray(x, dtype=float)
    y = (x + np.pi) % TAU - np.pi
    return float(y) if y.ndim == 0 else y


def value_at(probe: float, target: float) -> float:
    return 0.5 + 0.5 * np.cos(circ(probe - target))


def vdc(n: int, base: int = 2) -> float:
    out = 0.0
    denom = 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        out += rem / denom
    return out


def interp_waypoints(points: list[float], n: int) -> np.ndarray:
    """Piecewise-linear path through waypoints, including endpoints."""
    xp = np.linspace(0.0, 1.0, len(points))
    x = np.linspace(0.0, 1.0, n)
    return np.interp(x, xp, np.asarray(points, dtype=float))


def raw_schedule(name: str, cycle: int, n: int, rng) -> np.ndarray:
    """Dimensionless schedule. Starts and ends at zero when path-matched."""
    u = np.linspace(0.0, 1.0, n)
    side = 1.0 if cycle % 2 == 0 else -1.0

    if name == "alternating_triangle":
        # One side per cycle: 0 -> side -> 0.
        return side * (1.0 - np.abs(2.0 * u - 1.0))

    if name == "alternating_sine":
        # Smooth version of the same one-side-per-cycle policy.
        return side * np.sin(np.pi * u)

    if name == "bilateral_triangle":
        # Both sides every cycle: 0 -> -1 -> +1 -> 0.
        return interp_waypoints([0.0, -1.0, 1.0, 0.0], n)

    if name == "bilateral_sine":
        return np.sin(TAU * u)

    if name == "two_harmonic":
        x = np.sin(TAU * u) + 0.33 * np.sin(3.0 * TAU * u)
        x /= max(1e-12, np.max(np.abs(x)))
        return x

    if name == "golden_phase_sine":
        # Same smooth oscillator, but its starting phase advances by the
        # golden ratio. Subtract the linear endpoint bridge so the path starts
        # and ends at zero, making cycle travel well-defined.
        phase = (cycle / PHI) % 1.0
        x = np.sin(TAU * (u + phase))
        bridge = (1.0 - u) * x[0] + u * x[-1]
        x = x - bridge
        m = np.max(np.abs(x))
        return x / m if m > 1e-12 else x

    if name == "vdc_low_discrepancy":
        vals = [2.0 * vdc(cycle * 7 + k + 1) - 1.0 for k in range(7)]
        return interp_waypoints([0.0] + vals + [0.0], n)

    if name == "vdc_sorted_coverage":
        vals = sorted(2.0 * vdc(cycle * 7 + k + 1) - 1.0 for k in range(7))
        return interp_waypoints([0.0] + vals + [0.0], n)

    if name == "center_out":
        mags = np.linspace(0.2, 1.0, 5)
        vals = []
        for k, m in enumerate(mags):
            s = 1.0 if (cycle + k) % 2 == 0 else -1.0
            vals.extend([s * m, -s * m])
        return interp_waypoints([0.0] + vals + [0.0], n)

    if name == "smooth_random_walk":
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] + rng.normal(0.0, 0.12)
            if x[i] > 1.0:
                x[i] = 2.0 - x[i]
            if x[i] < -1.0:
                x[i] = -2.0 - x[i]
        # Force a return to zero without discontinuity.
        x -= np.linspace(0.0, x[-1], n)
        m = np.max(np.abs(x))
        return x / m if m > 1e-12 else x

    raise ValueError(name)


PATH_MATCHED = [
    "alternating_triangle",
    "alternating_sine",
    "bilateral_triangle",
    "bilateral_sine",
    "two_harmonic",
    "golden_phase_sine",
    "vdc_low_discrepancy",
    "vdc_sorted_coverage",
    "center_out",
    "smooth_random_walk",
]

ALL = PATH_MATCHED + ["axis_only", "iid_random_unmatched"]


def match_path(raw: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
    """Scale a raw schedule to consume an exact total-variation budget."""
    if len(raw) < 2:
        return raw.copy(), 0.0
    tv = float(np.sum(np.abs(np.diff(raw))))
    if tv < 1e-12:
        return np.zeros_like(raw), 0.0
    scale = budget / tv
    x = raw * scale
    return x, float(np.sum(np.abs(np.diff(x))))


class Sampler:
    def __init__(self, mode: str, seed: int, period_ms: int = 100):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 41000)
        self.period = period_ms
        self.axis = 0.0
        self.confidence = 0.0
        self.cycle = 0
        self.prev_probe = None
        self.travel = 0.0
        self.wall_ms = 0
        self.radius_used = []
        self.path_per_cycle = []

    def run_cycle(
        self,
        target_fn,
        start_ms: int,
        cue: float | None,
        cue_confidence: float,
    ) -> dict[str, float]:
        c = float(np.clip(cue_confidence, 0.0, 1.0))
        self.confidence = 0.78 * self.confidence + 0.22 * c

        if cue is not None:
            gain = 0.78 * max(0.12, c)
            self.axis = circ(self.axis + gain * circ(float(cue) - self.axis))

        # Broad under uncertainty, narrower under confidence.
        desired_radius = 0.72 * (1.0 - 0.42 * self.confidence)
        # The strict matched travel budget is the one-sided alternator's
        # 0 -> radius -> 0 path.
        path_budget = 2.0 * desired_radius

        n = self.period
        if self.mode == "axis_only":
            offsets = np.zeros(n)
            matched_path = 0.0
        elif self.mode == "iid_random_unmatched":
            offsets = self.rng.uniform(-desired_radius, desired_radius, size=n)
            matched_path = float(np.sum(np.abs(np.diff(offsets))))
        else:
            raw = raw_schedule(self.mode, self.cycle, n, self.rng)
            offsets, matched_path = match_path(raw, path_budget)

        self.radius_used.append(float(np.max(np.abs(offsets))))
        self.path_per_cycle.append(matched_path)

        vals = []
        axis_err = []
        best = 0.0
        hit = 0.0
        for j, off in enumerate(offsets):
            t = start_ms + j
            target = float(target_fn(t))
            probe = circ(self.axis + float(off))
            value = float(value_at(probe, target))
            vals.append(value)
            axis_err.append(abs(circ(self.axis - target)))
            best = max(best, value)
            if abs(circ(probe - target)) < 0.20:
                hit = 1.0

            if self.prev_probe is not None:
                self.travel += abs(circ(probe - self.prev_probe))
            self.prev_probe = probe

        self.cycle += 1
        self.wall_ms += n
        return {
            "best_value": best,
            "mean_value": float(np.mean(vals)),
            "hit": hit,
            "axis_error": float(np.mean(axis_err)),
        }

    @property
    def travel_per_ms(self) -> float:
        return self.travel / max(1, self.wall_ms - 1)

    @property
    def mean_radius(self) -> float:
        return float(np.mean(self.radius_used))

    @property
    def mean_path_per_cycle(self) -> float:
        return float(np.mean(self.path_per_cycle))


def confidence_from_sigma(sigma: float | None) -> float:
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


def run_world(seed: int, mode: str, world: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 700)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    s = Sampler(mode, seed)
    t = 0
    bests, means, hits, axis_errors = [], [], [], []
    reacq = []
    return_start = None
    found = None

    while t < horizon:
        if world == "reliable":
            sigma = 0.12
            bias = 0.0
        elif world == "mixed":
            sigma = [0.12, 0.65, 0.28][(t // 1000) % 3]
            bias = 0.0
        elif world == "biased_cue":
            sigma = 0.12
            # Persistent cue error changes sign halfway through.
            bias = 0.34 if t < horizon // 2 else -0.34
        elif world == "loss":
            local = t % 1800
            sigma = None if 900 <= local < 1300 else 0.12
            bias = 0.0
            if local >= 1300 and return_start is None:
                return_start = t
                found = None
            if local < 900 and return_start is not None:
                reacq.append(500.0 if found is None else float(found))
                return_start = None
        else:
            raise ValueError(world)

        conf = confidence_from_sigma(sigma)
        cue = None if sigma is None else circ(
            target_fn(t) + bias + rng.normal(0.0, sigma)
        )

        row = s.run_cycle(target_fn, t, cue, conf)
        bests.append(row["best_value"])
        means.append(row["mean_value"])
        hits.append(row["hit"])
        axis_errors.append(row["axis_error"])

        if (
            world == "loss"
            and return_start is not None
            and found is None
            and row["hit"] > 0.5
        ):
            found = t - return_start

        t += s.period

    if world == "loss" and return_start is not None:
        reacq.append(500.0 if found is None else float(found))

    seconds = s.wall_ms / 1000.0
    out = {
        "mean_cycle_best_value": float(np.mean(bests)),
        "mean_within_sweep_value": float(np.mean(means)),
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_axis_error_rad": float(np.mean(axis_errors)),
        "cycle_utility_per_second": float(np.sum(bests) / seconds),
        "probe_travel_rad_per_ms": s.travel_per_ms,
        "mean_schedule_radius_rad": s.mean_radius,
        "mean_path_per_cycle_rad": s.mean_path_per_cycle,
    }
    if world == "loss":
        out["reacquisition_ms"] = float(np.mean(reacq))
    return out


WORLDS = ["reliable", "mixed", "biased_cue", "loss"]


def summarize(rows):
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 10
    result = {
        world: {
            mode: summarize([run_world(seed, mode, world) for seed in range(n_seeds)])
            for mode in ALL
        }
        for world in WORLDS
    }
    result["settings"] = {
        "n_seeds": n_seeds,
        "period_ms": 100,
        "probes_per_cycle": 100,
        "slow_weight_changes": 0,
        "upstream_reference_motion": 0,
        "path_matching": (
            "all continuous structured schedules are scaled every cycle to the "
            "same total-variation budget: 2 * current uncertainty radius"
        ),
        "iid_random_unmatched": (
            "kept as a deliberately high-travel coverage upper bound, not a "
            "path-matched competitor"
        ),
    }
    result["question"] = (
        "Under equal wall time, probe count and per-cycle path budget, is "
        "left-right alternation special, or can other deterministic / "
        "low-discrepancy schedules provide better uncertainty coverage?"
    )
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_structured_sequences.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
