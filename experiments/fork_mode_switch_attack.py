from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attractor_scanner import AdaptationRingScanner

TAU = 2.0 * np.pi


def circ(x: float) -> float:
    return float((x + np.pi) % TAU - np.pi)


def value_at(probe: float, target: float) -> float:
    return 0.5 + 0.5 * np.cos(circ(probe - target))


def reliability(sigma: float | None) -> float:
    if sigma is None:
        return 0.0
    return float(1.0 / (1.0 + (sigma / 0.25) ** 2))


class ModeSwitchSampler:
    """Attack the attractive interpretation with cheap smooth alternatives."""

    MODES = {
        "ji_frequency_plus_width",
        "ji_frequency_only",
        "ji_width_only",
        "ji_adaptation_switch",
        "engineered_speed_matched",
        "smooth_random_walk",
        "iid_random",
        "axis_only",
    }

    def __init__(self, mode: str, seed: int):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.rng = np.random.default_rng(seed + 31000)
        self.axis = 0.0
        self.confidence = 0.0
        self.cycle_index = 0
        self.wall_ms = 0
        self.cycles = 0
        self.offset = 0.0
        self.prev_probe = None
        self.travel = 0.0
        self.peaks = []

        self.s = None
        if mode.startswith("ji_"):
            self.s = AdaptationRingScanner(
                n_cells=48,
                seed=seed,
                noise_std=0.04,
            )
            for _ in range(500):
                self.s.step(0.0)

    def _policy(self, confidence: float):
        c = float(np.clip(confidence, 0.0, 1.0))
        # Empirical rate endpoints from Vollan: ~8.2 -> ~9.1 Hz.
        f = 8.2 + 0.9 * c
        period = int(round(1000.0 / f))
        # Empirical alternation-width ratio: 39.8 / 62.2 ~= 0.64.
        scale = 1.0 - 0.36 * c
        adapt_mbar = 12.0

        if self.mode == "ji_frequency_only":
            scale = 1.0
        elif self.mode == "ji_width_only":
            period = 100
        elif self.mode == "ji_adaptation_switch":
            period = 100
            scale = 1.0
            adapt_mbar = 15.0 - 6.0 * c
        elif self.mode in {
            "engineered_speed_matched",
            "smooth_random_walk",
            "iid_random",
            "axis_only",
        }:
            pass

        return period, scale, adapt_mbar

    def run_cycle(
        self,
        target_fn,
        start_ms: int,
        cue: float | None,
        confidence: float,
    ) -> dict[str, float]:
        c = float(np.clip(confidence, 0.0, 1.0))
        self.confidence = 0.78 * self.confidence + 0.22 * c

        if cue is not None:
            gain = 0.78 * max(0.12, c)
            self.axis = circ(
                self.axis + gain * circ(float(cue) - self.axis)
            )

        period, scale, adapt_mbar = self._policy(self.confidence)
        # For cheap attackers, use the same broad->focused amplitude ratio.
        amp = 0.70 * (1.0 - 0.40 * self.confidence)

        if self.s is not None:
            self.s.theta_period_ms = float(period)
            self.s.adaptation_gain = (
                adapt_mbar
                * self.s.tau_ms
                / self.s.adaptation_tau_ms
            )
            self.s.external_gain = 3.0
            self.s.theta_modulation = 0.4
            self.s.time_ms = 0.0

        best = 0.0
        vals = []
        axis_err = []
        raw_abs = []
        side_target = amp if self.cycle_index % 2 == 0 else -amp

        for j in range(period):
            t = start_ms + j
            target = float(target_fn(t))

            if self.mode.startswith("ji_"):
                raw = float(self.s.step(0.0)["center"])
                probe = circ(self.axis + scale * raw)

            elif self.mode == "engineered_speed_matched":
                # No attractor, no adaptation, no theta-driven bump. Just a
                # continuous state moving toward alternating targets at roughly
                # the same per-ms travel budget as the Ji sweep.
                delta = side_target - self.offset
                self.offset += float(np.clip(delta, -0.013, 0.013))
                probe = circ(self.axis + self.offset)
                raw = self.offset

            elif self.mode == "smooth_random_walk":
                # Same continuity scale, but no left/right oscillator.
                self.offset += float(self.rng.normal(0.0, 0.016))
                if self.offset > amp:
                    self.offset = amp - (self.offset - amp)
                if self.offset < -amp:
                    self.offset = -amp - (self.offset + amp)
                probe = circ(self.axis + self.offset)
                raw = self.offset

            elif self.mode == "iid_random":
                raw = float(self.rng.uniform(-amp, amp))
                probe = circ(self.axis + raw)

            elif self.mode == "axis_only":
                raw = 0.0
                probe = self.axis

            else:
                raise ValueError(self.mode)

            v = float(value_at(probe, target))
            best = max(best, v)
            vals.append(v)
            axis_err.append(abs(circ(self.axis - target)))
            raw_abs.append(abs(raw))

            if self.prev_probe is not None:
                self.travel += abs(circ(probe - self.prev_probe))
            self.prev_probe = probe

        self.cycle_index += 1
        self.cycles += 1
        self.wall_ms += period
        self.peaks.append(float(np.max(raw_abs)))

        return {
            "duration_ms": float(period),
            "best_value": best,
            "mean_value": float(np.mean(vals)),
            "hit": float(best >= value_at(0.0, 0.20)),
            "axis_error": float(np.mean(axis_err)),
        }

    @property
    def cycles_per_second(self) -> float:
        return 1000.0 * self.cycles / max(1, self.wall_ms)

    @property
    def probe_travel(self) -> float:
        return self.travel / max(1, self.wall_ms - 1)

    @property
    def mean_peak(self) -> float:
        return float(np.mean(self.peaks))


MODES = [
    "ji_frequency_plus_width",
    "ji_frequency_only",
    "ji_width_only",
    "ji_adaptation_switch",
    "engineered_speed_matched",
    "smooth_random_walk",
    "iid_random",
    "axis_only",
]


def make_target(seed: int, horizon: int):
    rng = np.random.default_rng(seed + 500)
    x = np.zeros(horizon + 300)
    x[0] = float(rng.uniform(-2.0, 2.0))
    velocity = float(rng.choice([-1.0, 1.0]) * 0.0018)
    for t in range(1, len(x)):
        if t % 900 == 0:
            velocity = float(
                np.clip(
                    0.50 * velocity + rng.normal(0.0, 0.0015),
                    -0.004,
                    0.004,
                )
            )
        x[t] = circ(x[t - 1] + velocity)
    return x


def run_world(seed: int, mode: str, world: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 700)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    p = ModeSwitchSampler(mode, seed)
    t = 0
    bests, means, hits, axis_errors = [], [], [], []
    reacq = []
    return_start = None
    found = None

    while t < horizon:
        if world == "reliable":
            sigma = 0.12
        elif world == "mixed":
            sigma = [0.12, 0.65, 0.28][(t // 1000) % 3]
        elif world == "loss":
            local = t % 1800
            sigma = None if 900 <= local < 1300 else 0.12
            if local >= 1300 and return_start is None:
                return_start = t
                found = None
            if local < 900 and return_start is not None:
                reacq.append(500.0 if found is None else float(found))
                return_start = None
        else:
            raise ValueError(world)

        conf = reliability(sigma)
        cue = None if sigma is None else circ(
            target_fn(t) + rng.normal(0.0, sigma)
        )

        row = p.run_cycle(target_fn, t, cue, conf)
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

        t += int(row["duration_ms"])

    if world == "loss" and return_start is not None:
        reacq.append(500.0 if found is None else float(found))

    seconds = p.wall_ms / 1000.0
    out = {
        "mean_cycle_best_value": float(np.mean(bests)),
        "mean_within_sweep_value": float(np.mean(means)),
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_axis_error_rad": float(np.mean(axis_errors)),
        "cycle_utility_per_second": float(np.sum(bests) / seconds),
        "cycles_per_second": p.cycles_per_second,
        "mean_cycle_peak_abs_rad": p.mean_peak,
        "probe_travel_rad_per_ms": p.probe_travel,
    }
    if world == "loss":
        out["reacquisition_ms"] = float(np.mean(reacq))
    return out


WORLDS = ["reliable", "mixed", "loss"]


def summarize(rows):
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 8
    out = {
        world: {
            mode: summarize([
                run_world(seed, mode, world)
                for seed in range(n_seeds)
            ])
            for mode in MODES
        }
        for world in WORLDS
    }
    out["settings"] = {
        "n_seeds": n_seeds,
        "slow_weight_changes": 0,
        "upstream_reference_motion": 0,
        "empirical_rate_endpoints_hz": [8.2, 9.1],
        "empirical_width_ratio": 39.8 / 62.2,
        "engineered_speed_limit_rad_per_ms": 0.013,
    }
    out["question"] = (
        "If confidence controls both sampling rate and angular spread, is the "
        "Ji-like attractor needed, or can a cheap continuous engineered sweep "
        "or smooth random walk reproduce the active-sensing benefit?"
    )
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "fork_mode_switch_attack.json").write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
