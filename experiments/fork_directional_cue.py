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


class CueSampler:
    """Stable reference + fast directional control signal.

    Unlike earlier forks, the scanner is *not* asked to infer target direction
    from scalar reward. A noisy direction cue (or movement-plan-like vector)
    controls the fast sampling axis. This is closer to the interpretation that
    EC/HPC sweeps allocate samples using information computed elsewhere.

    No slow weights change.
    """

    MODES = {
        "forage_empirical",
        "chase_empirical",
        "adaptive_empirical_frequency",
        "adaptive_wide_frequency",
        "adaptive_adaptation",
        "adaptive_tether",
        "adaptive_downstream_width",
        "axis_only",
        "random_around_axis",
        "no_direction_cue",
    }

    def __init__(self, mode: str, seed: int):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.rng = np.random.default_rng(seed + 23000)
        self.axis = 0.0
        self.focus = 0.0
        self.wall_ms = 0
        self.cycles = 0
        self.probe_travel = 0.0
        self.prev_probe = None
        self.cycle_peaks = []

        self.s = None if mode == "random_around_axis" else AdaptationRingScanner(
            n_cells=48,
            seed=seed,
            noise_std=0.04,
        )
        if self.s is not None:
            for _ in range(500):
                self.s.step(0.0)

    def settings(self, confidence: float):
        c = float(np.clip(confidence, 0.0, 1.0))
        period = 100.0
        adapt_mbar = 12.0
        external_gain = 3.0
        scale = 1.0

        # Vollan means: foraging ~8.2 Hz, chasing ~9.1 Hz.
        if self.mode == "forage_empirical":
            period = 1000.0 / 8.2
        elif self.mode == "chase_empirical":
            period = 1000.0 / 9.1
        elif self.mode == "adaptive_empirical_frequency":
            f = 8.2 + (9.1 - 8.2) * c
            period = 1000.0 / f
        elif self.mode == "adaptive_wide_frequency":
            # Exaggerated sensitivity fork: the mechanism map showed a much
            # larger sweep-angle effect over this range.
            period = 130.0 - 60.0 * c
        elif self.mode == "adaptive_adaptation":
            adapt_mbar = 18.0 - 9.0 * c
        elif self.mode == "adaptive_tether":
            external_gain = 1.8 + 2.4 * c
        elif self.mode == "adaptive_downstream_width":
            scale = 1.45 - 0.75 * c
        elif self.mode == "axis_only":
            scale = 0.0
        elif self.mode == "random_around_axis":
            period = 100.0

        return int(round(period)), adapt_mbar, external_gain, scale

    def run_cycle(
        self,
        target_fn,
        start_ms: int,
        cue_angle: float | None,
        confidence: float,
    ) -> dict[str, float]:
        c = float(np.clip(confidence, 0.0, 1.0))
        self.focus = 0.8 * self.focus + 0.2 * c

        if self.mode != "no_direction_cue" and cue_angle is not None:
            # Fast state: align an elastic downstream axis to the cue while the
            # upstream reference stays fixed at zero.
            gain = 0.78 * max(0.12, c)
            self.axis = circ(
                self.axis + gain * circ(float(cue_angle) - self.axis)
            )

        period, adapt_mbar, external_gain, scale = self.settings(self.focus)

        if self.s is not None:
            self.s.theta_period_ms = float(period)
            self.s.adaptation_gain = (
                float(adapt_mbar)
                * self.s.tau_ms
                / self.s.adaptation_tau_ms
            )
            self.s.external_gain = float(external_gain)
            self.s.theta_modulation = 0.4
            self.s.time_ms = 0.0

        best = 0.0
        mean_values = []
        axis_errors = []
        raw_abs = []

        for j in range(period):
            t = start_ms + j
            target = float(target_fn(t))

            if self.mode == "random_around_axis":
                raw = float(self.rng.uniform(-0.60, 0.60))
            else:
                row = self.s.step(0.0)
                raw = float(row["center"])

            probe = circ(self.axis + scale * raw)
            value = float(value_at(probe, target))
            best = max(best, value)
            mean_values.append(value)
            axis_errors.append(abs(circ(self.axis - target)))
            raw_abs.append(abs(raw))

            if self.prev_probe is not None:
                self.probe_travel += abs(circ(probe - self.prev_probe))
            self.prev_probe = probe

        self.cycles += 1
        self.wall_ms += period
        self.cycle_peaks.append(
            float(np.max(raw_abs)) if raw_abs else 0.0
        )

        return {
            "duration_ms": float(period),
            "best_value": float(best),
            "mean_value": float(np.mean(mean_values)),
            "mean_axis_error": float(np.mean(axis_errors)),
            "hit": float(best >= value_at(0.0, 0.20)),
        }

    @property
    def cycles_per_second(self) -> float:
        return 1000.0 * self.cycles / max(1, self.wall_ms)

    @property
    def mean_probe_travel(self) -> float:
        return self.probe_travel / max(1, self.wall_ms - 1)

    @property
    def mean_cycle_peak(self) -> float:
        return float(np.mean(self.cycle_peaks))


MODES = [
    "forage_empirical",
    "chase_empirical",
    "adaptive_empirical_frequency",
    "adaptive_wide_frequency",
    "adaptive_adaptation",
    "adaptive_tether",
    "adaptive_downstream_width",
    "axis_only",
    "random_around_axis",
    "no_direction_cue",
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


def reliability_from_sigma(sigma: float | None) -> float:
    if sigma is None:
        return 0.0
    # Engineering confidence map; only ordering matters for this fork.
    return float(1.0 / (1.0 + (sigma / 0.25) ** 2))


def run_world(seed: int, mode: str, world: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 700)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    p = CueSampler(mode, seed)
    t = 0
    best_values = []
    mean_values = []
    axis_errors = []
    hits = []
    reacq = []
    return_start = None
    found_after_return = None

    while t < horizon:
        if world == "reliable_pursuit":
            sigma = 0.12
        elif world == "mixed_reliability":
            block = (t // 1000) % 3
            sigma = [0.12, 0.65, 0.28][block]
        elif world == "loss_reacquire":
            # Reliable tracking, then repeated 400-ms cue losses.
            local = t % 1800
            sigma = None if 900 <= local < 1300 else 0.12
            if local >= 1300 and return_start is None:
                return_start = t
                found_after_return = None
            if local < 900 and return_start is not None:
                if found_after_return is None:
                    reacq.append(500.0)
                else:
                    reacq.append(float(found_after_return))
                return_start = None
        else:
            raise ValueError(world)

        confidence = reliability_from_sigma(sigma)
        if sigma is None:
            cue = None
        else:
            cue = circ(
                target_fn(t) + rng.normal(0.0, sigma)
            )

        row = p.run_cycle(target_fn, t, cue, confidence)
        best_values.append(row["best_value"])
        mean_values.append(row["mean_value"])
        axis_errors.append(row["mean_axis_error"])
        hits.append(row["hit"])

        if (
            world == "loss_reacquire"
            and return_start is not None
            and found_after_return is None
            and row["hit"] > 0.5
        ):
            found_after_return = t - return_start

        t += int(row["duration_ms"])

    if world == "loss_reacquire" and return_start is not None:
        reacq.append(
            500.0 if found_after_return is None
            else float(found_after_return)
        )

    seconds = p.wall_ms / 1000.0
    out = {
        "mean_cycle_best_value": float(np.mean(best_values)),
        "mean_within_sweep_value": float(np.mean(mean_values)),
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_axis_error_rad": float(np.mean(axis_errors)),
        "cycle_utility_per_second": float(np.sum(best_values) / seconds),
        "cycles_per_second": p.cycles_per_second,
        "mean_cycle_peak_abs_rad": p.mean_cycle_peak,
        "probe_travel_rad_per_ms": p.mean_probe_travel,
    }
    if world == "loss_reacquire":
        out["reacquisition_ms"] = float(np.mean(reacq)) if reacq else 500.0
    return out


WORLDS = [
    "reliable_pursuit",
    "mixed_reliability",
    "loss_reacquire",
]


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        vals = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(vals.mean())
        out[key + "_std"] = float(vals.std())
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
        "cue_source": (
            "noisy fast directional signal supplied independently of sweep "
            "utility; intended as an artificial analogue of visual salience "
            "or an internally generated movement-plan vector"
        ),
    }
    out["question"] = (
        "Do Vollan-like width/frequency policies become computationally useful "
        "when the sampler allocates attention using a directional control "
        "signal, instead of being forced to discover target direction from "
        "scalar reward?"
    )
    out["warning"] = (
        "This tests an architectural interpretation, not the biological source "
        "or circuit implementation of the control signal."
    )
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "fork_directional_cue.json").write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
