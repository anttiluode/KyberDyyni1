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


class CyclePolicy:
    """Fast controller updated once per complete theta sweep.

    Neural activation/adaptation state persists across cycles.  Only the theta
    phase clock is restarted at the natural cycle boundary so a new period can
    be chosen without creating a mid-cycle phase discontinuity.

    This fixes a flaw in the earlier fork: relevance was previously sampled
    every millisecond, so faster theta did not actually create more complete
    internal sampling opportunities per unit wall time.
    """

    MODES = {
        "fixed_fast_70ms",
        "fixed_base_100ms",
        "fixed_slow_130ms",
        "adaptive_frequency",
        "inverse_frequency",
        "adaptive_tether",
        "adaptive_adaptation",
        "adaptive_theta_amplitude",
        "adaptive_downstream_scale",
        "random_cycle_dither",
        "no_fast_update",
    }

    def __init__(self, mode: str, seed: int):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.rng = np.random.default_rng(seed + 17000)
        self.bias = 0.0
        self.focus = 0.0
        self.baseline = 0.5
        self.cycle_count = 0
        self.wall_ms = 0
        self.probe_travel = 0.0
        self.prev_probe = None
        self.cycle_peak_offsets = []

        self.s = None if mode == "random_cycle_dither" else AdaptationRingScanner(
            n_cells=48,
            seed=seed,
            noise_std=0.04,
        )
        if self.s is not None:
            self.base_adapt = self.s.adaptation_gain
            self.base_external = self.s.external_gain
            self.base_theta_mod = self.s.theta_modulation
            # Warm the recurrent/adaptation state; do not warm the fast bias.
            for _ in range(500):
                self.s.step(0.0)

    def _settings(self) -> tuple[int, float, float, float]:
        f = float(np.clip(self.focus, 0.0, 1.0))
        period = 100.0
        external = 3.0
        adapt = 12.0
        theta_mod = 0.4

        if self.mode == "fixed_fast_70ms":
            period = 70.0
        elif self.mode == "fixed_slow_130ms":
            period = 130.0
        elif self.mode == "adaptive_frequency":
            # uncertain -> slow/broad ; confident -> fast/narrow
            period = 130.0 - 60.0 * f
        elif self.mode == "inverse_frequency":
            period = 70.0 + 60.0 * f
        elif self.mode == "adaptive_tether":
            # Mechanism map: stronger external tether narrows the sweep.
            external = 1.8 + 2.4 * f
        elif self.mode == "adaptive_adaptation":
            # Mechanism map: stronger adaptation broadens/mobilizes the bump.
            adapt = 18.0 - 9.0 * f
        elif self.mode == "adaptive_theta_amplitude":
            # Mechanism map: larger theta modulation broadens until instability.
            theta_mod = 0.55 - 0.27 * f

        return (
            int(round(period)),
            external,
            adapt,
            theta_mod,
        )

    def run_cycle(
        self,
        target_fn,
        visible_fn,
        start_ms: int,
        eta: float = 2.2,
    ) -> dict[str, float | list[float]]:
        period, external, adapt_mbar, theta_mod = self._settings()
        scale = 1.0
        if self.mode == "adaptive_downstream_scale":
            # Pure downstream translation: broad while uncertain, contract as
            # confidence rises; generator dynamics are untouched.
            scale = 1.45 - 0.55 * self.focus

        if self.s is not None:
            self.s.theta_period_ms = float(period)
            self.s.external_gain = float(external)
            self.s.adaptation_gain = (
                float(adapt_mbar)
                * self.s.tau_ms
                / self.s.adaptation_tau_ms
            )
            self.s.theta_modulation = float(theta_mod)
            # Natural cycle boundary. Population state is *not* reset.
            self.s.time_ms = 0.0

        signal_terms = []
        visible_values = []
        errors = []
        probes = []
        raw_offsets = []

        for j in range(period):
            t = start_ms + j
            target = float(target_fn(t))
            visible = bool(visible_fn(t))

            if self.mode == "random_cycle_dither":
                # Random proposal path with no continuity advantage.
                raw_offset = float(self.rng.uniform(-0.60, 0.60))
                probe = circ(self.bias + raw_offset)
            else:
                row = self.s.step(0.0)
                raw_offset = float(row["center"])
                probe = circ(self.bias + scale * raw_offset)

            if self.prev_probe is not None:
                self.probe_travel += abs(circ(probe - self.prev_probe))
            self.prev_probe = probe
            probes.append(probe)
            raw_offsets.append(abs(raw_offset))

            if visible:
                value = float(value_at(probe, target))
                visible_values.append(value)
                # Accumulate a complete-sweep correlation signal.  One weight
                # update occurs at the end of the theta cycle.
                signal_terms.append(
                    (value - self.baseline)
                    * circ(probe - self.bias)
                )

            errors.append(abs(circ(self.bias - target)))

        if signal_terms and self.mode != "no_fast_update":
            cycle_signal = float(np.mean(signal_terms))
            self.bias = circ(self.bias + eta * cycle_signal)
            self.bias *= 0.999

        if visible_values:
            cycle_value = float(np.mean(visible_values))
            confidence = np.clip(
                (cycle_value - 0.55) / 0.45,
                0.0,
                1.0,
            )
            self.focus = 0.82 * self.focus + 0.18 * confidence
            self.baseline = (
                0.88 * self.baseline + 0.12 * cycle_value
            )
        else:
            cycle_value = np.nan
            self.focus *= 0.98

        self.cycle_count += 1
        self.wall_ms += period
        self.cycle_peak_offsets.append(
            float(np.max(raw_offsets)) if raw_offsets else 0.0
        )

        return {
            "duration_ms": float(period),
            "errors": errors,
            "mean_value": cycle_value,
            "probe_count": float(period),
        }

    @property
    def cycles_per_second(self) -> float:
        return 1000.0 * self.cycle_count / max(1, self.wall_ms)

    @property
    def mean_probe_travel(self) -> float:
        return self.probe_travel / max(1, self.wall_ms - 1)

    @property
    def mean_cycle_peak(self) -> float:
        return float(np.mean(self.cycle_peak_offsets))


MODES = [
    "fixed_fast_70ms",
    "fixed_base_100ms",
    "fixed_slow_130ms",
    "adaptive_frequency",
    "inverse_frequency",
    "adaptive_tether",
    "adaptive_adaptation",
    "adaptive_theta_amplitude",
    "adaptive_downstream_scale",
    "random_cycle_dither",
    "no_fast_update",
]


def stationary(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 100)
    p = CyclePolicy(mode, seed)
    targets = rng.uniform(-2.7, 2.7, size=4)
    latency = []
    all_errors = []
    cycle_values = []
    t_global = 0

    for target in targets:
        acquired = None
        elapsed = 0
        horizon = 1600

        def target_fn(_t, target=target):
            return target

        def visible_fn(_t):
            return True

        while elapsed < horizon:
            before = p.bias
            row = p.run_cycle(target_fn, visible_fn, t_global)
            errors = row["errors"]
            if acquired is None:
                for k, err in enumerate(errors):
                    if err < 0.18:
                        acquired = elapsed + k
                        break
            all_errors.extend(errors)
            if np.isfinite(row["mean_value"]):
                cycle_values.append(row["mean_value"])
            dt = int(row["duration_ms"])
            elapsed += dt
            t_global += dt

        latency.append(horizon if acquired is None else acquired)

    return {
        "acquisition_ms": float(np.mean(latency)),
        "success_fraction": float(np.mean(np.asarray(latency) < 1600)),
        "mean_tracking_error_rad": float(np.mean(all_errors)),
        "mean_cycle_value": float(np.mean(cycle_values)),
        "cycles_per_second": p.cycles_per_second,
        "mean_cycle_peak_abs_rad": p.mean_cycle_peak,
        "probe_travel_rad_per_ms": p.mean_probe_travel,
    }


def pursuit(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 200)
    p = CyclePolicy(mode, seed)
    target0 = float(rng.uniform(-2.0, 2.0))
    velocity0 = float(rng.choice([-1.0, 1.0]) * 0.0025)
    change_points = {
        1200: float(rng.normal(0.0, 0.0018)),
        2400: float(rng.normal(0.0, 0.0018)),
        3600: float(rng.normal(0.0, 0.0018)),
    }

    # Precompute the moving target so all policies see exactly the same world.
    horizon = 4800
    target = np.zeros(horizon + 200)
    velocity = velocity0
    target[0] = target0
    for t in range(1, len(target)):
        if t in change_points:
            velocity = float(
                np.clip(
                    0.45 * velocity + change_points[t],
                    -0.005,
                    0.005,
                )
            )
        target[t] = circ(target[t - 1] + velocity)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    def visible_fn(_t):
        return True

    errors = []
    values = []
    t = 0
    while t < horizon:
        row = p.run_cycle(target_fn, visible_fn, t)
        errors.extend(row["errors"])
        if np.isfinite(row["mean_value"]):
            values.append(row["mean_value"])
        t += int(row["duration_ms"])

    usable = errors[min(700, len(errors)):]
    return {
        "mean_tracking_error_rad": float(np.mean(usable)),
        "close_fraction": float(np.mean(np.asarray(usable) < 0.22)),
        "mean_cycle_value": float(np.mean(values)),
        "cycles_per_second": p.cycles_per_second,
        "mean_cycle_peak_abs_rad": p.mean_cycle_peak,
        "probe_travel_rad_per_ms": p.mean_probe_travel,
    }


def reorientation(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 300)
    p = CyclePolicy(mode, seed)
    target = float(rng.uniform(-2.0, 2.0))
    t = 0
    reacq = []
    errors_after = []
    values = []

    for _episode in range(4):
        # Track old target.
        track_end = t + 700
        def target_old(_t, target=target):
            return target
        def vis(_t):
            return True
        while t < track_end:
            row = p.run_cycle(target_old, vis, t)
            t += int(row["duration_ms"])

        # Target is absent while the internal dynamics keep running.
        lost_end = t + 150
        def invisible(_t):
            return False
        while t < lost_end:
            row = p.run_cycle(target_old, invisible, t)
            t += int(row["duration_ms"])

        target = circ(
            target
            + float(rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.8))
        )
        return_start = t
        found = None
        return_end = return_start + 900

        def target_new(_t, target=target):
            return target

        while t < return_end:
            row = p.run_cycle(target_new, vis, t)
            for k, err in enumerate(row["errors"]):
                errors_after.append(err)
                if found is None and err < 0.20:
                    found = (t - return_start) + k
            if np.isfinite(row["mean_value"]):
                values.append(row["mean_value"])
            t += int(row["duration_ms"])

        reacq.append(900 if found is None else found)

    return {
        "reacquisition_ms": float(np.mean(reacq)),
        "success_fraction": float(np.mean(np.asarray(reacq) < 900)),
        "mean_post_return_error_rad": float(np.mean(errors_after)),
        "mean_cycle_value": float(np.mean(values)),
        "cycles_per_second": p.cycles_per_second,
        "mean_cycle_peak_abs_rad": p.mean_cycle_peak,
        "probe_travel_rad_per_ms": p.mean_probe_travel,
    }


WORLDS = {
    "stationary": stationary,
    "pursuit": pursuit,
    "reorientation": reorientation,
}


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 5
    out = {
        world: {
            mode: summarize([
                fn(seed, mode)
                for seed in range(n_seeds)
            ])
            for mode in MODES
        }
        for world, fn in WORLDS.items()
    }
    out["settings"] = {
        "n_seeds": n_seeds,
        "slow_weight_changes": 0,
        "credit_updates": "one per complete theta cycle",
        "population_state_reset": False,
        "stable_reference": True,
    }
    out["question"] = (
        "Does the Vollan-like slow/broad -> fast/narrow transition become "
        "useful when theta frequency changes the number of complete internal "
        "sweeps per unit wall time, rather than merely perturbing a learner "
        "that already receives feedback every millisecond?"
    )
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "fork_cycle_level_control.json").write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
