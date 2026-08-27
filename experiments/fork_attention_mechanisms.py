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


class DualDriveScanner:
    """Stable reference plus a second fast attention drive.

    The reference Gaussian never has to move. A separate drive can pull the
    same recurrent bump toward a transient internal sampling direction.
    """

    def __init__(self, seed: int, adaptation_mbar: float = 12.0):
        self.s = AdaptationRingScanner(
            n_cells=48,
            seed=seed,
            noise_std=0.04,
            adaptation_mbar=adaptation_mbar,
        )
        self.base_external = self.s.external_gain
        for _ in range(500):
            self.step(0.0, 0.0, attention_gain=0.0)

    def step(
        self,
        reference: float,
        attention_center: float,
        *,
        reference_gain: float = 1.0,
        attention_gain: float = 0.0,
        asymmetry_gain: float = 0.0,
    ) -> tuple[float, float]:
        s = self.s
        center_before = s.bump_center()
        phase = s.theta_phase()
        theta_gain = 1.0 + s.theta_modulation * np.cos(TAU * phase)

        d_ref = s.circular_distance(s.preferred - float(reference))
        ref = (
            theta_gain
            * self.base_external
            * float(reference_gain)
            * np.exp(-0.25 * (d_ref / s.width_rad) ** 2)
        )

        d_att = s.circular_distance(s.preferred - float(attention_center))
        att = (
            theta_gain
            * self.base_external
            * float(attention_gain)
            * np.exp(-0.25 * (d_att / s.width_rad) ** 2)
        )

        # Broad, non-Gaussian gain field: another plausible way of steering a
        # sampler without moving the stable reference representation itself.
        asym = (
            theta_gain
            * self.base_external
            * float(asymmetry_gain)
            * 0.5
            * (1.0 + np.cos(d_att))
        )

        recurrent = np.real(
            np.fft.ifft(np.fft.fft(s.rate) * s._kernel_fft)
        )
        total = (
            ref
            + att
            + asym
            + recurrent
            + s.rng.normal(scale=s.noise_std, size=s.n_cells)
        )

        du = (-s.activation + total - s.adaptation) / s.tau_ms
        dv = (
            -s.adaptation + s.adaptation_gain * s.activation
        ) / s.adaptation_tau_ms
        s.activation = np.maximum(
            s.activation + s.dt_ms * du, 0.0
        )
        s.adaptation = s.adaptation + s.dt_ms * dv
        squared = s.activation**2
        s.rate = squared / (1.0 + s.k * np.sum(squared))
        s.time_ms += s.dt_ms
        return center_before, phase


class AttentionController:
    MODES = {
        "move_anchor_axis",
        "downstream_explore_contract",
        "dual_drive_fixed",
        "dual_drive_precision",
        "dual_drive_release",
        "asymmetric_gain",
        "multiscale_modules",
        "predictive_downstream",
        "phase_reset_on_return",
        "random_dither",
    }

    def __init__(self, mode: str, seed: int):
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        self.rng = np.random.default_rng(seed + 5000)
        self.bias = 0.0
        self.velocity = 0.0
        self.focus = 0.0
        self.baseline = 0.5
        self.last_bias = 0.0
        self.prev_probe = None
        self.probe_travel = 0.0
        self.reference_travel = 0.0
        self.prev_reference_input = 0.0
        self.steps = 0

        if mode == "random_dither":
            self.scanner = None
        elif mode == "multiscale_modules":
            self.broad = DualDriveScanner(seed, adaptation_mbar=18.0)
            self.narrow = DualDriveScanner(seed + 100, adaptation_mbar=8.0)
            self.scanner = None
        else:
            self.scanner = DualDriveScanner(seed)

    def on_relevance_return(self) -> None:
        if self.mode == "phase_reset_on_return" and self.scanner is not None:
            # A separate hypothesis: salience may launch a fresh sampling
            # cycle rather than change its frequency or adaptation.
            self.scanner.s.time_ms = (
                np.floor(
                    self.scanner.s.time_ms
                    / self.scanner.s.theta_period_ms
                )
                * self.scanner.s.theta_period_ms
            )

    def sample(self, reference: float = 0.0) -> tuple[float, float, float]:
        predicted_bias = self.bias
        if self.mode == "predictive_downstream":
            predicted_bias = circ(self.bias + 22.0 * self.velocity)

        attention_center = circ(reference + predicted_bias)
        sampling_center = attention_center
        phase = 0.0

        if self.mode == "random_dither":
            probe = circ(
                sampling_center + self.rng.uniform(-0.60, 0.60)
            )
            reference_input = reference

        elif self.mode == "move_anchor_axis":
            # Conventional baseline: the generator's external reference moves.
            probe, phase = self.scanner.step(
                attention_center,
                attention_center,
                attention_gain=0.0,
            )
            reference_input = attention_center

        elif self.mode in {
            "downstream_explore_contract",
            "predictive_downstream",
            "phase_reset_on_return",
        }:
            raw, phase = self.scanner.step(
                reference,
                attention_center,
                attention_gain=0.0,
            )
            scale = 1.55 - 0.55 * self.focus
            probe = circ(
                sampling_center + scale * circ(raw - reference)
            )
            reference_input = reference

        elif self.mode == "dual_drive_fixed":
            probe, phase = self.scanner.step(
                reference,
                attention_center,
                reference_gain=1.0,
                attention_gain=0.65,
            )
            reference_input = reference

        elif self.mode == "dual_drive_precision":
            probe, phase = self.scanner.step(
                reference,
                attention_center,
                reference_gain=1.0,
                attention_gain=0.10 + 1.35 * self.focus,
            )
            reference_input = reference

        elif self.mode == "dual_drive_release":
            probe, phase = self.scanner.step(
                reference,
                attention_center,
                reference_gain=1.0 - 0.55 * self.focus,
                attention_gain=0.15 + 1.45 * self.focus,
            )
            reference_input = reference

        elif self.mode == "asymmetric_gain":
            probe, phase = self.scanner.step(
                reference,
                attention_center,
                reference_gain=1.0,
                attention_gain=0.0,
                asymmetry_gain=0.10 + 0.90 * self.focus,
            )
            reference_input = reference

        elif self.mode == "multiscale_modules":
            broad, p1 = self.broad.step(
                reference,
                attention_center,
                attention_gain=0.0,
            )
            narrow, p2 = self.narrow.step(
                reference,
                attention_center,
                attention_gain=0.0,
            )
            # Low confidence reads the high-adaptation / broader module;
            # high confidence reads the low-adaptation / narrower module.
            raw = (1.0 - self.focus) * broad + self.focus * narrow
            probe = circ(
                sampling_center + circ(raw - reference)
            )
            phase = (1.0 - self.focus) * p1 + self.focus * p2
            reference_input = reference

        else:
            raise ValueError(self.mode)

        if self.prev_probe is not None:
            self.probe_travel += abs(circ(probe - self.prev_probe))
        self.prev_probe = probe
        self.reference_travel += abs(
            circ(reference_input - self.prev_reference_input)
        )
        self.prev_reference_input = reference_input
        self.steps += 1
        return probe, float(phase), sampling_center

    def learn_fast(
        self,
        probe: float,
        phase: float,
        sampling_center: float,
        value: float | None,
        eta: float = 0.52,
    ) -> None:
        if value is None:
            self.focus *= 0.995
            self.velocity *= 0.98
            return

        advantage = float(value) - self.baseline
        offset = circ(probe - sampling_center)
        old_bias = self.bias
        self.bias = circ(self.bias + eta * advantage * offset)
        self.bias *= 0.9992

        delta = circ(self.bias - old_bias)
        self.velocity = 0.94 * self.velocity + 0.06 * delta

        confidence = np.clip((float(value) - 0.55) / 0.45, 0.0, 1.0)
        self.focus = 0.94 * self.focus + 0.06 * confidence
        self.baseline = 0.98 * self.baseline + 0.02 * float(value)

    @property
    def mean_probe_travel(self) -> float:
        return self.probe_travel / max(1, self.steps - 1)

    @property
    def mean_reference_travel(self) -> float:
        return self.reference_travel / max(1, self.steps)

    @property
    def compute_multiplier(self) -> float:
        return 2.0 if self.mode == "multiscale_modules" else 1.0


MODES = [
    "move_anchor_axis",
    "downstream_explore_contract",
    "dual_drive_fixed",
    "dual_drive_precision",
    "dual_drive_release",
    "asymmetric_gain",
    "multiscale_modules",
    "predictive_downstream",
    "phase_reset_on_return",
    "random_dither",
]


def stationary(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 100)
    c = AttentionController(mode, seed)
    targets = rng.uniform(-2.7, 2.7, size=5)
    latency, errors, values = [], [], []
    for target in targets:
        got = None
        for t in range(500):
            probe, phase, center = c.sample(0.0)
            value = value_at(probe, target)
            c.learn_fast(probe, phase, center, value)
            err = abs(circ(c.bias - target))
            errors.append(err)
            values.append(value)
            if got is None and err < 0.18:
                got = t
        latency.append(500 if got is None else got)
    return {
        "acquisition_steps": float(np.mean(latency)),
        "success_fraction": float(np.mean(np.asarray(latency) < 500)),
        "mean_error_rad": float(np.mean(errors)),
        "mean_value": float(np.mean(values)),
        "probe_travel": c.mean_probe_travel,
        "reference_travel": c.mean_reference_travel,
        "compute_multiplier": c.compute_multiplier,
    }


def pursuit(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 200)
    c = AttentionController(mode, seed)
    target = float(rng.uniform(-2.0, 2.0))
    velocity = float(rng.choice([-1.0, 1.0]) * 0.0025)
    errors, values = [], []
    for t in range(3000):
        if t > 0 and t % 600 == 0:
            velocity = float(
                np.clip(
                    0.45 * velocity + rng.normal(0.0, 0.0020),
                    -0.005,
                    0.005,
                )
            )
        target = circ(target + velocity)
        probe, phase, center = c.sample(0.0)
        value = value_at(probe, target)
        c.learn_fast(probe, phase, center, value)
        errors.append(abs(circ(c.bias - target)))
        values.append(value)
    return {
        "mean_tracking_error_rad": float(np.mean(errors[500:])),
        "close_fraction": float(np.mean(np.asarray(errors[500:]) < 0.22)),
        "mean_value": float(np.mean(values[500:])),
        "probe_travel": c.mean_probe_travel,
        "reference_travel": c.mean_reference_travel,
        "compute_multiplier": c.compute_multiplier,
    }


def reorient(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 300)
    c = AttentionController(mode, seed)
    target = float(rng.uniform(-2.0, 2.0))
    reacq, errors, values = [], [], []

    for _ in range(5):
        for _ in range(350):
            probe, phase, center = c.sample(0.0)
            value = value_at(probe, target)
            c.learn_fast(probe, phase, center, value)

        for _ in range(100):
            probe, phase, center = c.sample(0.0)
            c.learn_fast(probe, phase, center, None)

        target = circ(
            target
            + float(rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.8))
        )
        c.on_relevance_return()

        got = None
        for t in range(300):
            probe, phase, center = c.sample(0.0)
            value = value_at(probe, target)
            c.learn_fast(probe, phase, center, value)
            err = abs(circ(c.bias - target))
            errors.append(err)
            values.append(value)
            if got is None and err < 0.20:
                got = t
        reacq.append(300 if got is None else got)

    return {
        "reacquisition_steps": float(np.mean(reacq)),
        "success_fraction": float(np.mean(np.asarray(reacq) < 300)),
        "mean_post_return_error_rad": float(np.mean(errors)),
        "mean_value": float(np.mean(values)),
        "probe_travel": c.mean_probe_travel,
        "reference_travel": c.mean_reference_travel,
        "compute_multiplier": c.compute_multiplier,
    }


WORLDS = {
    "stationary": stationary,
    "pursuit": pursuit,
    "reorientation": reorient,
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
        "stable_reference_question": (
            "Can a distinct downstream attention drive steer continuous "
            "sampling while the upstream reference representation itself "
            "remains fixed?"
        ),
    }
    out["warning"] = (
        "These are artificial translations of population-level observations, "
        "not claims about the biological control circuit."
    )
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "fork_attention_mechanisms.json").write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
