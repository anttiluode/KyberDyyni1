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


class ForkController:
    """One base attractor, many interpretations of 'focused sampling'.

    The controller has a fast internal bias but no slow learning.  Some
    variants move the attractor's external anchor; others keep that reference
    fixed and translate/read the sweep downstream.  This distinction is the
    main biological/computational fork motivated by Vollan et al.
    """

    def __init__(self, mode: str, seed: int):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 9000)
        self.scanner = None if mode == "random_dither" else AdaptationRingScanner(
            n_cells=48,
            seed=seed,
            noise_std=0.04,
        )
        self.bias = 0.0
        self.focus = 0.0
        self.baseline = 0.5
        self.side_score = 0.0
        self.phase_vec = np.array([1.0, 0.0])
        self.prev_probe = None
        self.prev_external_anchor = 0.0
        self.probe_travel = 0.0
        self.external_anchor_travel = 0.0
        self.steps = 0
        self.cycles0 = 0.0

        if self.scanner is not None:
            self.base_period = self.scanner.theta_period_ms
            self.base_adapt = self.scanner.adaptation_gain
            self.base_external = self.scanner.external_gain
            for _ in range(500):
                self.scanner.step(0.0)
            self.cycles0 = self.scanner.time_ms / self.scanner.theta_period_ms

    def _settings(self) -> tuple[float, float]:
        """Return theta-rate factor and downstream sweep-scale."""
        f = self.focus
        rate = 1.0
        scale = 1.0

        if self.mode == "stable_ref_rate":
            rate = 1.0 + 0.75 * f
        elif self.mode == "stable_ref_width":
            scale = 1.0 - 0.65 * f
        elif self.mode == "stable_ref_budget":
            rate = 1.0 + 0.75 * f
            scale = 1.0 / rate
        elif self.mode == "stable_ref_explore_contract":
            # Broad while uncertain; contract toward the ordinary sweep as
            # confidence rises.
            scale = 1.55 - 0.55 * f
        elif self.mode == "old_naive_focus":
            rate = 1.0 + 0.75 * f
        return rate, scale

    def sample(self, reference: float = 0.0) -> tuple[float, float, float]:
        """Return effective probe, raw phase and sampling-center angle."""
        rate, scale = self._settings()

        moving_anchor_modes = {"move_anchor_axis", "old_naive_focus"}
        if self.mode in moving_anchor_modes:
            external_anchor = circ(reference + self.bias)
            sampling_center = external_anchor
        else:
            external_anchor = reference
            sampling_center = circ(reference + self.bias)

        if self.mode == "random_dither":
            raw_probe = circ(reference + self.rng.uniform(-0.60, 0.60))
            phase = 0.0
        else:
            self.scanner.theta_period_ms = self.base_period / rate
            self.scanner.adaptation_gain = self.base_adapt
            self.scanner.external_gain = self.base_external

            if self.mode == "old_naive_focus":
                # The first crude Gate-5 translation, kept as an explicit
                # attacker rather than silently discarded.
                self.scanner.adaptation_gain = self.base_adapt * (
                    1.0 - 0.55 * self.focus
                )
                self.scanner.external_gain = self.base_external * (
                    1.0 + 0.35 * self.focus
                )

            row = self.scanner.step(external_anchor)
            raw_probe = float(row["center"])
            phase = float(row["phase"])

        if self.mode in moving_anchor_modes:
            effective = raw_probe
        else:
            # Stable reference upstream; a downstream internal sampling
            # coordinate is displaced by bias, and may compress/expand the
            # raw sweep around that displaced center.
            raw_offset = circ(raw_probe - reference)
            effective = circ(sampling_center + scale * raw_offset)

        if self.prev_probe is not None:
            self.probe_travel += abs(circ(effective - self.prev_probe))
        self.prev_probe = effective
        self.external_anchor_travel += abs(
            circ(external_anchor - self.prev_external_anchor)
        )
        self.prev_external_anchor = external_anchor
        self.steps += 1
        return effective, phase, sampling_center

    def learn_fast(
        self,
        probe: float,
        phase: float,
        sampling_center: float,
        value: float | None,
        eta: float = 0.52,
    ) -> None:
        if value is None:
            # During occlusion the internal process keeps running, but there is
            # no relevance signal to steer it.
            self.focus *= 0.995
            return

        advantage = float(value) - self.baseline
        offset = circ(probe - sampling_center)

        # Confidence is a fast state, not a learned parameter.
        confidence = np.clip((float(value) - 0.55) / 0.45, 0.0, 1.0)
        self.focus = 0.94 * self.focus + 0.06 * confidence

        gain = advantage

        if self.mode == "stable_ref_gate":
            # Selective transmission: bad samples simply do not steer the
            # downstream internal axis.
            gain = 2.0 * max(0.0, advantage)

        elif self.mode == "stable_ref_side":
            side = np.sign(offset)
            self.side_score = 0.97 * self.side_score + advantage * side
            preferred = np.sign(self.side_score)
            if preferred != 0.0 and side != preferred:
                gain *= 0.12

        elif self.mode == "stable_ref_phase":
            # Learn which theta phase has recently carried useful probes, then
            # privilege steering updates occurring near that phase.
            a = TAU * phase
            unit = np.array([np.cos(a), np.sin(a)])
            if advantage > 0.0:
                self.phase_vec = 0.98 * self.phase_vec + 0.02 * advantage * unit
            norm = np.linalg.norm(self.phase_vec)
            pref = self.phase_vec / norm if norm > 1e-12 else np.array([1.0, 0.0])
            alignment = 0.5 * (1.0 + float(np.dot(pref, unit)))
            gain *= 0.20 + 1.60 * alignment

        if self.mode != "no_relevance":
            self.bias = circ(self.bias + eta * gain * offset)
            self.bias *= 0.9992

        self.baseline = 0.98 * self.baseline + 0.02 * float(value)

    @property
    def cycle_count(self) -> float:
        if self.scanner is None:
            return 0.0
        return (
            self.scanner.time_ms / self.scanner.theta_period_ms
            - self.cycles0
        )

    @property
    def mean_probe_travel(self) -> float:
        return self.probe_travel / max(1, self.steps - 1)

    @property
    def mean_reference_travel(self) -> float:
        return self.external_anchor_travel / max(1, self.steps)


CONTROLLERS = [
    "move_anchor_axis",
    "stable_ref_bias",
    "stable_ref_rate",
    "stable_ref_width",
    "stable_ref_budget",
    "stable_ref_explore_contract",
    "stable_ref_gate",
    "stable_ref_side",
    "stable_ref_phase",
    "old_naive_focus",
    "random_dither",
    "no_relevance",
]


def value_at(probe: float, target: float) -> float:
    return 0.5 + 0.5 * np.cos(circ(probe - target))


def stationary_search(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 100)
    c = ForkController(mode, seed)
    targets = rng.uniform(-2.7, 2.7, size=5)
    latencies = []
    success = []
    errors = []
    values = []

    for target in targets:
        acquired = None
        for t in range(500):
            probe, phase, center = c.sample(0.0)
            value = value_at(probe, target)
            c.learn_fast(probe, phase, center, value)
            err = abs(circ(c.bias - target))
            errors.append(err)
            values.append(value)
            if acquired is None and err < 0.18:
                acquired = t
        latencies.append(500 if acquired is None else acquired)
        success.append(float(acquired is not None))

    return {
        "acquisition_steps": float(np.mean(latencies)),
        "success_fraction": float(np.mean(success)),
        "mean_tracking_error_rad": float(np.mean(errors)),
        "mean_sample_value": float(np.mean(values)),
        "probe_travel_rad_per_step": c.mean_probe_travel,
        "reference_travel_rad_per_step": c.mean_reference_travel,
        "theta_cycles": c.cycle_count,
    }


def moving_pursuit(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 200)
    c = ForkController(mode, seed)
    target = float(rng.uniform(-2.0, 2.0))
    velocity = float(rng.choice([-1.0, 1.0]) * 0.0025)
    errors = []
    values = []
    close = []

    for t in range(3000):
        # Smooth target motion with occasional unpredictable course changes.
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
        err = abs(circ(c.bias - target))
        errors.append(err)
        values.append(value)
        close.append(float(err < 0.22))

    return {
        "mean_tracking_error_rad": float(np.mean(errors[500:])),
        "close_fraction": float(np.mean(close[500:])),
        "mean_sample_value": float(np.mean(values[500:])),
        "probe_travel_rad_per_step": c.mean_probe_travel,
        "reference_travel_rad_per_step": c.mean_reference_travel,
        "theta_cycles": c.cycle_count,
    }


def reorient_after_loss(seed: int, mode: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 300)
    c = ForkController(mode, seed)
    target = float(rng.uniform(-2.0, 2.0))
    reacq = []
    post_errors = []
    values = []

    # Establish tracking, then repeatedly remove relevance, move the target,
    # and restore the signal. The continuous scanner is never reset.
    for episode in range(5):
        for _ in range(350):
            probe, phase, center = c.sample(0.0)
            value = value_at(probe, target)
            c.learn_fast(probe, phase, center, value)

        # "lost target": no value signal while internal dynamics continue.
        for _ in range(100):
            probe, phase, center = c.sample(0.0)
            c.learn_fast(probe, phase, center, None)

        jump = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 1.8))
        target = circ(target + jump)

        found = None
        for t in range(300):
            probe, phase, center = c.sample(0.0)
            value = value_at(probe, target)
            c.learn_fast(probe, phase, center, value)
            err = abs(circ(c.bias - target))
            post_errors.append(err)
            values.append(value)
            if found is None and err < 0.20:
                found = t
        reacq.append(300 if found is None else found)

    return {
        "reacquisition_steps": float(np.mean(reacq)),
        "reacquisition_success": float(np.mean(np.asarray(reacq) < 300)),
        "post_return_error_rad": float(np.mean(post_errors)),
        "mean_sample_value": float(np.mean(values)),
        "probe_travel_rad_per_step": c.mean_probe_travel,
        "reference_travel_rad_per_step": c.mean_reference_travel,
        "theta_cycles": c.cycle_count,
    }


WORLDS = {
    "stationary_search": stationary_search,
    "moving_pursuit": moving_pursuit,
    "reorient_after_loss": reorient_after_loss,
}


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 6
    result = {
        world_name: {
            mode: summarize([
                world_fn(seed, mode)
                for seed in range(n_seeds)
            ])
            for mode in CONTROLLERS
        }
        for world_name, world_fn in WORLDS.items()
    }
    result["settings"] = {
        "n_seeds": n_seeds,
        "controllers": CONTROLLERS,
        "slow_weight_changes": 0,
        "base_scanner_cells": 48,
        "principle": (
            "separate stable reference, internal sampling axis, angular sweep "
            "scale, theta rate and downstream selection instead of mapping "
            "Vollan's population-level observations onto one cellular knob"
        ),
    }
    result["interpretation_rule"] = (
        "Do not select a winner from one world. Look for controller laws that "
        "survive stationary acquisition, moving pursuit and loss/reorientation, "
        "and report continuity/reference-motion costs alongside accuracy."
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_control_laws.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
