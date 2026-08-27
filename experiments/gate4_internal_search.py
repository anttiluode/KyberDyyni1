from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attractor_scanner import AdaptationRingScanner
from kyberdyyni import ThetaScanner


def circular(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def run(
    seed: int,
    mode: str,
    *,
    n_targets: int = 8,
    steps_per_target: int = 1200,
    learning_rate: float = 0.12,
    baseline_rate: float = 0.02,
) -> dict[str, float]:
    """Track a hidden moving target using only values at internal probes.

    There is no slow structural learning. All active search modes share the
    same zeroth-order steering rule:

        anchor += eta * (value - running_baseline) * (probe - anchor)

    The only difference is how candidate probes are generated.
    """
    rng = np.random.default_rng(seed + 123)
    anchor = 0.0
    baseline = 0.5

    if mode == "adaptation_attractor":
        scanner = AdaptationRingScanner(seed=seed, noise_std=0.05)
        # Let the population settle once. It is never reset at target jumps.
        for _ in range(1000):
            scanner.step(anchor)
    elif mode == "engineered_sweep":
        # Match the attractor's ~100-ms theta period and ~0.6-rad amplitude.
        scanner = ThetaScanner(
            base_frequency=0.01,
            base_width=0.6,
        )
    else:
        scanner = None

    targets = rng.uniform(-np.pi, np.pi, size=n_targets)
    acquisition = []
    errors = []
    values = []
    travel = []
    previous_probe = None

    for target in targets:
        acquired = None

        for t in range(steps_per_target):
            if mode == "adaptation_attractor":
                probe = float(scanner.step(anchor)["center"])
            elif mode == "engineered_sweep":
                probe = circular(
                    float(scanner.step(anchor)["probe"])
                )
            elif mode == "random_dither":
                probe = circular(
                    anchor + rng.uniform(-0.6, 0.6)
                )
            elif mode == "static_anchor":
                probe = anchor
            else:
                raise ValueError(mode)

            if previous_probe is not None:
                travel.append(
                    abs(circular(probe - previous_probe))
                )
            previous_probe = probe

            # Smooth hidden utility field. The searcher is not given target
            # direction or a derivative, only value at its current probe.
            distance = circular(probe - float(target))
            value = 0.5 + 0.5 * np.cos(distance)

            advantage = value - baseline
            probe_offset = circular(probe - anchor)
            anchor = circular(
                anchor
                + learning_rate * advantage * probe_offset
            )
            baseline = (
                (1.0 - baseline_rate) * baseline
                + baseline_rate * value
            )

            error = abs(circular(anchor - float(target)))
            errors.append(error)
            values.append(value)

            if acquired is None and error < 0.15:
                acquired = t

        acquisition.append(
            steps_per_target if acquired is None else acquired
        )

    acquisition = np.asarray(acquisition)
    return {
        "mean_acquisition_steps": float(
            np.mean(acquisition)
        ),
        "success_fraction": float(
            np.mean(acquisition < steps_per_target)
        ),
        "mean_tracking_error_rad": float(
            np.mean(errors)
        ),
        "mean_probe_travel_rad_per_step": float(
            np.mean(travel)
        ),
        "mean_sample_value": float(
            np.mean(values)
        ),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        x = np.asarray([row[key] for row in rows])
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    modes = {
        "adaptation_attractor": "adaptation_attractor",
        "engineered_sweep_attacker": "engineered_sweep",
        "random_dither_attacker": "random_dither",
        "static_no_sweep_control": "static_anchor",
    }
    result = {
        name: summarize([
            run(seed, mode)
            for seed in range(10)
        ])
        for name, mode in modes.items()
    }
    result["settings"] = {
        "n_seeds": 10,
        "target_jumps_per_seed": 8,
        "steps_per_target": 1200,
        "acquired_threshold_rad": 0.15,
        "probe_amplitude_rad": 0.6,
        "slow_weight_changes": 0,
    }
    result["claim"] = (
        "a continuous internally generated sweep can support useful "
        "zeroth-order target search before any slow structural learning"
    )
    result["boundary"] = (
        "the adaptation-driven scanner is not the fastest searcher here: "
        "engineered sweeps and especially random dither acquire targets "
        "faster; its distinguishing property in this gate is smooth low-travel "
        "sampling rather than raw search speed"
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "gate4_internal_search.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
