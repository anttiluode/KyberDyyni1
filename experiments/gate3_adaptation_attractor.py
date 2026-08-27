from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attractor_scanner import AdaptationRingScanner


def run_one(
    seed: int,
    *,
    adaptation_mbar: float = 12.0,
    theta_modulation: float = 0.4,
    recurrent_gain: float = 4.0,
    duration_ms: int = 6000,
    burn_ms: int = 1000,
) -> dict[str, float]:
    scanner = AdaptationRingScanner(
        seed=seed,
        adaptation_mbar=adaptation_mbar,
        theta_modulation=theta_modulation,
        recurrent_gain=recurrent_gain,
    )

    center = np.asarray([
        scanner.step(0.0)["center"]
        for _ in range(duration_ms)
    ])
    center = center[burn_ms:]

    samples_per_cycle = int(
        round(scanner.theta_period_ms / scanner.dt_ms)
    )
    n_cycles = len(center) // samples_per_cycle

    # For each theta cycle, extract the side and amplitude of the furthest
    # point reached by the bump. This is a measurement only; it does not feed
    # back into the scanner.
    peak_offset = []
    for i in range(n_cycles):
        segment = center[
            i * samples_per_cycle : (i + 1) * samples_per_cycle
        ]
        peak_offset.append(
            float(segment[np.argmax(np.abs(segment))])
        )
    peak_offset = np.asarray(peak_offset)

    sign = np.sign(peak_offset)
    alternation = float(
        np.mean(sign[1:] * sign[:-1] < 0)
    )

    return {
        "mean_cycle_peak_abs_rad": float(
            np.mean(np.abs(peak_offset))
        ),
        "cycle_to_cycle_alternation": alternation,
        "center_std_rad": float(np.std(center)),
        "mean_center_abs_rad": float(np.mean(np.abs(center))),
        "max_center_abs_rad": float(np.max(np.abs(center))),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        a = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(a.mean())
        out[key + "_std"] = float(a.std())
    return out


def main() -> None:
    conditions = {
        "full_adaptation_theta_recurrence": {},
        "no_adaptation": {
            "adaptation_mbar": 0.0,
        },
        "no_theta_modulation": {
            "theta_modulation": 0.0,
        },
        "no_recurrence": {
            "recurrent_gain": 0.0,
        },
    }

    result = {
        name: summarize([
            run_one(seed, **kwargs)
            for seed in range(10)
        ])
        for name, kwargs in conditions.items()
    }
    result["settings"] = {
        "n_seeds": 10,
        "duration_ms": 6000,
        "burn_ms": 1000,
        "theta_period_ms": 100,
        "n_cells": 100,
        "anchor_angle_rad": 0.0,
    }
    result["claim"] = (
        "bounded alternating bump sweeps emerge from recurrent attraction, "
        "slow adaptation and theta modulation; no explicit left/right parity "
        "rule appears in the scanner"
    )
    result["attacker_note"] = (
        "the engineered ThetaScanner remains much simpler and alternates by "
        "construction; Gate 3 establishes emergence, not engineering superiority"
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "gate3_adaptation_attractor.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
