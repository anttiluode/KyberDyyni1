from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attractor_scanner import AdaptationRingScanner


BASE = {
    "adaptation_mbar": 12.0,
    "theta_period_ms": 100.0,
    "theta_modulation": 0.4,
    "external_gain": 3.0,
    "recurrent_gain": 4.0,
    "width_rad": 0.4,
}

SWEEPS = {
    "adaptation_mbar": [6.0, 9.0, 12.0, 15.0, 18.0],
    "theta_period_ms": [70.0, 85.0, 100.0, 115.0, 130.0],
    "theta_modulation": [0.15, 0.28, 0.40, 0.55, 0.70],
    "external_gain": [1.8, 2.4, 3.0, 3.6, 4.2],
    "recurrent_gain": [2.5, 3.25, 4.0, 4.75, 5.5],
    "width_rad": [0.30, 0.35, 0.40, 0.45, 0.50],
}


def run_one(seed: int, overrides: dict[str, float]) -> dict[str, float]:
    pars = dict(BASE)
    pars.update(overrides)
    s = AdaptationRingScanner(
        n_cells=48,
        seed=seed,
        noise_std=0.04,
        **pars,
    )
    burn = 600
    duration = 3000
    centers = []
    coherence = []

    for t in range(duration):
        row = s.step(0.0)
        if t >= burn:
            center = float(row["center"])
            centers.append(center)
            rate = np.asarray(row["rate"], dtype=float)
            total = float(rate.sum())
            if total > 1e-12:
                z = np.sum(rate * np.exp(1j * s.preferred)) / total
                coherence.append(abs(z))

    centers = np.asarray(centers)
    step_travel = np.abs(
        (np.diff(centers) + np.pi) % (2.0 * np.pi) - np.pi
    )

    n_per = max(2, int(round(s.theta_period_ms / s.dt_ms)))
    n_cycles = len(centers) // n_per
    peaks = []
    for i in range(n_cycles):
        seg = centers[i * n_per : (i + 1) * n_per]
        peaks.append(float(seg[np.argmax(np.abs(seg))]))
    peaks = np.asarray(peaks)
    signs = np.sign(peaks)
    alternation = (
        float(np.mean(signs[1:] * signs[:-1] < 0))
        if len(signs) > 1 else 0.0
    )

    return {
        "mean_cycle_peak_abs_rad": float(np.mean(np.abs(peaks))),
        "alternation_fraction": alternation,
        "mean_step_travel_rad": float(np.mean(step_travel)),
        "mean_bump_coherence": float(np.mean(coherence)),
        "cycles_per_second": 1000.0 / s.theta_period_ms,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 4
    results = {}
    for knob, values in SWEEPS.items():
        rows = {}
        for value in values:
            rows[str(value)] = summarize([
                run_one(seed, {knob: value})
                for seed in range(n_seeds)
            ])
        results[knob] = rows

    out = {
        "one_factor_mechanism_map": results,
        "base_parameters": BASE,
        "n_seeds": n_seeds,
        "question": (
            "Which Ji-like mechanism knobs actually change sweep angle, "
            "alternation, temporal rate, continuity and bump coherence when "
            "varied one at a time?"
        ),
        "warning": (
            "These are mechanism sensitivities, not biological identifications. "
            "A population-level Vollan observation should not be assigned to "
            "one cellular parameter merely because that parameter can move the "
            "same summary statistic."
        ),
    }
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "fork_mechanism_map.json").write_text(
        json.dumps(out, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
