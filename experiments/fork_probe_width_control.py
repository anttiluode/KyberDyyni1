from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_highdim_probe_scaling import (
    ProbePolicy,
    make_bias,
    relevance,
)


DIMS = [128, 256]
NOISE_SIGMAS = [0.005, 0.01, 0.02, 0.04]
WORLDS = ["dense_bias", "sparse4_bias"]
BUDGET = 512
N_SEEDS = 32

MODES = [
    "fixed_0.10",
    "fixed_0.20",
    "fixed_0.28",
    "fixed_0.40",
    "fixed_0.60",
    "coarse_to_fine",
    "adaptive_contrast",
    "adaptive_contrast_capped",
]


def noisy_relevance(
    distance: float,
    rng: np.random.Generator,
    noise_sigma: float,
) -> float:
    value = relevance(distance)
    if noise_sigma > 0.0:
        value += float(rng.normal(0.0, noise_sigma))
    return float(np.clip(value, 0.0, 1.0))


def fixed_radius(mode: str) -> float | None:
    if not mode.startswith("fixed_"):
        return None
    return float(mode.split("_", 1)[1])


def choose_scheduled_radius(update: int) -> float:
    # 30 full 17-probe updates fit in the 512-measurement budget.
    if update < 8:
        return 0.60
    if update < 16:
        return 0.40
    if update < 24:
        return 0.28
    return 0.20


def run_trial(
    seed: int,
    dim: int,
    world: str,
    noise_sigma: float,
    mode: str,
    *,
    probe_budget: int = BUDGET,
    step_size: float = 0.14,
    success_radius: float = 0.18,
) -> dict[str, float]:
    env_rng = np.random.default_rng(seed + 100000 * dim)
    target = env_rng.normal(0.0, 0.4, size=dim)
    bias = make_bias(env_rng, dim, world)
    cue_noise = env_rng.normal(0.0, 0.08 / np.sqrt(dim), size=dim)
    cue = target + bias + cue_noise

    noise_key = int(round(noise_sigma * 1_000_000))
    measure_rng = np.random.default_rng(
        seed + 900000 + 1000 * dim + 31 * noise_key
    )

    policy = ProbePolicy(dim, "hadamard_block8", seed + 600000)
    fast = np.zeros(dim)

    probes = 0
    updates = 0
    success_at = None
    radius_state = 0.28

    radii = []
    contrast_ratios = []
    gradient_cosines = []

    initial_error = float(np.linalg.norm(target - cue))
    success_value = relevance(success_radius)

    while probes + 17 <= probe_budget:
        working = cue + fast
        correction = target - working
        error = float(np.linalg.norm(correction))

        if success_at is None and error < success_radius:
            success_at = probes

        center_value = noisy_relevance(error, measure_rng, noise_sigma)
        probes += 1

        if center_value >= success_value:
            continue

        radius = fixed_radius(mode)
        if radius is None:
            if mode == "coarse_to_fine":
                radius = choose_scheduled_radius(updates)
            else:
                radius = radius_state

        directions = policy.directions()
        g = np.zeros(dim)
        differences = []

        for u in directions:
            plus_dist = float(np.linalg.norm(
                working + radius * u - target
            ))
            minus_dist = float(np.linalg.norm(
                working - radius * u - target
            ))
            vp = noisy_relevance(plus_dist, measure_rng, noise_sigma)
            vm = noisy_relevance(minus_dist, measure_rng, noise_sigma)
            diff = vp - vm

            differences.append(diff)
            g += diff * u
            probes += 2

        gnorm = float(np.linalg.norm(g))
        if gnorm > 1e-12:
            cosine = float(np.dot(g, correction) / (
                gnorm * max(float(np.linalg.norm(correction)), 1e-12)
            ))
            gradient_cosines.append(cosine)
            fast += step_size * (g / gnorm)
        else:
            gradient_cosines.append(0.0)

        updates += 1
        radii.append(radius)

        diff_noise = max(np.sqrt(2.0) * noise_sigma, 1e-12)
        contrast = float(np.median(np.abs(differences)) / diff_noise)
        contrast_ratios.append(contrast)

        if mode == "adaptive_contrast":
            # Literal version of the planned mechanism: if the paired
            # consequence difference is weak relative to known measurement
            # noise, widen the next sweep; if it is very clear, narrow.
            if contrast < 1.2:
                radius_state = min(0.80, radius_state * 1.50)
            elif contrast > 3.0:
                radius_state = max(0.08, radius_state / 1.25)

        elif mode == "adaptive_contrast_capped":
            # Conservative attacker against the literal controller. It is
            # allowed to adapt, but not to run out to huge radii after a few
            # unlucky noisy measurements.
            if contrast < 1.5:
                radius_state = min(0.50, radius_state * 1.20)
            elif contrast > 3.5:
                radius_state = max(0.16, radius_state / 1.15)

    final_error = float(np.linalg.norm(target - (cue + fast)))
    if success_at is None and final_error < success_radius:
        success_at = probes

    return {
        "initial_error": initial_error,
        "final_error": final_error,
        "fraction_error_removed": float(
            (initial_error - final_error) / max(initial_error, 1e-12)
        ),
        "success": float(final_error < success_radius),
        "probes_to_success": float(
            probe_budget + 1 if success_at is None else success_at
        ),
        "probes_used": float(probes),
        "updates": float(updates),
        "mean_gradient_cosine": float(
            np.mean(gradient_cosines) if gradient_cosines else 0.0
        ),
        "mean_radius": float(np.mean(radii) if radii else 0.0),
        "mean_contrast_to_noise": float(
            np.mean(contrast_ratios) if contrast_ratios else 0.0
        ),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(values.mean())
        out[key + "_std"] = float(values.std())
    return out


def main() -> None:
    result = {
        world: {
            str(dim): {
                str(noise_sigma): {
                    mode: summarize([
                        run_trial(
                            seed,
                            dim,
                            world,
                            noise_sigma,
                            mode,
                        )
                        for seed in range(N_SEEDS)
                    ])
                    for mode in MODES
                }
                for noise_sigma in NOISE_SIGMAS
            }
            for dim in DIMS
        }
        for world in WORLDS
    }

    result["settings"] = {
        "n_seeds": N_SEEDS,
        "dimensions": DIMS,
        "noise_sigmas": NOISE_SIGMAS,
        "probe_budget": BUDGET,
        "success_error_radius": 0.18,
        "basis": "progressive Hadamard blocks of 8 directions",
        "fixed_radii": [0.10, 0.20, 0.28, 0.40, 0.60],
        "slow_weight_changes": 0,
    }
    result["question"] = (
        "Can a fast sampler actively widen probes when scalar-consequence SNR "
        "is poor and narrow them when it is strong, beating fixed probe widths "
        "under the same measurement budget?"
    )
    result["kill_condition"] = (
        "If one fixed radius or a simple coarse-to-fine schedule matches or "
        "beats the adaptive controllers, sweep-width control has not earned "
        "an architectural role in this toy."
    )

    out_path = ROOT / "results" / "fork_probe_width_control.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    compact = {
        world: {
            str(dim): {
                str(noise_sigma): {
                    mode: {
                        "final_error": result[world][str(dim)][
                            str(noise_sigma)
                        ][mode]["final_error"],
                        "success": result[world][str(dim)][
                            str(noise_sigma)
                        ][mode]["success"],
                        "mean_radius": result[world][str(dim)][
                            str(noise_sigma)
                        ][mode]["mean_radius"],
                    }
                    for mode in MODES
                }
                for noise_sigma in NOISE_SIGMAS
            }
            for dim in DIMS
        }
        for world in WORLDS
    }

    print("PROBE WIDTH CONTROL SUMMARY")
    print(json.dumps(compact, indent=2))
    print("\nFull receipt:", out_path)


if __name__ == "__main__":
    main()
