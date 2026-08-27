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


DIMS = [64, 128, 256]
NOISE_SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.04]
WORLDS = ["dense_bias", "sparse4_bias"]
BUDGET = 512
N_SEEDS = 32

METHODS = [
    "full_coordinate",
    "coordinate_block8",
    "hadamard_block8",
    "hadamard_repeat2",
    "hadamard_adaptive_repeat",
    "random_orthogonal8",
    "rademacher8",
    "spsa_two_probe",
    "point_estimate",
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


def base_policy_mode(method: str) -> str:
    if method in {"hadamard_repeat2", "hadamard_adaptive_repeat"}:
        return "hadamard_block8"
    if method == "full_coordinate":
        return "full_coordinate_unmatched"
    return method


def complete_batch_floor(dim: int, method: str) -> int:
    if method == "point_estimate":
        return 1
    if method == "full_coordinate":
        return 1 + 2 * dim
    if method == "hadamard_repeat2":
        return 1 + 4 * min(8, dim)
    if method == "hadamard_adaptive_repeat":
        # One repetition is the minimum. Extra repetitions are chosen online.
        return 1 + 2 * min(8, dim)
    if method in {
        "coordinate_block8",
        "hadamard_block8",
        "random_orthogonal8",
        "rademacher8",
    }:
        return 17
    if method == "spsa_two_probe":
        return 3
    raise ValueError(method)


def paired_difference(
    working: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
    radius: float,
    rng: np.random.Generator,
    noise_sigma: float,
    repeats: int,
) -> tuple[float, int]:
    diffs = []
    for _ in range(repeats):
        plus_dist = float(np.linalg.norm(working + radius * direction - target))
        minus_dist = float(np.linalg.norm(working - radius * direction - target))
        vp = noisy_relevance(plus_dist, rng, noise_sigma)
        vm = noisy_relevance(minus_dist, rng, noise_sigma)
        diffs.append(vp - vm)
    return float(np.mean(diffs)), 2 * repeats


def adaptive_paired_difference(
    working: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
    radius: float,
    rng: np.random.Generator,
    noise_sigma: float,
    remaining_budget: int,
    *,
    max_repeats: int = 4,
    z: float = 1.5,
) -> tuple[float, int, int]:
    diffs = []
    used = 0

    for repeat in range(1, max_repeats + 1):
        if remaining_budget - used < 2:
            break

        diff, cost = paired_difference(
            working,
            target,
            direction,
            radius,
            rng,
            noise_sigma,
            repeats=1,
        )
        diffs.append(diff)
        used += cost

        if noise_sigma <= 0.0:
            break

        # Difference of two independent scalar measurements has sqrt(2)*sigma
        # noise. Repetition lowers uncertainty as 1/sqrt(n). Stop once the
        # observed sign is reasonably resolved; otherwise spend another pair.
        stderr = np.sqrt(2.0) * noise_sigma / np.sqrt(repeat)
        if abs(float(np.mean(diffs))) >= z * stderr:
            break

    if not diffs:
        return 0.0, 0, 0
    return float(np.mean(diffs)), used, len(diffs)


def run_trial(
    seed: int,
    dim: int,
    method: str,
    world: str,
    noise_sigma: float,
    *,
    probe_budget: int = BUDGET,
    radius: float = 0.28,
    step_size: float = 0.14,
    success_radius: float = 0.18,
) -> dict[str, float]:
    # Hold the underlying world fixed across noise levels for paired comparison.
    env_rng = np.random.default_rng(seed + 100000 * dim)
    target = env_rng.normal(0.0, 0.4, size=dim)
    bias = make_bias(env_rng, dim, world)
    cue_noise = env_rng.normal(0.0, 0.08 / np.sqrt(dim), size=dim)
    cue = target + bias + cue_noise

    # Measurement noise has its own reproducible stream.
    noise_key = int(round(noise_sigma * 1_000_000))
    measure_rng = np.random.default_rng(
        seed + 700000 + 1000 * dim + 31 * noise_key
    )

    fast = np.zeros(dim)
    policy = ProbePolicy(dim, base_policy_mode(method), seed + 500000)

    probes = 0
    updates = 0
    gradient_cosines = []
    repeat_counts = []
    success_at = None

    initial_error = float(np.linalg.norm(target - cue))
    success_value = relevance(success_radius)

    while probes < probe_budget:
        working = cue + fast
        correction = target - working
        error = float(np.linalg.norm(correction))

        if success_at is None and error < success_radius:
            success_at = probes

        if method == "point_estimate":
            noisy_relevance(error, measure_rng, noise_sigma)
            probes += 1
            continue

        # Full-coordinate and fixed-repeat methods only update if a complete
        # batch fits. This keeps the common scalar-evaluation budget honest.
        minimum_cost = complete_batch_floor(dim, method)
        if probes + minimum_cost > probe_budget:
            break

        center_value = noisy_relevance(error, measure_rng, noise_sigma)
        probes += 1

        # The machine may decide it is already calibrated only from the noisy
        # scalar consequence. It does not get privileged access to true error.
        if center_value >= success_value:
            continue

        directions = policy.directions()
        g = np.zeros(dim)
        direction_repeats = []

        complete = True
        for u in directions:
            if method == "hadamard_repeat2":
                if probes + 4 > probe_budget:
                    complete = False
                    break
                diff, cost = paired_difference(
                    working,
                    target,
                    u,
                    radius,
                    measure_rng,
                    noise_sigma,
                    repeats=2,
                )
                reps = 2

            elif method == "hadamard_adaptive_repeat":
                diff, cost, reps = adaptive_paired_difference(
                    working,
                    target,
                    u,
                    radius,
                    measure_rng,
                    noise_sigma,
                    probe_budget - probes,
                )
                if cost == 0:
                    complete = False
                    break

            else:
                if probes + 2 > probe_budget:
                    complete = False
                    break
                diff, cost = paired_difference(
                    working,
                    target,
                    u,
                    radius,
                    measure_rng,
                    noise_sigma,
                    repeats=1,
                )
                reps = 1

            g += diff * u
            probes += cost
            direction_repeats.append(reps)

        if not complete:
            break

        if direction_repeats:
            repeat_counts.extend(direction_repeats)

        gnorm = float(np.linalg.norm(g))
        if gnorm <= 1e-12:
            gradient_cosines.append(0.0)
            updates += 1
            continue

        cosine = float(np.dot(g, correction) / (
            gnorm * max(float(np.linalg.norm(correction)), 1e-12)
        ))
        gradient_cosines.append(cosine)

        fast += step_size * (g / gnorm)
        updates += 1

    final_error = float(np.linalg.norm(target - (cue + fast)))
    if success_at is None and final_error < success_radius:
        success_at = probes

    removed = (initial_error - final_error) / max(initial_error, 1e-12)

    return {
        "initial_error": initial_error,
        "final_error": final_error,
        "fraction_error_removed": float(removed),
        "success": float(final_error < success_radius),
        "probes_to_success": float(
            probe_budget + 1 if success_at is None else success_at
        ),
        "probes_used": float(probes),
        "updates": float(updates),
        "mean_gradient_cosine": float(
            np.mean(gradient_cosines) if gradient_cosines else 0.0
        ),
        "mean_repetitions_per_direction": float(
            np.mean(repeat_counts) if repeat_counts else 0.0
        ),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(values.mean())
        out[key + "_std"] = float(values.std())
    return out


def compact_receipt(result: dict) -> dict:
    keep_methods = [
        "full_coordinate",
        "coordinate_block8",
        "hadamard_block8",
        "hadamard_repeat2",
        "hadamard_adaptive_repeat",
        "random_orthogonal8",
        "spsa_two_probe",
        "point_estimate",
    ]
    compact = {}
    for world in WORLDS:
        compact[world] = {}
        for dim in DIMS:
            compact[world][str(dim)] = {}
            for noise_sigma in NOISE_SIGMAS:
                key = str(noise_sigma)
                compact[world][str(dim)][key] = {
                    method: {
                        "final_error": result[world][str(dim)][key][method][
                            "final_error"
                        ],
                        "success": result[world][str(dim)][key][method]["success"],
                        "gradient_cosine": result[world][str(dim)][key][method][
                            "mean_gradient_cosine"
                        ],
                        "mean_repeats": result[world][str(dim)][key][method][
                            "mean_repetitions_per_direction"
                        ],
                    }
                    for method in keep_methods
                }
    return compact


def main() -> None:
    result = {
        world: {
            str(dim): {
                str(noise_sigma): {
                    method: summarize([
                        run_trial(
                            seed,
                            dim,
                            method,
                            world,
                            noise_sigma,
                        )
                        for seed in range(N_SEEDS)
                    ])
                    for method in METHODS
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
        "paired_probe_radius": 0.28,
        "fast_step_size": 0.14,
        "success_error_radius": 0.18,
        "measurement_noise": (
            "independent additive Gaussian noise on each scalar relevance "
            "measurement, clipped to [0,1]"
        ),
        "world_is_paired_across_noise_levels": True,
        "slow_weight_changes": 0,
    }
    result["question"] = (
        "How much scalar consequence noise can the dynamic partial-information "
        "calibration machine tolerate as latent dimension grows?"
    )
    result["interpretation_rule"] = (
        "A compressed probe method earns a robustness claim only if it improves "
        "true error under the same 512 scalar-evaluation budget and remains "
        "better than point-estimate/no-probe control. Repetition only earns its "
        "cost if improved consequence SNR outweighs the lost update cycles."
    )

    output_path = ROOT / "results" / "fork_highdim_measurement_noise.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("NOISY CONSEQUENCE SUMMARY")
    print(json.dumps(compact_receipt(result), indent=2))
    print("\nFull receipt:", output_path)


if __name__ == "__main__":
    main()
