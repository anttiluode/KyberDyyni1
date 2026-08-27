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


DIMS = [16, 32, 64, 128, 256]
BUDGETS = [64, 128, 256, 512]
MODES = [
    "full_coordinate_unmatched",
    "coordinate_block8",
    "hadamard_block8",
    "random_orthogonal8",
    "rademacher8",
    "gaussian8",
    "spsa_two_probe",
    "point_estimate",
]
WORLDS = ["dense_bias", "sparse4_bias"]


def batch_cost(dim: int, mode: str) -> int:
    if mode == "point_estimate":
        return 1
    if mode == "full_coordinate_unmatched":
        return 1 + 2 * dim
    if mode in {
        "coordinate_block8",
        "hadamard_block8",
        "random_orthogonal8",
    }:
        return 1 + 2 * min(8, dim)
    if mode in {"rademacher8", "gaussian8"}:
        return 17
    if mode == "spsa_two_probe":
        return 3
    raise ValueError(mode)


def run_budget(
    seed: int,
    dim: int,
    mode: str,
    world: str,
    probe_budget: int,
    *,
    radius: float = 0.28,
    step_size: float = 0.14,
    success_radius: float = 0.18,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 100000 * dim + 1000 * probe_budget)
    target = rng.normal(0.0, 0.4, size=dim)
    bias = make_bias(rng, dim, world)
    cue_noise = rng.normal(0.0, 0.08 / np.sqrt(dim), size=dim)
    cue = target + bias + cue_noise

    fast = np.zeros(dim)
    policy = ProbePolicy(dim, mode, seed + 400000 + probe_budget)

    probes = 0
    updates = 0
    gradient_cosines = []
    success_at = None
    error_at_half_budget = None
    full_batch_cost = batch_cost(dim, mode)

    while probes < probe_budget:
        working = cue + fast
        correction = target - working
        error = float(np.linalg.norm(correction))

        if success_at is None and error < success_radius:
            success_at = probes

        if probes >= probe_budget // 2 and error_at_half_budget is None:
            error_at_half_budget = error

        # A point-estimate method can consume its budget checking the center,
        # but it has no directional information to update from.
        if mode == "point_estimate":
            probes += 1
            continue

        # Do not silently let an algorithm exceed the common budget. A full
        # coordinate method therefore cannot update at all when 2D+1 does not
        # fit inside the available budget.
        if probes + full_batch_cost > probe_budget:
            break

        center_error = error
        center_value = relevance(center_error)
        probes += 1

        if center_error < success_radius:
            # Once already good enough, preserve the calibration rather than
            # forcing extra zeroth-order motion just to consume budget.
            continue

        directions = policy.directions()
        g = np.zeros(dim)

        for u in directions:
            plus_dist = float(np.linalg.norm(
                working + radius * u - target
            ))
            minus_dist = float(np.linalg.norm(
                working - radius * u - target
            ))
            vp = relevance(plus_dist)
            vm = relevance(minus_dist)
            g += (vp - vm) * u
            probes += 2

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

    if error_at_half_budget is None:
        error_at_half_budget = final_error

    return {
        "initial_error": float(np.linalg.norm(target - cue)),
        "half_budget_error": float(error_at_half_budget),
        "final_error": final_error,
        "success": float(final_error < success_radius),
        "probes_to_success": float(
            probe_budget + 1 if success_at is None else success_at
        ),
        "probes_used": float(probes),
        "updates": float(updates),
        "mean_gradient_cosine": float(
            np.mean(gradient_cosines) if gradient_cosines else 0.0
        ),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 24
    result = {
        world: {
            str(dim): {
                str(budget): {
                    mode: summarize([
                        run_budget(
                            seed,
                            dim,
                            mode,
                            world,
                            budget,
                        )
                        for seed in range(n_seeds)
                    ])
                    for mode in MODES
                }
                for budget in BUDGETS
            }
            for dim in DIMS
        }
        for world in WORLDS
    }
    result["settings"] = {
        "n_seeds": n_seeds,
        "dimensions": DIMS,
        "probe_budgets": BUDGETS,
        "paired_probe_radius": 0.28,
        "fast_step_size": 0.14,
        "success_error_radius": 0.18,
        "measurement_noise": 0.0,
        "cue_error_norm": "approximately 0.6",
        "budget_rule": (
            "all scalar relevance evaluations count equally; algorithms stop "
            "rather than exceed the common total probe budget"
        ),
        "slow_weight_changes": 0,
    }
    result["question"] = (
        "When total scalar-relevance evaluations rather than cycles are held "
        "fixed, can progressive structured probe blocks recover high-D cue "
        "corrections with fewer probes than a full O(D) coordinate stencil?"
    )
    result["interpretation_rule"] = (
        "The 10-cycle failure from Fork 1 is a horizon artifact if fixed-block "
        "methods recover under the same total probe budget that full coordinates "
        "consume. Any apparent win remains provisional until scalar measurement "
        "noise is added."
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_highdim_equal_probe_budget.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
