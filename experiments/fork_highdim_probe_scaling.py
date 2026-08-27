from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DIMS = [2, 4, 8, 16, 32, 64]
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


def hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.ones((1, 1), dtype=float)
    h = hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def relevance(distance: float, sigma: float = 0.40) -> float:
    return float(np.exp(-0.5 * (distance / sigma) ** 2))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def make_bias(rng, dim: int, world: str, magnitude: float = 0.60) -> np.ndarray:
    if world == "dense_bias":
        return magnitude * unit(rng.normal(size=dim))
    if world == "sparse4_bias":
        k = min(4, dim)
        idx = rng.choice(dim, size=k, replace=False)
        v = np.zeros(dim)
        v[idx] = rng.normal(size=k)
        return magnitude * unit(v)
    raise ValueError(world)


class ProbePolicy:
    def __init__(self, dim: int, mode: str, seed: int):
        self.dim = dim
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.cursor = 0
        self.h = hadamard(dim) / np.sqrt(dim)

    def directions(self) -> np.ndarray:
        d = self.dim
        if self.mode == "point_estimate":
            return np.zeros((0, d))

        if self.mode == "full_coordinate_unmatched":
            return np.eye(d)

        if self.mode == "coordinate_block8":
            m = min(8, d)
            idx = [(self.cursor + i) % d for i in range(m)]
            self.cursor = (self.cursor + m) % d
            return np.eye(d)[idx]

        if self.mode == "hadamard_block8":
            m = min(8, d)
            idx = [(self.cursor + i) % d for i in range(m)]
            self.cursor = (self.cursor + m) % d
            return self.h[idx]

        if self.mode == "random_orthogonal8":
            m = min(8, d)
            q, _ = np.linalg.qr(self.rng.normal(size=(d, m)))
            return q[:, :m].T

        if self.mode == "rademacher8":
            m = 8
            return self.rng.choice([-1.0, 1.0], size=(m, d)) / np.sqrt(d)

        if self.mode == "gaussian8":
            m = 8
            rows = self.rng.normal(size=(m, d))
            rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
            return rows

        if self.mode == "spsa_two_probe":
            return self.rng.choice([-1.0, 1.0], size=(1, d)) / np.sqrt(d)

        raise ValueError(self.mode)


def probe_travel(directions: np.ndarray, radius: float) -> float:
    """Path length through center,+u,-u pairs in the order actually sampled."""
    pos = np.zeros(directions.shape[1] if len(directions) else 1)
    total = 0.0
    for u in directions:
        plus = radius * u
        minus = -radius * u
        total += float(np.linalg.norm(plus - pos))
        total += float(np.linalg.norm(minus - plus))
        pos = minus
    total += float(np.linalg.norm(pos))
    return total


def run_trial(
    seed: int,
    dim: int,
    mode: str,
    world: str,
    *,
    cycles: int = 10,
    radius: float = 0.28,
    step_size: float = 0.14,
    success_radius: float = 0.18,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 100000 * dim)
    target = rng.normal(0.0, 0.4, size=dim)
    bias = make_bias(rng, dim, world)
    noise = rng.normal(0.0, 0.08 / np.sqrt(dim), size=dim)
    cue = target + bias + noise

    fast = np.zeros(dim)
    policy = ProbePolicy(dim, mode, seed + 300000)
    errors = []
    gradient_cosines = []
    probes_used = 0
    path_used = 0.0
    success_cycle = None
    total_probes_at_success = None

    for cycle in range(cycles):
        working = cue + fast
        true_correction = target - working
        error = float(np.linalg.norm(true_correction))
        errors.append(error)

        if success_cycle is None and error < success_radius:
            success_cycle = cycle
            total_probes_at_success = probes_used

        center_value = relevance(error)
        probes_used += 1

        if center_value >= relevance(success_radius):
            gradient_cosines.append(1.0)
            continue

        directions = policy.directions()
        if len(directions) == 0:
            gradient_cosines.append(0.0)
            continue

        g = np.zeros(dim)
        for u in directions:
            plus_dist = float(np.linalg.norm(working + radius * u - target))
            minus_dist = float(np.linalg.norm(working - radius * u - target))
            vp = relevance(plus_dist)
            vm = relevance(minus_dist)
            g += (vp - vm) * u
            probes_used += 2

        path_used += probe_travel(directions, radius)

        gnorm = float(np.linalg.norm(g))
        if gnorm <= 1e-12:
            gradient_cosines.append(0.0)
            continue

        cosine = float(np.dot(g, true_correction) / (
            gnorm * max(float(np.linalg.norm(true_correction)), 1e-12)
        ))
        gradient_cosines.append(cosine)

        fast += step_size * (g / gnorm)

    final_error = float(np.linalg.norm(target - (cue + fast)))
    if success_cycle is None and final_error < success_radius:
        success_cycle = cycles
        total_probes_at_success = probes_used

    return {
        "initial_error": errors[0],
        "error_after_1": float(errors[1] if len(errors) > 1 else final_error),
        "error_after_4": float(errors[4] if len(errors) > 4 else final_error),
        "final_error": final_error,
        "success": float(final_error < success_radius),
        "cycles_to_success": float(
            cycles + 1 if success_cycle is None else success_cycle
        ),
        "probes_to_success": float(
            probes_used if total_probes_at_success is None
            else total_probes_at_success
        ),
        "total_probes": float(probes_used),
        "mean_gradient_cosine": float(np.mean(gradient_cosines)),
        "path_per_cycle": float(path_used / cycles),
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
                mode: summarize([
                    run_trial(seed, dim, mode, world)
                    for seed in range(n_seeds)
                ])
                for mode in MODES
            }
            for dim in DIMS
        }
        for world in WORLDS
    }
    result["settings"] = {
        "n_seeds": n_seeds,
        "dimensions": DIMS,
        "cycles": 10,
        "paired_probe_radius": 0.28,
        "fast_step_size": 0.14,
        "success_error_radius": 0.18,
        "fixed_block": (
            "8 directions / 16 directional probes + one center probe per cycle"
        ),
        "full_coordinate_unmatched": "2D directional probes + center probe",
        "spsa_two_probe": "2 directional probes + center probe",
        "slow_weight_changes": 0,
    }
    result["question"] = (
        "Can fast cue calibration survive increasing latent dimension with a "
        "fixed directional-probe budget, or does useful calibration require "
        "O(D) coordinate probing?"
    )
    result["interpretation_rule"] = (
        "Compare both error and probes-to-success. Full coordinates are an "
        "expensive O(D) upper bound. A fixed-budget method earns a scaling "
        "advantage only if its accuracy degrades substantially slower than "
        "coordinate-block coverage as D increases."
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_highdim_probe_scaling.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
