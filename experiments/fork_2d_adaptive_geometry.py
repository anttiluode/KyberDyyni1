from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_2d_sampling_geometry import (
    PATHS,
    WaypointFollower,
    confidence_from_sigma,
    make_target,
)


class AdaptiveFollower:
    def __init__(self):
        self.geometry = None
        self.follower = None
        self.offset = np.zeros(2, dtype=float)

    def set_geometry(self, name: str, radius: float) -> None:
        if name == self.geometry:
            return
        points, pingpong = PATHS[name]
        follower = WaypointFollower(points, pingpong)
        follower.offset = self.offset.copy()

        # Enter the new path near the current continuous state rather than
        # teleporting to waypoint zero.
        scaled = radius * points
        idx = int(np.argmin(np.linalg.norm(scaled - self.offset[None, :], axis=1)))
        follower.index = idx
        self.geometry = name
        self.follower = follower

    def step(self, radius: float, distance_budget: float) -> np.ndarray:
        self.follower.offset = self.offset.copy()
        self.offset = self.follower.step(radius, distance_budget)
        return self.offset.copy()


MODES = [
    "radial_fixed",
    "spiral_fixed",
    "lissajous_fixed",
    "adaptive_radial_spiral",
    "adaptive_radial_lissajous",
    "adaptive_three_way",
    "adaptive_radius_radial",
    "smooth_random_walk",
    "iid_random_unmatched",
    "point_estimate",
]


class FastGeometrySampler:
    """No slow learning: misses change only fast sampling policy."""

    def __init__(self, mode: str, seed: int, period_ms: int = 100):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 81000)
        self.period = period_ms
        self.center = np.zeros(2, dtype=float)
        self.confidence = 0.0
        self.miss_state = 0.0
        self.offset = np.zeros(2, dtype=float)
        self.prev_offset = np.zeros(2, dtype=float)
        self.internal_travel = 0.0
        self.wall_ms = 0
        self.path_per_cycle = []
        self.geometry_counts = {
            "radial_golden": 0,
            "square_spiral": 0,
            "lissajous": 0,
        }
        self.follower = AdaptiveFollower()

    def update_center(self, cue, confidence: float) -> None:
        c = float(np.clip(confidence, 0.0, 1.0))
        self.confidence = 0.80 * self.confidence + 0.20 * c
        if cue is not None:
            gain = 0.82 * max(0.10, c)
            self.center = self.center + gain * (np.asarray(cue) - self.center)

    def choose_geometry(self) -> tuple[str | None, float]:
        miss = float(np.clip(self.miss_state, 0.0, 1.0))
        radius_scale = 1.0

        if self.mode == "radial_fixed":
            return "radial_golden", radius_scale
        if self.mode == "spiral_fixed":
            return "square_spiral", radius_scale
        if self.mode == "lissajous_fixed":
            return "lissajous", radius_scale

        if self.mode == "adaptive_radial_spiral":
            radius_scale = 1.0 + 0.45 * miss
            return (
                "radial_golden" if miss < 0.30 else "square_spiral",
                radius_scale,
            )

        if self.mode == "adaptive_radial_lissajous":
            radius_scale = 1.0 + 0.45 * miss
            return (
                "radial_golden" if miss < 0.30 else "lissajous",
                radius_scale,
            )

        if self.mode == "adaptive_three_way":
            radius_scale = 1.0 + 0.55 * miss
            if miss < 0.22:
                return "radial_golden", radius_scale
            if miss < 0.55:
                return "lissajous", radius_scale
            return "square_spiral", radius_scale

        if self.mode == "adaptive_radius_radial":
            return "radial_golden", 1.0 + 0.70 * miss

        return None, radius_scale

    def reflect_square(self, x: np.ndarray, radius: float) -> np.ndarray:
        y = x.copy()
        for k in range(2):
            if y[k] > radius:
                y[k] = radius - (y[k] - radius)
            if y[k] < -radius:
                y[k] = -radius - (y[k] + radius)
            y[k] = float(np.clip(y[k], -radius, radius))
        return y

    def run_cycle(self, target_fn, start_ms: int, cue, cue_confidence: float):
        self.update_center(cue, cue_confidence)

        base_radius = 0.82 * (1.0 - 0.42 * self.confidence)
        geometry, radius_scale = self.choose_geometry()
        coverage_radius = min(1.25, base_radius * radius_scale)

        # Fairness rule: misses may redistribute/expand the geometry, but they
        # do not buy extra movement. Budget depends only on cue confidence.
        path_budget = 4.0 * base_radius
        step_budget = path_budget / max(1, self.period - 1)

        if geometry is not None:
            self.follower.offset = self.offset.copy()
            self.follower.set_geometry(geometry, coverage_radius)
            self.geometry_counts[geometry] += 1

        hit = 0.0
        best_distance = np.inf
        distances = []
        cycle_travel = 0.0

        for j in range(self.period):
            if geometry is not None:
                offset = self.follower.step(coverage_radius, step_budget)
                self.offset = offset.copy()

            elif self.mode == "smooth_random_walk":
                theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
                proposal = self.offset + step_budget * np.array([
                    np.cos(theta), np.sin(theta)
                ])
                self.offset = self.reflect_square(proposal, coverage_radius)
                offset = self.offset.copy()

            elif self.mode == "iid_random_unmatched":
                offset = self.rng.uniform(
                    -coverage_radius, coverage_radius, size=2
                )

            elif self.mode == "point_estimate":
                offset = np.zeros(2)

            else:
                raise ValueError(self.mode)

            dstep = float(np.linalg.norm(offset - self.prev_offset))
            self.internal_travel += dstep
            cycle_travel += dstep
            self.prev_offset = offset.copy()

            target = np.asarray(target_fn(start_ms + j), dtype=float)
            probe = self.center + offset
            dist = float(np.linalg.norm(probe - target))
            best_distance = min(best_distance, dist)
            distances.append(dist)
            if dist < 0.20:
                hit = 1.0

        # One bit of local consequence changes only fast policy state.
        self.miss_state = 0.84 * self.miss_state + 0.16 * (1.0 - hit)

        self.wall_ms += self.period
        self.path_per_cycle.append(cycle_travel)
        return {
            "hit": hit,
            "best_distance": float(best_distance),
            "mean_distance": float(np.mean(distances)),
            "miss_state": float(self.miss_state),
        }

    @property
    def travel_per_ms(self):
        return self.internal_travel / max(1, self.wall_ms - 1)

    @property
    def mean_path(self):
        return float(np.mean(self.path_per_cycle))


def run_world(seed: int, mode: str, world: str):
    rng = np.random.default_rng(seed + 900)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    sampler = FastGeometrySampler(mode, seed)
    t = 0
    hits, bests, means, miss_states = [], [], [], []
    reacq = []
    return_start = None
    found = None

    while t < horizon:
        if world == "reliable":
            sigma = 0.13
            bias = np.zeros(2)
        elif world == "mixed":
            sigma = [0.13, 0.65, 0.30][(t // 1000) % 3]
            bias = np.zeros(2)
        elif world == "biased":
            sigma = 0.13
            sign = 1.0 if t < horizon // 2 else -1.0
            bias = sign * np.array([0.34, -0.28])
        elif world == "loss":
            local = t % 1800
            sigma = None if 900 <= local < 1300 else 0.13
            bias = np.zeros(2)
            if local >= 1300 and return_start is None:
                return_start, found = t, None
            if local < 900 and return_start is not None:
                reacq.append(600.0 if found is None else float(found))
                return_start = None
        else:
            raise ValueError(world)

        conf = confidence_from_sigma(sigma)
        cue = None if sigma is None else (
            target_fn(t) + bias + rng.normal(0.0, sigma, size=2)
        )

        row = sampler.run_cycle(target_fn, t, cue, conf)
        hits.append(row["hit"])
        bests.append(row["best_distance"])
        means.append(row["mean_distance"])
        miss_states.append(row["miss_state"])

        if (
            world == "loss"
            and return_start is not None
            and found is None
            and row["hit"] > 0.5
        ):
            found = t - return_start

        t += sampler.period

    if world == "loss" and return_start is not None:
        reacq.append(600.0 if found is None else float(found))

    total_geom = max(1, sum(sampler.geometry_counts.values()))
    out = {
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_best_distance": float(np.mean(bests)),
        "mean_probe_distance": float(np.mean(means)),
        "probe_travel_per_ms": sampler.travel_per_ms,
        "mean_path_per_cycle": sampler.mean_path,
        "mean_fast_miss_state": float(np.mean(miss_states)),
        "radial_fraction": sampler.geometry_counts["radial_golden"] / total_geom,
        "spiral_fraction": sampler.geometry_counts["square_spiral"] / total_geom,
        "lissajous_fraction": sampler.geometry_counts["lissajous"] / total_geom,
    }
    if world == "loss":
        out["reacquisition_ms"] = float(np.mean(reacq))
    return out


WORLDS = ["reliable", "mixed", "biased", "loss"]


def summarize(rows):
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main():
    n_seeds = 12
    result = {
        world: {
            mode: summarize([
                run_world(seed, mode, world)
                for seed in range(n_seeds)
            ])
            for mode in MODES
        }
        for world in WORLDS
    }
    result["settings"] = {
        "n_seeds": n_seeds,
        "period_ms": 100,
        "slow_weight_changes": 0,
        "fast_feedback": "one binary hit/miss consequence per cycle",
        "movement_budget": (
            "4 * cue-derived base radius; adaptive expansion gets no extra path"
        ),
    }
    result["question"] = (
        "Can failure of the current 2-D sampling geometry become a fast state "
        "signal that switches or expands the geometry, improving robustness "
        "without changing slow weights?"
    )
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_2d_adaptive_geometry.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
