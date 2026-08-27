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


MODES = [
    "radial_fixed",
    "radial_hit_recenter",
    "radial_contrast_recenter",
    "spiral_contrast_recenter",
    "smooth_contrast_recenter",
    "iid_contrast_recenter",
    "point_estimate",
]


def relevance(distance: float) -> float:
    # Smooth local consequence. No target direction or derivative is exposed.
    return float(np.exp(-0.5 * (distance / 0.30) ** 2))


class FastRecenterSampler:
    """Structured sampler + fast local calibration state, no slow learning."""

    def __init__(self, mode: str, seed: int, period_ms: int = 100):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 91000)
        self.period = period_ms
        self.cue_center = np.zeros(2, dtype=float)
        self.fast_offset = np.zeros(2, dtype=float)
        self.confidence = 0.0
        self.local_offset = np.zeros(2, dtype=float)
        self.prev_local_offset = np.zeros(2, dtype=float)
        self.internal_travel = 0.0
        self.wall_ms = 0
        self.path_per_cycle = []
        self.fast_offset_norm = []

        if mode.startswith("radial"):
            points, pingpong = PATHS["radial_golden"]
            self.follower = WaypointFollower(points, pingpong)
        elif mode.startswith("spiral"):
            points, pingpong = PATHS["square_spiral"]
            self.follower = WaypointFollower(points, pingpong)
        else:
            self.follower = None

    def update_cue_center(self, cue, confidence: float):
        c = float(np.clip(confidence, 0.0, 1.0))
        self.confidence = 0.80 * self.confidence + 0.20 * c
        if cue is not None:
            gain = 0.82 * max(0.10, c)
            self.cue_center = self.cue_center + gain * (
                np.asarray(cue) - self.cue_center
            )

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
        self.update_cue_center(cue, cue_confidence)

        radius = 0.82 * (1.0 - 0.42 * self.confidence)
        path_budget = 4.0 * radius
        step_budget = path_budget / max(1, self.period - 1)

        values = []
        local_offsets = []
        hit_offsets = []
        distances = []
        hit = 0.0
        cycle_travel = 0.0

        working_center = self.cue_center + self.fast_offset

        for j in range(self.period):
            if self.follower is not None:
                self.follower.offset = self.local_offset.copy()
                local = self.follower.step(radius, step_budget)
                self.local_offset = local.copy()
            elif self.mode == "smooth_contrast_recenter":
                theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
                proposal = self.local_offset + step_budget * np.array([
                    np.cos(theta), np.sin(theta)
                ])
                self.local_offset = self.reflect_square(proposal, radius)
                local = self.local_offset.copy()
            elif self.mode == "iid_contrast_recenter":
                local = self.rng.uniform(-radius, radius, size=2)
            elif self.mode == "point_estimate":
                local = np.zeros(2)
            else:
                raise ValueError(self.mode)

            dstep = float(np.linalg.norm(local - self.prev_local_offset))
            self.internal_travel += dstep
            cycle_travel += dstep
            self.prev_local_offset = local.copy()

            target = np.asarray(target_fn(start_ms + j), dtype=float)
            probe = working_center + local
            dist = float(np.linalg.norm(probe - target))
            val = relevance(dist)

            values.append(val)
            local_offsets.append(local.copy())
            distances.append(dist)
            if dist < 0.20:
                hit = 1.0
                hit_offsets.append(local.copy())

        vals = np.asarray(values)
        offs = np.asarray(local_offsets)

        if self.mode == "radial_hit_recenter" and hit_offsets:
            delta = np.mean(np.asarray(hit_offsets), axis=0)
            self.fast_offset += 0.55 * delta

        elif "contrast_recenter" in self.mode:
            # Local zeroth-order steering. The baseline is computed inside the
            # current sweep, so only relative relevance of self-generated
            # samples matters.
            advantage = vals - float(np.mean(vals))
            denom = float(np.sum(np.abs(advantage)))
            if denom > 1e-10:
                delta = np.sum(
                    advantage[:, None] * offs, axis=0
                ) / denom
                self.fast_offset += 0.75 * delta

        # Fast calibration is elastic, not slow memory.
        self.fast_offset *= 0.992
        norm = float(np.linalg.norm(self.fast_offset))
        if norm > 1.5:
            self.fast_offset *= 1.5 / norm

        self.wall_ms += self.period
        self.path_per_cycle.append(cycle_travel)
        self.fast_offset_norm.append(float(np.linalg.norm(self.fast_offset)))

        return {
            "hit": hit,
            "best_distance": float(np.min(distances)),
            "mean_distance": float(np.mean(distances)),
            "mean_relevance": float(np.mean(vals)),
            "fast_offset_norm": float(np.linalg.norm(self.fast_offset)),
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

    sampler = FastRecenterSampler(mode, seed)
    t = 0
    hits, bests, means, relevance_rows, offset_norms = [], [], [], [], []
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
        relevance_rows.append(row["mean_relevance"])
        offset_norms.append(row["fast_offset_norm"])

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

    out = {
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_best_distance": float(np.mean(bests)),
        "mean_probe_distance": float(np.mean(means)),
        "mean_relevance": float(np.mean(relevance_rows)),
        "mean_fast_offset_norm": float(np.mean(offset_norms)),
        "probe_travel_per_ms": sampler.travel_per_ms,
        "mean_path_per_cycle": sampler.mean_path,
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
        "continuous_budget": "4 * cue-derived radius per cycle",
        "fast_state": (
            "2-D cue-calibration offset updated only from local sample "
            "relevance; decays elastically and is never consolidated"
        ),
    }
    result["question"] = (
        "Can structured samples calibrate a noisy or biased fast control cue "
        "through an elastic 2-D state, without asking the sampler to discover "
        "the target from scratch or changing slow weights?"
    )
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_2d_fast_recenter.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
