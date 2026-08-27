from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_2d_sampling_geometry import (
    PATHS,
    confidence_from_sigma,
    make_target,
)
from experiments.fork_2d_fast_recenter import relevance


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    return np.asarray(points) @ r.T


CARDINAL = np.asarray([
    [0.0, 0.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [0.0, 0.0],
])

OCTAGON = np.asarray([
    [np.cos(a), np.sin(a)]
    for a in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
])

MODES = [
    "radial_golden",
    "cardinal_cross",
    "diagonal_cross",
    "rotating_cross",
    "octagon_ring",
    "spsa_line",
    "smooth_random_walk",
    "iid_random_unmatched",
    "point_estimate",
]


class DynamicFollower:
    def __init__(self):
        self.offset = np.zeros(2, dtype=float)
        self.index = 0
        self.direction = 1

    def step(
        self,
        points: np.ndarray,
        radius: float,
        distance_budget: float,
        *,
        pingpong: bool = False,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        self.index = int(np.clip(self.index, 0, len(points) - 1))
        remaining = float(distance_budget)
        guard = 0

        while remaining > 1e-12 and guard < len(points) * 3:
            target = radius * points[self.index]
            delta = target - self.offset
            dist = float(np.linalg.norm(delta))

            if dist <= remaining + 1e-12:
                self.offset = target.copy()
                remaining -= dist
                if pingpong:
                    nxt = self.index + self.direction
                    if nxt >= len(points):
                        self.direction = -1
                        nxt = len(points) - 2
                    elif nxt < 0:
                        self.direction = 1
                        nxt = 1
                    self.index = int(np.clip(nxt, 0, len(points) - 1))
                else:
                    self.index = (self.index + 1) % len(points)
            else:
                self.offset += delta * (remaining / max(dist, 1e-12))
                remaining = 0.0
            guard += 1

        return self.offset.copy()


class ProbeBasisSampler:
    def __init__(self, mode: str, seed: int, period_ms: int = 100):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 130000)
        self.period = period_ms
        self.cue_center = np.zeros(2)
        self.fast_offset = np.zeros(2)
        self.confidence = 0.0
        self.follower = DynamicFollower()
        self.prev_local = np.zeros(2)
        self.local = np.zeros(2)
        self.travel = 0.0
        self.wall = 0
        self.cycle = 0

    def update_cue(self, cue, confidence):
        c = float(np.clip(confidence, 0.0, 1.0))
        self.confidence = 0.80 * self.confidence + 0.20 * c
        if cue is not None:
            gain = 0.82 * max(0.10, c)
            self.cue_center += gain * (np.asarray(cue) - self.cue_center)

    def path_points(self) -> tuple[np.ndarray | None, bool]:
        if self.mode == "radial_golden":
            return PATHS["radial_golden"][0], False
        if self.mode == "cardinal_cross":
            return CARDINAL, False
        if self.mode == "diagonal_cross":
            return rotate(CARDINAL, np.pi / 4.0), False
        if self.mode == "rotating_cross":
            angle = self.cycle * np.pi * (3.0 - np.sqrt(5.0))
            return rotate(CARDINAL, angle), False
        if self.mode == "octagon_ring":
            return OCTAGON, False
        if self.mode == "spsa_line":
            a = float(self.rng.uniform(0.0, 2.0 * np.pi))
            d = np.array([np.cos(a), np.sin(a)])
            return np.vstack([d, -d]), True
        return None, False

    def reflect(self, x: np.ndarray, radius: float) -> np.ndarray:
        y = x.copy()
        for k in range(2):
            if y[k] > radius:
                y[k] = radius - (y[k] - radius)
            if y[k] < -radius:
                y[k] = -radius - (y[k] + radius)
            y[k] = float(np.clip(y[k], -radius, radius))
        return y

    def run_cycle(self, target_fn, start_ms, cue, confidence):
        self.update_cue(cue, confidence)
        radius = 0.82 * (1.0 - 0.42 * self.confidence)
        step_budget = (4.0 * radius) / max(1, self.period - 1)

        points, pingpong = self.path_points()
        working_center = self.cue_center + self.fast_offset
        vals, offs, dists = [], [], []
        hit = 0.0

        for j in range(self.period):
            if points is not None:
                local = self.follower.step(
                    points, radius, step_budget, pingpong=pingpong
                )
                self.local = local.copy()
            elif self.mode == "smooth_random_walk":
                a = float(self.rng.uniform(0.0, 2.0 * np.pi))
                proposal = self.local + step_budget * np.array([
                    np.cos(a), np.sin(a)
                ])
                self.local = self.reflect(proposal, radius)
                local = self.local.copy()
            elif self.mode == "iid_random_unmatched":
                local = self.rng.uniform(-radius, radius, size=2)
            elif self.mode == "point_estimate":
                local = np.zeros(2)
            else:
                raise ValueError(self.mode)

            self.travel += float(np.linalg.norm(local - self.prev_local))
            self.prev_local = local.copy()

            target = np.asarray(target_fn(start_ms + j))
            dist = float(np.linalg.norm(working_center + local - target))
            vals.append(relevance(dist))
            offs.append(local.copy())
            dists.append(dist)
            if dist < 0.20:
                hit = 1.0

        if self.mode != "point_estimate":
            vals_a = np.asarray(vals)
            offs_a = np.asarray(offs)
            advantage = vals_a - float(np.mean(vals_a))
            denom = float(np.sum(np.abs(advantage)))
            if denom > 1e-10:
                delta = np.sum(
                    advantage[:, None] * offs_a, axis=0
                ) / denom
                self.fast_offset += 0.75 * delta

        self.fast_offset *= 0.992
        norm = float(np.linalg.norm(self.fast_offset))
        if norm > 1.5:
            self.fast_offset *= 1.5 / norm

        self.wall += self.period
        self.cycle += 1
        return {
            "hit": hit,
            "best_distance": float(np.min(dists)),
            "mean_relevance": float(np.mean(vals)),
        }

    @property
    def travel_per_ms(self):
        return self.travel / max(1, self.wall - 1)


def run_world(seed: int, mode: str, world: str):
    rng = np.random.default_rng(seed + 900)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t):
        return target[min(t, len(target) - 1)]

    s = ProbeBasisSampler(mode, seed)
    t = 0
    hits, bests, rels = [], [], []
    reacq = []
    return_start = None
    found = None

    while t < horizon:
        if world == "reliable":
            sigma, bias = 0.13, np.zeros(2)
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

        row = s.run_cycle(target_fn, t, cue, conf)
        hits.append(row["hit"])
        bests.append(row["best_distance"])
        rels.append(row["mean_relevance"])

        if (
            world == "loss"
            and return_start is not None
            and found is None
            and row["hit"] > 0.5
        ):
            found = t - return_start

        t += s.period

    if world == "loss" and return_start is not None:
        reacq.append(600.0 if found is None else float(found))

    out = {
        "hit_cycle_fraction": float(np.mean(hits)),
        "mean_best_distance": float(np.mean(bests)),
        "mean_relevance": float(np.mean(rels)),
        "probe_travel_per_ms": s.travel_per_ms,
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
        "continuous_path_budget": "4 * cue-derived radius per cycle",
        "same_fast_contrast_rule": True,
        "slow_weight_changes": 0,
    }
    result["question"] = (
        "Is the golden radial sampler doing anything beyond a conventional "
        "finite-difference / probe-basis operation once the fast contrast "
        "recenter rule has been discovered?"
    )
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_2d_probe_basis_attack.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
