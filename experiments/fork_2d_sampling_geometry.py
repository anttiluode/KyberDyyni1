from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPS = 1e-12


def halton(n: int, base: int) -> float:
    out = 0.0
    f = 1.0
    while n > 0:
        f /= base
        n, rem = divmod(n, base)
        out += f * rem
    return out


def hilbert_d2xy(order: int, d: int) -> tuple[int, int]:
    """Hilbert index -> integer grid coordinate."""
    n = 1 << order
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def normalize_square(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    return 2.0 * (p - lo) / scale - 1.0


def boustrophedon_waypoints(rows: int = 7) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, rows)
    pts = []
    for i, y in enumerate(ys):
        xs = (-1.0, 1.0) if i % 2 == 0 else (1.0, -1.0)
        pts.append([xs[0], y])
        pts.append([xs[1], y])
    return np.asarray(pts, dtype=float)


def hilbert_waypoints(order: int = 3) -> np.ndarray:
    n = 1 << order
    pts = [hilbert_d2xy(order, d) for d in range(n * n)]
    return normalize_square(np.asarray(pts, dtype=float))


def square_spiral_waypoints(rings: int = 5) -> np.ndarray:
    pts = [[0.0, 0.0]]
    for k in range(1, rings + 1):
        r = k / rings
        pts.extend([
            [-r, -r],
            [ r, -r],
            [ r,  r],
            [-r,  r],
        ])
    return np.asarray(pts, dtype=float)


def lissajous_waypoints(n: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([
        np.sin(3.0 * t + 0.25),
        np.sin(4.0 * t),
    ])


def radial_golden_waypoints(spokes: int = 18) -> np.ndarray:
    golden = np.pi * (3.0 - np.sqrt(5.0))
    pts = [[0.0, 0.0]]
    for k in range(spokes):
        a = k * golden
        pts.extend([
            [np.cos(a), np.sin(a)],
            [0.0, 0.0],
        ])
    return np.asarray(pts, dtype=float)


def halton_greedy_waypoints(n: int = 64) -> np.ndarray:
    pts = np.asarray([
        [2.0 * halton(i + 1, 2) - 1.0,
         2.0 * halton(i + 1, 3) - 1.0]
        for i in range(n)
    ])
    remaining = list(range(n))
    current = np.array([0.0, 0.0])
    order = []
    while remaining:
        j = min(
            remaining,
            key=lambda idx: float(np.linalg.norm(pts[idx] - current)),
        )
        order.append(j)
        current = pts[j]
        remaining.remove(j)
    return pts[order]


def make_paths() -> dict[str, tuple[np.ndarray, bool]]:
    return {
        "boustrophedon": (boustrophedon_waypoints(), True),
        "hilbert": (hilbert_waypoints(), True),
        "square_spiral": (square_spiral_waypoints(), True),
        "lissajous": (lissajous_waypoints(), False),
        "radial_golden": (radial_golden_waypoints(), False),
        "halton_greedy": (halton_greedy_waypoints(), True),
    }


PATHS = make_paths()
STRUCTURED = list(PATHS)
MODES = STRUCTURED + ["smooth_random_walk", "iid_random_unmatched", "point_estimate"]


class WaypointFollower:
    """Persistent continuous follower with an exact per-step travel cap."""

    def __init__(self, points: np.ndarray, pingpong: bool):
        self.points = np.asarray(points, dtype=float)
        self.pingpong = bool(pingpong)
        self.offset = np.zeros(2, dtype=float)
        self.index = 0
        self.direction = 1

    def _advance_index(self) -> None:
        if not self.pingpong:
            self.index = (self.index + 1) % len(self.points)
            return
        nxt = self.index + self.direction
        if nxt >= len(self.points):
            self.direction = -1
            nxt = len(self.points) - 2
        elif nxt < 0:
            self.direction = 1
            nxt = 1
        self.index = max(0, min(len(self.points) - 1, nxt))

    def step(self, radius: float, distance_budget: float) -> np.ndarray:
        remaining = float(distance_budget)
        guard = 0
        while remaining > EPS and guard < len(self.points) * 3:
            target = radius * self.points[self.index]
            delta = target - self.offset
            dist = float(np.linalg.norm(delta))
            if dist <= remaining + EPS:
                self.offset = target.copy()
                remaining -= dist
                self._advance_index()
            else:
                self.offset = self.offset + delta * (remaining / max(dist, EPS))
                remaining = 0.0
            guard += 1
        return self.offset.copy()


class Sampler2D:
    def __init__(self, mode: str, seed: int, period_ms: int = 100):
        self.mode = mode
        self.rng = np.random.default_rng(seed + 71000)
        self.period = period_ms
        self.center = np.zeros(2, dtype=float)
        self.confidence = 0.0
        self.offset = np.zeros(2, dtype=float)
        self.prev_offset = np.zeros(2, dtype=float)
        self.internal_travel = 0.0
        self.wall_ms = 0
        self.cycle_path = []
        self.radius_history = []

        if mode in PATHS:
            points, pingpong = PATHS[mode]
            self.follower = WaypointFollower(points, pingpong)
        else:
            self.follower = None

    def update_center(self, cue: np.ndarray | None, confidence: float) -> None:
        c = float(np.clip(confidence, 0.0, 1.0))
        self.confidence = 0.80 * self.confidence + 0.20 * c
        if cue is not None:
            gain = 0.82 * max(0.10, c)
            self.center = self.center + gain * (np.asarray(cue) - self.center)

    def _reflect_square(self, x: np.ndarray, radius: float) -> np.ndarray:
        y = x.copy()
        for k in range(2):
            if y[k] > radius:
                y[k] = radius - (y[k] - radius)
            if y[k] < -radius:
                y[k] = -radius - (y[k] + radius)
            y[k] = float(np.clip(y[k], -radius, radius))
        return y

    def run_cycle(
        self,
        target_fn,
        start_ms: int,
        cue: np.ndarray | None,
        cue_confidence: float,
    ) -> dict[str, float]:
        self.update_center(cue, cue_confidence)

        # Broad under uncertainty, focused under confidence.
        radius = 0.82 * (1.0 - 0.42 * self.confidence)
        # Every continuous schedule may travel the same internal distance per cycle.
        path_budget = 4.0 * radius
        step_budget = path_budget / max(1, self.period - 1)

        hits = 0.0
        best_dist = np.inf
        mean_dist = []
        cycle_travel = 0.0

        for j in range(self.period):
            if self.mode in PATHS:
                offset = self.follower.step(radius, step_budget)
            elif self.mode == "smooth_random_walk":
                theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
                proposal = self.offset + step_budget * np.array([
                    np.cos(theta), np.sin(theta)
                ])
                self.offset = self._reflect_square(proposal, radius)
                offset = self.offset.copy()
            elif self.mode == "iid_random_unmatched":
                offset = self.rng.uniform(-radius, radius, size=2)
            elif self.mode == "point_estimate":
                offset = np.zeros(2)
            else:
                raise ValueError(self.mode)

            dstep = float(np.linalg.norm(offset - self.prev_offset))
            self.internal_travel += dstep
            cycle_travel += dstep
            self.prev_offset = offset.copy()
            self.offset = offset.copy()

            target = np.asarray(target_fn(start_ms + j), dtype=float)
            probe = self.center + offset
            dist = float(np.linalg.norm(probe - target))
            best_dist = min(best_dist, dist)
            mean_dist.append(dist)
            if dist < 0.20:
                hits = 1.0

        self.wall_ms += self.period
        self.cycle_path.append(cycle_travel)
        self.radius_history.append(radius)

        return {
            "hit": hits,
            "best_distance": float(best_dist),
            "mean_distance": float(np.mean(mean_dist)),
        }

    @property
    def travel_per_ms(self) -> float:
        return self.internal_travel / max(1, self.wall_ms - 1)

    @property
    def mean_cycle_path(self) -> float:
        return float(np.mean(self.cycle_path))

    @property
    def mean_radius(self) -> float:
        return float(np.mean(self.radius_history))


def confidence_from_sigma(sigma: float | None) -> float:
    if sigma is None:
        return 0.0
    return float(1.0 / (1.0 + (sigma / 0.28) ** 2))


def make_target(seed: int, horizon: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 800)
    p = np.zeros((horizon + 300, 2), dtype=float)
    p[0] = rng.uniform(-1.5, 1.5, size=2)
    v = rng.normal(0.0, 0.0014, size=2)
    for t in range(1, len(p)):
        if t % 900 == 0:
            v = 0.45 * v + rng.normal(0.0, 0.0013, size=2)
            speed = float(np.linalg.norm(v))
            if speed > 0.0045:
                v *= 0.0045 / speed
        p[t] = p[t - 1] + v
        for k in range(2):
            if p[t, k] > 2.5:
                p[t, k] = 2.5 - (p[t, k] - 2.5)
                v[k] *= -1.0
            elif p[t, k] < -2.5:
                p[t, k] = -2.5 - (p[t, k] + 2.5)
                v[k] *= -1.0
    return p


def run_world(seed: int, mode: str, world: str) -> dict[str, float]:
    rng = np.random.default_rng(seed + 900)
    horizon = 6000
    target = make_target(seed, horizon)

    def target_fn(t: int):
        return target[min(t, len(target) - 1)]

    sampler = Sampler2D(mode, seed)
    t = 0
    hits = []
    bests = []
    means = []
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
                return_start = t
                found = None
            if local < 900 and return_start is not None:
                reacq.append(600.0 if found is None else float(found))
                return_start = None
        else:
            raise ValueError(world)

        conf = confidence_from_sigma(sigma)
        if sigma is None:
            cue = None
        else:
            cue = (
                target_fn(t)
                + bias
                + rng.normal(0.0, sigma, size=2)
            )

        row = sampler.run_cycle(target_fn, t, cue, conf)
        hits.append(row["hit"])
        bests.append(row["best_distance"])
        means.append(row["mean_distance"])

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
        "probe_travel_per_ms": sampler.travel_per_ms,
        "mean_path_per_cycle": sampler.mean_cycle_path,
        "mean_uncertainty_radius": sampler.mean_radius,
    }
    if world == "loss":
        out["reacquisition_ms"] = float(np.mean(reacq))
    return out


WORLDS = ["reliable", "mixed", "biased", "loss"]


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main() -> None:
    n_seeds = 10
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
        "probes_per_cycle": 100,
        "continuous_path_budget": "4 * current uncertainty radius per cycle",
        "state_persists_across_cycles": True,
        "iid_random_unmatched": True,
        "slow_weight_changes": 0,
    }
    result["question"] = (
        "What replaces the 1-D boundary-to-boundary shuttle in a 2-D "
        "uncertainty region when continuous samplers receive equal path, "
        "probe-count and wall-time budgets?"
    )
    result["interpretation_rule"] = (
        "A useful generalization should beat matched-travel smooth random walk "
        "across more than one cue regime and approach IID random coverage "
        "without paying IID random's internal travel."
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_2d_sampling_geometry.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
