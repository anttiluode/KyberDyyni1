from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kyberdyyni import _project_l1
from experiments.fork_2d_sampling_geometry import (
    PATHS,
    WaypointFollower,
)


def relevance(distance: float) -> float:
    return float(np.exp(-0.5 * (distance / 0.30) ** 2))


class SlowCalibration:
    def __init__(
        self,
        n_contexts: int,
        *,
        learning_rate: float = 0.28,
        budget: float = 4.5,
        bounded: bool = True,
    ):
        self.learning_rate = learning_rate
        self.budget = budget
        self.bounded = bounded
        self.weight = np.zeros((n_contexts, 2), dtype=float)

    def predict(self, context_id: int) -> np.ndarray:
        return self.weight[context_id].copy()

    def update(
        self,
        context_id: int,
        correction: np.ndarray,
        quality: float,
    ) -> None:
        self.weight[context_id] += (
            self.learning_rate
            * float(quality)
            * (np.asarray(correction) - self.weight[context_id])
        )
        if self.bounded:
            flat = _project_l1(self.weight.ravel(), self.budget)
            self.weight = flat.reshape(self.weight.shape)

    @property
    def used_capacity(self) -> float:
        return float(np.sum(np.abs(self.weight)))


class RadialFastCalibrator:
    """Continuous structured sampler with an elastic 2-D correction state."""

    def __init__(self):
        points, pingpong = PATHS["radial_golden"]
        self.follower = WaypointFollower(points, pingpong)
        self.fast_offset = np.zeros(2, dtype=float)
        self.local_offset = np.zeros(2, dtype=float)

    def context_switch(self) -> None:
        # Fast state mostly decays between unrelated encounters, but the
        # sampling trajectory itself is not reset.
        self.fast_offset *= 0.10

    def run_cycle(
        self,
        target: np.ndarray,
        base_center: np.ndarray,
        *,
        radius: float = 0.56,
        period: int = 100,
    ) -> dict[str, float]:
        step_budget = (4.0 * radius) / max(1, period - 1)
        values = []
        offsets = []
        distances = []
        hit = 0.0
        working_center = base_center + self.fast_offset

        for _ in range(period):
            self.follower.offset = self.local_offset.copy()
            local = self.follower.step(radius, step_budget)
            self.local_offset = local.copy()

            probe = working_center + local
            dist = float(np.linalg.norm(probe - target))
            val = relevance(dist)
            values.append(val)
            offsets.append(local.copy())
            distances.append(dist)
            if dist < 0.20:
                hit = 1.0

        vals = np.asarray(values)
        offs = np.asarray(offsets)
        advantage = vals - float(np.mean(vals))
        denom = float(np.sum(np.abs(advantage)))
        if denom > 1e-10:
            delta = np.sum(
                advantage[:, None] * offs, axis=0
            ) / denom
            self.fast_offset += 0.75 * delta

        self.fast_offset *= 0.992
        norm = float(np.linalg.norm(self.fast_offset))
        if norm > 1.5:
            self.fast_offset *= 1.5 / norm

        return {
            "hit": hit,
            "best_distance": float(np.min(distances)),
            "mean_relevance": float(np.mean(vals)),
            "center_error_after": float(np.linalg.norm(
                base_center + self.fast_offset - target
            )),
        }


def make_context_biases(seed: int, n_contexts: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 110000)
    angles = np.linspace(0.0, 2.0 * np.pi, n_contexts, endpoint=False)
    angles += rng.normal(0.0, 0.16, size=n_contexts)
    magnitudes = rng.uniform(0.34, 0.52, size=n_contexts)
    return np.column_stack([
        magnitudes * np.cos(angles),
        magnitudes * np.sin(angles),
    ])


def run(
    seed: int,
    *,
    slow_mode: str,
    shuffle_delayed_context: bool = False,
    n_contexts: int = 6,
    rounds: int = 8,
    cycles_per_encounter: int = 5,
    reward_delay_encounters: int = 3,
):
    rng = np.random.default_rng(seed + 120000)
    biases = make_context_biases(seed, n_contexts)

    if slow_mode == "bounded":
        slow = SlowCalibration(n_contexts, bounded=True)
        allow_slow = True
    elif slow_mode == "ema":
        slow = SlowCalibration(
            n_contexts,
            learning_rate=0.28,
            budget=999.0,
            bounded=False,
        )
        allow_slow = True
    elif slow_mode == "frozen":
        slow = SlowCalibration(n_contexts, bounded=True)
        allow_slow = False
    else:
        raise ValueError(slow_mode)

    fast = RadialFastCalibrator()
    credit = deque()
    records = []
    slow_updates = 0

    for round_id in range(rounds):
        for context_id in rng.permutation(n_contexts):
            fast.context_switch()

            target = rng.uniform(-1.5, 1.5, size=2)
            cue = (
                target
                + biases[context_id]
                + rng.normal(0.0, 0.08, size=2)
            )
            slow_prior = slow.predict(context_id)
            base_center = cue + slow_prior

            start_error = float(np.linalg.norm(base_center - target))
            first_cycle_hit = 0.0
            first_cycle_best = np.inf
            last = None

            for cycle in range(cycles_per_encounter):
                last = fast.run_cycle(target, base_center)
                if cycle == 0:
                    first_cycle_hit = last["hit"]
                    first_cycle_best = last["best_distance"]

            final_error = float(last["center_error_after"])
            quality = relevance(final_error)

            # The temporary calibration that worked now becomes a delayed local
            # packet. No within-encounter trajectory is stored for reverse
            # differentiation.
            correction = fast.fast_offset.copy()
            credit.append((context_id, correction, quality))

            if len(credit) > reward_delay_encounters:
                credit_context, delayed_correction, delayed_quality = (
                    credit.popleft()
                )
                if shuffle_delayed_context:
                    credit_context = int(rng.integers(0, n_contexts))
                if allow_slow:
                    slow.update(
                        credit_context,
                        delayed_correction,
                        delayed_quality,
                    )
                    slow_updates += 1

            records.append({
                "round": round_id,
                "start_error": start_error,
                "first_cycle_hit": float(first_cycle_hit),
                "first_cycle_best": float(first_cycle_best),
                "final_error": final_error,
                "quality": quality,
            })

    first = [r for r in records if r["round"] == 0]
    late = [r for r in records if r["round"] >= rounds - 2]

    def mean(key: str, rows) -> float:
        return float(np.mean([r[key] for r in rows]))

    return {
        "first_start_error": mean("start_error", first),
        "late_start_error": mean("start_error", late),
        "first_first_cycle_hit": mean("first_cycle_hit", first),
        "late_first_cycle_hit": mean("first_cycle_hit", late),
        "first_first_cycle_best": mean("first_cycle_best", first),
        "late_first_cycle_best": mean("first_cycle_best", late),
        "first_final_error": mean("final_error", first),
        "late_final_error": mean("final_error", late),
        "used_slow_capacity": slow.used_capacity,
        "slow_updates": float(slow_updates),
    }


def summarize(rows):
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def evaluate(condition, n_seeds: int = 12):
    return summarize([
        run(seed, **condition)
        for seed in range(n_seeds)
    ])


def main():
    conditions = {
        "bounded_delayed_calibration": {
            "slow_mode": "bounded",
        },
        "fast_only_slow_frozen": {
            "slow_mode": "frozen",
        },
        "shuffled_delayed_context": {
            "slow_mode": "bounded",
            "shuffle_delayed_context": True,
        },
        "explicit_ema_attacker": {
            "slow_mode": "ema",
        },
    }

    result = {
        name: evaluate(condition)
        for name, condition in conditions.items()
    }
    result["settings"] = {
        "n_seeds": 12,
        "n_contexts": 6,
        "rounds": 8,
        "cycles_per_encounter": 5,
        "reward_delay_encounters": 3,
        "cue_noise_std": 0.08,
        "context_bias_magnitude_range": [0.34, 0.52],
        "slow_weight_changes_inside_encounter": 0,
        "BPTT": False,
    }
    result["question"] = (
        "Can a temporary 2-D calibration discovered by structured fast "
        "sampling be crystallized after delayed local consequence into a "
        "context-specific slow cue calibration, so later encounters begin "
        "closer to the target without re-solving the whole bias?"
    )
    result["interpretation_rule"] = (
        "The delayed learner earns a role only if late encounters start closer "
        "and hit sooner than fast-only, while shuffling delayed context removes "
        "that benefit. The explicit EMA table remains the ordinary attacker."
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_2d_fast_slow_calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
