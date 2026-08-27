from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attractor_scanner import AdaptationRingScanner
from kyberdyyni import ThetaScanner, _project_l1


def circular(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


class BoundedContextPrior:
    """Small slow context -> circular prior with a global structural budget."""

    def __init__(self, n_contexts: int, learning_rate: float = 0.22, budget: float = 8.0):
        self.learning_rate = learning_rate
        self.budget = budget
        self.weight = np.zeros((n_contexts, 2), dtype=float)

    def predict(self, context_id: int) -> float:
        v = self.weight[context_id]
        if np.linalg.norm(v) < 1e-8:
            return 0.0
        return float(np.arctan2(v[1], v[0]))

    def update(self, context_id: int, candidate_angle: float, reward: float) -> None:
        target = np.array([np.cos(candidate_angle), np.sin(candidate_angle)])
        self.weight[context_id] += (
            self.learning_rate * float(reward) * (target - self.weight[context_id])
        )
        flat = _project_l1(self.weight.ravel(), self.budget)
        self.weight = flat.reshape(self.weight.shape)

    @property
    def used_capacity(self) -> float:
        return float(np.sum(np.abs(self.weight)))


class ExplicitEMATable:
    """Ordinary attacker: one unbounded circular EMA per explicit context."""

    def __init__(self, n_contexts: int, learning_rate: float = 0.35):
        self.learning_rate = learning_rate
        self.weight = np.zeros((n_contexts, 2), dtype=float)

    def predict(self, context_id: int) -> float:
        v = self.weight[context_id]
        if np.linalg.norm(v) < 1e-8:
            return 0.0
        return float(np.arctan2(v[1], v[0]))

    def update(self, context_id: int, candidate_angle: float, reward: float) -> None:
        target = np.array([np.cos(candidate_angle), np.sin(candidate_angle)])
        self.weight[context_id] += (
            self.learning_rate * float(reward) * (target - self.weight[context_id])
        )

    @property
    def used_capacity(self) -> float:
        return float(np.sum(np.abs(self.weight)))


class ControlledAttractor:
    """Expose a fast relevance control without changing slow structure.

    axis is supplied by the caller through the external anchor.
    focus is deliberately simple:
      - faster theta
      - less adaptation -> smaller excursions
      - slightly stronger sensory tether

    This is an engineering attack on the Vollan-inspired direction/width/
    frequency idea, not a claim about the biological control circuit.
    """

    def __init__(self, seed: int):
        self.scanner = AdaptationRingScanner(
            n_cells=64,
            seed=seed,
            noise_std=0.05,
        )
        self.base_period = self.scanner.theta_period_ms
        self.base_adaptation_gain = self.scanner.adaptation_gain
        self.base_external_gain = self.scanner.external_gain

        for _ in range(600):
            self.scanner.step(0.0)

    def step(self, anchor: float, focus: float = 0.0) -> float:
        focus = float(np.clip(focus, 0.0, 1.0))
        self.scanner.theta_period_ms = self.base_period / (1.0 + 0.75 * focus)
        self.scanner.adaptation_gain = self.base_adaptation_gain * (1.0 - 0.55 * focus)
        self.scanner.external_gain = self.base_external_gain * (1.0 + 0.35 * focus)
        return float(self.scanner.step(anchor)["center"])


def make_scanner(mode: str, seed: int):
    if mode == "adaptation":
        return ControlledAttractor(seed)
    if mode == "engineered":
        return ThetaScanner(base_frequency=0.01, base_width=0.6)
    return None


def probe_step(scanner, mode: str, anchor: float, focus: float, rng) -> float:
    if mode == "adaptation":
        return scanner.step(anchor, focus)
    if mode == "engineered":
        return circular(scanner.step(anchor, focus_control=focus)["probe"])
    if mode == "random":
        return circular(anchor + rng.uniform(-0.6, 0.6))
    if mode == "static":
        return anchor
    raise ValueError(mode)


def run(
    seed: int,
    *,
    scanner_mode: str = "adaptation",
    fast_mode: str = "axis",
    slow_mode: str = "bounded",
    shuffle_credit_context: bool = False,
    n_contexts: int = 6,
    rounds: int = 6,
    steps_per_encounter: int = 700,
    reward_delay_encounters: int = 2,
    fast_learning_rate: float = 0.5,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    target_means = (
        np.linspace(-2.6, 2.6, n_contexts)
        + rng.normal(0.0, 0.12, size=n_contexts)
    )

    if slow_mode == "bounded":
        prior = BoundedContextPrior(n_contexts)
        allow_slow_learning = True
    elif slow_mode == "ema":
        prior = ExplicitEMATable(n_contexts)
        allow_slow_learning = True
    elif slow_mode == "frozen":
        prior = BoundedContextPrior(n_contexts)
        allow_slow_learning = False
    else:
        raise ValueError(slow_mode)

    scanner = make_scanner(scanner_mode, seed)
    delayed_credit = deque()

    # These are fast state. They decay across context changes but are never
    # backpropagated through and the scanner population itself is never reset.
    fast_offset = 0.0
    focus = 0.0
    records = []
    slow_updates = 0

    for round_id in range(rounds):
        for context_id in rng.permutation(n_contexts):
            target = circular(
                target_means[context_id] + rng.normal(0.0, 0.10)
            )
            base_anchor = prior.predict(context_id)

            # Re-anchor to the new context without resetting the continuous
            # scanner. Temporary state mostly decays rather than being erased.
            fast_offset *= 0.25
            focus *= 0.50
            baseline = 0.50

            initial_error = abs(circular(base_anchor - target))
            acquired = None
            best_value = -np.inf
            best_probe = base_anchor
            value_sum = 0.0
            travel = 0.0
            previous_probe = None

            for t in range(steps_per_encounter):
                working_anchor = circular(base_anchor + fast_offset)
                active_focus = focus if fast_mode == "axis_focus" else 0.0
                probe = probe_step(
                    scanner,
                    scanner_mode,
                    working_anchor,
                    active_focus,
                    rng,
                )

                if previous_probe is not None:
                    travel += abs(circular(probe - previous_probe))
                previous_probe = probe

                distance = circular(probe - target)
                value = 0.5 + 0.5 * np.cos(distance)
                value_sum += value

                advantage = value - baseline
                probe_offset = circular(probe - working_anchor)

                if fast_mode != "none":
                    # Fast temporary computation: reward-modulated movement in
                    # the direction indicated by the internally generated probe.
                    fast_offset = circular(
                        fast_offset
                        + fast_learning_rate * advantage * probe_offset
                    )
                    fast_offset *= 0.999

                if fast_mode == "axis_focus":
                    desired_focus = np.clip((value - 0.65) / 0.35, 0.0, 1.0)
                    focus = 0.92 * focus + 0.08 * desired_focus

                baseline = 0.98 * baseline + 0.02 * value

                if value > best_value:
                    best_value = value
                    best_probe = probe

                working_error = abs(
                    circular(
                        circular(base_anchor + fast_offset) - target
                    )
                )
                if acquired is None and working_error < 0.18:
                    acquired = t

            if acquired is None:
                acquired = steps_per_encounter

            # The successful fast event becomes a small local credit packet.
            # Its scalar consequence arrives only after later encounters.
            reward = max(0.0, (best_value - 0.5) * 2.0)
            delayed_credit.append(
                (context_id, best_probe, reward)
            )

            if len(delayed_credit) > reward_delay_encounters:
                credit_context, candidate_angle, delayed_reward = (
                    delayed_credit.popleft()
                )
                if shuffle_credit_context:
                    credit_context = int(
                        rng.integers(0, n_contexts)
                    )
                if allow_slow_learning:
                    prior.update(
                        credit_context,
                        candidate_angle,
                        delayed_reward,
                    )
                    slow_updates += 1

            records.append(
                {
                    "round": round_id,
                    "acquisition": float(acquired),
                    "success": float(
                        acquired < steps_per_encounter
                    ),
                    "initial_error": initial_error,
                    "final_error": abs(
                        circular(
                            circular(base_anchor + fast_offset)
                            - target
                        )
                    ),
                    "mean_value": value_sum / steps_per_encounter,
                    "travel": travel / steps_per_encounter,
                }
            )

    first = [r for r in records if r["round"] == 0]
    late = [r for r in records if r["round"] >= rounds - 2]

    def mean(key: str, rows) -> float:
        return float(np.mean([r[key] for r in rows]))

    return {
        "first_acquisition_steps": mean("acquisition", first),
        "first_success_fraction": mean("success", first),
        "first_mean_sample_value": mean("mean_value", first),
        "late_acquisition_steps": mean("acquisition", late),
        "late_success_fraction": mean("success", late),
        "late_mean_sample_value": mean("mean_value", late),
        "first_initial_error_rad": mean("initial_error", first),
        "late_initial_error_rad": mean("initial_error", late),
        "mean_probe_travel_rad_per_step": mean("travel", records),
        "used_slow_capacity": prior.used_capacity,
        "slow_updates": float(slow_updates),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        x = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def evaluate(condition: dict, n_seeds: int = 8) -> dict[str, float]:
    return summarize([
        run(seed, **condition)
        for seed in range(n_seeds)
    ])


def main() -> None:
    fast_conditions = {
        "axis_only_fast_state": {
            "scanner_mode": "adaptation",
            "fast_mode": "axis",
            "slow_mode": "frozen",
        },
        "axis_plus_naive_focus": {
            "scanner_mode": "adaptation",
            "fast_mode": "axis_focus",
            "slow_mode": "frozen",
        },
        "no_fast_relevance": {
            "scanner_mode": "adaptation",
            "fast_mode": "none",
            "slow_mode": "frozen",
        },
    }

    slow_conditions = {
        "adaptation_bounded_slow": {
            "scanner_mode": "adaptation",
            "fast_mode": "axis",
            "slow_mode": "bounded",
        },
        "adaptation_frozen_slow": {
            "scanner_mode": "adaptation",
            "fast_mode": "axis",
            "slow_mode": "frozen",
        },
        "adaptation_shuffled_credit_context": {
            "scanner_mode": "adaptation",
            "fast_mode": "axis",
            "slow_mode": "bounded",
            "shuffle_credit_context": True,
        },
        "explicit_ema_table_attacker": {
            "scanner_mode": "adaptation",
            "fast_mode": "axis",
            "slow_mode": "ema",
        },
        "engineered_sweep_attacker": {
            "scanner_mode": "engineered",
            "fast_mode": "axis",
            "slow_mode": "bounded",
        },
        "random_dither_attacker": {
            "scanner_mode": "random",
            "fast_mode": "axis",
            "slow_mode": "bounded",
        },
    }

    result = {
        "gate5a_fast_temporary_computation": {
            name: evaluate(condition)
            for name, condition in fast_conditions.items()
        },
        "gate5b_delayed_consolidation": {
            name: evaluate(condition)
            for name, condition in slow_conditions.items()
        },
        "settings": {
            "n_seeds": 8,
            "n_contexts": 6,
            "rounds": 6,
            "steps_per_encounter": 700,
            "reward_delay_encounters": 2,
            "fast_learning_rate": 0.5,
            "acquired_threshold_rad": 0.18,
            "target_jitter_std_rad": 0.10,
        },
        "question": (
            "Can fast continuously evolving state do useful temporary "
            "computation, and can delayed local consequence crystallize "
            "successful fast states into slow structure without BPTT?"
        ),
        "interpretation_rule": (
            "Gate 5A passes only if fast-state relevance improves the first "
            "encounter with slow weights frozen. Gate 5B passes only if "
            "repeated contexts begin closer/faster after delayed slow updates, "
            "and context-credit shuffling removes that benefit. Ordinary EMA, "
            "engineered sweep and random-dither attackers remain decisive."
        ),
    }

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "gate5_fast_slow_memory.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
