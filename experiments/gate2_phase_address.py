from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kyberdyyni import ThetaScanner, _project_l1
from phase_credit import PhaseAddressedSelector


class ExplicitSlotSelector:
    """Ordinary attacker: identify candidates by an explicit within-cycle slot."""

    def __init__(
        self,
        context_dim: int,
        n_slots: int,
        seed: int,
        learning_rate: float = 0.04,
        delay_cycles: int = 4,
        structural_budget: float = 20.0,
        temperature: float = 0.8,
    ) -> None:
        self.context_dim = context_dim
        self.n_slots = n_slots
        self.rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        self.delay_cycles = delay_cycles
        self.structural_budget = structural_budget
        self.temperature = temperature
        self.weight = np.zeros(
            1 + context_dim + n_slots + context_dim * n_slots,
            dtype=float,
        )
        self._credit_queue: deque[np.ndarray] = deque()
        self.reward_baseline = 1.0 / n_slots

    def _feature(self, context: np.ndarray, slot: int) -> np.ndarray:
        one = np.zeros(self.n_slots)
        one[int(slot)] = 1.0
        return np.concatenate((
            [1.0],
            context,
            one,
            np.outer(context, one).ravel(),
        ))

    def step_cycle(
        self,
        context: np.ndarray,
        delayed_reward: float | None,
    ) -> int:
        features = np.vstack([
            self._feature(context, i) for i in range(self.n_slots)
        ])
        logits = (features @ self.weight) / self.temperature
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        choice = int(self.rng.choice(self.n_slots, p=probs))

        expected = np.sum(features * probs[:, None], axis=0)
        self._credit_queue.append(features[choice] - expected)

        if delayed_reward is not None and len(self._credit_queue) > self.delay_cycles:
            credit = self._credit_queue.popleft()
            reward = float(delayed_reward)
            advantage = reward - self.reward_baseline
            self.reward_baseline = 0.995 * self.reward_baseline + 0.005 * reward
            proposal = self.weight + self.learning_rate * advantage * credit
            self.weight = _project_l1(proposal, self.structural_budget)
        return choice

    def choose(self, context: np.ndarray) -> int:
        features = np.vstack([
            self._feature(context, i) for i in range(self.n_slots)
        ])
        return int(np.argmax(features @ self.weight))

    @property
    def used_capacity(self) -> float:
        return float(np.sum(np.abs(self.weight)))


def cycle_phases(scanner: ThetaScanner, n_slots: int) -> np.ndarray:
    """Let the same scanner clock run; never reset it between cycles."""
    return np.asarray([
        scanner.step(0.0, axis_control=0.0, focus_control=0.0)["phase"]
        for _ in range(n_slots)
    ])


def run(
    seed: int,
    mode: str,
    *,
    train_cycles: int = 1500,
    test_cycles: int = 1000,
    n_slots: int = 8,
    n_contexts: int = 4,
    reward_delay_cycles: int = 4,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 1000)
    scanner = ThetaScanner(base_frequency=1.0 / n_slots)
    target_slot = np.asarray([0, 2, 4, 6], dtype=int)

    if mode == "explicit_slot":
        selector = ExplicitSlotSelector(
            n_contexts,
            n_slots,
            seed,
            delay_cycles=reward_delay_cycles,
        )
    else:
        selector = PhaseAddressedSelector(
            n_contexts,
            seed,
            delay_cycles=reward_delay_cycles,
        )

    reward_queue: deque[float] = deque()

    for _ in range(train_cycles):
        actual_phases = cycle_phases(scanner, n_slots)
        context_id = int(rng.integers(0, n_contexts))
        context = np.eye(n_contexts)[context_id]
        delayed_reward = (
            reward_queue.popleft()
            if len(reward_queue) >= reward_delay_cycles
            else None
        )

        if mode == "explicit_slot":
            choice = selector.step_cycle(context, delayed_reward)
        else:
            if mode == "phase":
                observed_phases = actual_phases
            elif mode == "no_phase":
                observed_phases = np.zeros(n_slots)
            elif mode == "shuffled_phase":
                observed_phases = rng.permutation(actual_phases)
            else:
                raise ValueError(mode)

            choice = int(selector.step_cycle(
                context,
                observed_phases,
                delayed_reward=delayed_reward,
            )["choice"])

        reward_queue.append(float(choice == target_slot[context_id]))

    correct = 0
    for _ in range(test_cycles):
        actual_phases = cycle_phases(scanner, n_slots)
        context_id = int(rng.integers(0, n_contexts))
        context = np.eye(n_contexts)[context_id]

        if mode == "explicit_slot":
            choice = selector.choose(context)
        else:
            if mode == "phase":
                observed_phases = actual_phases
            elif mode == "no_phase":
                observed_phases = np.zeros(n_slots)
            else:
                observed_phases = rng.permutation(actual_phases)
            choice = selector.choose(context, observed_phases)

        correct += int(choice == target_slot[context_id])

    return {
        "accuracy": correct / test_cycles,
        "used_capacity": selector.used_capacity,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(values.mean())
        out[key + "_std"] = float(values.std())
    return out


def main() -> None:
    modes = {
        "phase_address": "phase",
        "no_phase_address": "no_phase",
        "shuffled_phase_address": "shuffled_phase",
        "explicit_slot_address_attacker": "explicit_slot",
    }
    result = {
        name: summarize([run(seed, mode) for seed in range(10)])
        for name, mode in modes.items()
    }
    result["chance_accuracy"] = 1.0 / 8.0
    result["settings"] = {
        "n_seeds": 10,
        "n_contexts": 4,
        "candidate_events_per_theta_cycle": 8,
        "reward_delay_theta_cycles": 4,
        "train_cycles": 1500,
        "test_cycles": 1000,
    }
    result["claim"] = (
        "phase can serve as a local address for delayed credit when otherwise "
        "identical candidate events coexist within one sweep; explicit slot "
        "addresses solve the same problem, so phase is not uniquely privileged"
    )

    print(json.dumps(result, indent=2))
    (ROOT / "results" / "gate2_phase_address.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
