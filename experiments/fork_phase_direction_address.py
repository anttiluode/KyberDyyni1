from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kyberdyyni import _project_l1
from phase_credit import PhaseAddressedSelector


class ExplicitSelector:
    def __init__(
        self,
        context_dim: int,
        n_slots: int,
        seed: int,
        *,
        delay_cycles: int = 4,
        learning_rate: float = 0.04,
        budget: float = 20.0,
        temperature: float = 0.8,
    ):
        self.context_dim = context_dim
        self.n_slots = n_slots
        self.rng = np.random.default_rng(seed)
        self.delay_cycles = delay_cycles
        self.learning_rate = learning_rate
        self.budget = budget
        self.temperature = temperature
        self.weight = np.zeros(
            1 + context_dim + n_slots + context_dim * n_slots
        )
        self.queue = deque()
        self.baseline = 1.0 / n_slots

    def feature(self, context, slot):
        one = np.zeros(self.n_slots)
        one[int(slot)] = 1.0
        return np.concatenate([
            [1.0],
            context,
            one,
            np.outer(context, one).ravel(),
        ])

    def step(self, context, slots, delayed_reward):
        features = np.vstack([self.feature(context, s) for s in slots])
        logits = features @ self.weight / self.temperature
        p = np.exp(logits - np.max(logits))
        p /= p.sum()
        choice = int(self.rng.choice(len(slots), p=p))
        expected = np.sum(features * p[:, None], axis=0)
        self.queue.append(features[choice] - expected)

        if delayed_reward is not None and len(self.queue) > self.delay_cycles:
            credit = self.queue.popleft()
            advantage = float(delayed_reward) - self.baseline
            self.baseline = 0.995 * self.baseline + 0.005 * float(delayed_reward)
            self.weight = _project_l1(
                self.weight + self.learning_rate * advantage * credit,
                self.budget,
            )
        return choice

    def choose(self, context, slots):
        features = np.vstack([self.feature(context, s) for s in slots])
        return int(np.argmax(features @ self.weight))

    @property
    def used_capacity(self):
        return float(np.sum(np.abs(self.weight)))


def cycle_candidates(
    direction: int,
    n_slots: int,
    radius: float,
):
    """Return raw phase, oriented phase, offset and raw/oriented slot ids."""
    raw_phase = np.linspace(0.0, 1.0, n_slots, endpoint=False) + 0.5 / n_slots

    if direction > 0:
        # left -> right
        offset = -radius + 2.0 * radius * raw_phase
        oriented_phase = raw_phase.copy()
        oriented_slot = np.arange(n_slots)
    else:
        # right -> left
        offset = radius - 2.0 * radius * raw_phase
        oriented_phase = 1.0 - raw_phase
        oriented_slot = np.arange(n_slots - 1, -1, -1)

    raw_slot = np.arange(n_slots)
    return {
        "raw_phase": raw_phase,
        "oriented_phase": oriented_phase,
        "offset": offset,
        "raw_slot": raw_slot,
        "oriented_slot": oriented_slot,
    }


def run(
    seed: int,
    mode: str,
    *,
    train_cycles: int = 2200,
    test_cycles: int = 1200,
    n_slots: int = 8,
    n_contexts: int = 6,
    delay_cycles: int = 4,
):
    rng = np.random.default_rng(seed + 61000)
    radius = 0.50

    # Contexts prefer different *spatial positions* in the sweep. Avoid the
    # exact endpoints/center so raw phase flips to a genuinely different slot
    # on the opposite-direction cycle.
    target_oriented_slot = np.asarray([1, 2, 3, 4, 5, 6], dtype=int)

    if mode.startswith("explicit"):
        selector = ExplicitSelector(
            n_contexts,
            n_slots,
            seed,
            delay_cycles=delay_cycles,
        )
    else:
        selector = PhaseAddressedSelector(
            n_contexts,
            seed,
            delay_cycles=delay_cycles,
        )

    reward_queue = deque()
    direction = 1

    for _ in range(train_cycles):
        cand = cycle_candidates(direction, n_slots, radius)
        context_id = int(rng.integers(0, n_contexts))
        context = np.eye(n_contexts)[context_id]
        delayed_reward = (
            reward_queue.popleft()
            if len(reward_queue) >= delay_cycles
            else None
        )

        if mode == "oriented_phase":
            phases = cand["oriented_phase"]
            choice = int(selector.step_cycle(
                context, phases, delayed_reward=delayed_reward
            )["choice"])
        elif mode == "raw_phase":
            phases = cand["raw_phase"]
            choice = int(selector.step_cycle(
                context, phases, delayed_reward=delayed_reward
            )["choice"])
        elif mode == "no_phase":
            choice = int(selector.step_cycle(
                context,
                np.zeros(n_slots),
                delayed_reward=delayed_reward,
            )["choice"])
        elif mode == "shuffled_oriented_phase":
            phases = rng.permutation(cand["oriented_phase"])
            choice = int(selector.step_cycle(
                context, phases, delayed_reward=delayed_reward
            )["choice"])
        elif mode == "explicit_oriented_slot":
            choice = selector.step(
                context,
                cand["oriented_slot"],
                delayed_reward,
            )
        elif mode == "explicit_raw_slot":
            choice = selector.step(
                context,
                cand["raw_slot"],
                delayed_reward,
            )
        else:
            raise ValueError(mode)

        chosen_oriented_slot = int(cand["oriented_slot"][choice])
        reward_queue.append(float(
            chosen_oriented_slot == target_oriented_slot[context_id]
        ))
        direction *= -1

    correct = 0
    spatial_error = []
    direction = 1

    for _ in range(test_cycles):
        cand = cycle_candidates(direction, n_slots, radius)
        context_id = int(rng.integers(0, n_contexts))
        context = np.eye(n_contexts)[context_id]

        if mode == "oriented_phase":
            choice = selector.choose(context, cand["oriented_phase"])
        elif mode == "raw_phase":
            choice = selector.choose(context, cand["raw_phase"])
        elif mode == "no_phase":
            choice = selector.choose(context, np.zeros(n_slots))
        elif mode == "shuffled_oriented_phase":
            phases = rng.permutation(cand["oriented_phase"])
            choice = selector.choose(context, phases)
        elif mode == "explicit_oriented_slot":
            choice = selector.choose(context, cand["oriented_slot"])
        elif mode == "explicit_raw_slot":
            choice = selector.choose(context, cand["raw_slot"])
        else:
            raise ValueError(mode)

        chosen_oriented = int(cand["oriented_slot"][choice])
        target = int(target_oriented_slot[context_id])
        correct += int(chosen_oriented == target)
        spatial_error.append(abs(chosen_oriented - target))
        direction *= -1

    return {
        "accuracy": correct / test_cycles,
        "mean_slot_error": float(np.mean(spatial_error)),
        "used_capacity": selector.used_capacity,
    }


def summarize(rows):
    out = {}
    for key in rows[0]:
        x = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(x.mean())
        out[key + "_std"] = float(x.std())
    return out


def main():
    modes = [
        "oriented_phase",
        "raw_phase",
        "no_phase",
        "shuffled_oriented_phase",
        "explicit_oriented_slot",
        "explicit_raw_slot",
    ]
    result = {
        mode: summarize([run(seed, mode) for seed in range(12)])
        for mode in modes
    }
    result["chance_accuracy"] = 1.0 / 8.0
    result["settings"] = {
        "n_seeds": 12,
        "n_contexts": 6,
        "candidate_events_per_cycle": 8,
        "delayed_reward_cycles": 4,
        "train_cycles": 2200,
        "test_cycles": 1200,
        "sweep": (
            "continuous cross-cycle shuttle; direction reverses every cycle"
        ),
    }
    result["question"] = (
        "When a continuous structured sweep reverses direction, can delayed "
        "credit still use theta phase as an address, or must phase be bound "
        "to the sweep's current direction / coordinate frame?"
    )
    result["interpretation_rule"] = (
        "If oriented phase succeeds but raw phase does not, the composed "
        "address is phase x sweep-direction rather than phase alone. Explicit "
        "oriented slots remain the symbolic attacker."
    )
    print(json.dumps(result, indent=2))
    (ROOT / "results" / "fork_phase_direction_address.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
