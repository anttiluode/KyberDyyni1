from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

from kyberdyyni import _project_l1

Array = np.ndarray


@dataclass
class PhaseAddressedSelector:
    """Delayed local credit whose action identity is supplied by theta phase.

    Each theta cycle contains several candidate internal events. The selector
    sees a context plus the phase at which each candidate occurred, chooses one,
    and stores only a local credit vector. A scalar reward can arrive several
    theta cycles later; the retained local vector receives the update.

    This is not BPTT. It is a small reward-modulated local competition rule.
    """

    context_dim: int
    seed: int = 0
    learning_rate: float = 0.04
    delay_cycles: int = 4
    structural_budget: float = 12.0
    temperature: float = 0.8

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        # bias + context + sin/cos(theta) + context x sin/cos(theta)
        self.weight = np.zeros(1 + self.context_dim + 2 + 2 * self.context_dim)
        self._credit_queue: deque[Array] = deque()
        self.reward_baseline: float | None = None
        self.update_count = 0

    def feature(self, context: Array, phase: float) -> Array:
        c = np.asarray(context, dtype=float).ravel()
        if len(c) != self.context_dim:
            raise ValueError("context dimension mismatch")
        angle = 2.0 * np.pi * float(phase)
        return np.concatenate((
            [1.0],
            c,
            [np.sin(angle), np.cos(angle)],
            c * np.sin(angle),
            c * np.cos(angle),
        ))

    def step_cycle(
        self,
        context: Array,
        phases: Array,
        *,
        delayed_reward: float | None = None,
        learn: bool = True,
    ) -> dict[str, Array | int]:
        phases = np.asarray(phases, dtype=float).ravel()
        features = np.vstack([self.feature(context, p) for p in phases])
        logits = (features @ self.weight) / self.temperature
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        choice = int(self.rng.choice(len(phases), p=probs))

        # Local eligibility-like record for the choice relative to its
        # within-cycle competitors. No old network state is replayed later.
        expected_feature = np.sum(features * probs[:, None], axis=0)
        self._credit_queue.append(features[choice] - expected_feature)

        if self.reward_baseline is None:
            self.reward_baseline = 1.0 / len(phases)

        if delayed_reward is not None and len(self._credit_queue) > self.delay_cycles:
            credit = self._credit_queue.popleft()
            reward = float(delayed_reward)
            advantage = reward - self.reward_baseline
            self.reward_baseline = 0.995 * self.reward_baseline + 0.005 * reward

            if learn:
                proposal = self.weight + self.learning_rate * advantage * credit
                self.weight = _project_l1(proposal, self.structural_budget)
                self.update_count += 1

        return {
            "choice": choice,
            "probabilities": probs,
            "scores": logits,
        }

    def choose(self, context: Array, phases: Array) -> int:
        """Frozen greedy choice; random only when candidates are truly tied."""
        phases = np.asarray(phases, dtype=float).ravel()
        features = np.vstack([self.feature(context, p) for p in phases])
        score = features @ self.weight
        winners = np.flatnonzero(np.isclose(score, np.max(score), atol=1e-12))
        return int(self.rng.choice(winners))

    @property
    def used_capacity(self) -> float:
        return float(np.sum(np.abs(self.weight)))
