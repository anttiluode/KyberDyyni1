from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

Array = np.ndarray


def _project_l1(v: Array, budget: float) -> Array:
    """Project onto a finite structural L1 budget."""
    v = np.asarray(v, dtype=float)
    if np.sum(np.abs(v)) <= budget:
        return v
    u = np.abs(v)
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s)
    idx = np.arange(1, len(v) + 1)
    cond = s - (cssv - budget) / idx > 0
    rho = idx[cond][-1]
    theta = (cssv[rho - 1] - budget) / rho
    return np.sign(v) * np.maximum(u - theta, 0.0)


@dataclass
class ThetaScanner:
    """Fast phase-organized internal sampler.

    Engineered analogue, not a biological reconstruction. A theta clock runs
    continuously; successive cycles alternate sides. Attention-like control can
    retarget the central axis, narrow the width, and increase sweep frequency
    without changing slow structural weights.
    """

    base_frequency: float = 0.08
    base_width: float = 1.0
    phase: float = 0.0
    cycle_index: int = 0
    axis_state: float = 0.0
    axis_decay: float = 0.88
    focus_state: float = 0.0
    focus_decay: float = 0.90

    def step(
        self,
        anchor: float,
        *,
        axis_control: float = 0.0,
        focus_control: float = 0.0,
    ) -> dict[str, float]:
        self.axis_state = self.axis_decay * self.axis_state + (1.0 - self.axis_decay) * float(axis_control)
        self.focus_state = self.focus_decay * self.focus_state + (1.0 - self.focus_decay) * float(focus_control)
        focus = float(np.clip(self.focus_state, 0.0, 1.0))

        frequency = self.base_frequency * (1.0 + 0.75 * focus)
        old_phase = self.phase
        self.phase = (self.phase + frequency) % 1.0
        if self.phase < old_phase:
            self.cycle_index += 1

        side = 1.0 if self.cycle_index % 2 == 0 else -1.0
        # Out-and-back within one theta cycle; no teleport at cycle boundary.
        radial = 1.0 - abs(2.0 * self.phase - 1.0)
        width = self.base_width * (1.0 - 0.65 * focus)
        probe = float(anchor) + self.axis_state + side * width * radial

        return {
            "anchor": float(anchor),
            "probe": probe,
            "phase": self.phase,
            "cycle": float(self.cycle_index),
            "side": side,
            "radial": radial,
            "axis": self.axis_state,
            "width": width,
            "frequency": frequency,
            "focus": focus,
        }


@dataclass
class PhaseBridge:
    """Bind a fast sweep sample to a local phase/context coordinate."""

    n_phase_harmonics: int = 2

    def encode(self, sweep: dict[str, float], context: Array | None = None) -> Array:
        p = 2.0 * math.pi * sweep["phase"]
        feat = [
            1.0,
            sweep["anchor"],
            sweep["probe"] - sweep["anchor"],
            sweep["side"],
            sweep["axis"],
            sweep["focus"],
        ]
        for k in range(1, self.n_phase_harmonics + 1):
            feat.extend((math.sin(k * p), math.cos(k * p)))
        if context is not None:
            c = np.asarray(context, dtype=float).ravel()
            feat.extend(c.tolist())
            feat.extend((c * math.sin(p)).tolist())
            feat.extend((c * math.cos(p)).tolist())
        return np.asarray(feat, dtype=float)


@dataclass
class DelayedSlowPrior:
    """Slow context->axis structure updated by delayed local consequence."""

    context_dim: int
    seed: int = 0
    learning_rate: float = 0.003
    delay: int = 8
    structural_budget: float = 4.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weight = rng.normal(scale=0.01, size=self.context_dim)
        self._queue: deque[Array] = deque(maxlen=self.delay + 1)

    def predict(self, context: Array) -> float:
        c = np.asarray(context, dtype=float).ravel()
        return float(np.tanh(np.dot(self.weight, c)))

    def step(
        self,
        context: Array,
        *,
        delayed_target: float | None = None,
        learn: bool = True,
        shuffle_address: bool = False,
        rng: np.random.Generator | None = None,
    ) -> float:
        c = np.asarray(context, dtype=float).ravel()
        y = self.predict(c)
        self._queue.append(c.copy())
        if delayed_target is not None and len(self._queue) == self.delay + 1:
            if shuffle_address:
                if rng is None:
                    raise ValueError("shuffle_address=True requires rng")
                credited = self._queue[int(rng.integers(0, len(self._queue)))]
            else:
                credited = self._queue[0]
            old_y = self.predict(credited)
            err = float(delayed_target) - old_y
            if learn:
                self.weight = _project_l1(
                    self.weight + self.learning_rate * err * credited,
                    self.structural_budget,
                )
        return y

    @property
    def used_capacity(self) -> float:
        return float(np.sum(np.abs(self.weight)))


@dataclass
class PhaseTaggedEligibility:
    """A phase-addressed local trace for future delayed-credit experiments.

    Each local event is stored together with sine/cosine theta coordinates.
    Nothing here claims that biological EC literally implements this encoding;
    it is the architectural hypothesis KyberDyyni will attack.
    """

    feature_dim: int
    decay: float = 0.96

    def __post_init__(self) -> None:
        self.trace = np.zeros(self.feature_dim + 2)

    def deposit(self, feature: Array, phase: float) -> Array:
        f = np.asarray(feature, dtype=float).ravel()
        if len(f) != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        p = 2.0 * math.pi * float(phase)
        tagged = np.concatenate((f, [math.sin(p), math.cos(p)]))
        self.trace = self.decay * self.trace + tagged
        return self.trace.copy()


@dataclass
class KyberDyyni:
    """Fast scanner + phase bridge + slow cortical-style prior.

    Fast state can retarget immediately. Slow structure changes only from
    delayed local credit. There is no autograd/BPTT.
    """

    context_dim: int = 2
    seed: int = 0
    delay: int = 8
    slow_learning_rate: float = 0.003

    def __post_init__(self) -> None:
        self.scanner = ThetaScanner()
        self.bridge = PhaseBridge()
        self.slow = DelayedSlowPrior(
            context_dim=self.context_dim,
            seed=self.seed + 31,
            delay=self.delay,
            learning_rate=self.slow_learning_rate,
        )

    def step(
        self,
        anchor: float,
        context: Array,
        *,
        fast_axis_control: float = 0.0,
        fast_focus_control: float = 0.0,
        delayed_target: float | None = None,
        learn: bool = True,
        shuffle_address: bool = False,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        slow_prior = self.slow.step(
            context,
            delayed_target=delayed_target,
            learn=learn,
            shuffle_address=shuffle_address,
            rng=rng,
        )
        sweep = self.scanner.step(
            anchor,
            axis_control=slow_prior + float(fast_axis_control),
            focus_control=fast_focus_control,
        )
        phi = self.bridge.encode(sweep, context)
        return {
            "sweep": sweep,
            "phi": phi,
            "slow_prior": slow_prior,
            "used_capacity": self.slow.used_capacity,
        }
