from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass
class AdaptationRingScanner:
    """Small NumPy port of the Ji et al. theta-modulated HD ring mechanism.

    This is intentionally a *mechanism* port, not a full reproduction of the
    entorhinal-hippocampal model. The ingredients match the authors' published
    HD-cell module:

        recurrent ring attraction
        + slow firing-rate adaptation
        + theta-modulated external anchor

    No left/right alternation rule is present. If alternating bump motion
    appears, it comes from the state dynamics.

    Default parameters follow the simple HD demonstration in the authors'
    public GridCellThetaSweeps code:
        dt=1 ms, tau=10 ms, tau_adapt=100 ms, mbar=12,
        width=0.4 rad, A=3, J0=4, k=1,
        theta period=100 ms, theta modulation=0.4.
    """

    n_cells: int = 100
    seed: int = 0
    dt_ms: float = 1.0
    tau_ms: float = 10.0
    adaptation_tau_ms: float = 100.0
    adaptation_mbar: float = 12.0
    noise_std: float = 0.1
    width_rad: float = 0.4
    external_gain: float = 3.0
    recurrent_gain: float = 4.0
    global_inhibition: float = 1.0
    theta_period_ms: float = 100.0
    theta_modulation: float = 0.4

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.preferred = np.linspace(-np.pi, np.pi, self.n_cells, endpoint=False)

        # Match the scaling used by the authors' HD_cell implementation.
        self.k = self.global_inhibition / self.n_cells * 20.0
        self.J0 = self.recurrent_gain / self.n_cells * 20.0
        self.adaptation_gain = (
            self.adaptation_mbar * self.tau_ms / self.adaptation_tau_ms
        )

        d = self.circular_distance(
            np.abs(self.preferred[0] - self.preferred)
        )
        kernel = (
            self.J0
            * np.exp(-0.5 * (d / self.width_rad) ** 2)
            / (2.0 * np.pi * self.width_rad**2)
        )
        self._kernel_fft = np.fft.fft(kernel)

        self.rate = np.zeros(self.n_cells)
        self.activation = np.zeros(self.n_cells)
        self.adaptation = np.zeros(self.n_cells)
        self.time_ms = 0.0

    @staticmethod
    def circular_distance(x: Array | float) -> Array:
        x = np.asarray(x, dtype=float)
        x = np.where(x > np.pi, x - 2.0 * np.pi, x)
        x = np.where(x < -np.pi, x + 2.0 * np.pi, x)
        return x

    def bump_center(self) -> float:
        z = np.sum(np.exp(1j * self.preferred) * self.rate)
        if abs(z) < 1e-12:
            return 0.0
        return float(np.angle(z))

    def theta_phase(self) -> float:
        return float(
            (self.time_ms % self.theta_period_ms) / self.theta_period_ms
        )

    def step(self, anchor_angle: float = 0.0) -> dict[str, float | Array]:
        """Advance the continuously running population by one dt."""

        center_before = self.bump_center()
        phase = self.theta_phase()
        theta_gain = 1.0 + self.theta_modulation * np.cos(
            2.0 * np.pi * phase
        )

        distance = self.circular_distance(
            self.preferred - float(anchor_angle)
        )
        external = (
            theta_gain
            * self.external_gain
            * np.exp(
                -0.25
                * (distance / self.width_rad) ** 2
            )
        )

        recurrent = np.real(
            np.fft.ifft(
                np.fft.fft(self.rate) * self._kernel_fft
            )
        )
        total = (
            external
            + recurrent
            + self.rng.normal(scale=self.noise_std, size=self.n_cells)
        )

        # Joint Euler step of the rate state and the slower negative feedback.
        du = (
            -self.activation
            + total
            - self.adaptation
        ) / self.tau_ms
        dv = (
            -self.adaptation
            + self.adaptation_gain * self.activation
        ) / self.adaptation_tau_ms

        self.activation = np.maximum(
            self.activation + self.dt_ms * du,
            0.0,
        )
        self.adaptation = (
            self.adaptation + self.dt_ms * dv
        )

        squared = self.activation**2
        self.rate = squared / (
            1.0 + self.k * np.sum(squared)
        )
        self.time_ms += self.dt_ms

        return {
            "center": center_before,
            "phase": phase,
            "theta_gain": float(theta_gain),
            "rate": self.rate.copy(),
            "adaptation": self.adaptation.copy(),
        }
