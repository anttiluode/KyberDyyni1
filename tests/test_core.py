import unittest
import numpy as np

from kyberdyyni import ThetaScanner, PhaseBridge, DelayedSlowPrior, KyberDyyni
from phase_credit import PhaseAddressedSelector


class CoreTests(unittest.TestCase):
    def test_scanner_runs_continuously(self):
        s = ThetaScanner()
        phases = [s.step(0.0)["phase"] for _ in range(30)]
        self.assertGreater(s.cycle_index, 0)
        self.assertTrue(all(0.0 <= p < 1.0 for p in phases))

    def test_fast_focus_changes_width_and_frequency_without_weights(self):
        s = ThetaScanner()
        a = [s.step(0.0, focus_control=0.0) for _ in range(200)]
        b = [s.step(0.0, focus_control=1.0) for _ in range(200)]
        self.assertLess(np.mean([x["width"] for x in b[-50:]]),
                        np.mean([x["width"] for x in a[-50:]]))
        self.assertGreater(np.mean([x["frequency"] for x in b[-50:]]),
                           np.mean([x["frequency"] for x in a[-50:]]))

    def test_phase_bridge_distinguishes_phase(self):
        b = PhaseBridge()
        sweep = dict(anchor=0.0, probe=0.5, side=1.0, axis=0.0, focus=0.0)
        p1 = b.encode({**sweep, "phase": 0.1}, np.array([1.0, 0.0]))
        p2 = b.encode({**sweep, "phase": 0.6}, np.array([1.0, 0.0]))
        self.assertFalse(np.allclose(p1, p2))

    def test_slow_prior_budget(self):
        s = DelayedSlowPrior(2, structural_budget=0.5, delay=1, learning_rate=1.0)
        for _ in range(100):
            s.step(np.array([1.0, 1.0]), delayed_target=10.0)
        self.assertLessEqual(s.used_capacity, 0.5 + 1e-9)

    def test_fast_control_does_not_change_slow_weights_when_learning_off(self):
        m = KyberDyyni(context_dim=2, seed=1)
        before = m.slow.weight.copy()
        for _ in range(100):
            m.step(0.0, np.array([1.0, 0.0]), fast_axis_control=1.0, learn=False)
        self.assertTrue(np.allclose(before, m.slow.weight))

    def test_phase_selector_distinguishes_identical_events_by_phase(self):
        s = PhaseAddressedSelector(context_dim=2, seed=2)
        c = np.array([1.0, 0.0])
        a = s.feature(c, 0.125)
        b = s.feature(c, 0.625)
        self.assertFalse(np.allclose(a, b))

    def test_phase_selector_budget_is_bounded(self):
        s = PhaseAddressedSelector(
            context_dim=2,
            seed=3,
            delay_cycles=0,
            learning_rate=2.0,
            structural_budget=0.75,
        )
        c = np.array([1.0, 0.0])
        phases = np.array([0.125, 0.375, 0.625, 0.875])
        for _ in range(200):
            s.step_cycle(c, phases, delayed_reward=1.0, learn=True)
        self.assertLessEqual(s.used_capacity, 0.75 + 1e-9)


if __name__ == "__main__":
    unittest.main()
