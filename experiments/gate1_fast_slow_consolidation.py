from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kyberdyyni import KyberDyyni


def stream(seed: int, n: int, hazard: float = 0.035):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, 2))
    for _ in range(n):
        if rng.random() < hazard:
            k = 1 - k
        c = np.array([1.0, 0.0]) if k == 0 else np.array([0.0, 1.0])
        target = 1.0 if k == 0 else -1.0
        yield c, target


def axis_accuracy(rows):
    ok = []
    for target, axis in rows:
        if abs(axis) < 0.05:
            ok.append(0.5)
        else:
            ok.append(float(np.sign(axis) == np.sign(target)))
    return float(np.mean(ok))


def run(seed: int, shuffled: bool = False):
    m = KyberDyyni(context_dim=2, seed=seed, delay=12, slow_learning_rate=0.0012)
    rng = np.random.default_rng(seed + 7000)

    # A: fast retargeting, no slow learning.
    fast_rows = []
    for c, target in stream(seed + 1, 1200):
        o = m.step(0.0, c, fast_axis_control=target, fast_focus_control=1.0, learn=False)
        fast_rows.append((target, o["sweep"]["axis"]))
    fast_weight_change = float(np.linalg.norm(m.slow.weight))

    # B: cue removed before consolidation: the same slow structure should not know.
    naked_rows = []
    for c, target in stream(seed + 2, 800):
        o = m.step(0.0, c, learn=False)
        naked_rows.append((target, o["sweep"]["axis"]))

    # C: repeated delayed consequence slowly teaches the context->axis prior.
    hist = list(stream(seed + 3, 9000))
    delay = m.delay
    for t, (c, target) in enumerate(hist):
        delayed_target = hist[t - delay][1] if t >= delay else None
        m.step(
            0.0, c,
            fast_axis_control=target,
            fast_focus_control=1.0,
            delayed_target=delayed_target,
            learn=True,
            shuffle_address=shuffled,
            rng=rng,
        )

    # D: remove the fast cue again and freeze weights.
    test_rows = []
    for c, target in stream(seed + 4, 2500):
        o = m.step(0.0, c, learn=False)
        test_rows.append((target, o["sweep"]["axis"]))

    return {
        "fast_before_slow": axis_accuracy(fast_rows),
        "no_cue_before_consolidation": axis_accuracy(naked_rows),
        "no_cue_after_consolidation": axis_accuracy(test_rows),
        "used_slow_capacity": m.slow.used_capacity,
        "initial_weight_norm": fast_weight_change,
    }


def summary(rows):
    out = {}
    for k in rows[0]:
        a = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(a.mean())
        out[k + "_std"] = float(a.std())
    return out


def main():
    good = [run(s, False) for s in range(10)]
    bad = [run(s, True) for s in range(10)]
    out = {
        "delayed_address_intact": summary(good),
        "shuffled_delayed_address": summary(bad),
        "note": "fast control changes state immediately; slow context structure changes only under delayed local updates",
    }
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "gate1_fast_slow_consolidation.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
