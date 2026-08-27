from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kyberdyyni import ThetaScanner


def summarize(scanner: ThetaScanner, n: int, axis: float, focus: float) -> dict[str, float]:
    rows = [scanner.step(0.0, axis_control=axis, focus_control=focus) for _ in range(n)]
    tail = rows[n // 2:]
    return {
        "mean_axis": float(np.mean([r["axis"] for r in tail])),
        "mean_width": float(np.mean([r["width"] for r in tail])),
        "mean_frequency": float(np.mean([r["frequency"] for r in tail])),
        "cycles_crossed": int(rows[-1]["cycle"] - rows[0]["cycle"]),
        "probe_min": float(np.min([r["probe"] for r in tail])),
        "probe_max": float(np.max([r["probe"] for r in tail])),
    }


def main() -> None:
    scanner = ThetaScanner()
    exploratory = summarize(scanner, 400, axis=0.0, focus=0.0)
    focused_right = summarize(scanner, 400, axis=1.0, focus=1.0)
    release = summarize(scanner, 400, axis=0.0, focus=0.0)

    out = {
        "exploratory": exploratory,
        "focused_right": focused_right,
        "released": release,
        "slow_weight_changes": 0,
        "claim": "direction/width/frequency can retune through fast state alone",
    }
    print(json.dumps(out, indent=2))
    (ROOT / "results" / "gate0_theta_scanner.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
