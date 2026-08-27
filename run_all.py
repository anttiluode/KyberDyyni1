from experiments.gate0_theta_scanner import main as gate0
from experiments.gate1_fast_slow_consolidation import main as gate1

if __name__ == "__main__":
    print("\n=== GATE 0: FAST THETA SCANNER ===")
    gate0()
    print("\n=== GATE 1: FAST / SLOW CONSOLIDATION ===")
    gate1()
