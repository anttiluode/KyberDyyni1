from experiments.gate0_theta_scanner import main as gate0
from experiments.gate1_fast_slow_consolidation import main as gate1
from experiments.gate2_phase_address import main as gate2
from experiments.gate3_adaptation_attractor import main as gate3
from experiments.gate4_internal_search import main as gate4

if __name__ == "__main__":
    print("\n=== GATE 0: FAST THETA SCANNER ===")
    gate0()
    print("\n=== GATE 1: FAST / SLOW CONSOLIDATION ===")
    gate1()
    print("\n=== GATE 2: PHASE-ADDRESSED DELAYED CREDIT ===")
    gate2()
    print("\n=== GATE 3: ADAPTATION-GENERATED ATTRACTOR SWEEPS ===")
    gate3()
    print("\\n=== GATE 4: FAST INTERNAL SEARCH ===")
    gate4()
