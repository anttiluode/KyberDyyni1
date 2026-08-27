from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_highdim_probe_scaling import ProbePolicy, make_bias, relevance, unit


DIM = 128
NOISE_SIGMAS = [0.01, 0.02]
WORLDS = ["dense_bias", "sparse4_bias"]
N_SEEDS = 10
N_CONTEXTS = 6
ROUNDS = 8
PROBE_BUDGET = 512
PROBE_RADIUS = 0.28
FAST_STEP = 0.14
SUCCESS_RADIUS = 0.18
REWARD_DELAY = 2
SLOW_RATE = 0.35
HALT_CONFIRMATIONS = 2

METHODS = [
    "bounded_delayed",
    "ema_delayed",
    "ema_immediate",
    "running_mean_delayed",
    "kalman_delayed",
    "frozen",
    "shuffled_delayed",
]


def noisy_relevance(distance: float, rng: np.random.Generator, sigma: float) -> float:
    value = relevance(distance)
    if sigma > 0.0:
        value += float(rng.normal(0.0, sigma))
    return float(np.clip(value, 0.0, 1.0))


def clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= limit:
        return v
    return v * (limit / max(n, 1e-12))


class SlowMemory:
    def __init__(self, mode: str, n_contexts: int, dim: int):
        self.mode = mode
        self.state = np.zeros((n_contexts, dim), dtype=float)
        self.count = np.zeros(n_contexts, dtype=float)
        self.weight_sum = np.zeros(n_contexts, dtype=float)
        self.sum_candidate = np.zeros((n_contexts, dim), dtype=float)
        self.kalman_var = np.full(n_contexts, (0.24 ** 2) / dim, dtype=float)

    def predict(self, context_id: int) -> np.ndarray:
        return self.state[context_id].copy()

    def update(self, context_id: int, candidate: np.ndarray, quality: float) -> None:
        quality = float(np.clip(quality, 0.0, 1.0))

        if self.mode == "frozen":
            return

        if self.mode in {"bounded_delayed", "ema_delayed", "ema_immediate", "shuffled_delayed"}:
            alpha = SLOW_RATE * quality
            self.state[context_id] += alpha * (candidate - self.state[context_id])

            if self.mode == "bounded_delayed":
                self.state[context_id] = clip_norm(self.state[context_id], 0.85)
                flat = self.state.ravel()
                total = float(np.linalg.norm(flat))
                if total > 1.60:
                    self.state *= 1.60 / total
            return

        if self.mode == "running_mean_delayed":
            # One-hot ridge / shrinkage attacker.  The scalar consequence is
            # used as the sample weight, but no exotic update rule is used.
            self.weight_sum[context_id] += quality
            self.sum_candidate[context_id] += quality * candidate
            ridge = 1.0
            self.state[context_id] = (
                self.sum_candidate[context_id]
                / (ridge + self.weight_sum[context_id])
            )
            return

        if self.mode == "kalman_delayed":
            # Deliberately ordinary scalar-gain state estimator.  It treats the
            # fast candidate as a noisy measurement of the persistent context
            # correction.  Quality only inflates/deflates measurement noise.
            p = float(self.kalman_var[context_id])
            q = (0.025 ** 2) / DIM
            r = ((0.18 + 0.22 * (1.0 - quality)) ** 2) / DIM
            p = p + q
            gain = p / max(p + r, 1e-12)
            self.state[context_id] += gain * (
                candidate - self.state[context_id]
            )
            self.kalman_var[context_id] = (1.0 - gain) * p
            return

        raise ValueError(self.mode)

    @property
    def capacity(self) -> float:
        return float(np.linalg.norm(self.state))


def make_world(seed: int, world: str) -> dict:
    rng = np.random.default_rng(seed + (0 if world == "dense_bias" else 900000))
    stable = np.stack([
        make_bias(rng, DIM, world, magnitude=0.60)
        for _ in range(N_CONTEXTS)
    ])

    schedule = []
    episode_id = 0
    for round_id in range(ROUNDS):
        for context_id in rng.permutation(N_CONTEXTS):
            target = rng.normal(0.0, 0.40, size=DIM)
            jitter = 0.08 * unit(rng.normal(size=DIM))
            schedule.append({
                "round": round_id,
                "episode": episode_id,
                "context": int(context_id),
                "target": target,
                "jitter": jitter,
            })
            episode_id += 1

    return {"stable": stable, "schedule": schedule}


def fast_episode(
    *,
    target: np.ndarray,
    cue: np.ndarray,
    prior: np.ndarray,
    seed: int,
    episode_id: int,
    noise_sigma: float,
) -> dict:
    measure_rng = np.random.default_rng(
        seed + 3000000 + 7919 * episode_id + int(noise_sigma * 1_000_000)
    )

    fast = np.zeros(DIM, dtype=float)
    policy = ProbePolicy(DIM, "hadamard_block8", seed + 4000000 + episode_id)
    # Do not always begin with the same eight rows.  Across related episodes
    # the fast machine sees different partial views of the same correction.
    policy.cursor = (episode_id * 8) % DIM

    probes = 0
    updates = 0
    true_success_at = None
    declared = False
    confirmations = 0
    success_value = relevance(SUCCESS_RADIUS)

    start_error = float(np.linalg.norm(target - (cue + prior)))
    if start_error < SUCCESS_RADIUS:
        true_success_at = 0

    while probes < PROBE_BUDGET:
        working = cue + prior + fast
        correction = target - working
        error = float(np.linalg.norm(correction))

        center = noisy_relevance(error, measure_rng, noise_sigma)
        probes += 1

        if center >= success_value:
            confirmations += 1
            if confirmations >= HALT_CONFIRMATIONS:
                declared = True
                break
            # Spend another center observation before trusting a possible
            # threshold crossing.  This is a real measurement cost.
            continue
        confirmations = 0

        if probes + 16 > PROBE_BUDGET:
            break

        g = np.zeros(DIM, dtype=float)
        for direction in policy.directions():
            plus = float(np.linalg.norm(
                working + PROBE_RADIUS * direction - target
            ))
            minus = float(np.linalg.norm(
                working - PROBE_RADIUS * direction - target
            ))
            vp = noisy_relevance(plus, measure_rng, noise_sigma)
            vm = noisy_relevance(minus, measure_rng, noise_sigma)
            probes += 2
            g += (vp - vm) * direction

        gnorm = float(np.linalg.norm(g))
        if gnorm > 1e-12:
            fast += FAST_STEP * (g / gnorm)
        updates += 1

        after_error = float(np.linalg.norm(target - (cue + prior + fast)))
        if true_success_at is None and after_error < SUCCESS_RADIUS:
            true_success_at = probes

    final_working = cue + prior + fast
    final_error = float(np.linalg.norm(target - final_working))
    candidate = prior + fast

    if true_success_at is None and final_error < SUCCESS_RADIUS:
        true_success_at = probes

    return {
        "candidate": candidate,
        "start_error": start_error,
        "final_error": final_error,
        "probes_used": float(probes),
        "updates": float(updates),
        "true_success": float(final_error < SUCCESS_RADIUS),
        "declared_success": float(declared),
        "false_halt": float(declared and final_error >= SUCCESS_RADIUS),
        "zero_probe_success": float(start_error < SUCCESS_RADIUS),
        "probes_to_true_success": float(
            PROBE_BUDGET + 1 if true_success_at is None else true_success_at
        ),
    }


def run_method(
    seed: int,
    world: str,
    noise_sigma: float,
    method: str,
) -> dict[str, float]:
    env = make_world(seed, world)
    memory_mode = "ema_delayed" if method == "shuffled_delayed" else method
    memory = SlowMemory(memory_mode, N_CONTEXTS, DIM)
    delayed = deque()
    shuffle_rng = np.random.default_rng(seed + 7000000)
    records = []

    for item in env["schedule"]:
        context_id = item["context"]
        target = item["target"]
        # The hidden stable correction is repeated by context; jitter is an
        # episode-specific nuisance component that slow memory should *not*
        # memorize.
        true_correction = env["stable"][context_id] + item["jitter"]
        cue = target - true_correction
        prior = memory.predict(context_id)

        episode = fast_episode(
            target=target,
            cue=cue,
            prior=prior,
            seed=seed,
            episode_id=item["episode"],
            noise_sigma=noise_sigma,
        )

        consequence_rng = np.random.default_rng(
            seed + 8000000 + 6151 * item["episode"]
            + int(noise_sigma * 1_000_000)
        )
        delayed_quality = noisy_relevance(
            episode["final_error"], consequence_rng, noise_sigma
        )
        packet = (
            context_id,
            episode["candidate"].copy(),
            delayed_quality,
        )

        if method == "ema_immediate":
            memory.update(*packet)
        elif method != "frozen":
            delayed.append(packet)
            if len(delayed) > REWARD_DELAY:
                credit_context, candidate, quality = delayed.popleft()
                if method == "shuffled_delayed":
                    choices = [c for c in range(N_CONTEXTS) if c != credit_context]
                    credit_context = int(shuffle_rng.choice(choices))
                memory.update(credit_context, candidate, quality)

        records.append({
            "round": item["round"],
            "start_error": episode["start_error"],
            "final_error": episode["final_error"],
            "probes_used": episode["probes_used"],
            "probes_to_true_success": episode["probes_to_true_success"],
            "true_success": episode["true_success"],
            "declared_success": episode["declared_success"],
            "false_halt": episode["false_halt"],
            "zero_probe_success": episode["zero_probe_success"],
        })

    first = [r for r in records if r["round"] == 0]
    late = [r for r in records if r["round"] >= ROUNDS - 2]

    def mean(key: str, rows) -> float:
        return float(np.mean([r[key] for r in rows]))

    return {
        "first_start_error": mean("start_error", first),
        "late_start_error": mean("start_error", late),
        "first_final_error": mean("final_error", first),
        "late_final_error": mean("final_error", late),
        "first_probes_used": mean("probes_used", first),
        "late_probes_used": mean("probes_used", late),
        "first_probes_to_true_success": mean("probes_to_true_success", first),
        "late_probes_to_true_success": mean("probes_to_true_success", late),
        "first_true_success": mean("true_success", first),
        "late_true_success": mean("true_success", late),
        "late_declared_success": mean("declared_success", late),
        "late_false_halt": mean("false_halt", late),
        "late_zero_probe_success": mean("zero_probe_success", late),
        "probe_savings_fraction": float(
            1.0 - mean("probes_used", late) / max(mean("probes_used", first), 1e-12)
        ),
        "slow_capacity": memory.capacity,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        values = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(values.mean())
        out[key + "_std"] = float(values.std())
    return out


def compact(result: dict) -> dict:
    keep = [
        "first_start_error",
        "late_start_error",
        "first_probes_used",
        "late_probes_used",
        "late_true_success",
        "late_zero_probe_success",
        "late_false_halt",
        "probe_savings_fraction",
    ]
    out = {}
    for world in WORLDS:
        out[world] = {}
        for sigma in NOISE_SIGMAS:
            key = str(sigma)
            out[world][key] = {
                method: {
                    k: result[world][key][method][k]
                    for k in keep
                }
                for method in METHODS
            }
    return out


def main() -> None:
    result = {
        world: {
            str(sigma): {
                method: summarize([
                    run_method(seed, world, sigma, method)
                    for seed in range(N_SEEDS)
                ])
                for method in METHODS
            }
            for sigma in NOISE_SIGMAS
        }
        for world in WORLDS
    }

    result["settings"] = {
        "dimension": DIM,
        "worlds": WORLDS,
        "noise_sigmas": NOISE_SIGMAS,
        "n_seeds": N_SEEDS,
        "n_contexts": N_CONTEXTS,
        "rounds": ROUNDS,
        "probe_budget_per_episode": PROBE_BUDGET,
        "probe_radius": PROBE_RADIUS,
        "fast_step": FAST_STEP,
        "success_radius": SUCCESS_RADIUS,
        "reward_delay_encounters": REWARD_DELAY,
        "halt_confirmations": HALT_CONFIRMATIONS,
        "stable_context_correction_norm": 0.60,
        "episode_nuisance_correction_norm": 0.08,
        "fast_probe_basis": "progressive Hadamard block8",
        "slow_learner_never_sees_true_target_or_true_correction": True,
    }
    result["question"] = (
        "Can slow context memory extract the stable part of repeated noisy "
        "partial fast corrections so later related episodes require fewer "
        "scalar probes, without memorizing episode-specific nuisance?"
    )
    result["kill_condition"] = (
        "The fast/slow role does not earn anything beyond ordinary estimation "
        "if EMA, running mean, or Kalman-style attackers match the same probe "
        "amortization. Correct context address must also beat shuffled credit."
    )

    path = ROOT / "results" / "fork_slow_consolidation_noise.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("SLOW CONSOLIDATION / PROBE AMORTIZATION SUMMARY")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
