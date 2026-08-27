from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_highdim_probe_scaling import ProbePolicy, relevance
from experiments.fork_slow_consolidation_noise import (
    DIM,
    NOISE_SIGMAS,
    WORLDS,
    N_SEEDS,
    N_CONTEXTS,
    ROUNDS,
    PROBE_BUDGET,
    SUCCESS_RADIUS,
    REWARD_DELAY,
    HALT_CONFIRMATIONS,
    SlowMemory,
    make_world,
    noisy_relevance,
)


CONDITIONS = {
    "kalman_legacy": ("kalman_delayed", "legacy", False),
    "kalman_scaled": ("kalman_delayed", "distance_scaled", False),
    "kalman_accept": ("kalman_delayed", "accept_reject", False),
    "kalman_scaled_accept": ("kalman_delayed", "scaled_accept", False),
    "kalman_small_fixed": ("kalman_delayed", "small_fixed", False),
    "ema_scaled_accept": ("ema_delayed", "scaled_accept", False),
    "frozen_scaled_accept": ("frozen", "scaled_accept", False),
    "shuffled_scaled_accept": ("kalman_delayed", "scaled_accept", True),
}


def distance_from_relevance(value: float, landscape_sigma: float = 0.40) -> float:
    # Conventional inverse of the known radial relevance curve.  This is an
    # optimizer attacker, not privileged access to true distance.
    value = float(np.clip(value, 1e-6, 0.999999))
    return float(landscape_sigma * np.sqrt(max(0.0, -2.0 * np.log(value))))


def controls(mode: str, center_value: float) -> tuple[float, float, bool]:
    if mode == "legacy":
        return 0.28, 0.14, False
    if mode == "accept_reject":
        return 0.28, 0.14, True
    if mode == "small_fixed":
        return 0.10, 0.05, False
    if mode in {"distance_scaled", "scaled_accept"}:
        d_est = distance_from_relevance(center_value)
        radius = float(np.clip(0.75 * d_est, 0.06, 0.28))
        step = float(np.clip(0.45 * d_est, 0.02, 0.14))
        return radius, step, mode == "scaled_accept"
    raise ValueError(mode)


def fast_episode(
    *,
    target: np.ndarray,
    cue: np.ndarray,
    prior: np.ndarray,
    seed: int,
    episode_id: int,
    noise_sigma: float,
    fast_mode: str,
) -> dict:
    measure_rng = np.random.default_rng(
        seed + 3000000 + 7919 * episode_id + int(noise_sigma * 1_000_000)
    )
    fast = np.zeros(DIM, dtype=float)
    policy = ProbePolicy(DIM, "hadamard_block8", seed + 4000000 + episode_id)
    policy.cursor = (episode_id * 8) % DIM

    probes = 0
    updates = 0
    rejected = 0
    true_success_at = None
    declared = False
    confirmations = 0
    success_value = relevance(SUCCESS_RADIUS)

    start_error = float(np.linalg.norm(target - (cue + prior)))
    if start_error < SUCCESS_RADIUS:
        true_success_at = 0

    while probes < PROBE_BUDGET:
        working = cue + prior + fast
        error = float(np.linalg.norm(target - working))
        center = noisy_relevance(error, measure_rng, noise_sigma)
        probes += 1

        if center >= success_value:
            confirmations += 1
            if confirmations >= HALT_CONFIRMATIONS:
                declared = True
                break
            continue
        confirmations = 0

        radius, step, use_accept = controls(fast_mode, center)
        extra = 1 if use_accept else 0
        if probes + 16 + extra > PROBE_BUDGET:
            break

        g = np.zeros(DIM, dtype=float)
        for direction in policy.directions():
            plus = float(np.linalg.norm(working + radius * direction - target))
            minus = float(np.linalg.norm(working - radius * direction - target))
            vp = noisy_relevance(plus, measure_rng, noise_sigma)
            vm = noisy_relevance(minus, measure_rng, noise_sigma)
            probes += 2
            g += (vp - vm) * direction

        gnorm = float(np.linalg.norm(g))
        if gnorm <= 1e-12:
            updates += 1
            continue

        proposal = fast + step * (g / gnorm)

        if use_accept:
            proposal_error = float(np.linalg.norm(target - (cue + prior + proposal)))
            proposal_value = noisy_relevance(
                proposal_error, measure_rng, noise_sigma
            )
            probes += 1
            # A one-measurement stochastic line-search attacker.  If the
            # proposed residual correction does not look better than the
            # pre-update state, preserve the slow prior instead of overwriting it.
            if proposal_value <= center:
                rejected += 1
                updates += 1
                continue

        fast = proposal
        updates += 1
        after_error = float(np.linalg.norm(target - (cue + prior + fast)))
        if true_success_at is None and after_error < SUCCESS_RADIUS:
            true_success_at = probes

    final_error = float(np.linalg.norm(target - (cue + prior + fast)))
    if true_success_at is None and final_error < SUCCESS_RADIUS:
        true_success_at = probes

    return {
        "candidate": prior + fast,
        "start_error": start_error,
        "final_error": final_error,
        "probes_used": float(probes),
        "true_success": float(final_error < SUCCESS_RADIUS),
        "zero_probe_success": float(start_error < SUCCESS_RADIUS),
        "false_halt": float(declared and final_error >= SUCCESS_RADIUS),
        "rejected_fraction": float(rejected / max(updates, 1)),
        "probes_to_true_success": float(
            PROBE_BUDGET + 1 if true_success_at is None else true_success_at
        ),
    }


def run_condition(
    seed: int,
    world: str,
    noise_sigma: float,
    condition: str,
) -> dict[str, float]:
    memory_mode, fast_mode, shuffle_credit = CONDITIONS[condition]
    env = make_world(seed, world)
    memory = SlowMemory(memory_mode, N_CONTEXTS, DIM)
    delayed = deque()
    shuffle_rng = np.random.default_rng(seed + 7000000)
    records = []

    for item in env["schedule"]:
        context_id = item["context"]
        target = item["target"]
        true_correction = env["stable"][context_id] + item["jitter"]
        cue = target - true_correction
        prior = memory.predict(context_id)

        ep = fast_episode(
            target=target,
            cue=cue,
            prior=prior,
            seed=seed,
            episode_id=item["episode"],
            noise_sigma=noise_sigma,
            fast_mode=fast_mode,
        )

        consequence_rng = np.random.default_rng(
            seed + 8000000 + 6151 * item["episode"]
            + int(noise_sigma * 1_000_000)
        )
        quality = noisy_relevance(ep["final_error"], consequence_rng, noise_sigma)
        packet = (context_id, ep["candidate"].copy(), quality)

        if memory_mode != "frozen":
            delayed.append(packet)
            if len(delayed) > REWARD_DELAY:
                credit_context, candidate, delayed_quality = delayed.popleft()
                if shuffle_credit:
                    choices = [c for c in range(N_CONTEXTS) if c != credit_context]
                    credit_context = int(shuffle_rng.choice(choices))
                memory.update(credit_context, candidate, delayed_quality)

        records.append({
            "round": item["round"],
            "start_error": ep["start_error"],
            "final_error": ep["final_error"],
            "probes_used": ep["probes_used"],
            "true_success": ep["true_success"],
            "zero_probe_success": ep["zero_probe_success"],
            "false_halt": ep["false_halt"],
            "rejected_fraction": ep["rejected_fraction"],
            "probes_to_true_success": ep["probes_to_true_success"],
        })

    first = [r for r in records if r["round"] == 0]
    late = [r for r in records if r["round"] >= ROUNDS - 2]

    def mean(key: str, rows) -> float:
        return float(np.mean([r[key] for r in rows]))

    first_probes = mean("probes_used", first)
    late_probes = mean("probes_used", late)
    return {
        "first_start_error": mean("start_error", first),
        "late_start_error": mean("start_error", late),
        "first_final_error": mean("final_error", first),
        "late_final_error": mean("final_error", late),
        "first_probes_used": first_probes,
        "late_probes_used": late_probes,
        "late_true_success": mean("true_success", late),
        "late_zero_probe_success": mean("zero_probe_success", late),
        "late_false_halt": mean("false_halt", late),
        "late_rejected_fraction": mean("rejected_fraction", late),
        "late_probes_to_true_success": mean("probes_to_true_success", late),
        "probe_savings_fraction": float(
            1.0 - late_probes / max(first_probes, 1e-12)
        ),
        "slow_capacity": memory.capacity,
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = float(values.mean())
        out[key + "_std"] = float(values.std())
    return out


def compact(result: dict) -> dict:
    keep = [
        "late_start_error",
        "late_final_error",
        "late_probes_used",
        "late_true_success",
        "late_zero_probe_success",
        "late_false_halt",
        "late_rejected_fraction",
        "probe_savings_fraction",
    ]
    return {
        world: {
            str(sigma): {
                condition: {
                    key: result[world][str(sigma)][condition][key]
                    for key in keep
                }
                for condition in CONDITIONS
            }
            for sigma in NOISE_SIGMAS
        }
        for world in WORLDS
    }


def main() -> None:
    result = {
        world: {
            str(sigma): {
                condition: summarize([
                    run_condition(seed, world, sigma, condition)
                    for seed in range(N_SEEDS)
                ])
                for condition in CONDITIONS
            }
            for sigma in NOISE_SIGMAS
        }
        for world in WORLDS
    }
    result["settings"] = {
        "dimension": DIM,
        "noise_sigmas": NOISE_SIGMAS,
        "worlds": WORLDS,
        "n_seeds": N_SEEDS,
        "contexts": N_CONTEXTS,
        "rounds": ROUNDS,
        "probe_budget": PROBE_BUDGET,
        "success_radius": SUCCESS_RADIUS,
        "reward_delay": REWARD_DELAY,
        "fast_basis": "progressive Hadamard block8",
        "distance_scaled_rule": (
            "invert noisy known relevance curve; radius=clip(.75*d,.06,.28), "
            "step=clip(.45*d,.02,.14)"
        ),
        "accept_reject_rule": (
            "one extra noisy scalar measurement; reject proposed fast update "
            "unless its measured relevance exceeds pre-update center"
        ),
    }
    result["question"] = (
        "When slow memory already places the system near the target, can an "
        "ordinary trust-region / stochastic accept-reject fast controller "
        "avoid destroying the learned prior and turn slow learning into real "
        "probe amortization?"
    )
    result["interpretation_rule"] = (
        "If a conventional cautious residual optimizer fixes the handoff, do "
        "not invent a special biological controller. If even these attackers "
        "cannot preserve the slow prior, the fast/slow combination remains "
        "internally incompatible in this toy."
    )

    path = ROOT / "results" / "fork_slow_fast_handoff.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("FAST/SLOW HANDOFF ATTACK")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
