from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RANKS = [4, 8, 16, 32]
WORLDS = ["dense", "sparse4", "heavy_tail"]
NOISE_SIGMAS = [0.02, 0.04]
N_SEEDS = 48
N_CALIBRATION = 128
N_TEST = 512
CALIBRATION_COUNTS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
CORRECTION_NORM = 0.60
SUCCESS_RADIUS = 0.18
RELEVANCE_SIGMA = 0.40
ACTIVE_BUDGET = 4

METHODS = [
    "full_bitwise",
    "top4",
    "active4",
    "random4",
    "shuffled_full",
    "oracle_sign",
]


def make_coefficients(
    rng: np.random.Generator,
    n: int,
    rank: int,
    world: str,
) -> np.ndarray:
    rows = []
    if world == "heavy_tail":
        scales = np.exp(-np.linspace(0.0, 3.0, rank))
    else:
        scales = np.ones(rank)

    for _ in range(n):
        if world == "dense":
            v = rng.normal(size=rank)

        elif world == "sparse4":
            v = np.zeros(rank, dtype=float)
            k = min(4, rank)
            idx = rng.choice(rank, size=k, replace=False)
            v[idx] = rng.normal(size=k)

        elif world == "heavy_tail":
            v = rng.normal(size=rank) * scales

        else:
            raise ValueError(world)

        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            v[0] = 1.0
            norm = 1.0
        rows.append(CORRECTION_NORM * v / norm)

    return np.asarray(rows)


def relevance(
    prediction: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float,
) -> float:
    error = float(np.linalg.norm(prediction - truth))
    value = float(np.exp(-0.5 * (error / RELEVANCE_SIGMA) ** 2))
    if noise_sigma > 0.0:
        value += float(rng.normal(0.0, noise_sigma))
    return float(np.clip(value, 0.0, 1.0))


def test_metrics(
    signs: np.ndarray,
    truth_signs: np.ndarray,
    coeff: np.ndarray,
    probed: np.ndarray,
) -> dict[str, float]:
    prediction = coeff * signs
    truth = coeff * truth_signs
    error = np.linalg.norm(prediction - truth, axis=1)

    energy = np.mean(coeff ** 2, axis=0)
    weighted_sign = float(
        np.sum(energy * (signs == truth_signs))
        / max(float(np.sum(energy)), 1e-12)
    )

    return {
        "transfer_error": float(np.mean(error)),
        "success": float(np.mean(error < SUCCESS_RADIUS)),
        "sign_accuracy": float(np.mean(signs == truth_signs)),
        "energy_weighted_sign_accuracy": weighted_sign,
        "bit_coverage": float(np.mean(probed > 0)),
    }


def choose_bits(
    method: str,
    coeff: np.ndarray,
    probed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    rank = len(coeff)

    if method in {"full_bitwise", "shuffled_full"}:
        return np.arange(rank)

    k = min(ACTIVE_BUDGET, rank)

    if method == "top4":
        return np.argsort(np.abs(coeff))[-k:]

    if method == "active4":
        # Spend the tiny fixed probe budget where the current context gives a
        # strong signal, but discount repeatedly tested bits so unexplored
        # degrees of freedom eventually get a turn.
        score = np.abs(coeff) / np.sqrt(1.0 + probed)
        return np.argsort(score)[-k:]

    if method == "random4":
        return rng.choice(rank, size=k, replace=False)

    raise ValueError(method)


def simulate(
    seed: int,
    rank: int,
    world: str,
    noise_sigma: float,
    method: str,
) -> dict[str, dict[str, float]]:
    env_rng = np.random.default_rng(
        seed + 100000 * rank + 1000 * WORLDS.index(world)
    )
    truth_signs = env_rng.choice([-1.0, 1.0], size=rank)
    calibration = make_coefficients(
        env_rng,
        N_CALIBRATION,
        rank,
        world,
    )
    test = make_coefficients(
        env_rng,
        N_TEST,
        rank,
        world,
    )

    rng = np.random.default_rng(
        seed
        + 500000
        + 10000 * rank
        + int(noise_sigma * 1_000_000)
        + 1000000 * METHODS.index(method)
    )

    evidence = np.zeros(rank, dtype=float)
    probed = np.zeros(rank, dtype=float)
    signs = (
        truth_signs.copy()
        if method == "oracle_sign"
        else np.ones(rank, dtype=float)
    )
    scalar_evals = 0

    snapshots = {}

    def save(k: int) -> None:
        metrics = test_metrics(
            signs,
            truth_signs,
            test,
            probed,
        )
        metrics["scalar_evaluations"] = float(scalar_evals)
        metrics["evals_per_rank"] = float(scalar_evals / rank)
        snapshots[str(k)] = metrics

    save(0)

    for j in range(N_CALIBRATION):
        coeff = calibration[j]
        truth = coeff * truth_signs

        if method != "oracle_sign":
            bits = choose_bits(method, coeff, probed, rng)

            for bit in bits:
                plus_sign = np.ones(rank, dtype=float)
                minus_sign = np.ones(rank, dtype=float)
                minus_sign[bit] = -1.0

                plus = relevance(
                    coeff * plus_sign,
                    truth,
                    rng,
                    noise_sigma,
                )
                minus = relevance(
                    coeff * minus_sign,
                    truth,
                    rng,
                    noise_sigma,
                )
                scalar_evals += 2

                diff = plus - minus
                if method == "shuffled_full" and rng.random() < 0.5:
                    diff *= -1.0

                evidence[bit] += diff
                probed[bit] += 1.0

            signs = np.where(evidence >= 0.0, 1.0, -1.0)

        k = j + 1
        if k in CALIBRATION_COUNTS:
            save(k)

    return snapshots


def summarize(rows: list[dict[str, dict[str, float]]]) -> dict:
    out = {}
    for count in CALIBRATION_COUNTS:
        key = str(count)
        out[key] = {}
        for metric in rows[0][key]:
            values = np.asarray([row[key][metric] for row in rows], dtype=float)
            out[key][metric] = float(values.mean())
            out[key][metric + "_std"] = float(values.std())
    return out


def compact(result: dict) -> dict:
    keep_counts = ["4", "16", "64", "128"]
    keep_methods = [
        "full_bitwise",
        "top4",
        "active4",
        "random4",
        "shuffled_full",
    ]
    keep_metrics = [
        "transfer_error",
        "success",
        "sign_accuracy",
        "energy_weighted_sign_accuracy",
        "bit_coverage",
        "scalar_evaluations",
        "evals_per_rank",
    ]

    out = {}
    for sigma in NOISE_SIGMAS:
        skey = str(sigma)
        out[skey] = {}
        for world in WORLDS:
            out[skey][world] = {}
            for rank in RANKS:
                rkey = str(rank)
                out[skey][world][rkey] = {}
                for method in keep_methods:
                    out[skey][world][rkey][method] = {
                        count: {
                            metric: result[skey][world][rkey][method][count][metric]
                            for metric in keep_metrics
                        }
                        for count in keep_counts
                    }
    return out


def main() -> None:
    result = {
        str(sigma): {
            world: {
                str(rank): {
                    method: summarize([
                        simulate(
                            seed,
                            rank,
                            world,
                            sigma,
                            method,
                        )
                        for seed in range(N_SEEDS)
                    ])
                    for method in METHODS
                }
                for rank in RANKS
            }
            for world in WORLDS
        }
        for sigma in NOISE_SIGMAS
    }

    result["settings"] = {
        "ranks": RANKS,
        "worlds": WORLDS,
        "noise_sigmas": NOISE_SIGMAS,
        "n_seeds": N_SEEDS,
        "calibration_contexts": N_CALIBRATION,
        "test_contexts": N_TEST,
        "calibration_counts": CALIBRATION_COUNTS,
        "correction_norm": CORRECTION_NORM,
        "success_radius": SUCCESS_RADIUS,
        "relevance_sigma": RELEVANCE_SIGMA,
        "budgeted_bits_per_context": ACTIVE_BUDGET,
        "premise": (
            "Gate 11 temporal alignment has already reduced the cross-view "
            "relation to one global sign bit per recovered component. This "
            "experiment isolates the scaling of that residual calibration."
        ),
    }

    result["question"] = (
        "How does scalar-consequence sign calibration scale with transferable "
        "rank, and can a tiny active bit-probe budget avoid wasting O(R) "
        "measurements on inactive or already-known components?"
    )

    result["interpretation_rule"] = (
        "Full bitwise is the simple O(R)-per-context attacker. Budgeted active "
        "probing earns a role only if it reaches comparable transfer quality "
        "with materially fewer scalar evaluations. Dense high-rank failure "
        "under fixed consequence noise should be treated as an SNR/sample-"
        "complexity boundary, not hidden with extra machinery."
    )

    path = ROOT / "results" / "fork_rank_scaling_sign.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("RANK-SCALING SIGN CALIBRATION SUMMARY")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
