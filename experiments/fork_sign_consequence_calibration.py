from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_unlabeled_temporal_alignment import (
    DIM,
    RANK,
    orthonormal_basis,
    gaussian_ar_sources,
    render,
    temporal_components,
    autocorr_profile,
    best_permutation,
)


T = 4500
N_SEEDS = 48
N_CALIBRATION = 32
N_TEST = 256
CALIBRATION_COUNTS = [0, 1, 2, 4, 8, 16, 32]
NOISE_SIGMAS = [0.0, 0.01, 0.02, 0.04]
CORRECTION_NORM = 0.60
SUCCESS_RADIUS = 0.18
RELEVANCE_SIGMA = 0.40

METHODS = [
    "random_sign",
    "hillclimb",
    "bitwise",
    "bitwise_repeat2",
    "exhaustive16",
    "shuffled_bitwise",
    "paired_vector_oracle",
    "oracle_sign",
]


def all_signs() -> np.ndarray:
    return np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=RANK)),
        dtype=float,
    )


SIGN_PATTERNS = all_signs()


def make_coefficients(
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    z = rng.normal(size=(n, RANK))
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    z *= CORRECTION_NORM
    return z


def consequence(
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


def predict(
    xa: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    q = xa @ np.linalg.pinv(ma.T)
    return (q * signs) @ mb.T


def transfer_metrics(
    xa: np.ndarray,
    truth: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
    signs: np.ndarray,
) -> dict[str, float]:
    pred = predict(xa, ma, mb, signs)
    error = np.linalg.norm(pred - truth, axis=1)
    cosine = np.sum(pred * truth, axis=1) / np.maximum(
        np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1),
        1e-12,
    )
    return {
        "transfer_error": float(np.mean(error)),
        "success": float(np.mean(error < SUCCESS_RADIUS)),
        "cosine": float(np.mean(cosine)),
    }


def oracle_signs(
    xa: np.ndarray,
    truth: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
) -> np.ndarray:
    best = None
    best_error = np.inf
    for signs in SIGN_PATTERNS:
        pred = predict(xa, ma, mb, signs)
        error = float(np.mean(np.linalg.norm(pred - truth, axis=1)))
        if error < best_error:
            best_error = error
            best = signs.copy()
    assert best is not None
    return best


def paired_vector_signs(
    xa: np.ndarray,
    yb: np.ndarray,
    ma: np.ndarray,
    mb: np.ndarray,
) -> np.ndarray:
    qa = xa @ np.linalg.pinv(ma.T)
    qb = yb @ np.linalg.pinv(mb.T)
    products = qa * qb
    signs = np.where(products >= 0.0, 1.0, -1.0)
    # A single generic vector almost always touches all four components.  If
    # one coefficient is numerically tiny, leave that bit at +1 rather than
    # smuggling in another observation.
    signs[np.abs(products) < 1e-8] = 1.0
    return signs


def make_problem(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    basis_a = orthonormal_basis(rng, DIM, RANK)
    basis_b = orthonormal_basis(rng, DIM, RANK)

    sa = gaussian_ar_sources(np.random.default_rng(seed + 10000), T)
    sb = gaussian_ar_sources(np.random.default_rng(seed + 20000), T)

    xa_stream = render(sa, basis_a, np.random.default_rng(seed + 30000))
    xb_stream = render(sb, basis_b, np.random.default_rng(seed + 40000))

    comp_a, mix_a = temporal_components(xa_stream)
    comp_b, mix_b = temporal_components(xb_stream)

    perm = best_permutation(
        autocorr_profile(comp_a),
        autocorr_profile(comp_b),
    )
    mix_b = mix_b[:, list(perm)]
    comp_b = comp_b[:, list(perm)]

    cal_coeff = make_coefficients(
        np.random.default_rng(seed + 50000),
        N_CALIBRATION,
    )
    test_coeff = make_coefficients(
        np.random.default_rng(seed + 60000),
        N_TEST,
    )

    xa_cal = cal_coeff @ basis_a.T
    yb_cal = cal_coeff @ basis_b.T
    xa_test = test_coeff @ basis_a.T
    yb_test = test_coeff @ basis_b.T

    true_sign = oracle_signs(
        xa_test,
        yb_test,
        mix_a,
        mix_b,
    )

    return {
        "mix_a": mix_a,
        "mix_b": mix_b,
        "xa_cal": xa_cal,
        "yb_cal": yb_cal,
        "xa_test": xa_test,
        "yb_test": yb_test,
        "oracle_sign": true_sign,
    }


def sign_accuracy(
    signs: np.ndarray,
    oracle: np.ndarray,
) -> float:
    return float(np.mean(signs == oracle))


def simulate_method(
    problem: dict,
    seed: int,
    noise_sigma: float,
    method: str,
) -> dict[str, dict[str, float]]:
    ma = problem["mix_a"]
    mb = problem["mix_b"]
    xa_cal = problem["xa_cal"]
    yb_cal = problem["yb_cal"]
    oracle = problem["oracle_sign"]

    rng = np.random.default_rng(
        seed + 700000 + int(noise_sigma * 1_000_000)
        + 100000 * METHODS.index(method)
    )

    if method == "random_sign":
        signs = rng.choice([-1.0, 1.0], size=RANK)
    else:
        signs = np.ones(RANK, dtype=float)

    evidence = np.zeros(RANK, dtype=float)
    pattern_score = np.zeros(len(SIGN_PATTERNS), dtype=float)
    scalar_evals = 0
    vector_pairs = 0

    snapshots: dict[int, dict[str, float]] = {}

    def save(k: int) -> None:
        metrics = transfer_metrics(
            problem["xa_test"],
            problem["yb_test"],
            ma,
            mb,
            signs,
        )
        metrics.update({
            "sign_accuracy": sign_accuracy(signs, oracle),
            "scalar_evaluations": float(scalar_evals),
            "vector_pairs": float(vector_pairs),
        })
        snapshots[k] = metrics

    save(0)

    for j in range(N_CALIBRATION):
        xa = xa_cal[j:j + 1]
        truth = yb_cal[j]

        if method == "random_sign":
            pass

        elif method == "hillclimb":
            current_pred = predict(xa, ma, mb, signs)[0]
            current_value = consequence(
                current_pred, truth, rng, noise_sigma
            )
            scalar_evals += 1

            for bit in range(RANK):
                trial = signs.copy()
                trial[bit] *= -1.0
                trial_pred = predict(xa, ma, mb, trial)[0]
                trial_value = consequence(
                    trial_pred, truth, rng, noise_sigma
                )
                scalar_evals += 1
                if trial_value > current_value:
                    signs = trial
                    current_value = trial_value

        elif method in {"bitwise", "bitwise_repeat2", "shuffled_bitwise"}:
            repeats = 2 if method == "bitwise_repeat2" else 1
            for bit in range(RANK):
                plus = np.ones(RANK, dtype=float)
                minus = np.ones(RANK, dtype=float)
                plus[bit] = 1.0
                minus[bit] = -1.0

                plus_values = []
                minus_values = []
                for _ in range(repeats):
                    plus_values.append(consequence(
                        predict(xa, ma, mb, plus)[0],
                        truth,
                        rng,
                        noise_sigma,
                    ))
                    minus_values.append(consequence(
                        predict(xa, ma, mb, minus)[0],
                        truth,
                        rng,
                        noise_sigma,
                    ))
                    scalar_evals += 2

                diff = float(np.mean(plus_values) - np.mean(minus_values))
                if method == "shuffled_bitwise" and rng.random() < 0.5:
                    diff *= -1.0
                evidence[bit] += diff

            signs = np.where(evidence >= 0.0, 1.0, -1.0)

        elif method == "exhaustive16":
            for idx, candidate in enumerate(SIGN_PATTERNS):
                pattern_score[idx] += consequence(
                    predict(xa, ma, mb, candidate)[0],
                    truth,
                    rng,
                    noise_sigma,
                )
                scalar_evals += 1
            signs = SIGN_PATTERNS[int(np.argmax(pattern_score))].copy()

        elif method == "paired_vector_oracle":
            if vector_pairs == 0:
                signs = paired_vector_signs(
                    xa_cal[j],
                    yb_cal[j],
                    ma,
                    mb,
                )
                vector_pairs = 1

        elif method == "oracle_sign":
            signs = oracle.copy()

        else:
            raise ValueError(method)

        k = j + 1
        if k in CALIBRATION_COUNTS:
            save(k)

    return {
        str(k): snapshots[k]
        for k in CALIBRATION_COUNTS
    }


def summarize(
    rows: list[dict[str, dict[str, float]]],
) -> dict:
    out = {}
    for k in CALIBRATION_COUNTS:
        key = str(k)
        out[key] = {}
        for metric in rows[0][key]:
            values = np.asarray([row[key][metric] for row in rows], dtype=float)
            out[key][metric] = float(values.mean())
            out[key][metric + "_std"] = float(values.std())
    return out


def compact(result: dict) -> dict:
    keep_methods = [
        "random_sign",
        "hillclimb",
        "bitwise",
        "bitwise_repeat2",
        "exhaustive16",
        "shuffled_bitwise",
        "paired_vector_oracle",
    ]
    keep_counts = ["0", "1", "2", "4", "8", "16", "32"]
    keep_metrics = [
        "transfer_error",
        "success",
        "sign_accuracy",
        "scalar_evaluations",
    ]
    return {
        str(sigma): {
            method: {
                k: {
                    metric: result[str(sigma)][method][k][metric]
                    for metric in keep_metrics
                }
                for k in keep_counts
            }
            for method in keep_methods
        }
        for sigma in NOISE_SIGMAS
    }


def main() -> None:
    problems = [make_problem(seed) for seed in range(N_SEEDS)]

    result = {
        str(sigma): {
            method: summarize([
                simulate_method(
                    problems[seed],
                    seed,
                    sigma,
                    method,
                )
                for seed in range(N_SEEDS)
            ])
            for method in METHODS
        }
        for sigma in NOISE_SIGMAS
    }

    result["settings"] = {
        "observed_dimension": DIM,
        "rank": RANK,
        "sequence_length_for_temporal_alignment": T,
        "n_seeds": N_SEEDS,
        "calibration_contexts": N_CALIBRATION,
        "test_contexts": N_TEST,
        "calibration_counts": CALIBRATION_COUNTS,
        "consequence_noise_sigmas": NOISE_SIGMAS,
        "correction_norm": CORRECTION_NORM,
        "success_radius": SUCCESS_RADIUS,
        "relevance_sigma": RELEVANCE_SIGMA,
        "temporal_alignment": (
            "independent Gaussian AR streams -> multi-lag basis -> "
            "autocorrelation-profile permutation match; sign left unresolved"
        ),
        "slow_state_to_learn": "one global sign bit per recovered component",
    }

    result["question"] = (
        "After blind temporal alignment has reduced an arbitrary coordinate "
        "relation to R unresolved sign bits, can local scalar consequence "
        "resolve and consolidate those bits cheaply for future transfer?"
    )

    result["interpretation_rule"] = (
        "Exhaustive 2^R search and ordinary bitwise comparisons are decisive "
        "attackers. A Gate-12 role is earned only if scalar consequence turns "
        "the residual sign ambiguity into a reusable global map; shuffled "
        "consequence must destroy the effect. No novelty is claimed if simple "
        "binary evidence accumulation wins."
    )

    path = ROOT / "results" / "fork_sign_consequence_calibration.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("SIGN-CONSEQUENCE CALIBRATION SUMMARY")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
