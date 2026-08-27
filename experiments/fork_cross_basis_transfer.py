from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_highdim_probe_scaling import unit


DIM = 32
RANK = 4
N_CONTEXTS = 16
N_TRAIN_VIEWS = 3
N_TEST_VIEWS = 1
REPEATS = 6
N_SEEDS = 64
CALIBRATION_COUNTS = [0, 2, 3, 4, 6, 8]
CORRECTION_NORM = 0.60
FAST_NOISE_NORM = 0.16
NUISANCE_NORM = 0.08
SUCCESS_RADIUS = 0.18


def orthonormal_basis(rng: np.random.Generator, dim: int, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q[:, :rank]


def make_world(seed: int) -> dict:
    rng = np.random.default_rng(seed)

    latent_scale = np.asarray([1.0, 0.75, 0.50, 0.30])[:RANK]
    coeff = rng.normal(size=(N_CONTEXTS, RANK)) * latent_scale
    coeff /= np.maximum(np.linalg.norm(coeff, axis=1, keepdims=True), 1e-12)
    coeff *= CORRECTION_NORM

    bases = [
        orthonormal_basis(rng, DIM, RANK)
        for _ in range(N_TRAIN_VIEWS + N_TEST_VIEWS)
    ]

    # Mean fast correction packet per (view, context).  This stands in for the
    # noisy partial-search output established in Gate 9 so this experiment can
    # isolate the coordinate-transfer problem itself.
    observed = np.zeros((len(bases), N_CONTEXTS, DIM), dtype=float)
    for view_id, basis in enumerate(bases):
        for context_id in range(N_CONTEXTS):
            rows = []
            true = basis @ coeff[context_id]
            for _ in range(REPEATS):
                fast_noise = FAST_NOISE_NORM * unit(rng.normal(size=DIM))
                nuisance = NUISANCE_NORM * unit(rng.normal(size=DIM))
                rows.append(true + fast_noise + nuisance)
            observed[view_id, context_id] = np.mean(rows, axis=0)

    return {
        "coeff": coeff,
        "bases": bases,
        "observed": observed,
    }


def ridge_map(x: np.ndarray, y: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    # Row-vector convention: x @ W ~= y.
    gram = x.T @ x + lam * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def procrustes_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Orthogonal row-vector map minimizing ||xW-y||_F.
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vt


def build_reference_memory(world: dict, mode: str) -> np.ndarray:
    observed = world["observed"]

    if mode == "view0_only":
        return observed[0].copy()

    if mode == "blind_rendered_average":
        return np.mean(observed[:N_TRAIN_VIEWS], axis=0)

    if mode == "aligned_ridge":
        ref = observed[0]
        aligned = [ref]
        for view_id in range(1, N_TRAIN_VIEWS):
            w = ridge_map(observed[view_id], ref)
            aligned.append(observed[view_id] @ w)
        return np.mean(aligned, axis=0)

    if mode == "oracle_latent":
        # Return a reference-view rendering of the latent estimate after using
        # the exact train-view bases as an upper bound.
        z = np.zeros((N_CONTEXTS, RANK), dtype=float)
        for view_id in range(N_TRAIN_VIEWS):
            z += observed[view_id] @ world["bases"][view_id]
        z /= N_TRAIN_VIEWS
        return z @ world["bases"][0].T

    raise ValueError(mode)


def oracle_prediction(world: dict, heldout: np.ndarray, test_view: int) -> np.ndarray:
    z = np.zeros((N_CONTEXTS, RANK), dtype=float)
    for view_id in range(N_TRAIN_VIEWS):
        z += world["observed"][view_id] @ world["bases"][view_id]
    z /= N_TRAIN_VIEWS
    return z[heldout] @ world["bases"][test_view].T


def evaluate_predictions(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    error = np.linalg.norm(prediction - truth, axis=1)
    cosine = np.sum(prediction * truth, axis=1) / np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1),
        1e-12,
    )
    return {
        "error": float(np.mean(error)),
        "success": float(np.mean(error < SUCCESS_RADIUS)),
        "cosine": float(np.mean(cosine)),
    }


def run(seed: int, calibration_count: int) -> dict[str, dict[str, float]]:
    world = make_world(seed)
    rng = np.random.default_rng(seed + 900000)

    test_view = N_TRAIN_VIEWS
    test_truth = np.stack([
        world["bases"][test_view] @ world["coeff"][c]
        for c in range(N_CONTEXTS)
    ])

    perm = rng.permutation(N_CONTEXTS)
    calibration = perm[:calibration_count]
    heldout = perm[calibration_count:]
    if len(heldout) == 0:
        heldout = perm[-1:]

    result = {}

    # 1) A single shared table in raw rendered coordinates.
    blind = build_reference_memory(world, "blind_rendered_average")
    result["blind_shared_rendered"] = evaluate_predictions(
        blind[heldout],
        test_truth[heldout],
    )

    # 2) Explicit context x view lookup cannot zero-shot an unseen view.
    zeros = np.zeros((len(heldout), DIM), dtype=float)
    result["explicit_view_table_unseen"] = evaluate_predictions(
        zeros,
        test_truth[heldout],
    )

    # 3) Exact basis knowledge upper bound.
    result["oracle_basis"] = evaluate_predictions(
        oracle_prediction(world, heldout, test_view),
        test_truth[heldout],
    )

    # 4) Learn a common reference representation across the seen views using
    # ordinary ridge alignment and then calibrate the unseen view.
    ref = build_reference_memory(world, "aligned_ridge")

    if calibration_count == 0:
        result["ridge_calibration"] = evaluate_predictions(
            np.zeros((len(heldout), DIM)),
            test_truth[heldout],
        )
        result["procrustes_calibration"] = result["ridge_calibration"].copy()
    else:
        x = ref[calibration]
        y = world["observed"][test_view, calibration]

        w_ridge = ridge_map(x, y)
        result["ridge_calibration"] = evaluate_predictions(
            ref[heldout] @ w_ridge,
            test_truth[heldout],
        )

        w_proc = procrustes_map(x, y)
        result["procrustes_calibration"] = evaluate_predictions(
            ref[heldout] @ w_proc,
            test_truth[heldout],
        )

    # 5) Simpler attacker: use only view-0 memory and calibrate directly.
    ref0 = build_reference_memory(world, "view0_only")
    if calibration_count == 0:
        result["view0_ridge_calibration"] = evaluate_predictions(
            np.zeros((len(heldout), DIM)),
            test_truth[heldout],
        )
    else:
        w0 = ridge_map(ref0[calibration], world["observed"][test_view, calibration])
        result["view0_ridge_calibration"] = evaluate_predictions(
            ref0[heldout] @ w0,
            test_truth[heldout],
        )

    return result


def summarize(rows: list[dict[str, dict[str, float]]]) -> dict:
    methods = rows[0].keys()
    out = {}
    for method in methods:
        out[method] = {}
        for metric in rows[0][method]:
            x = np.asarray([row[method][metric] for row in rows], dtype=float)
            out[method][metric] = float(x.mean())
            out[method][metric + "_std"] = float(x.std())
    return out


def compact(result: dict) -> dict:
    methods = [
        "blind_shared_rendered",
        "explicit_view_table_unseen",
        "ridge_calibration",
        "procrustes_calibration",
        "view0_ridge_calibration",
        "oracle_basis",
    ]
    return {
        str(k): {
            method: {
                "error": result[str(k)][method]["error"],
                "success": result[str(k)][method]["success"],
                "cosine": result[str(k)][method]["cosine"],
            }
            for method in methods
        }
        for k in CALIBRATION_COUNTS
    }


def main() -> None:
    result = {
        str(k): summarize([
            run(seed, k)
            for seed in range(N_SEEDS)
        ])
        for k in CALIBRATION_COUNTS
    }

    result["settings"] = {
        "observed_dimension": DIM,
        "shared_correction_rank": RANK,
        "contexts": N_CONTEXTS,
        "train_views": N_TRAIN_VIEWS,
        "unseen_test_views": N_TEST_VIEWS,
        "fast_packets_per_view_context": REPEATS,
        "fast_packet_noise_norm": FAST_NOISE_NORM,
        "episode_nuisance_norm": NUISANCE_NORM,
        "correction_norm": CORRECTION_NORM,
        "success_radius": SUCCESS_RADIUS,
        "n_seeds": N_SEEDS,
        "calibration_context_counts": CALIBRATION_COUNTS,
    }
    result["question"] = (
        "Can a context correction learned in several changing latent bases "
        "transfer to an unseen basis, and how much cross-view calibration is "
        "needed before ordinary matrix alignment solves the problem?"
    )
    result["interpretation_rule"] = (
        "Arbitrary unseen orthogonal coordinates are unidentifiable without "
        "some relation signal. A cross-basis mechanism earns a role only if it "
        "beats raw shared tables and explicit view lookup; ordinary ridge or "
        "Procrustes alignment remains the decisive attacker."
    )

    path = ROOT / "results" / "fork_cross_basis_transfer.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("CROSS-BASIS TRANSFER SUMMARY")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
