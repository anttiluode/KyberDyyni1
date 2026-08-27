from __future__ import annotations

import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RANKS = [4, 8, 16, 32]
LENGTHS = [512, 2048, 4096]
NOISE_STDS = [0.0, 0.10]
SIGNATURES = ["wide", "crowded", "degenerate_pair"]
METHODS = ["pca_static", "amuse_lag1", "sobi_multilag"]
LAGS = [1, 2, 4, 8, 16]
N_SEEDS = 8
BURN = 300
CORRECTION_NORM = 0.60
SUCCESS_RADIUS = 0.18
TEST_CONTEXTS = 256


def orthonormal_basis(rng: np.random.Generator, rank: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
    return q


def rho_schedule(rank: int, signature: str) -> np.ndarray:
    if signature == "wide":
        rho = np.linspace(0.10, 0.95, rank)
    elif signature == "crowded":
        rho = np.linspace(0.65, 0.90, rank)
    elif signature == "degenerate_pair":
        rho = np.linspace(0.10, 0.95, rank)
        i = rank // 2 - 1
        shared = 0.5 * (rho[i] + rho[i + 1])
        rho[i] = shared
        rho[i + 1] = shared
    else:
        raise ValueError(signature)
    return rho.astype(float)


def ar_sources(
    rng: np.random.Generator,
    n: int,
    rho: np.ndarray,
) -> np.ndarray:
    rank = len(rho)
    state = rng.normal(size=rank)
    scale = np.sqrt(np.maximum(1.0 - rho ** 2, 1e-12))
    rows = np.empty((n + BURN, rank), dtype=float)
    for t in range(n + BURN):
        state = rho * state + scale * rng.normal(size=rank)
        rows[t] = state
    return rows[BURN:]


def render(
    sources: np.ndarray,
    basis: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    x = sources @ basis.T
    if noise_std > 0.0:
        x = x + rng.normal(0.0, noise_std, size=x.shape)
    return x


def whiten(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xc = x - np.mean(x, axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    w = eigvec @ np.diag(1.0 / np.sqrt(np.maximum(eigval, 1e-12)))
    y = xc @ w
    return y, xc


def regression_axes(components: np.ndarray, xc: np.ndarray) -> np.ndarray:
    # X ~= S @ M^T
    mix = np.linalg.lstsq(components, xc, rcond=None)[0].T
    return normalize_columns(mix)


def normalize_columns(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-12)


def lag_autocorr(components: np.ndarray, lag: int = 1) -> np.ndarray:
    s = components - np.mean(components, axis=0, keepdims=True)
    denom = np.mean(s ** 2, axis=0)
    return np.mean(s[:-lag] * s[lag:], axis=0) / np.maximum(denom, 1e-12)


def decompose_all(x: np.ndarray) -> dict[str, np.ndarray]:
    rank = x.shape[1]
    xc = x - np.mean(x, axis=0, keepdims=True)

    # Static PCA attacker.  Ordering is purely by zero-lag variance/eigenvalue.
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    pca_axes = normalize_columns(eigvec[:, order])

    y, xc = whiten(x)

    # AMUSE: one symmetric lagged covariance after whitening.
    r1 = (y[:-1].T @ y[1:]) / max(len(y) - 1, 1)
    r1 = 0.5 * (r1 + r1.T)
    _, e1 = np.linalg.eigh(r1)
    s1 = y @ e1
    a1 = regression_axes(s1, xc)
    order1 = np.argsort(lag_autocorr(s1, 1))
    a1 = a1[:, order1]

    # SOBI-like multi-lag operator.  It is not a full iterative joint
    # diagonalizer; it is the same conservative multi-lag second-order idea
    # used in the earlier fork.
    q = np.zeros((rank, rank), dtype=float)
    for lag in LAGS:
        if lag >= len(y):
            continue
        r = (y[:-lag].T @ y[lag:]) / max(len(y) - lag, 1)
        r = 0.5 * (r + r.T)
        q += r @ r.T
    _, em = np.linalg.eigh(q)
    sm = y @ em
    am = regression_axes(sm, xc)
    orderm = np.argsort(lag_autocorr(sm, 1))
    am = am[:, orderm]

    return {
        "pca_static": pca_axes,
        "amuse_lag1": a1,
        "sobi_multilag": am,
    }


def projection_score(axes: np.ndarray, true_basis: np.ndarray) -> float:
    qa, _ = np.linalg.qr(axes)
    qb, _ = np.linalg.qr(true_basis)
    return float(np.trace((qa @ qa.T) @ (qb @ qb.T)) / axes.shape[1])


def axis_metrics(
    axes: np.ndarray,
    true_basis: np.ndarray,
) -> dict[str, float]:
    axes = normalize_columns(axes)
    truth = normalize_columns(true_basis)
    corr = np.abs(axes.T @ truth)

    diagonal = np.diag(corr)
    labels = np.argmax(corr, axis=1)
    expected = np.arange(len(labels))

    return {
        "axis_recovery": float(np.mean(diagonal)),
        "worst_axis_recovery": float(np.min(diagonal)),
        "identity_accuracy": float(np.mean(labels == expected)),
        "subspace_recovery": projection_score(axes, truth),
    }


def degenerate_pair_subspace(
    axes: np.ndarray,
    true_basis: np.ndarray,
) -> float:
    rank = axes.shape[1]
    i = rank // 2 - 1
    pair = true_basis[:, [i, i + 1]]
    scores = np.sum((normalize_columns(axes).T @ pair) ** 2, axis=1)
    chosen = np.argsort(scores)[-2:]
    qa, _ = np.linalg.qr(axes[:, chosen])
    qb, _ = np.linalg.qr(pair)
    return float(np.trace((qa @ qa.T) @ (qb @ qb.T)) / 2.0)


def orient_with_truth(
    axes: np.ndarray,
    true_basis: np.ndarray,
) -> np.ndarray:
    out = normalize_columns(axes.copy())
    for j in range(out.shape[1]):
        if float(out[:, j] @ true_basis[:, j]) < 0.0:
            out[:, j] *= -1.0
    return out


def oracle_sign_transfer(
    axes_a: np.ndarray,
    axes_b: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    aa = orient_with_truth(axes_a, basis_a)
    ab = orient_with_truth(axes_b, basis_b)
    mapping = aa @ ab.T

    coeff = rng.normal(size=(TEST_CONTEXTS, len(aa)))
    coeff /= np.maximum(np.linalg.norm(coeff, axis=1, keepdims=True), 1e-12)
    coeff *= CORRECTION_NORM

    xa = coeff @ basis_a.T
    truth = coeff @ basis_b.T
    pred = xa @ mapping

    error = np.linalg.norm(pred - truth, axis=1)
    return {
        "oracle_sign_transfer_error": float(np.mean(error)),
        "oracle_sign_transfer_success": float(np.mean(error < SUCCESS_RADIUS)),
    }


def run_case(
    seed: int,
    rank: int,
    n: int,
    noise_std: float,
    signature: str,
    *,
    shuffle_time: bool = False,
) -> dict[str, dict[str, float]]:
    rho = rho_schedule(rank, signature)

    basis_a = orthonormal_basis(np.random.default_rng(seed + 1000 + rank), rank)
    basis_b = orthonormal_basis(np.random.default_rng(seed + 2000 + rank), rank)

    sa = ar_sources(
        np.random.default_rng(seed + 3000 + 17 * rank),
        n,
        rho,
    )
    sb = ar_sources(
        np.random.default_rng(seed + 4000 + 19 * rank),
        n,
        rho,
    )

    xa = render(
        sa,
        basis_a,
        np.random.default_rng(seed + 5000 + 23 * rank),
        noise_std,
    )
    xb = render(
        sb,
        basis_b,
        np.random.default_rng(seed + 6000 + 29 * rank),
        noise_std,
    )

    if shuffle_time:
        xa = xa[np.random.default_rng(seed + 7000).permutation(n)]
        xb = xb[np.random.default_rng(seed + 8000).permutation(n)]

    da = decompose_all(xa)
    db = decompose_all(xb)

    out = {}
    for method in METHODS:
        ma = axis_metrics(da[method], basis_a)
        mb = axis_metrics(db[method], basis_b)
        transfer = oracle_sign_transfer(
            da[method],
            db[method],
            basis_a,
            basis_b,
            np.random.default_rng(seed + 9000 + METHODS.index(method)),
        )

        row = {
            "axis_recovery": 0.5 * (ma["axis_recovery"] + mb["axis_recovery"]),
            "worst_axis_recovery": 0.5 * (
                ma["worst_axis_recovery"] + mb["worst_axis_recovery"]
            ),
            "identity_accuracy": 0.5 * (
                ma["identity_accuracy"] + mb["identity_accuracy"]
            ),
            "subspace_recovery": 0.5 * (
                ma["subspace_recovery"] + mb["subspace_recovery"]
            ),
            **transfer,
        }

        if signature == "degenerate_pair":
            row["degenerate_pair_subspace"] = 0.5 * (
                degenerate_pair_subspace(da[method], basis_a)
                + degenerate_pair_subspace(db[method], basis_b)
            )

        out[method] = row

    return out


def summarize(rows: list[dict[str, dict[str, float]]]) -> dict:
    out = {}
    for method in METHODS:
        out[method] = {}
        for metric in rows[0][method]:
            values = np.asarray([row[method][metric] for row in rows], dtype=float)
            out[method][metric] = float(values.mean())
            out[method][metric + "_std"] = float(values.std())
    return out


def signature_meta(rank: int, signature: str, n: int) -> dict[str, float]:
    rho = rho_schedule(rank, signature)
    gaps = np.diff(np.sort(rho))
    min_gap = float(np.min(gaps)) if len(gaps) else 0.0
    return {
        "min_rho_gap": min_gap,
        "gap_times_sqrt_n": float(min_gap * np.sqrt(n)),
    }


def compact(result: dict) -> dict:
    selected = {}
    for signature in SIGNATURES:
        selected[signature] = {}
        for rank in RANKS:
            rkey = str(rank)
            selected[signature][rkey] = {}
            for n, noise in [(512, 0.0), (2048, 0.0), (4096, 0.10)]:
                key = f"n{n}_noise{noise}"
                selected[signature][rkey][key] = {}
                for method in METHODS:
                    row = result[signature][rkey][str(n)][str(noise)][method]
                    selected[signature][rkey][key][method] = {
                        "axis": row["axis_recovery"],
                        "identity": row["identity_accuracy"],
                        "transfer_success": row["oracle_sign_transfer_success"],
                        "transfer_error": row["oracle_sign_transfer_error"],
                    }
                    if "degenerate_pair_subspace" in row:
                        selected[signature][rkey][key][method]["pair_subspace"] = (
                            row["degenerate_pair_subspace"]
                        )
    selected["shuffled_control"] = result["shuffled_control"]
    return selected


def main() -> None:
    result = {}

    for signature in SIGNATURES:
        result[signature] = {}
        for rank in RANKS:
            rkey = str(rank)
            result[signature][rkey] = {}
            for n in LENGTHS:
                result[signature][rkey][str(n)] = {}
                for noise_std in NOISE_STDS:
                    rows = [
                        run_case(
                            seed,
                            rank,
                            n,
                            noise_std,
                            signature,
                        )
                        for seed in range(N_SEEDS)
                    ]
                    summary = summarize(rows)
                    summary["signature_meta"] = signature_meta(rank, signature, n)
                    result[signature][rkey][str(n)][str(noise_std)] = summary

    # Causal control: preserve static distribution but destroy temporal order.
    result["shuffled_control"] = {
        "rank": 8,
        "length": 4096,
        "noise_std": 0.0,
        "signature": "wide",
        "summary": summarize([
            run_case(
                seed,
                8,
                4096,
                0.0,
                "wide",
                shuffle_time=True,
            )
            for seed in range(N_SEEDS)
        ]),
    }

    result["settings"] = {
        "ranks": RANKS,
        "lengths": LENGTHS,
        "noise_stds": NOISE_STDS,
        "signatures": SIGNATURES,
        "methods": METHODS,
        "lags": LAGS,
        "n_seeds": N_SEEDS,
        "test_contexts": TEST_CONTEXTS,
        "correction_norm": CORRECTION_NORM,
        "success_radius": SUCCESS_RADIUS,
        "source_family": "independent stationary unit-variance Gaussian AR(1)",
        "views_are_independent_runs": True,
        "oracle_sign_only_for_scoring_transfer": True,
    }

    result["question"] = (
        "Does unlabeled temporal source identity survive increasing rank, "
        "finite observation windows, observation noise, crowded temporal "
        "signatures, and exact second-order degeneracy?"
    )

    result["interpretation_rule"] = (
        "Source-axis recovery must be distinguished from mere subspace "
        "recovery. Exact equal temporal signatures must retain a rotational "
        "ambiguity inside the degenerate subspace. Shuffling time must destroy "
        "AMUSE/SOBI source identity. The useful scaling variable is expected "
        "to involve temporal-signature separation relative to finite-window "
        "estimation noise rather than ambient rank by itself."
    )

    path = ROOT / "results" / "fork_temporal_separator_scaling.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("TEMPORAL SEPARATOR SCALING SUMMARY")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
