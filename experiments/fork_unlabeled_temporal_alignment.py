from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fork_highdim_probe_scaling import unit


DIM = 16
RANK = 4
T = 5000
BURN = 400
N_CONTEXTS = 128
N_SEEDS = 24
OBS_NOISE = 0.01
CORRECTION_NORM = 0.60
SUCCESS_RADIUS = 0.18
LAGS = [1, 2, 4, 8, 16, 32, 64]
RATES = np.asarray([1 / 47, 1 / 71, 1 / 103, 1 / 149], dtype=float)
AR_RHO = np.asarray([0.15, 0.45, 0.75, 0.93], dtype=float)

WORLDS = [
    "sawtooth",
    "gaussian_ar",
    "sawtooth_shuffled_time",
]

METHODS = [
    "pca_static",
    "fastica_static",
    "fastica_temporal_unsigned",
    "fastica_temporal_signed",
    "temporal_unsigned",
    "temporal_signed",
    "paired_time_ridge",
]


def orthonormal_basis(
    rng: np.random.Generator,
    dim: int,
    rank: int,
) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    return q[:, :rank]


def standardize(s: np.ndarray) -> np.ndarray:
    s = s - np.mean(s, axis=0, keepdims=True)
    s = s / np.maximum(np.std(s, axis=0, keepdims=True), 1e-12)
    return s


def phase_sources(
    rng: np.random.Generator,
    n: int,
    *,
    triangle: bool = False,
) -> np.ndarray:
    phase = rng.uniform(0.0, 1.0, size=RANK)
    rows = []
    for _ in range(n + BURN):
        phase = (
            phase
            + RATES
            + rng.normal(0.0, 0.00035, size=RANK)
        ) % 1.0
        if triangle:
            value = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
        else:
            value = 2.0 * phase - 1.0
        rows.append(value.copy())
    return standardize(np.asarray(rows[BURN:], dtype=float))


def gaussian_ar_sources(
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    state = rng.normal(size=RANK)
    rows = []
    innovation_scale = np.sqrt(1.0 - AR_RHO ** 2)
    for _ in range(n + BURN):
        state = (
            AR_RHO * state
            + innovation_scale * rng.normal(size=RANK)
        )
        rows.append(state.copy())
    return standardize(np.asarray(rows[BURN:], dtype=float))


def make_sources(
    rng: np.random.Generator,
    n: int,
    world: str,
) -> np.ndarray:
    if world in {"sawtooth", "sawtooth_shuffled_time"}:
        return phase_sources(rng, n, triangle=False)
    if world == "gaussian_ar":
        return gaussian_ar_sources(rng, n)
    raise ValueError(world)


def render(
    s: np.ndarray,
    basis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = s @ basis.T
    if OBS_NOISE > 0.0:
        x = x + rng.normal(0.0, OBS_NOISE, size=x.shape)
    return x


def whiten(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xc = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    subspace = vt[:RANK].T
    z = xc @ subspace
    cov = (z.T @ z) / max(len(z) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    k = eigvec @ np.diag(1.0 / np.sqrt(np.maximum(eigval, 1e-12))) @ eigvec.T
    y = z @ k
    return y, xc


def regression_mixing(
    components: np.ndarray,
    xc: np.ndarray,
) -> np.ndarray:
    # X ~= S @ M^T
    return np.linalg.lstsq(components, xc, rcond=None)[0].T


def sym_decorrelate(w: np.ndarray) -> np.ndarray:
    s, u = np.linalg.eigh(w @ w.T)
    return (
        u
        @ np.diag(1.0 / np.sqrt(np.maximum(s, 1e-12)))
        @ u.T
        @ w
    )


def fastica_components(
    x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y, xc = whiten(x)
    rng = np.random.default_rng(seed)
    w = sym_decorrelate(rng.normal(size=(RANK, RANK)))

    for _ in range(160):
        old = w.copy()
        wx = y @ w.T
        g = np.tanh(wx)
        gp = 1.0 - g ** 2
        w = (g.T @ y) / len(y) - np.diag(np.mean(gp, axis=0)) @ w
        w = sym_decorrelate(w)
        lim = np.max(np.abs(np.abs(np.diag(w @ old.T)) - 1.0))
        if lim < 1e-7:
            break

    s = y @ w.T
    mix = regression_mixing(s, xc)
    return s, mix


def temporal_components(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y, xc = whiten(x)
    q = np.zeros((RANK, RANK), dtype=float)

    for lag in LAGS:
        r = (y[:-lag].T @ y[lag:]) / (len(y) - lag)
        r = 0.5 * (r + r.T)
        q += r @ r.T

    eigval, eigvec = np.linalg.eigh(q)
    order = np.argsort(eigval)[::-1]
    e = eigvec[:, order]
    s = y @ e
    mix = regression_mixing(s, xc)
    return s, mix


def pca_components(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xc = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    mix = vt[:RANK].T
    s = xc @ mix
    return standardize(s), mix


def autocorr_profile(s: np.ndarray) -> np.ndarray:
    profiles = []
    for j in range(RANK):
        v = s[:, j] - np.mean(s[:, j])
        denom = float(np.mean(v ** 2))
        row = []
        for lag in LAGS:
            row.append(
                float(np.mean(v[:-lag] * v[lag:]) / max(denom, 1e-12))
            )
        profiles.append(row)
    return np.asarray(profiles)


def static_profile(s: np.ndarray) -> np.ndarray:
    z = standardize(s)
    return np.stack([
        np.mean(z ** 3, axis=0),
        np.mean(z ** 4, axis=0) - 3.0,
        np.mean(np.abs(z), axis=0),
    ], axis=1)


def best_permutation(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[int, ...]:
    best = None
    best_cost = np.inf
    for perm in itertools.permutations(range(RANK)):
        bp = b[list(perm)]
        cost = float(np.sum((a - bp) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = perm
    assert best is not None
    return best


def temporal_orientation_stat(s: np.ndarray) -> np.ndarray:
    d = np.diff(standardize(s), axis=0)
    scale = np.maximum(np.std(d, axis=0), 1e-12)
    return np.mean((d / scale) ** 3, axis=0)


def marginal_orientation_stat(s: np.ndarray) -> np.ndarray:
    z = standardize(s)
    return np.mean(z ** 3, axis=0)


def orient(
    s: np.ndarray,
    mix: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "temporal":
        stat = temporal_orientation_stat(s)
    elif mode == "static":
        stat = marginal_orientation_stat(s)
    elif mode == "none":
        return s, mix, np.zeros(RANK)
    else:
        raise ValueError(mode)

    s = s.copy()
    mix = mix.copy()

    # Canonical orientation: make the chosen odd statistic negative.
    # Sawtooth increments have a strong negative third moment.  For symmetric
    # Gaussian AR dynamics this statistic is approximately zero, so the sign
    # remains effectively random across independent views.
    for j in range(RANK):
        if stat[j] > 0.0:
            s[:, j] *= -1.0
            mix[:, j] *= -1.0
            stat[j] *= -1.0
    return s, mix, stat


def normalize_columns(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=0, keepdims=True), 1e-12)


def axis_recovery_abs(
    mix: np.ndarray,
    true_basis: np.ndarray,
) -> float:
    m = normalize_columns(mix)
    b = normalize_columns(true_basis)
    corr = np.abs(m.T @ b)
    best = 0.0
    for perm in itertools.permutations(range(RANK)):
        best = max(best, float(np.mean([
            corr[i, perm[i]]
            for i in range(RANK)
        ])))
    return best


def align_two_views(
    xa: np.ndarray,
    xb: np.ndarray,
    *,
    method: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if method == "pca_static":
        sa, ma = pca_components(xa)
        sb, mb = pca_components(xb)
        sa, ma, oa = orient(sa, ma, "static")
        sb, mb, ob = orient(sb, mb, "static")
        perm = best_permutation(static_profile(sa), static_profile(sb))

    elif method.startswith("fastica"):
        sa, ma = fastica_components(xa, seed + 11)
        sb, mb = fastica_components(xb, seed + 29)

        if method == "fastica_static":
            sa, ma, oa = orient(sa, ma, "static")
            sb, mb, ob = orient(sb, mb, "static")
            perm = best_permutation(static_profile(sa), static_profile(sb))
        elif method == "fastica_temporal_unsigned":
            sa, ma, oa = orient(sa, ma, "none")
            sb, mb, ob = orient(sb, mb, "none")
            perm = best_permutation(autocorr_profile(sa), autocorr_profile(sb))
        elif method == "fastica_temporal_signed":
            sa, ma, oa = orient(sa, ma, "temporal")
            sb, mb, ob = orient(sb, mb, "temporal")
            perm = best_permutation(autocorr_profile(sa), autocorr_profile(sb))
        else:
            raise ValueError(method)

    elif method.startswith("temporal"):
        sa, ma = temporal_components(xa)
        sb, mb = temporal_components(xb)

        if method == "temporal_unsigned":
            sa, ma, oa = orient(sa, ma, "none")
            sb, mb, ob = orient(sb, mb, "none")
        elif method == "temporal_signed":
            sa, ma, oa = orient(sa, ma, "temporal")
            sb, mb, ob = orient(sb, mb, "temporal")
        else:
            raise ValueError(method)

        perm = best_permutation(autocorr_profile(sa), autocorr_profile(sb))

    else:
        raise ValueError(method)

    mb = mb[:, list(perm)]
    sb = sb[:, list(perm)]
    ob = ob[list(perm)]

    mapping = np.linalg.pinv(ma.T) @ mb.T
    diagnostics = {
        "mean_abs_temporal_orientation_a": float(np.mean(np.abs(oa))),
        "mean_abs_temporal_orientation_b": float(np.mean(np.abs(ob))),
    }
    return mapping, diagnostics


def ridge_map(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-4,
) -> np.ndarray:
    return np.linalg.solve(
        x.T @ x + lam * np.eye(x.shape[1]),
        x.T @ y,
    )


def evaluate_mapping(
    mapping: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    coeff = rng.normal(size=(N_CONTEXTS, RANK))
    coeff /= np.maximum(np.linalg.norm(coeff, axis=1, keepdims=True), 1e-12)
    coeff *= CORRECTION_NORM

    xa = coeff @ basis_a.T
    truth = coeff @ basis_b.T
    pred = xa @ mapping

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


def run(
    seed: int,
    world: str,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    basis_a = orthonormal_basis(rng, DIM, RANK)
    basis_b = orthonormal_basis(rng, DIM, RANK)

    sa = make_sources(np.random.default_rng(seed + 10000), T, world)
    sb = make_sources(np.random.default_rng(seed + 20000), T, world)

    xa = render(sa, basis_a, np.random.default_rng(seed + 30000))
    xb = render(sb, basis_b, np.random.default_rng(seed + 40000))

    if world == "sawtooth_shuffled_time":
        xa = xa[np.random.default_rng(seed + 50000).permutation(T)]
        xb = xb[np.random.default_rng(seed + 60000).permutation(T)]

    result = {}

    for method in METHODS:
        if method == "paired_time_ridge":
            sync = make_sources(
                np.random.default_rng(seed + 70000),
                T,
                "sawtooth" if world == "sawtooth_shuffled_time" else world,
            )
            x_sync = render(
                sync,
                basis_a,
                np.random.default_rng(seed + 80000),
            )
            y_sync = render(
                sync,
                basis_b,
                np.random.default_rng(seed + 90000),
            )
            mapping = ridge_map(x_sync, y_sync)
            diag = {
                "axis_recovery_a": 1.0,
                "axis_recovery_b": 1.0,
                "mean_abs_temporal_orientation_a": 0.0,
                "mean_abs_temporal_orientation_b": 0.0,
            }
        else:
            mapping, diag = align_two_views(
                xa,
                xb,
                method=method,
                seed=seed + 100000,
            )

            if method == "pca_static":
                _, ma = pca_components(xa)
                _, mb = pca_components(xb)
            elif method.startswith("fastica"):
                _, ma = fastica_components(xa, seed + 100011)
                _, mb = fastica_components(xb, seed + 100029)
            else:
                _, ma = temporal_components(xa)
                _, mb = temporal_components(xb)

            diag["axis_recovery_a"] = axis_recovery_abs(ma, basis_a)
            diag["axis_recovery_b"] = axis_recovery_abs(mb, basis_b)

        metrics = evaluate_mapping(
            mapping,
            basis_a,
            basis_b,
            np.random.default_rng(seed + 110000),
        )
        metrics.update(diag)
        result[method] = metrics

    return result


def summarize(
    rows: list[dict[str, dict[str, float]]],
) -> dict:
    out = {}
    for method in METHODS:
        out[method] = {}
        for metric in rows[0][method]:
            x = np.asarray([row[method][metric] for row in rows], dtype=float)
            out[method][metric] = float(x.mean())
            out[method][metric + "_std"] = float(x.std())
    return out


def compact(result: dict) -> dict:
    keep = [
        "transfer_error",
        "success",
        "cosine",
        "axis_recovery_a",
        "axis_recovery_b",
        "mean_abs_temporal_orientation_a",
    ]
    return {
        world: {
            method: {
                k: result[world][method][k]
                for k in keep
            }
            for method in METHODS
        }
        for world in WORLDS
    }


def main() -> None:
    result = {
        world: summarize([
            run(seed, world)
            for seed in range(N_SEEDS)
        ])
        for world in WORLDS
    }

    result["settings"] = {
        "observed_dimension": DIM,
        "rank": RANK,
        "sequence_length": T,
        "n_seeds": N_SEEDS,
        "contexts_for_transfer_metric": N_CONTEXTS,
        "observation_noise_std": OBS_NOISE,
        "correction_norm": CORRECTION_NORM,
        "success_radius": SUCCESS_RADIUS,
        "lags": LAGS,
        "sawtooth_periods_approx": [float(1.0 / r) for r in RATES],
        "gaussian_ar_rho": AR_RHO.tolist(),
        "views_are_independent_trajectories": True,
        "explicit_context_correspondence_labels": 0,
    }

    result["question"] = (
        "Can independent views of the same family of hidden freedoms recover "
        "a usable coordinate relation from dynamics alone, without paired "
        "context labels or synchronized samples?"
    )

    result["interpretation_rule"] = (
        "Static covariance/ICA attackers must be separated from temporal "
        "fingerprints. Second-order temporal methods only earn axis/permutation "
        "identification; oriented vector transfer additionally requires a "
        "sign-sensitive asymmetry. Shuffling time must destroy any temporal "
        "claim. Paired-time ridge is an upper attacker showing how easy the "
        "problem becomes once synchronization itself supplies correspondence."
    )

    path = ROOT / "results" / "fork_unlabeled_temporal_alignment.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("UNLABELED TEMPORAL ALIGNMENT SUMMARY")
    print(json.dumps(compact(result), indent=2))
    print("\nFull receipt:", path)


if __name__ == "__main__":
    main()
