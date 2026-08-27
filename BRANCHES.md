# Branch archive and consolidation

`main` is the canonical KyberDyyni1 tree.

The experimental branches are intentionally retained as frozen provenance, but their useful code, receipts, and conclusions have now been copied into the canonical tree so the project no longer requires branch archaeology.

Consolidation source heads:

| branch | frozen head | status in main |
| --- | --- | --- |
| `control-law-fork` | `7f086f06093b8cf0be9dba2cb2cff88541dd33ae` | preserved |
| `structured-sampling-fork` | `ab8a17d2235796818393e4298eff029bf69e31af` | preserved |
| `two-dimensional-sampling-fork` | `e1fdc2044bf273836e7c9abac6e99838b2f5028b` | preserved |
| `high-dimensional-probe-fork` | `ece0e4ee0fce79f800f53988b6541362a73de875` | preserved |
| `probe-width-control-fork` | `f94942e0fa3fd64c3fa836cfb7d97475dd3c8909` | preserved |
| `slow-consolidation-noise-fork` | `114e05fdf7f426e978a97801915d95b64767ebb5` | merged as Gate 9 |
| `cross-basis-transfer-fork` | `3d31aad757d808c395fb739ca317458755e5e8b9` | merged as Gate 10 |
| `unlabeled-temporal-alignment-fork` | `0cc316c8718eccff35dd530749eaf0aed43060e8` | merged as Gate 11 |
| `sign-consequence-calibration-fork` | `07f8ba7399c457d6ff47b70b93b9d411b0f88977` | merged as Gate 12 |
| `rank-scaling-sign-fork` | `5049f6347b3205e2bf1bd7d3ec6e5f1fdfcb76c7` | merged as Gate 13 |
| `temporal-separator-scaling-fork` | `1f45f09d147e0058343c479f53a8f95412f86a7e` | merged as Gate 14 |

## Control-law line

Preserved narrative:

- [FORK.md](FORK.md)
- [FORK_CONCLUSION.md](FORK_CONCLUSION.md)

The corresponding `experiments/fork_*.py` and `results/fork_*.json` files were already present in the canonical Gate-6/7 tree before this consolidation.

## Structured-sampling line

Preserved narrative:

- [STRUCTURED_FORK.md](STRUCTURED_FORK.md)
- [STRUCTURED_CONCLUSION.md](STRUCTURED_CONCLUSION.md)

Key experiment families include persistent cross-cycle shuttles, structured sequences, and oriented phase addressing.

## Two-dimensional line

Preserved narrative:

- [TWO_D_FORK.md](TWO_D_FORK.md)
- [TWO_D_CONCLUSION.md](TWO_D_CONCLUSION.md)
- [GATE7.md](GATE7.md)

This line established the 2-D local probe-basis / fast-calibration result and the delayed slow-calibration result.

## High-dimensional line

Preserved narrative:

- [HIGHDIM_FORK.md](HIGHDIM_FORK.md)

Preserved experiment/receipt families:

- `fork_highdim_probe_scaling.py`
- `fork_highdim_equal_probe_budget.py`
- `fork_highdim_measurement_noise.py`
- matching committed result receipts

This line established the finite-probe-budget and consequence-SNR boundary.

## Probe-width line

Preserved narrative:

- [PROBE_WIDTH_FORK.md](PROBE_WIDTH_FORK.md)

Preserved experiment/receipt:

- `experiments/fork_probe_width_control.py`
- `results/fork_probe_width_control_summary.json`

This is a negative result: adaptive sweep width did not beat strong fixed-radius attackers in the stationary relevance landscape.

## Slow-consolidation line

Preserved narrative:

- [SLOW_CONSOLIDATION_FORK.md](SLOW_CONSOLIDATION_FORK.md)

Preserved experiment/receipt families:

- `experiments/fork_slow_consolidation_noise.py`
- `experiments/fork_slow_fast_handoff.py`
- `results/fork_slow_consolidation_summary.json`

This line established that repeated noisy fast corrections can be amortized into context-specific slow priors, but also exposed a fast/slow handoff failure: an exploratory controller tuned for large residuals can erase a good learned prior. Ordinary distance scaling plus stochastic accept/reject fixed that interaction; ordinary EMA/Kalman slow estimators were sufficient.

## Cross-basis transfer line

Preserved narrative:

- [CROSS_BASIS_FORK.md](CROSS_BASIS_FORK.md)

Preserved experiment/receipt:

- `experiments/fork_cross_basis_transfer.py`
- `results/fork_cross_basis_transfer_summary.json`

This line established an identifiability boundary for arbitrary unseen latent bases and showed that ordinary ridge/Procrustes alignment solves the low-rank case once enough cross-view calibration contexts relate the new coordinates to the learned reference.

## Unlabeled temporal-alignment line

Preserved narrative:

- [UNLABELED_ALIGNMENT_FORK.md](UNLABELED_ALIGNMENT_FORK.md)

Preserved experiment/receipt:

- `experiments/fork_unlabeled_temporal_alignment.py`
- `results/fork_unlabeled_temporal_alignment_summary.json`

This line established that independent unpaired streams can reveal cross-view component identity through temporal fingerprints and can reveal orientation when the dynamics contain a sign-sensitive temporal asymmetry. The Gaussian AR control recovers axes but leaves an irreducible sign ambiguity; shuffling time kills the positive result.

## Sign-consequence calibration line

Preserved narrative:

- [SIGN_CONSEQUENCE_FORK.md](SIGN_CONSEQUENCE_FORK.md)

Preserved experiment/receipt:

- `experiments/fork_sign_consequence_calibration.py`
- `results/fork_sign_consequence_calibration_summary.json`

This line established that the residual sign ambiguity left by blind temporal alignment can be resolved with ordinary scalar consequence. Simple bitwise evidence accumulation reaches the oracle transfer floor; shuffled consequence stays at chance.

## Rank-scaling sign-calibration line

Preserved narrative:

- [RANK_SCALING_FORK.md](RANK_SCALING_FORK.md)

Preserved experiment/receipt:

- `experiments/fork_rank_scaling_sign.py`
- `results/fork_rank_scaling_sign_summary.json`

This line established that a tiny informed consequence-probe budget can match or beat full O(R) sign probing at higher rank, with particularly strong gains under sparse and heavy-tailed excitation. The earned role is ordinary active measurement allocation, not a new learner.

## Temporal-separator scaling line

Preserved narrative:

- [TEMPORAL_SCALING_FORK.md](TEMPORAL_SCALING_FORK.md)

Preserved experiment/receipt:

- `experiments/fork_temporal_separator_scaling.py`
- `results/fork_temporal_separator_scaling_summary.json`

This line established the upstream boundary for unlabeled temporal alignment: distinct dynamical fingerprints can identify reusable source coordinates, crowded fingerprints fail as finite-window estimation noise approaches the signature gap, and exact equal Gaussian dynamics leave a rotationally ambiguous subspace. Time shuffling destroys source identity while preserving the latent subspace.

## Workflow files

The branch-specific GitHub Actions workflows are also retained under `.github/workflows/` as executable provenance. Their branch filters remain branch-specific; the canonical `tests.yml` is the main-line CI.

## Rule going forward

Branches may still be used to isolate risky attacks, but a finished fork should leave three things on `main`:

1. the experiment code;
2. a compact committed receipt;
3. the conclusion, including negative results and kill conditions.

That keeps branching useful without fragmenting the research record.
