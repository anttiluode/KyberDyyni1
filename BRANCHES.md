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

## Workflow files

The branch-specific GitHub Actions workflows are also retained under `.github/workflows/` as executable provenance. Their branch filters remain branch-specific; the canonical `tests.yml` is the main-line CI.

## Rule going forward

Branches may still be used to isolate risky attacks, but a finished fork should leave three things on `main`:

1. the experiment code;
2. a compact committed receipt;
3. the conclusion, including negative results and kill conditions.

That keeps branching useful without fragmenting the research record.
