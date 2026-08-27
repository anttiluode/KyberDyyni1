# Temporal separator scaling and signature crowding

Gate 13 showed that the downstream scalar-consequence stage can scale to rank 32 if measurements are allocated selectively.

Gate 14 attacks the upstream assumption:

> **What if the temporal separator can no longer tell the latent freedoms apart?**

## Deliberately Gaussian sources

This fork uses independent stationary Gaussian AR(1) components.

That removes marginal non-Gaussianity as a rescue route. At zero lag the source covariance is the identity, so PCA sees only the source **subspace**, not the individual source axes.

The only source identity comes from dynamics.

Each view is an independent run of the same latent process family rendered through its own random orthonormal basis.

There are no synchronized samples.

## Axes of attack

Ranks:

```text
4, 8, 16, 32
```

Observation windows:

```text
512, 2048, 4096
```

Observation noise:

```text
0.00, 0.10
```

Temporal-signature families:

### wide

AR coefficients span roughly 0.10 to 0.95.

### crowded

AR coefficients span only 0.65 to 0.90.

As rank rises, adjacent temporal fingerprints become increasingly similar.

### exact degenerate pair

Two central components are assigned exactly the same AR coefficient.

For Gaussian AR(1), those two processes are statistically identical under any orthogonal rotation inside their two-dimensional subspace.

No blind second-order algorithm is allowed to invent a unique pair of axes there.

## Attackers

- static PCA;
- one-lag AMUSE;
- a conservative multi-lag SOBI-like second-order operator.

The multi-lag method is intentionally simple. This gate is about the information boundary, not about claiming a new joint-diagonalization implementation.

## Metrics

The fork separates:

- source **subspace** recovery;
- individual source-axis recovery;
- component identity accuracy;
- oracle-sign transfer through the recovered cross-view axes;
- recovery of the two-dimensional degenerate subspace.

The oracle sign is used only for scoring transfer because Gate 11 already proved Gaussian sign orientation is unidentifiable from the blind streams.

## Required controls

### Shuffled time

Independent temporal shuffling preserves each view's zero-lag distribution but destroys lag structure.

AMUSE/SOBI source identity must collapse.

### Exact degeneracy

When two source time constants are identical:

- the pair's two-dimensional subspace may still be recovered;
- the individual axes should remain rotationally ambiguous.

If not, the experiment is cheating or measuring the wrong thing.

## Working hypothesis

The likely scaling quantity is not simply rank.

For adjacent AR coefficients separated by `delta rho` and a finite window `N`, the relevant difficulty should track something like:

```text
temporal signature gap
----------------------
finite-window estimation noise
```

The receipt therefore records both the minimum rho gap and `gap * sqrt(N)`.

Gate 14 should end with a boundary, not a heroic algorithm.


## Result — signature separation, not rank by itself, is the boundary

The fork gives a much cleaner scaling story than "high dimension is hard."

### Static covariance keeps the subspace and loses the freedoms

The strongest control is the shuffled-time run at rank 8 / 4096 samples.

```text
                         axis recovery   identity   subspace   transfer

PCA static                   0.266          14.1%      1.000       0%
AMUSE after time shuffle     0.276           7.8%      1.000       0%
multi-lag after shuffle      0.302           8.6%      1.000       0%
```

All methods still span the complete latent space because zero-lag covariance contains that subspace.

But once temporal order is destroyed, the individual freedoms disappear inside arbitrary rotations.

That is exactly the distinction this project has been circling for several days:

> **Knowing the latent subspace is not the same as knowing the coordinates that carry reusable freedoms.**

### Well-separated temporal signatures

With four broadly separated AR time constants and 2048 samples:

```text
                         axis recovery   identity   oracle-sign transfer

PCA                          0.414          23.4%            0.4%
AMUSE                        0.997         100.0%          100.0%
multi-lag                    0.997         100.0%          100.0%
```

At rank 8 / 2048 samples the component identities are still perfect, but the small axis errors now accumulate across more freedoms:

```text
AMUSE      axis 0.985   identity 100%   transfer 82.1%
multi-lag  axis 0.983   identity 100%   transfer 80.2%
```

At rank 16 / 4096 samples / observation noise 0.10:

```text
AMUSE      axis 0.944   identity 98.4%   transfer 10.4%
multi-lag  axis 0.939   identity 97.7%   transfer  7.9%
```

That is a useful warning. "Almost every component label is correct" is not enough for a reusable high-rank vector map. Small angular errors accumulate.

### Crowded temporal signatures

When the AR coefficients are squeezed into 0.65--0.90, the separator collapses much earlier.

```text
crowded signatures, 2048 samples

rank       AMUSE axis recovery   identity   transfer

8                0.880            93.8%       4.2%
16               0.527            48.4%       0.0%
```

At rank 32 / 4096 samples / noise 0.10:

```text
AMUSE      axis 0.370   identity 31.4%   transfer 0%
multi-lag  axis 0.355   identity 28.9%   transfer 0%
```

A useful rough dimensionless quantity is:

```text
minimum temporal-signature gap × sqrt(observation length)
```

For the crowded family:

```text
rank 8,  N=2048      ~1.62   -> identity 93.8%
rank 16, N=2048      ~0.75   -> identity 48.4%
rank 32, N=4096      ~0.52   -> identity 31.4%
```

By contrast, the broad rank-32 family at N=4096 is around 1.75 and retains about 80% identity.

This is not presented as a universal threshold, but it captures the direction of the boundary much better than rank alone.

### Exact degeneracy behaves correctly

The strongest scientific control is the pair with exactly equal AR coefficients.

At rank 4, AMUSE gives:

```text
samples       mean axis recovery   pair-subspace recovery

512                 0.863                0.993
2048                0.861                0.999
4096 + noise         0.849                0.999
```

The **two-dimensional pair subspace converges almost perfectly**, while the individual pair axes do not converge to unique true directions.

More data does not remove the ambiguity because there is no information available to remove it.

That is the result we required before running the experiment:

> **When two Gaussian components have identical temporal statistics, blind second-order separation can recover their joint subspace but not a privileged rotation inside it.**

### AMUSE beats the fancy attacker here

The conservative multi-lag SOBI-like operator does not earn a special role in this AR(1) world.

One-lag AMUSE is usually equal or slightly better.

That makes sense after seeing the result. For a pure AR(1) source, every later autocorrelation is just a power of the same coefficient. The extra lags are largely redundant and add finite-sample estimation variance.

This does **not** establish that full SOBI is generally worse. It says only:

> **When one lag already contains the identifying statistic, adding more lags need not help.**

## What Gate 14 earns

The cleanest surviving statement is:

> **Unlabeled temporal alignment works when latent freedoms have sufficiently distinct dynamical fingerprints relative to the amount and quality of observation. It fails smoothly as those fingerprints crowd, and it fails fundamentally inside exactly degenerate dynamical subspaces.**

The architecture has therefore acquired a real boundary condition.

Temporal structure is useful because it can turn an otherwise arbitrary latent rotation into identifiable freedoms.

But it cannot manufacture distinctions the process itself does not contain.

And high rank introduces a second practical problem: even when component identity is mostly right, many small axis errors can accumulate into a bad full-vector transfer map.

That is a good stopping point.

The next step should not be Gate 15.

It should be to ask what useful machine is actually implied by Gates 1--14.
