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
