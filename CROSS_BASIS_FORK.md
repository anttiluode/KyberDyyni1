# Cross-basis transfer fork

Gate 9 showed that repeated noisy fast corrections can become cheap context-specific slow priors. But it still gave each context one fixed coordinate system.

This fork attacks that loophole.

## World

There is a rank-4 family of hidden context corrections embedded in a 32-D observed latent space.

Each view renders the same four-dimensional correction family through a different random orthonormal basis.

```text
hidden correction coefficients z_c
             |
             v
      view basis B_v
             |
             v
 observed correction B_v z_c
```

The fast stage is deliberately collapsed to a noisy correction packet so the coordinate-transfer question is isolated from the Gate-8/9 search mechanics.

Three views are seen during training. A fourth basis is held out completely.

## Question

> **Can slow information learned under changing latent bases be reused in an unseen basis?**

## Attackers

- one raw shared context table averaging incompatible rendered coordinates;
- explicit context x view table, which has no row for the unseen view;
- oracle exact basis knowledge;
- ordinary ridge alignment between views;
- ordinary orthogonal Procrustes alignment;
- a simpler view-0-only ridge calibration attacker.

The unseen view is given 0, 2, 3, 4, 6, or 8 calibration contexts. The remaining contexts are held out for transfer evaluation.

The shared correction family has rank 4, so there is a sharp matrix question hiding here:

> Does ordinary cross-view calibration begin to work once the calibration set spans the underlying correction subspace?

## Kill condition

If ordinary ridge/Procrustes alignment solves transfer after a small calibration set, no novel cross-basis mechanism has been earned.

If nothing transfers without exact basis knowledge, then arbitrary unseen coordinate changes impose an identifiability boundary rather than an architectural failure.
