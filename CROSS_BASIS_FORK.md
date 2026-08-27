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


## Result — arbitrary new bases need a relation signal

The zero-calibration result is decisive:

```text
unseen view, zero calibration

blind shared rendered table     error 0.684   success 0%
explicit context x view table   error 0.600   success 0%
ordinary ridge                  error 0.600   success 0%
oracle exact basis              error 0.014   success 100%
```

A raw slow memory averaged across incompatible coordinate systems is worse than doing nothing. An explicit view table cannot invent a row for a view it has never seen.

The oracle says the underlying information is absolutely reusable if the coordinate relation is known.

So the failure is not "memory cannot transfer." It is an **identifiability problem**:

> **An arbitrary unseen basis cannot be decoded from no relation signal at all.**

## Calibration attack

Give the new view a few contexts whose identity is shared with the learned reference, fit an ordinary linear map, then test on the remaining contexts.

The hidden correction family has rank 4.

```text
paired calibration contexts     ridge error    success    cosine

0                                  0.600         0.0%      0.000
2                                  0.358        11.7%      0.736
3                                  0.247        37.3%      0.879
4                                  0.173        63.4%      0.942
6                                  0.108        90.3%      0.981
8                                  0.088        97.9%      0.989

oracle exact basis                0.014       100.0%      1.000
```

Orthogonal Procrustes also works, but ordinary ridge is better under the noisy finite calibration used here:

```text
K=4   Procrustes success 42.8%
K=6                      83.0%
K=8                      95.1%
```

The transition occurs exactly where one would expect a matrix method to become plausible: around the point where the calibration examples begin to span the rank-4 correction family.

A simpler attacker using only view 0 as the reference performs essentially identically to the more elaborate alignment of all three training views. So even the extra multi-view machinery is not needed for this toy.

## What Gate 10 earns

It does **not** earn a novel alignment algorithm.

It earns a clean boundary:

> **Slow knowledge can transfer across changing latent bases, but only after the machine obtains enough information to relate the coordinate systems. In a low-rank correction family, a small set of cross-view correspondences lets ordinary linear alignment re-render the learned correction into an unseen basis.**

This is the first place the current fast/slow line reconnects directly to the earlier matrix/source-separation work.

The architecture now looks like:

```text
        recurring hidden freedom / correction
                       |
              slow reusable memory
                       |
            coordinate relation / alignment
                       |
             current rendered latent basis
                       |
             cautious fast residual search
```

The remaining question is more interesting than adding another optimizer:

> **Can the coordinate relation itself be discovered from structure in the signals, without being handed explicit paired context labels?**

That is the IVA/SOBI/temporal-alignment shaped question.

If ordinary PCA/CCA/Procrustes after weak unsupervised matching solves it, use them. If temporal dependence across views is enough to align changing latent bases without labels, then the Tuesday matrix line and KyberDyyni fast/slow line have actually met.
